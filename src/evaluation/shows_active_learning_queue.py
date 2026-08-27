"""
Shows Active-Learning Queue  ("what should I watch next?")
==========================================================

The TV counterpart of src/evaluation/movie_active_learning_queue.py. It pulls your
Serializd **watchlist** (unwatched candidate shows), runs each through the deployed
TV feature pipeline + ensemble, and ranks them by how much *rating* one would teach
the model.

Two signals, combined (identical scheme to the movie queue):

  1. UNCERTAINTY — query-by-committee disagreement. The shows ensemble has diverse
     base learners (XGB, SVR, CatBoost, an ordinal classifier). For a show they
     agree on, your rating mostly confirms what the model knows; where they disagree
     (high std), your true rating resolves the conflict. Uncertainty = std of the
     base-model predictions.
  2. NOVELTY — cosine distance in the MiniLM→PCA text space to the 5 nearest already
     rated shows. A show unlike anything you've rated covers new taste-space.

Composite priority = z(uncertainty) + z(novelty). High => maximally informative.

Outputs
-------
- reports/SHOW_ACTIVE_LEARNING_QUEUE.md
- data/predictions/shows/watchlist_active_learning_queue.csv

Run:  python -m src.evaluation.shows_active_learning_queue
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.serializd_sync import make_client
from src.shows.ingestion import get_tv_show_metadata_ultimate
from src.shows.predict_ratings import safe_eval, get_primary_network, parse_people

import os
from dotenv import load_dotenv
load_dotenv()

QUEUE_CSV_PATH = config.PREDICTIONS_DIR / "shows" / "watchlist_active_learning_queue.csv"
QUEUE_MD_PATH = config.BASE_DIR / "reports" / "SHOW_ACTIVE_LEARNING_QUEUE.md"

# Base learners whose disagreement defines uncertainty (stacking is the meta-model,
# a deterministic function of the others — a point estimate, not a committee vote).
BASE_LEARNER_FILES = {
    "XGB": "xgb_base_regressor.joblib",
    "SVR": "svr_base_regressor.joblib",
    "CatBoost": "catboost_base_regressor.joblib",
}
BUCKET_MAP = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}


# --------------------------------------------------------------------------- #
# Pool: Serializd watchlist
# --------------------------------------------------------------------------- #
def fetch_watchlist(refresh=False):
    """Candidate pool = the persisted Serializd watchlist (data/raw/shows/watchlist.csv).

    Reads the file written by src/shows/pull_watchlist.py; pulls live from Serializd
    and (re)writes it when the file is missing or refresh=True.
    """
    from src.shows.pull_watchlist import WATCHLIST_PATH, fetch_watchlist_rows
    if refresh or not WATCHLIST_PATH.exists():
        rows = fetch_watchlist_rows()
        wdf = pd.DataFrame(rows).dropna(subset=["showId"]).drop_duplicates(subset=["showId"])
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        wdf.to_csv(WATCHLIST_PATH, index=False)
    else:
        wdf = pd.read_csv(WATCHLIST_PATH)
    out = {int(r["showId"]): r["showName"] for _, r in wdf.iterrows() if pd.notna(r["showId"])}

    # Exclude shows already watched/rated (in the enriched CSV). A show can sit on
    # both the Serializd watchlist and watched list; the pool must be unwatched only.
    if config.TV_SHOWS_ENRICHED_DATA_PATH.exists():
        en = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        watched = {int(float(x)) for x in en["tmdb_id"].dropna()}
        before = len(out)
        out = {sid: nm for sid, nm in out.items() if sid not in watched}
        if before - len(out):
            print(f"   Excluded {before - len(out)} already-watched shows "
                  f"({len(out)} unwatched candidates remain).")
    return out


def enrich_pool(watchlist, limit=None):
    items = list(watchlist.items())
    if limit:
        items = items[:limit]
    rows = []
    for sid, name in items:
        try:
            meta = get_tv_show_metadata_ultimate(str(name), None, tmdb_id=sid)
        except Exception as e:  # noqa: BLE001
            print(f"   ⚠️ enrich failed {name} ({sid}): {e}")
            continue
        meta["tmdb_id"] = meta.get("tmdb_id") or sid
        meta["name"] = meta.get("name") or name
        # get_tv_show_metadata_ultimate returns some fields as lists; the featurizer
        # (mirroring predict_ratings on the CSV) expects their string-repr form.
        for col in ("genres", "created_by", "actors", "writer", "production_companies"):
            if isinstance(meta.get(col), list):
                meta[col] = str(meta[col])
        rows.append(meta)
        time.sleep(0.15)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Featurization — mirrors src/shows/predict_ratings.py exactly
# --------------------------------------------------------------------------- #
def build_features(df, state):
    df = df.copy()
    med = state.get("median_values", {})
    df["vote_count_log"] = np.log1p(pd.to_numeric(df.get("vote_count"), errors="coerce").fillna(0))
    df["vote_average"] = pd.to_numeric(df.get("vote_average"), errors="coerce").fillna(med.get("vote_average"))
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").fillna(med.get("year"))
    df["season_count"] = pd.to_numeric(df.get("number_of_seasons"), errors="coerce").fillna(1)
    df["imdb_rating"] = pd.to_numeric(df.get("imdb_rating"), errors="coerce").fillna(med.get("imdb_rating"))
    df["imdb_votes_log"] = np.log1p(pd.to_numeric(df.get("imdb_votes"), errors="coerce").fillna(0))
    df["runtime"] = pd.to_numeric(df.get("runtime"), errors="coerce").fillna(med.get("runtime"))
    df["episode_count"] = pd.to_numeric(df.get("number_of_episodes"), errors="coerce").fillna(med.get("episode_count"))
    awards = df.get("awards", pd.Series("", index=df.index)).astype(str)
    df["total_wins"] = awards.str.extract(r"(\d+)\s+win", flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df["total_nominations"] = awards.str.extract(r"(\d+)\s+nomination", flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df["is_adult"] = df.get("age_rating", pd.Series("", index=df.index)).apply(
        lambda x: 1 if str(x) in ["TV-MA", "R", "18+"] else 0)

    df["primary_network"] = df.get("network", pd.Series("", index=df.index)).apply(get_primary_network)
    df["network_clean"] = df["primary_network"].apply(lambda x: x if x in state["top_networks"] else "Other")
    top_ctry = state.get("top_countries", [])
    df["primary_country"] = (df.get("country", pd.Series("", index=df.index)).fillna("").astype(str)
                             .str.replace(r"[\[\]'\"<>]", "", regex=True)
                             .str.split(",").str[0].str.strip())
    df["country_clean"] = df["primary_country"].apply(lambda x: x if x in top_ctry else "Other")
    for c in top_ctry:
        df[f"ctry_{c}"] = (df["country_clean"] == c).astype(int)
    df["ctry_Other"] = (df["country_clean"] == "Other").astype(int)

    df["genres_list"] = df.get("genres", pd.Series("", index=df.index)).apply(safe_eval)
    for col_name in state["kept_genres"]:
        raw_genre = col_name.replace("g_", "")
        df[col_name] = df["genres_list"].apply(lambda x: 1 if raw_genre in x else 0)
    for net in state["top_networks"]:
        df[f"net_{net}"] = (df["network_clean"] == net).astype(int)
    if "net_Other" not in df.columns:
        df["net_Other"] = (df["network_clean"] == "Other").astype(int)

    pca_vec = None
    if "pca" in state and "pca_cols" in state:
        from sentence_transformers import SentenceTransformer
        from src.shows.feature_engineering import _show_header, _show_body
        from src.common.text_embedding import encode_pooled
        from src.shows.season_data import attach_season_narrative
        # Same chunk-pooled recipe the show PCA in `state` was fit on, season
        # narrative included.
        df = attach_season_narrative(df)
        _records = df.to_dict("records")
        _headers = [_show_header(r) for r in _records]
        _bodies = [_show_body(r) for r in _records]
        stm = SentenceTransformer(state.get("sentence_transformer", "all-MiniLM-L6-v2"))
        emb = encode_pooled(_headers, _bodies, stm, show_progress_bar=False)
        pca_vec = state["pca"].transform(emb)
        pca_df = pd.DataFrame(pca_vec, columns=state["pca_cols"], index=df.index)
        df = pd.concat([df, pca_df], axis=1)

    # People target-encodings, same as src/shows/predict_ratings.py. The trainer
    # fits these OOF and persists the global smoothed maps in state so serving
    # can rebuild the columns; without this the reindex below raises KeyError.
    te = state.get("target_encodings", {})
    if te:
        g = te.get("global_mean", 3.5)

        def encode(raw_col, map_key):
            mapping = te.get(map_key, {})
            raw = df.get(raw_col, pd.Series("", index=df.index))
            return raw.apply(parse_people).apply(
                lambda items: float(np.mean([mapping.get(i, g) for i in items]))
                if items else g)

        df["creator_encoded"] = encode("created_by", "creator")
        df["actors_encoded"] = encode("actors", "actors")
        df["writer_encoded"] = encode("writer", "writer")

    X = df[state["features_columns"]].fillna(0)
    return df, X, pca_vec


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def zscore(s):
    std = s.std()
    return pd.Series(0.0, index=s.index) if std < 1e-9 else (s - s.mean()) / std


def align_to_model(X, model):
    m = getattr(model, "clf", model)
    names = None
    if hasattr(m, "feature_names_in_"):
        names = list(m.feature_names_in_)
    elif hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                names = list(step.feature_names_in_)
                break
    if not names:
        try:
            names = list(m.get_booster().feature_names)
        except Exception:  # noqa: BLE001
            names = None
    if not names and hasattr(m, "feature_names_") and m.feature_names_:
        names = list(m.feature_names_)
    return X.reindex(columns=names, fill_value=0) if names else X


def score(df, X, pca_vec):
    ens = config.TV_SHOWS_MODEL_DIR / "ensemble"
    base = {}
    for name, fname in BASE_LEARNER_FILES.items():
        p = ens / fname
        if p.exists():
            base[name] = np.asarray(joblib.load(p).predict(align_to_model(X, joblib.load(p))), dtype=float)
    # Ordinal committee member: expected value from class probabilities.
    ord_p, ord_cls = ens / "ordinal_classifier.joblib", ens / "ordinal_classes.joblib"
    if ord_p.exists() and ord_cls.exists():
        oc = joblib.load(ord_p)
        classes = joblib.load(ord_cls)
        bucket_vals = np.array([BUCKET_MAP[c] for c in classes])
        probs = oc.predict_proba(align_to_model(X, oc))
        base["Ordinal_EV"] = np.sum(probs * bucket_vals, axis=1)

    if len(base) < 2:
        raise RuntimeError("Need >=2 base learners for a disagreement signal.")
    mat = np.column_stack(list(base.values()))
    df["uncertainty"] = mat.std(axis=1)
    for name, preds in base.items():
        df[f"pred_{name}"] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2

    stack = ens / "stacking_ensemble_regressor.joblib"
    if stack.exists():
        sm = joblib.load(stack)
        raw = np.asarray(sm.predict(align_to_model(X, sm)), dtype=float)
    else:
        raw = mat.mean(axis=1)
    df["predicted_rating"] = np.round(np.clip(raw, 0.5, 5.0) * 2) / 2

    # Novelty: cosine distance to nearest rated shows in PCA text space.
    if pca_vec is not None:
        train = pd.read_csv(config.TV_SHOWS_TRAINING_DATA_PATH)
        cols = [c for c in state_pca_cols if c in train.columns]
        if len(cols) >= 2:
            knn = NearestNeighbors(n_neighbors=min(5, len(train)), metric="cosine")
            knn.fit(train[cols].fillna(0).values)
            dist, _ = knn.kneighbors(pca_vec[:, : len(cols)])
            df["novelty"] = dist.mean(axis=1)
        else:
            df["novelty"] = 0.0
    else:
        df["novelty"] = 0.0

    df["uncertainty_z"] = zscore(df["uncertainty"])
    df["novelty_z"] = zscore(df["novelty"])
    df["priority_score"] = df["uncertainty_z"] + df["novelty_z"]

    def why(r):
        bits = []
        if r["uncertainty_z"] > 0.5:
            bits.append("models disagree")
        if r["novelty_z"] > 0.5:
            bits.append("unlike your rated shows")
        if bits:
            return " + ".join(bits)
        if r["uncertainty_z"] < -0.5 and r["novelty_z"] < -0.5:
            return "low info (confident + familiar)"
        return "moderate"

    df["why"] = df.apply(why, axis=1)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


def write_outputs(ranked):
    out = ranked.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    cols = (["rank", "name", "year", "predicted_rating"] + pred_cols
            + ["uncertainty", "novelty", "priority_score", "why"])
    cols = [c for c in cols if c in out.columns]
    QUEUE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[cols].round({"uncertainty": 4, "novelty": 4, "priority_score": 4}).to_csv(QUEUE_CSV_PATH, index=False)

    show = out[["rank", "name", "year", "predicted_rating", "uncertainty", "novelty", "priority_score", "why"]].copy()
    show = show.round({"uncertainty": 3, "novelty": 3, "priority_score": 3})
    with open(QUEUE_MD_PATH, "w", encoding="utf-8") as f:
        f.write("# TV Shows — Active Learning: What to Watch Next\n\n")
        f.write(f"Your Serializd **watchlist** ({len(out)} unwatched shows), ranked by "
                "**information gain** = z(uncertainty) + z(novelty). *Uncertainty* is the "
                "disagreement (std) across the TV ensemble's base learners (XGB, SVR, CatBoost, "
                "ordinal); *novelty* is the cosine distance from each show to your 5 nearest rated "
                "shows in the MiniLM→PCA text space. Rating the top shows first teaches the TV "
                "model the most per watch. `predicted_rating` is the stacking ensemble's guess "
                "(0.5★ grid) — informational, not the ranking key.\n\n")
        f.write(show.to_markdown(index=False))
        f.write("\n")
    print(f"\n✅ Shows watch-next queue ({len(out)} shows) saved:\n   {QUEUE_CSV_PATH}\n   {QUEUE_MD_PATH}")


state_pca_cols = []  # populated in main()


def main():
    global state_pca_cols
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--refresh", action="store_true",
                    help="Re-pull the watchlist from Serializd before scoring.")
    args = ap.parse_args()

    print("🎯 Building shows active-learning queue from the Serializd watchlist...")
    if not config.TV_SHOWS_PREPROCESSOR_STATE.exists():
        sys.exit(f"❌ Missing {config.TV_SHOWS_PREPROCESSOR_STATE}")
    state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    state_pca_cols = state.get("pca_cols", [])

    watch = fetch_watchlist(refresh=args.refresh)
    print(f"   {len(watch)} watchlist shows fetched.")
    if not watch:
        sys.exit("❌ Empty Serializd watchlist — nothing to queue.")

    df = enrich_pool(watch, limit=args.limit)
    if df.empty:
        sys.exit("❌ No watchlist shows could be enriched.")
    print(f"   Scoring {len(df)} shows...")

    df, X, pca_vec = build_features(df, state)
    ranked = score(df, X, pca_vec)
    write_outputs(ranked)


if __name__ == "__main__":
    main()
