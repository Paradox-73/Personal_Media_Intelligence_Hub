"""
Movie Active-Learning Queue
===========================

Ranks the films on your Letterboxd *watchlist* by how much rating them would
help the movie rating-model improve -- i.e. "what should I watch next?".

The idea (pool-based active learning)
-------------------------------------
The model has already learned from ~1000 rated films. The watchlist is a pool
of *unlabeled* candidates. Watching + rating a film gives the model one new
training label. Some labels teach it a lot; some teach it almost nothing.

We score each candidate on two signals and combine them:

1. UNCERTAINTY  (a.k.a. "Query-by-Committee" disagreement)
   We have an ensemble of diverse base models (XGB, SVR, CatBoost, MLP, an
   ordinal classifier). For a film they *agree* on, the model is already
   confident -- your rating would mostly confirm what it knows, so it learns
   little. For a film they *disagree* on (high std across their predictions),
   the model is genuinely torn; your true rating resolves that conflict and
   moves the decision boundary. Uncertainty = std of the base-model predictions.

2. NOVELTY  (representativeness / coverage)
   Uncertainty alone chases noisy edge cases. Novelty asks: does this film sit
   in a region of taste-space the model has *few examples of*? We embed each
   film (title + director + plot -> sentence-transformer -> PCA) and measure the
   cosine distance to its 5 nearest already-rated neighbours. A film far from
   everything you've rated covers new ground; a near-duplicate of films you've
   already rated adds little.

Composite priority = z(uncertainty) + z(novelty), computed within the movie
domain. High score => watching it is maximally informative. Low score => the
model can already predict it confidently *and* it looks like things you've
rated, so labelling it barely changes the model.

Why not just "expected rating"? Because a film you'll obviously love (high
predicted rating, low uncertainty) is a great *recommendation* but a poor
*training example* -- the model already knows. Active learning optimises for
information gain, not enjoyment. (See the note at the bottom about mixing in
enjoyment if you'd rather not sit through informative-but-mediocre films.)

Outputs
-------
- reports/MOVIE_ACTIVE_LEARNING_QUEUE.md   (human-readable ranked queue)
- data/predictions/movies/watchlist_active_learning_queue.csv

Enrichment (TMDB/OMDB) is cached to data/cache/watchlist_enriched.json so
re-runs are cheap; only newly-added watchlist films hit the APIs.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# Project root on path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.ingestion import get_movie_metadata, clean_year
from src.movies.feature_engineering import transform_single_movie
# Needed so joblib can unpickle the ordinal / stacking estimators.
from src.movies.advanced_movie_model_trainer import (
    OrdinalExpectedValueRegressor, MeanMetaRegressor, AnchoredRegressor)  # noqa: F401

WATCHLIST_PATH = config.MOVIES_RAW_DIR / "watchlist.csv"
ENRICH_CACHE_PATH = config.CACHE_DIR / "watchlist_enriched.json"
QUEUE_CSV_PATH = config.MOVIES_PREDICTIONS_DIR / "watchlist_active_learning_queue.csv"
QUEUE_MD_PATH = config.BASE_DIR / "reports" / "MOVIE_ACTIVE_LEARNING_QUEUE.md"

# Base learners whose disagreement defines epistemic uncertainty. All expose a
# uniform .predict(X) -> rating-scale float (the ordinal one returns its
# expected value). The stacking meta-model is the point estimate, NOT a
# committee member (it's a deterministic function of the others).
BASE_LEARNER_FILES = {
    "XGB": "xgb_base_regressor.joblib",
    "SVR": "svr_base_regressor.joblib",
    "CatBoost": "catboost_base_regressor.joblib",
    "MLP": "mlp_base_regressor.joblib",
    "Ordinal_EV": "ordinal_classifier.joblib",
}


# --------------------------------------------------------------------------- #
# Enrichment
# --------------------------------------------------------------------------- #

def load_cache():
    if ENRICH_CACHE_PATH.exists():
        try:
            return json.loads(ENRICH_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache):
    ENRICH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENRICH_CACHE_PATH.write_text(json.dumps(cache, default=str, indent=0),
                                 encoding="utf-8")


def _stringify(val, sep=", "):
    if isinstance(val, (list, tuple)):
        return sep.join(str(v) for v in val)
    if val is None:
        return ""
    return str(val)


def _watched_keys():
    """Keys (letterboxd_uri + title/year) for films already watched/rated.

    Letterboxd leaves a film on your watchlist after you watch it, so the raw
    watchlist contains already-seen films. The active-learning pool must be
    *unlabelled* candidates only — exclude anything already in the enriched CSV.
    """
    uris, ty = set(), set()
    if config.MOVIES_ENRICHED_DATA_PATH.exists():
        en = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        uris = {str(u).strip() for u in en.get("letterboxd_uri", pd.Series(dtype=str)).dropna()}
        for _, r in en.iterrows():
            for nm in (r.get("title"), r.get("letterboxd_name")):
                if pd.notna(nm):
                    ty.add((str(nm).strip().lower(), clean_year(r.get("year"))))
    return uris, ty


def enrich_watchlist(limit=None, no_fetch=False):
    """Return a DataFrame of enriched watchlist films (cached)."""
    wl = pd.read_csv(WATCHLIST_PATH)

    watched_uris, watched_ty = _watched_keys()
    before = len(wl)
    keep = []
    for _, row in wl.iterrows():
        uri = str(row.get("Letterboxd URI", "")).strip()
        key = (str(row.get("Name", "")).strip().lower(), clean_year(row.get("Year")))
        keep.append(not ((uri and uri in watched_uris) or key in watched_ty))
    wl = wl[keep].reset_index(drop=True)
    dropped = before - len(wl)
    if dropped:
        print(f"   Excluded {dropped} already-watched films from the watchlist pool "
              f"({len(wl)} unwatched candidates remain).")

    if limit:
        wl = wl.head(limit)

    cache = load_cache()
    rows = []
    fetched = 0
    for i, row in wl.iterrows():
        uri = str(row.get("Letterboxd URI", "")).strip()
        name = str(row.get("Name", "")).strip()
        year = clean_year(row.get("Year"))
        key = uri or f"{name} ({year})"

        meta = cache.get(key)
        if meta is None and not no_fetch:
            try:
                meta = get_movie_metadata(name, year)
                fetched += 1
                time.sleep(0.05)  # be gentle with TMDB rate limits
            except Exception as e:  # noqa: BLE001
                meta = {"processing_status": "error", "error": str(e)}
            cache[key] = meta
            if fetched % 25 == 0:
                save_cache(cache)
                print(f"   ...enriched {fetched} new films")

        if not meta:
            continue
        meta = dict(meta)
        meta.setdefault("letterboxd_name", name)
        meta["letterboxd_uri"] = uri
        # Fall back to the watchlist year/title if the API missed.
        meta["year"] = meta.get("year") or year
        meta["title"] = meta.get("title") or name
        rows.append(meta)

    save_cache(cache)
    if fetched:
        print(f"✅ Enriched {fetched} new films (rest from cache).")

    df = pd.DataFrame(rows)
    # Normalise list-valued fields to strings for the featurizer.
    for col in ("director", "actors", "writer", "production", "genre",
                "language", "country"):
        if col in df.columns:
            df[col] = df[col].apply(_stringify)
    # Drop films we couldn't resolve at all (no plot/overview to embed).
    df["plot"] = df.get("plot", pd.Series("", index=df.index)).fillna("")
    df["overview"] = df.get("overview", pd.Series("", index=df.index)).fillna("")
    df["_has_text"] = (df["plot"].str.len() + df["overview"].str.len()) > 0
    resolved = df[df["_has_text"]].copy().reset_index(drop=True)
    unresolved = df[~df["_has_text"]].copy()
    if len(unresolved):
        print(f"⚠️  {len(unresolved)} watchlist films could not be resolved "
              f"(no TMDB/OMDB match) and are skipped.")
    return resolved, unresolved


# --------------------------------------------------------------------------- #
# Featurization (the same per-row builder as the Oracle and batch scorer)
# --------------------------------------------------------------------------- #

def build_features(df, state):
    X_final = pd.concat([transform_single_movie(r, state) for r in df.to_dict("records")],
                        ignore_index=True)
    X_final.index = df.index
    pca_cols = [c for c in X_final.columns if c.startswith("pca_")]
    return X_final, X_final[pca_cols].to_numpy()


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def zscore(s):
    std = s.std()
    if std < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std


def expected_feature_names(model):
    """Best-effort recovery of the exact column names a model was fit on.

    Handles schema drift: the SVR/MLP pipelines and the older base learners can
    expect a different column set than the current preprocessor_state. We align
    to each model's own expectation rather than assuming one shared schema.
    """
    m = getattr(model, "clf", model)  # unwrap OrdinalExpectedValueRegressor
    if hasattr(m, "feature_names_in_"):
        return list(m.feature_names_in_)
    if hasattr(m, "named_steps"):  # sklearn Pipeline
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    try:
        bn = m.get_booster().feature_names  # xgboost
        if bn:
            return list(bn)
    except Exception:  # noqa: BLE001
        pass
    if hasattr(m, "feature_names_") and m.feature_names_:  # catboost
        return list(m.feature_names_)
    return None


def align_to_model(X, model):
    names = expected_feature_names(model)
    if not names:
        return X
    return X.reindex(columns=names, fill_value=0)


def score_queue(df, X_final, pca_vec):
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"

    # 1. Committee disagreement -> uncertainty
    base_preds = {}
    for name, fname in BASE_LEARNER_FILES.items():
        path = ensemble_dir / fname
        if not path.exists():
            print(f"⚠️  Missing base learner {name} ({path}); skipping it.")
            continue
        model = joblib.load(path)
        base_preds[name] = np.asarray(model.predict(align_to_model(X_final, model)), dtype=float)
    if len(base_preds) < 2:
        raise RuntimeError("Need >=2 base learners for a disagreement signal.")
    pred_matrix = np.column_stack(list(base_preds.values()))
    df["uncertainty"] = pred_matrix.std(axis=1)

    # Keep each base learner's own estimate (fun to compare against your rating).
    for bl_name, preds in base_preds.items():
        df[f"pred_{bl_name}"] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2

    # Point estimate from the stacking meta-model (informational only).
    stack_path = ensemble_dir / "stacking_ensemble_regressor.joblib"
    if stack_path.exists():
        stack = joblib.load(stack_path)
        raw = np.asarray(stack.predict(align_to_model(X_final, stack)), dtype=float)
    else:
        raw = pred_matrix.mean(axis=1)
    df["predicted_rating"] = np.round(np.clip(raw, 0.5, 5.0) * 2) / 2

    # 2. Novelty -> cosine distance to nearest rated films in embedding space
    train = pd.read_csv(config.TRAINING_DATA_PATH)
    n = pca_vec.shape[1]
    train_pca_cols = [f"pca_{i}" for i in range(n) if f"pca_{i}" in train.columns]
    if len(train_pca_cols) >= 2:
        X_train = train[train_pca_cols].fillna(0).values
        knn = NearestNeighbors(n_neighbors=min(5, len(X_train)), metric="cosine")
        knn.fit(X_train)
        dist, _ = knn.kneighbors(pca_vec[:, : len(train_pca_cols)])
        df["novelty"] = dist.mean(axis=1)
    else:
        print("⚠️  No PCA columns in training data; novelty flat.")
        df["novelty"] = 0.0

    # 3. Composite priority
    df["uncertainty_z"] = zscore(df["uncertainty"])
    df["novelty_z"] = zscore(df["novelty"])
    df["priority_score"] = df["uncertainty_z"] + df["novelty_z"]

    def reason(r):
        bits = []
        if r["uncertainty_z"] > 0.5:
            bits.append("models disagree")
        if r["novelty_z"] > 0.5:
            bits.append("unlike your rated films")
        if not bits:
            if r["uncertainty_z"] < -0.5 and r["novelty_z"] < -0.5:
                return "low info (confident + familiar)"
            return "moderate"
        return " + ".join(bits)

    df["why"] = df.apply(reason, axis=1)
    return df.sort_values("priority_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_outputs(ranked, top_n=None):
    """Write the FULL watch-next ranking (top-to-bottom) to a single CSV."""
    out = ranked.copy()
    out.insert(0, "rank", range(1, len(out) + 1))
    out["film"] = out["title"].fillna(out["letterboxd_name"])

    pred_cols = [c for c in out.columns if c.startswith("pred_")]
    cols = (["rank", "film", "year", "predicted_rating"] + pred_cols
            + ["uncertainty", "novelty", "priority_score", "why", "letterboxd_uri"])
    cols = [c for c in cols if c in out.columns]

    QUEUE_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[cols].round({"uncertainty": 4, "novelty": 4, "priority_score": 4}) \
        .to_csv(QUEUE_CSV_PATH, index=False)

    print(f"\n✅ Full ranked watch-next queue ({len(out)} films) saved:\n"
          f"   {QUEUE_CSV_PATH}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    try:  # emoji-safe on Windows cp1252 consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description="Movie active-learning queue")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N watchlist films (for testing).")
    ap.add_argument("--no-fetch", action="store_true",
                    help="Use cached enrichment only; do not call TMDB/OMDB.")
    ap.add_argument("--top", type=int, default=25,
                    help="How many rows to show in the markdown report.")
    args = ap.parse_args()

    print("🎯 Building movie active-learning queue...")
    if not config.PREPROCESSOR_STATE.exists():
        sys.exit(f"❌ Preprocessor state missing: {config.PREPROCESSOR_STATE}")
    state = joblib.load(config.PREPROCESSOR_STATE)

    df, _unresolved = enrich_watchlist(limit=args.limit, no_fetch=args.no_fetch)
    if df.empty:
        sys.exit("❌ No resolvable watchlist films to score.")
    print(f"   Scoring {len(df)} films...")

    X_final, pca_vec = build_features(df, state)
    ranked = score_queue(df, X_final, pca_vec)
    write_outputs(ranked)


if __name__ == "__main__":
    main()
