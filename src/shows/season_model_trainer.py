"""Season-level TV rating model — a SECOND model, alongside the show-level one.

The show-level model in models/shows/ predicts one rating per show. But 71% of
rated shows are multi-season and your rating moves season to season (Arcane 5.0 ->
4.5, Arrested Development 5.0 -> 3.5), so a single show rating is a noisy average.
This model predicts YOUR rating for an individual season.

It trains ONLY on seasons you actually watched (season_rating present and the
season has aired). Anthology shows are kept -- each of their seasons is a separate
story, which is exactly the signal this model is meant to capture.

Features are per-season, not per-show:
  season position   season_number, seasons_so_far, is_final_season, air_year, gap_since_prev
  size              episode_count
  crowd             tmdb_season_vote, imdb_season_mean/min/max, imdb_episode_spread,
                    imdb_delta_vs_prev   (did the crowd think quality moved?)
  churn             cast_overlap_prev, cast_turnover, cast_new_count, crew_change
  your history      prev_season_rating, show_mean_so_far  (leakage-safe: strictly
                    earlier seasons of the same show only)
  show context      genre multi-hot, language, network, runtime
  text              season_overview + top_cast + key_crew -> MiniLM -> PCA

Artifacts (NOTHING under models/shows/ is overwritten):
  models/shows/season/season_regressor.joblib
  models/shows/season/season_preprocessor_state.pkl
  reports/season_model_metrics.json

Run: python -m src.shows.season_model_trainer
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.season_data import load_seasons

OUT_DIR = config.TV_SHOWS_MODEL_DIR / "season"
MODEL_PATH = OUT_DIR / "season_regressor.joblib"
STATE_PATH = OUT_DIR / "season_preprocessor_state.pkl"
METRICS_PATH = config.BASE_DIR / "reports" / "season_model_metrics.json"

PCA_COMPONENTS = 12

# CONTENT features only — what the season IS, exactly like the movie model works
# from a film's own metadata. Season plot/cast/crew differ season to season; that
# is the signal this model is supposed to read.
NUMERIC = [
    "season_number", "seasons_so_far", "is_final_season", "air_year",
    "gap_since_prev", "episode_count", "tmdb_season_vote", "imdb_season_mean",
    "imdb_season_min", "imdb_season_max", "imdb_episode_spread",
    "imdb_delta_vs_prev", "cast_overlap_prev", "cast_turnover", "cast_new_count",
    "crew_change", "is_anthology", "runtime", "show_vote_average",
    "show_imdb_rating",
    # season-shape signals added by src/shows/season_fill.py: whether quality
    # rises or falls ACROSS the season, which mean/min/max cannot express
    "imdb_ep_first", "imdb_ep_last", "imdb_ep_trend", "guest_star_count",
]

# DELIBERATELY EXCLUDED from the content model: prev_season_rating and
# show_mean_so_far. Under GroupKFold-by-show the whole show is held out, so its
# earlier seasons have no known rating at prediction time either — feeding those
# in leaks the test labels into their own fold. They are legitimate only in the
# forward-chaining protocol below, where earlier seasons really are in train.
HISTORY = ["prev_season_rating", "show_mean_so_far"]


def _parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    s = str(x).strip()
    if s.startswith("["):
        try:
            import ast
            return list(ast.literal_eval(s))
        except (ValueError, SyntaxError):
            return []
    return [i.strip() for i in s.split(",") if i.strip()]


def build_frame():
    """Watched seasons, with per-season history features attached."""
    s = load_seasons()
    if s.empty:
        sys.exit("❌ season_features.csv not found — run src.shows.season_ingestion first.")

    s = s.sort_values(["tmdb_id", "season_number"]).reset_index(drop=True)
    g = s.groupby("tmdb_id")
    s["seasons_so_far"] = g.cumcount() + 1
    s["is_final_season"] = (s["season_number"] == g["season_number"].transform("max")).astype(int)
    s["gap_since_prev"] = g["air_year"].diff()

    # Your own history, strictly from EARLIER seasons of the same show (shift(1)),
    # so a season never sees its own rating or a later one.
    s["prev_season_rating"] = g["season_rating"].shift(1)
    s["show_mean_so_far"] = (g["season_rating"]
                             .transform(lambda x: x.shift(1).expanding().mean()))

    # Show-level context from the enriched CSV.
    en = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    en_idx = en.set_index("tmdb_id")
    for src_col, dst in (("runtime", "runtime"), ("vote_average", "show_vote_average"),
                         ("imdb_rating", "show_imdb_rating")):
        s[dst] = pd.to_numeric(s["tmdb_id"].map(en_idx[src_col]), errors="coerce")
    s["genres_list"] = s["tmdb_id"].map(en_idx["genres"].apply(_parse_list))
    s["language"] = s["tmdb_id"].map(en_idx["language"])
    s["network"] = s["tmdb_id"].map(en_idx["network"])

    train = s[s["watched"]].copy().reset_index(drop=True)
    print(f"   {len(train)} watched seasons across {train['tmdb_id'].nunique()} shows "
          f"({len(s) - len(train)} unwatched/unaired excluded).")
    return train, s


def featurize(df, state=None, fit=False, extra_numeric=()):
    from sentence_transformers import SentenceTransformer

    cols = list(NUMERIC) + list(extra_numeric)
    X_num = df.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
    if fit:
        medians = X_num.median().to_dict()
    else:
        medians = state["medians"]
    X_num = X_num.fillna(pd.Series(medians)).fillna(0)

    text = ("Season " + df["season_number"].astype("Int64").astype(str) +
            " of " + df["show_name"].fillna("") +
            ". " + df["season_overview"].fillna("") +
            " Cast: " + df["top_cast"].fillna("") +
            " Crew: " + df["key_crew"].fillna(""))
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = st_model.encode(text.tolist(), show_progress_bar=False)
    if fit:
        pca = PCA(n_components=min(PCA_COMPONENTS, emb.shape[0], emb.shape[1]))
        vec = pca.fit_transform(emb)
    else:
        pca = state["pca"]
        vec = pca.transform(emb)
    X_text = pd.DataFrame(vec, columns=[f"pca_{i}" for i in range(vec.shape[1])],
                          index=df.index)

    genres = df["genres_list"].apply(lambda v: v if isinstance(v, list) else [])
    if fit:
        mlb = MultiLabelBinarizer()
        gen = mlb.fit_transform(genres)
    else:
        mlb = state["mlb_genre"]
        gen = mlb.transform(genres)
    X_gen = pd.DataFrame(gen, columns=[f"gen_{c}" for c in mlb.classes_], index=df.index)

    cats = []
    for col in ("language", "network"):
        d = pd.get_dummies(df[col].fillna("Unknown").astype(str), prefix=col)
        cats.append(d)
    X_cat = pd.concat(cats, axis=1) if cats else pd.DataFrame(index=df.index)

    X = pd.concat([X_num, X_gen, X_cat, X_text], axis=1)
    X = X.loc[:, ~X.columns.duplicated()]

    if fit:
        state = {"medians": medians, "pca": pca, "mlb_genre": mlb,
                 "columns": list(X.columns), "sentence_transformer": "all-MiniLM-L6-v2"}
        return X, state
    return X.reindex(columns=state["columns"], fill_value=0), state


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    print("📺 Training SEASON-level TV model (the show-level model is left untouched)...")
    train, _all_seasons = build_frame()
    X, state = featurize(train, fit=True)
    y = train["season_rating"].astype(float)
    groups = train["tmdb_id"]

    # Group by SHOW: seasons of one show must never straddle the split, or the
    # model just memorises that show's level from its other seasons.
    n_splits = min(5, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    oof = pd.Series(np.nan, index=train.index)
    for tr, va in gkf.split(X, y, groups):
        m = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[va] = m.predict(X.iloc[va])

    mae = mean_absolute_error(y, oof)
    r2 = r2_score(y, oof)
    within_half = float((np.abs(y - oof) <= 0.5).mean())

    # Honest baseline for a CONTENT model: predict the global mean season rating.
    # (Predicting the show's own rating is NOT a usable baseline -- for a show you
    # have not rated there is no show rating to predict with, and it is the very
    # thing the model is meant to supply.)
    base = pd.Series(y.mean(), index=y.index)
    base_mae, base_r2 = mean_absolute_error(y, base), r2_score(y, base)
    skill = 1 - (mae / base_mae)

    print(f"\n   SEASON content model (grouped {n_splits}-fold OOF by show, N={len(y)})")
    print(f"     MAE {mae:.4f} | R2 {r2:.4f} | within 0.5 stars {within_half:.1%}")
    print(f"   Baseline (global mean season rating)")
    print(f"     MAE {base_mae:.4f} | R2 {base_r2:.4f}")
    print(f"   -> skill over mean baseline: {skill:+.3f}")
    verdict = ("content model beats the mean baseline" if mae < base_mae
               else "content model does NOT beat the mean baseline")

    # --- Second protocol: "next season" forecasting -------------------------
    # Train on every season EXCEPT each show's last rated one, test on that last
    # one. Here the earlier seasons genuinely are in train, so prev_season_rating
    # and show_mean_so_far are legitimately known and are included.
    fwd = {}
    last_idx = (train.sort_values(["tmdb_id", "season_number"])
                .groupby("tmdb_id").tail(1).index)
    multi = train.loc[last_idx]
    multi = multi[multi["season_number"] > 1]  # needs at least one earlier season
    if len(multi) >= 20:
        Xf, state_f = featurize(train, fit=True, extra_numeric=HISTORY)
        te_mask = train.index.isin(multi.index)
        mf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2,
                                   random_state=42, n_jobs=-1)
        mf.fit(Xf[~te_mask], y[~te_mask])
        pf = mf.predict(Xf[te_mask])
        yf = y[te_mask]
        # Carry-forward baseline: "next season = your previous season's rating".
        carry = pd.to_numeric(train.loc[te_mask, "prev_season_rating"],
                              errors="coerce").fillna(y.mean())
        fwd = {"n": int(len(yf)),
               "model_mae": round(float(mean_absolute_error(yf, pf)), 4),
               "carry_forward_mae": round(float(mean_absolute_error(yf, carry)), 4)}
        print(f"\n   NEXT-SEASON forecast (predict each show's latest rated season, "
              f"N={fwd['n']})")
        print(f"     model MAE {fwd['model_mae']:.4f}  vs  "
              f"carry-forward-previous-season MAE {fwd['carry_forward_mae']:.4f}")

    final = RandomForestRegressor(n_estimators=600, min_samples_leaf=2,
                                  random_state=42, n_jobs=-1)
    final.fit(X, y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    joblib.dump(state, STATE_PATH)

    imp = (pd.Series(final.feature_importances_, index=X.columns)
           .sort_values(ascending=False).head(15))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps({
        "n_seasons": int(len(y)), "n_shows": int(groups.nunique()),
        "cv": f"GroupKFold({n_splits}) by show",
        "season_model": {"mae": round(mae, 4), "r2": round(r2, 4),
                         "within_half_star": round(within_half, 4)},
        "mean_baseline": {"mae": round(base_mae, 4), "r2": round(base_r2, 4)},
        "skill_over_mean": round(skill, 4),
        "next_season_forecast": fwd,
        "excluded_features": HISTORY,
        "exclusion_reason": ("leak under GroupKFold-by-show: a held-out show's "
                             "earlier seasons are held out too, so their ratings "
                             "are not known at prediction time"),
        "verdict": verdict,
        "top_features": {k: round(float(v), 4) for k, v in imp.items()},
    }, indent=2), encoding="utf-8")

    print(f"\n   saved -> {MODEL_PATH}")
    print(f"   saved -> {STATE_PATH}")
    print(f"   saved -> {METRICS_PATH}")
    print("\n   top features:")
    for k, v in imp.head(10).items():
        print(f"     {k:26} {v:.4f}")


if __name__ == "__main__":
    main()
