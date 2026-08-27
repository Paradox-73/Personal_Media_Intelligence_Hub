"""
Text/plot vs crowd-quality — where does YOUR personal taste signal actually live?

Hypothesis (a good one): crowd-quality features (IMDb rating, votes, Metascore) measure
"is this generally good", not "will *I* like it". The personal signal should live in the
plot/title/crew **text embedding**. If so, a text-only model should hold its own — and we
may be under-weighting text by compressing it to a few PCA components and dropping it into
a tree that splits on the loud crowd features first.

This runs, per domain, a feature-group ablation with repeated CV:
  * TEXT-only   : just the pca_* plot/title/crew embedding components
  * CROWD-only  : just the crowd-quality numerics (imdb/critic/votes/metascore/…)
  * NON-TEXT    : everything that is NOT the text embedding (crowd + genres + year + …)
  * FULL        : the deployed feature set
  * kNN-TEXT    : purest "plot similarity → taste" — predict the mean rating of the k
                  nearest items in embedding space (no crowd features at all)

Metric = skill vs predict-the-mean (1 - MAE/MAE_mean); higher = more personal signal.
Output: reports/text_vs_crowd.json  (+ console table)
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

DROP = ["target_reg", "target_class", "target_ordinal", "source_id", "global_id",
        "media_type", "rating_date"]
LOCAL = {"movie": config.TRAINING_DATA_PATH, "tv": config.TV_SHOWS_TRAINING_DATA_PATH,
         "game": config.GAMES_TRAINING_DATA_PATH, "book": config.BOOKS_TRAINING_DATA_PATH}
# crowd-quality numerics (everyone's average opinion), per domain — the "is this good" signal
CROWD = {
    "movie": ["imdb_rating", "metascore", "rotten_tomatoes_rating", "vote_average",
              "imdb_votes", "critic_avg_5", "vote_count", "popularity"],
    "tv": ["vote_average", "imdb_rating", "vote_count_log", "imdb_votes_log"],
    "game": ["metacritic", "rating", "ratings_count", "reviews_count"],
    "book": ["averageRating", "ratingsCount"],
    "unified": ["critic_avg_5", "imdb_rating", "vote_average", "popularity"],
}


def _xgb():
    return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                            subsample=0.85, colsample_bytree=0.85,
                            objective="reg:absoluteerror", random_state=0)


def _svr():
    return Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1))])


BUILD = {"movie": _xgb, "unified": _xgb, "tv": _svr, "game": _svr, "book": _svr}


def _load(domain):
    df = (pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH) if domain == "unified"
          else pd.read_csv(LOCAL[domain]))
    df = df[df["target_reg"].notna()].reset_index(drop=True)
    feats = [c for c in df.columns if c not in DROP]
    return df[feats].fillna(0), df["target_reg"].astype(float).values


def _skill(y, pred):
    mae = mean_absolute_error(y, np.clip(pred, 0.5, 5.0))
    base = mean_absolute_error(y, np.full_like(y, y.mean()))
    return 1 - mae / base, mae


def _cv_skill(X, y, cols, build, knn=False):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    preds = np.zeros(len(y)); counts = np.zeros(len(y))
    for tr, te in rkf.split(X):
        Xtr, Xte = X.iloc[tr][cols], X.iloc[te][cols]
        if knn:
            k = max(3, min(10, len(tr) // 8))
            m = Pipeline([("s", StandardScaler()),
                          ("m", KNeighborsRegressor(n_neighbors=k, weights="distance", metric="cosine"))])
        else:
            m = build()
        m.fit(Xtr, y[tr]); preds[te] += m.predict(Xte); counts[te] += 1
    preds /= np.maximum(counts, 1)
    return _skill(y, preds)


def main():
    out = {"models": {}}
    print("=" * 96)
    print("TEXT/PLOT vs CROWD-QUALITY — skill (1 - MAE/MAE_mean) by feature group, repeated 5x5 CV")
    print("=" * 96)
    print(f"{'domain':8} {'#text':>5} {'#crowd':>6} | {'TEXT':>6} {'CROWD':>6} {'NONTEXT':>7} "
          f"{'FULL':>6} {'kNN-TEXT':>8}   winner")
    for d in ["movie", "tv", "game", "book", "unified"]:
        X, y = _load(d)
        build = BUILD[d]
        text = [c for c in X.columns if c.startswith("pca_") or c.startswith("emb_")]
        crowd = [c for c in CROWD[d] if c in X.columns]
        nontext = [c for c in X.columns if c not in text]
        res = {}
        res["text"], _ = _cv_skill(X, y, text, build) if text else (float("nan"), 0)
        res["crowd"], _ = _cv_skill(X, y, crowd, build) if crowd else (float("nan"), 0)
        res["nontext"], _ = _cv_skill(X, y, nontext, build)
        res["full"], full_mae = _cv_skill(X, y, list(X.columns), build)
        res["knn_text"], _ = _cv_skill(X, y, text, build, knn=True) if text else (float("nan"), 0)
        res = {k: (round(float(v), 3) if v == v else None) for k, v in res.items()}
        res["n_text"] = len(text); res["n_crowd"] = len(crowd); res["full_mae"] = round(full_mae, 3)
        winner = max([("TEXT", res["text"]), ("CROWD", res["crowd"]), ("FULL", res["full"])],
                     key=lambda t: (t[1] if t[1] is not None else -9))[0]
        res["winner"] = winner
        out["models"][d] = res
        f = lambda v: f"{v:>6.3f}" if v is not None else "   n/a"
        print(f"{d:8} {len(text):>5} {len(crowd):>6} | {f(res['text'])} {f(res['crowd'])} "
              f"{f(res['nontext']):>7} {f(res['full'])} {f(res['knn_text']):>8}   {winner}")
    Path("reports/text_vs_crowd.json").write_text(json.dumps(out, indent=2))
    print("\nRead: if TEXT ≈ or > CROWD/FULL, personal taste lives in plot/crew and text is "
          "under-weighted.\nkNN-TEXT is the purest 'items with plots like ones I rated high' predictor.")
    print("Wrote reports/text_vs_crowd.json")


if __name__ == "__main__":
    main()
