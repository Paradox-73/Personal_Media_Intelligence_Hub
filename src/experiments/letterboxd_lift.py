"""
Do Letterboxd crowd columns (lb_*) improve the n=1 movie rating model?

Uses only per-film crowd aggregates from the user's own Films page export, never other
users' individual ratings. training_features.csv has no id column: its row i is the i-th
rated row of enriched_data.csv (positional join), guarded by asserting target_reg == user_rating.

Repeated 5x5 CV with the same seeds as text_vs_crowd.py. Metric = skill vs predict-the-mean
(1 - MAE/MAE_mean) and Spearman of out-of-fold predictions, plus paired per-repeat deltas.
Output: reports/letterboxd_lift.json  (+ console table)
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.experiments.text_vs_crowd import _load, _cv_skill, _xgb, _skill, CROWD

LB_COUNTS = ["lb_rating_count", "lb_review_count", "lb_fans", "lb_watched_by",
             "lb_list_count", "lb_like_count"]
LB_RAW = ["lb_avg_rating", "lb_rating_std", "lb_top500_rank"]
N_REPEATS = 5


def _with_lb():
    X, y = _load("movie")
    e = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    rated = e[e["user_rating"].notna()].reset_index(drop=True)
    if len(rated) < len(X):
        sys.exit(f"ABORT: {len(rated)} rated enriched rows < {len(X)} training rows")
    rated = rated.iloc[:len(X)]
    ur = pd.to_numeric(rated["user_rating"], errors="coerce").values
    bad = np.flatnonzero(~np.isclose(ur, y))
    if len(bad):
        sys.exit(f"ABORT: positional join broken, target_reg != user_rating at {len(bad)} rows "
                 f"(first {bad[:5].tolist()})")
    lb = rated[LB_RAW + LB_COUNTS].apply(pd.to_numeric, errors="coerce")
    lb[LB_COUNTS] = np.log1p(lb[LB_COUNTS])
    lb["lb_top500_rank"] = lb["lb_top500_rank"].fillna(501)  # unranked sits past the list end
    # other lb_* NaNs stay NaN: XGBoost routes missing values natively, 0 would read as a real value
    return pd.concat([X.drop(columns=[c for c in X.columns if c.startswith(('lb_', 'collection_name_'))]), lb], axis=1), y  # baseline = pre-Letterboxd set


def _per_repeat(X, y, cols):
    """OOF predictions per repeat, same splits as _cv_skill (each row predicted once per repeat)."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=0)
    P = np.zeros((N_REPEATS, len(y)))
    for i, (tr, te) in enumerate(rkf.split(X)):
        m = _xgb(); m.fit(X.iloc[tr][cols], y[tr])
        P[i // 5, te] = m.predict(X.iloc[te][cols])
    return P


def _summary(y, P):
    avg = P.mean(axis=0)
    return {"skill": float(_skill(y, avg)[0]), "mae": float(_skill(y, avg)[1]),
            "spearman": float(spearmanr(y, avg).correlation),
            "per_repeat_skill": [float(_skill(y, p)[0]) for p in P],
            "per_repeat_spearman": [float(spearmanr(y, p).correlation) for p in P]}


def _paired(a, b, key):
    d = np.array(b[key]) - np.array(a[key])
    return {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "improved": int((d > 0).sum()), "of": len(d)}


def main():
    X, y = _with_lb()
    full = [c for c in X.columns if not c.startswith("lb_")]
    crowd = [c for c in CROWD["movie"] if c in X.columns]
    lb_all = LB_RAW + LB_COUNTS
    configs = {
        "1_full": full,
        "2_full+lb_numeric": full + lb_all,
        "3_full+lb_avg_rating": full + ["lb_avg_rating"],
        "4a_crowd": crowd,
        "4b_crowd+lb_avg_rating": crowd + ["lb_avg_rating"],
    }
    res = {k: _summary(y, _per_repeat(X, y, cols)) for k, cols in configs.items()}

    ref = _cv_skill(X, y, full, _xgb)[0]
    assert np.isclose(ref, res["1_full"]["skill"], atol=1e-6), \
        f"per-repeat loop disagrees with _cv_skill on FULL: {ref} vs {res['1_full']['skill']}"

    paired = {}
    for a, b in [("1_full", "2_full+lb_numeric"), ("1_full", "3_full+lb_avg_rating"),
                 ("4a_crowd", "4b_crowd+lb_avg_rating")]:
        paired[f"{b} vs {a}"] = {k: _paired(res[a], res[b], f"per_repeat_{k}")
                                 for k in ("skill", "spearman")}

    print(f"n={len(y)} rated films, lb_avg_rating present for "
          f"{int(X['lb_avg_rating'].notna().sum())}")
    print(f"{'config':26} {'#feat':>5} {'skill':>7} {'MAE':>6} {'spearman':>8}")
    for k, cols in configs.items():
        r = res[k]
        print(f"{k:26} {len(cols):>5} {r['skill']:>7.4f} {r['mae']:>6.3f} {r['spearman']:>8.4f}")
    print("\nPaired per-repeat delta (B - A), mean ± sd, repeats improved:")
    for name, d in paired.items():
        s, p = d["skill"], d["spearman"]
        print(f"  {name:48} skill {s['mean']:+.4f} ± {s['sd']:.4f} ({s['improved']}/{s['of']})"
              f"   spearman {p['mean']:+.4f} ± {p['sd']:.4f} ({p['improved']}/{p['of']})")

    out = {"n": len(y), "configs": {k: len(v) for k, v in configs.items()},
           "results": res, "paired": paired}
    Path("reports/letterboxd_lift.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/letterboxd_lift.json")


if __name__ == "__main__":
    main()
