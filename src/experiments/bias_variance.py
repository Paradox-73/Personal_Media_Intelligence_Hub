"""
Bias / variance decomposition per domain — the "what do I actually need?" diagnostic.

For each domain we bootstrap-train its representative deployed learner on its OWN
local feature matrix and decompose expected squared error on held-out items into:

    E[(y - pred)^2]  ≈  bias^2  +  variance  (+ irreducible noise we can't measure)

  * bias^2   = how far the *average* prediction (over many training resamples) is
               from the truth. HIGH bias ⇒ underfitting ⇒ the model/features are the
               ceiling ⇒ collecting more *rows* won't help; you need better FEATURES
               or a stronger MODEL.
  * variance = how much the prediction wobbles as the training set is resampled.
               HIGH variance ⇒ overfitting / data-starved ⇒ more DATA (or more
               regularization) is the lever.

We also report the train-vs-test MAE gap (a second over/underfit read) and pull in
N, feature count, and the shared-space learning-curve slope so each domain gets a
concrete prescription: more data-points / more features / fix model / explore.

Representative core learner per domain (kept light for the bootstrap refits;
bias/variance is a property of model-family + features + N, robust to this choice):
    movie, unified : XGBRegressor (the tree core of the deployed stack/ensemble)
    tv, game, book : SVR(rbf) (game/book deploy this; tv's strongest single member)

Output: reports/bias_variance.json  (+ console table)
Run:    python -m src.experiments.bias_variance
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

DROP = ["target_reg", "target_class", "target_ordinal", "source_id", "global_id",
        "media_type", "rating_date"]
LOCAL_PATHS = {
    "movie": config.TRAINING_DATA_PATH,
    "tv": config.TV_SHOWS_TRAINING_DATA_PATH,
    "game": config.GAMES_TRAINING_DATA_PATH,
    "book": config.BOOKS_TRAINING_DATA_PATH,
}
# Learning-curve edge slope: how much still improves between 75% and 100% of the
# training data. POSITIVE = still learning from more rows.
#
# This used to be a hardcoded dict copied out of LEARNING_CURVE_RESULTS.md, measured
# in the SHARED (unified) space in skill units. It went stale and then drove
# _diagnose() to print confident, WRONG prescriptions: it claimed the tv/game/book
# curves were flat when a direct per-domain measurement puts them at -0.0241,
# -0.0177 and -0.0609 MAE between 75% and 100% -- i.e. all three are still clearly
# learning. Read the measured value instead, and fall back loudly.
_LC_FALLBACK = {"movie": +0.013, "tv": +0.000, "game": -0.032, "book": +0.003,
                "unified": None}


def _lc_edge_slopes():
    """Per-domain improvement rate from reports/ceiling_analysis.json.

    That file stores `tail_slope` as a DELTA MAE (negative = error still falling),
    so negate it to get an improvement rate comparable to the old skill-units
    convention _diagnose() thresholds against.
    """
    p = Path("reports/ceiling_analysis.json")
    if not p.exists():
        print("   ⚠️ reports/ceiling_analysis.json missing -- falling back to the stale\n"
              "      hardcoded learning-curve slopes; prescriptions may be wrong.\n"
              "      Regenerate it before trusting them.")
        return dict(_LC_FALLBACK)
    data = json.loads(p.read_text())
    out = {}
    for dom, rec in data.items():
        ts = rec.get("tail_slope")
        out[dom] = None if ts is None else round(-float(ts), 4)
    return out


LC_EDGE_SLOPE = _lc_edge_slopes()


# Anchor columns for the residual reframing the movie ensemble now deploys.
ANCHOR_COLS = ["imdb_rating", "metascore", "rotten_tomatoes_rating", "vote_average",
               "critic_avg_5"]


def _xgb():
    return xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                            subsample=0.85, colsample_bytree=0.85,
                            objective="reg:absoluteerror", random_state=0)


def _svr(c=1.0):
    return Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=c, epsilon=0.1))])


def _svr_tuned():
    """Books and games deploy SVR at C=3.0 (nested-CV tuned)."""
    return _svr(3.0)


class _Anchored:
    """Mirror of AnchoredRegressor: learn (y - crowd anchor), add it back.

    The movie ensemble deploys this, so the decomposition has to include it or it
    describes a model that is not shipped. Kept as a tiny local class to avoid
    importing the trainer (which pulls in matplotlib/optuna).
    """

    def __init__(self, base):
        self.base = base

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression
        self.cols_ = [c for c in ANCHOR_COLS if c in X.columns]
        y = np.asarray(y, dtype=float)
        if self.cols_:
            self.lin_ = LinearRegression().fit(X[self.cols_], y)
            y = y - self.lin_.predict(X[self.cols_])
        else:
            self.lin_ = None
        self.base.fit(X, y)
        return self

    def predict(self, X):
        p = np.asarray(self.base.predict(X), dtype=float)
        if self.lin_ is not None:
            p = p + self.lin_.predict(X[self.cols_])
        return p


def _xgb_anchored():
    return _Anchored(_xgb())


# Representative learner per domain, matched to what each domain actually DEPLOYS:
#   movie   -> residual-anchored trees (the production ensemble's regression members)
#   unified -> plain trees (no residual there; never measured on its mean ensemble)
#   game/book -> SVR at the tuned C=3.0
#   tv      -> SVR at C=1.0, a proxy for its strongest single member (tv deploys a stack)
BUILDER = {"movie": _xgb_anchored, "unified": _xgb, "tv": _svr,
           "game": _svr_tuned, "book": _svr_tuned}


def _load(domain):
    if domain == "unified":
        df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
        # `notna()` is not enough: unrated items are stored as target_reg == 0.0
        # (a fillna in the builder, not NaN), so six unrated games survived this
        # filter and were scored as genuine 0-star items -- 1,413 rows instead of
        # the 1,407 the fold registry and every other report use.
        df = df[df["target_reg"].notna() & (df["target_reg"] > 0)]
    else:
        df = pd.read_csv(LOCAL_PATHS[domain])
        df = df[df["target_reg"].notna() & (df["target_reg"] > 0)]
    feats = [c for c in df.columns if c not in DROP]
    return df[feats].fillna(0), df["target_reg"].astype(float).values


def decompose(domain, n_outer=5, n_boot=30, test_frac=0.30, seed=0):
    X, y = _load(domain)
    build = BUILDER[domain]
    rng = np.random.RandomState(seed)
    biases, varis, mses, train_maes, test_maes = [], [], [], [], []

    for o in range(n_outer):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_frac, random_state=seed + o)
        Xtr = Xtr.reset_index(drop=True); ytr = np.asarray(ytr)
        preds = np.zeros((n_boot, len(Xte)))
        # one full-train fit for the train/test gap read
        m0 = build(); m0.fit(Xtr, ytr)
        train_maes.append(mean_absolute_error(ytr, m0.predict(Xtr)))
        test_maes.append(mean_absolute_error(yte, m0.predict(Xte)))
        for b in range(n_boot):
            idx = rng.randint(0, len(Xtr), len(Xtr))   # bootstrap resample
            mb = build(); mb.fit(Xtr.iloc[idx], ytr[idx])
            preds[b] = mb.predict(Xte)
        main = preds.mean(axis=0)                       # average prediction
        bias2 = np.mean((main - yte) ** 2)
        var = np.mean(preds.var(axis=0))
        mse = np.mean((preds - yte) ** 2)              # ≈ bias2 + var
        biases.append(bias2); varis.append(var); mses.append(mse)

    bias2 = float(np.mean(biases)); var = float(np.mean(varis)); mse = float(np.mean(mses))
    tot = bias2 + var
    rec = {
        "N": int(len(X)), "n_features": int(X.shape[1]),
        "bias2": round(bias2, 4), "variance": round(var, 4),
        "mse_est": round(mse, 4),
        "bias_share": round(bias2 / tot, 3) if tot else None,
        "var_share": round(var / tot, 3) if tot else None,
        "train_mae": round(float(np.mean(train_maes)), 4),
        "test_mae": round(float(np.mean(test_maes)), 4),
        "train_test_gap": round(float(np.mean(test_maes) - np.mean(train_maes)), 4),
        "lc_edge_slope": LC_EDGE_SLOPE.get(domain),
    }
    return rec


def _diagnose(r):
    """Prescription from the decomposition. The authoritative signal is bias% vs var%
    (NOT the train/test gap: a large gap with LOW bootstrap variance means the learner
    interpolates the training set but reverts to a stable, biased guess on test items —
    that's a bias/signal ceiling, not classic high-variance instability). Note bias^2
    here conflates reducible model/feature bias with the irreducible 0.5★-label noise —
    either way, more rows of the SAME features (which only cut variance) barely helps."""
    var_share = r["var_share"] or 0.0
    slope = r["lc_edge_slope"]
    if var_share >= 0.30:
        return "VARIANCE-limited → more DATA (rows) or stronger regularization is the lever"
    # bias-dominated (the case for every domain here)
    if slope is not None and slope >= 0.01:
        return ("BIAS-limited, but the learning curve is still rising → better FEATURES first; "
                "a little more data helps only at the margin")
    return ("BIAS-limited and the learning curve is flat → FEATURES/signal ceiling (or the "
            "aleatoric 0.5★-rating noise); more rows of the same features won't move it")


def main():
    domains = ["movie", "tv", "game", "book", "unified"]
    out = {"models": {}}
    print("=" * 100)
    print("BIAS / VARIANCE DECOMPOSITION  (bootstrap on each domain's OWN local features)")
    print("=" * 100)
    print(f"{'domain':8} {'N':>5} {'feat':>5} {'bias^2':>7} {'var':>7} {'bias%':>6} {'var%':>5} "
          f"{'trainMAE':>8} {'testMAE':>7} {'gap':>6} {'LCslope':>7}")
    for d in domains:
        r = decompose(d)
        out["models"][d] = r
        r["prescription"] = _diagnose(r)
        lc = "n/a" if r["lc_edge_slope"] is None else f"{r['lc_edge_slope']:+.3f}"
        print(f"{d:8} {r['N']:>5} {r['n_features']:>5} {r['bias2']:>7.3f} {r['variance']:>7.3f} "
              f"{(r['bias_share'] or 0)*100:>5.0f}% {(r['var_share'] or 0)*100:>4.0f}% "
              f"{r['train_mae']:>8.3f} {r['test_mae']:>7.3f} {r['train_test_gap']:>6.3f} {lc:>7}")
    print("\nPer-domain prescription:")
    for d in domains:
        print(f"  {d:8}: {out['models'][d]['prescription']}")
    Path("reports/bias_variance.json").write_text(json.dumps(out, indent=2))
    print("\nWrote reports/bias_variance.json")


if __name__ == "__main__":
    main()
