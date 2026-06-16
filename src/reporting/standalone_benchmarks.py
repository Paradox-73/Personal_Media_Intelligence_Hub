"""
Standalone per-domain benchmarks on the FROZEN registry folds.

Phase 0 bug fix: the previous writer published the *unified* model's per-domain
OOF slice as if it were the standalone domain models (benchmarks == slices, with
N inflated 5x by counting OOF rows instead of items). This module regenerates
genuine standalone-model OOF on the same 50 registry folds, deduplicated to one
row per item, so the benchmark table reports the local models -- not the unified
slice.

Design decisions (documented so the docs can cite them):
  * Each domain runs its OWN deployed regressor architecture:
      - movie : XGBoost (reg:absoluteerror, balanced sample weights)   -> "XGBoost"
      - tv    : Simplex-weighted stack of XGB + CatBoost + SVR          -> "Simplex-Stack"
      - game  : local SVR(rbf), NO unified prior (distillation DROPPED) -> "Local SVR"
      - book  : local SVR(rbf), NO unified prior (distillation DROPPED) -> "Local SVR"
  * Standalone feature CSVs carry no source_id, so each row is mapped to its
    registry global_id positionally against the unified domain slice (the row
    order is identical and every standalone item is rated; validated by an exact
    target-value alignment check, same pattern as distillation_ablation.py).
  * Each item lands in 5 test folds (5 CV repeats). Predictions are averaged
    across repeats per item (mean of raw predictions), THEN rounded to the
    nearest 0.5 once, THEN scored. N is the unique-item count.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as ctb
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

DOMAIN_PATHS = {
    "movie": config.TRAINING_DATA_PATH,
    "tv": config.TV_SHOWS_TRAINING_DATA_PATH,
    "game": config.GAMES_TRAINING_DATA_PATH,
    "book": config.BOOKS_TRAINING_DATA_PATH,
}
DOMAIN_MODEL_LABEL = {
    "movie": "XGBoost",
    "tv": "Simplex-Stack",
    "game": "Local SVR",
    "book": "Local SVR",
}
DROP_COLS = ["target_reg", "target_class", "target_ordinal", "source_id", "global_id"]


def _round_half(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def _metrics(y_true, y_pred_raw):
    yp = _round_half(np.asarray(y_pred_raw))
    yt = np.asarray(y_true)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "Acc": float((np.abs(yt - yp) <= 0.5).mean() * 100),
    }


# ---- per-domain model builders -------------------------------------------------

def _fit_predict_xgb(X_tr, y_tr, X_te):
    w = _balanced_weights(y_tr)
    m = xgb.XGBRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:absoluteerror", random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=w)
    return m.predict(X_te)


def _fit_predict_svr(X_tr, y_tr, X_te):
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)),
    ])
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _fit_predict_simplex_stack(X_tr, y_tr, X_te):
    """XGB + CatBoost + SVR base models, simplex-weighted by inner 5-fold OOF MAE."""
    base = {
        "xgb": lambda: xgb.XGBRegressor(n_estimators=150, learning_rate=0.03,
                                        max_depth=5, random_state=42),
        "cat": lambda: ctb.CatBoostRegressor(iterations=200, learning_rate=0.05,
                                             depth=5, loss_function="MAE",
                                             verbose=0, random_seed=42),
        "svr": lambda: Pipeline([("scaler", StandardScaler()),
                                 ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))]),
    }
    # Inner OOF to fit simplex weights
    inner = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = {k: np.zeros(len(y_tr)) for k in base}
    y_arr = np.asarray(y_tr)
    X_arr = X_tr.reset_index(drop=True)
    for tr_idx, va_idx in inner.split(X_arr):
        for k, mk in base.items():
            mdl = mk()
            mdl.fit(X_arr.iloc[tr_idx], y_arr[tr_idx])
            oof[k][va_idx] = mdl.predict(X_arr.iloc[va_idx])
    Z = np.column_stack([oof[k] for k in base])

    def obj(w):
        return mean_absolute_error(y_arr, Z @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, 1)] * len(base)
    w0 = np.ones(len(base)) / len(base)
    weights = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons).x

    # Refit base models on full train, blend test predictions
    test_preds = []
    for k, mk in base.items():
        mdl = mk()
        mdl.fit(X_tr, y_tr)
        test_preds.append(mdl.predict(X_te))
    return np.column_stack(test_preds) @ weights


DOMAIN_FIT = {
    "movie": _fit_predict_xgb,
    "tv": _fit_predict_simplex_stack,
    "game": _fit_predict_svr,
    "book": _fit_predict_svr,
}


def _balanced_weights(y):
    from sklearn.utils.class_weight import compute_sample_weight
    classes = (np.asarray(y) * 2).round() / 2
    return compute_sample_weight(class_weight="balanced", y=classes)


# ---- driver --------------------------------------------------------------------

def _load_domain(domain, uni_df, registry):
    """Load a standalone domain frame and attach registry global_ids positionally."""
    loc = pd.read_csv(DOMAIN_PATHS[domain])
    uni_slice = uni_df[uni_df["media_type"] == domain].reset_index(drop=True)
    if len(uni_slice) != len(loc):
        raise ValueError(
            f"{domain}: standalone rows={len(loc)} != unified slice={len(uni_slice)}. "
            "Positional source_id mapping invalid -- re-run feature engineering."
        )
    loc = loc.reset_index(drop=True)
    loc["source_id"] = uni_slice["source_id"].values
    # Validate positional alignment with a NaN-SAFE check. The standalone Games
    # file keeps "Incomplete"-status rows with NaN target_reg (no rating); those
    # must not silently pass a plain abs-diff (NaN > tol is False). Require that
    # every RATED standalone row matches the unified slice target at its position.
    a = uni_slice["target_reg"].values.astype(float)
    b = loc["target_reg"].values.astype(float)
    rated = ~np.isnan(b)
    if not np.allclose(a[rated], b[rated], atol=1e-6):
        raise ValueError(f"{domain}: rated target alignment failed (positional map invalid).")
    n_unrated = int((~rated).sum())
    if n_unrated:
        print(f"      ({domain}: {n_unrated} unrated/Incomplete rows excluded -- no target)")
    loc["global_id"] = domain + "_" + loc["source_id"].astype(str)
    loc = loc[loc["target_reg"].notna()].copy()
    return loc


def benchmarks_from_oof(oof_df):
    """Recompute benchmark rows from an existing per-item standalone OOF frame."""
    bms = []
    for domain in ["movie", "tv", "game", "book"]:
        d = oof_df[oof_df["media_type"] == domain]
        if d.empty:
            continue
        n = d["source_id"].nunique()
        bms.append({"Domain": domain, "N": int(n),
                    "Model": DOMAIN_MODEL_LABEL[domain],
                    **_metrics(d["target_reg"], d["pred"])})
    return bms


def generate_standalone_oof(uni_df, registry, oof_out_path="reports/standalone_oof.csv",
                            use_cache=False):
    if use_cache and Path(oof_out_path).exists():
        print(f"   [standalone] reusing cached per-item OOF <- {oof_out_path}")
        oof_df = pd.read_csv(oof_out_path)
        return benchmarks_from_oof(oof_df), oof_df
    return _generate_standalone_oof(uni_df, registry, oof_out_path)


def _generate_standalone_oof(uni_df, registry, oof_out_path="reports/standalone_oof.csv"):
    """Run all four standalone models on registry folds; return (benchmarks, oof_df)."""
    benchmarks = []
    all_oof = []
    for domain in ["movie", "tv", "game", "book"]:
        print(f"   [standalone] {domain} ...", flush=True)
        loc = _load_domain(domain, uni_df, registry)
        feat_cols = [c for c in loc.columns if c not in DROP_COLS]
        fit_fn = DOMAIN_FIT[domain]

        rows = []  # (source_id, target_reg, raw_pred) per (item, repeat)
        for fold_id in range(50):
            test_mask = loc["global_id"].apply(lambda g: fold_id in registry.get(g, []))
            if not test_mask.any():
                continue
            train_mask = ~test_mask
            X_tr, X_te = loc.loc[train_mask, feat_cols], loc.loc[test_mask, feat_cols]
            y_tr = loc.loc[train_mask, "target_reg"]
            preds = fit_fn(X_tr, y_tr, X_te)
            sub = loc.loc[test_mask, ["source_id", "target_reg"]].copy()
            sub["pred"] = np.asarray(preds)
            sub["media_type"] = domain
            rows.append(sub)

        oof = pd.concat(rows, ignore_index=True)
        # Per-item dedup: mean of raw predictions across the 5 CV repeats.
        per_item = oof.groupby("source_id").agg(
            target_reg=("target_reg", "first"),
            pred=("pred", "mean"),
        ).reset_index()
        per_item["media_type"] = domain

        n_items = per_item["source_id"].nunique()
        assert len(per_item) == n_items, f"{domain}: dedup left duplicate items"
        met = _metrics(per_item["target_reg"], per_item["pred"])
        benchmarks.append({"Domain": domain, "N": int(n_items),
                           "Model": DOMAIN_MODEL_LABEL[domain], **met})
        all_oof.append(per_item)
        print(f"      N={n_items}  MAE={met['MAE']:.4f}  R2={met['R2']:.4f}  Acc={met['Acc']:.1f}%")

    oof_df = pd.concat(all_oof, ignore_index=True)
    Path(oof_out_path).parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_out_path, index=False)
    print(f"   [standalone] per-item OOF saved -> {oof_out_path}")
    return benchmarks, oof_df


if __name__ == "__main__":
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        reg = json.load(f)
    bms, _ = generate_standalone_oof(uni, reg)
    print(json.dumps(bms, indent=2))
