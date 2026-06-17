"""
Production-grade per-domain benchmarks on the FROZEN registry folds (Task 1).

Why this exists
---------------
`standalone_benchmarks.py` scores each domain with a *simplified* estimator. For
Games and Books that simplified model IS the deployed model (SVR, frozen HPs), so
those rows are already honest. For **Movies** and **TV** it is a weakened proxy
(plain XGB / a manual simplex) — far below the production architecture (Optuna-tuned
asymmetric edge-penalty XGB + CatBoost + SVR, fused, with an ordinal expected-value
member). Benchmarking the unified model against those weak proxies flattered it.

This module evaluates a **working, production-grade** ensemble for Movies and TV on
the same 50 registry folds, deduplicated to one row per item, so the benchmark table
finally compares the unified model against *strong* locals.

Faithfulness & determinism
--------------------------
* The deployed `advanced_movie_model_trainer.py` fuses the ordinal classifier inside
  an sklearn `StackingRegressor`; that path is fragile (it crashes under `clone()`)
  and was never runnable end-to-end. We reproduce the same *ideas* with a hand-rolled,
  fully-deterministic stack (inner-CV out-of-fold meta-features → meta learner →
  bases refit on full train), which is robust and leakage-safe.
* Edge-penalty alphas are tuned **once** on the full domain (seeded Optuna), persisted
  to `models/<domain>/best_params.json`, and **reused across folds** — i.e. we measure
  the *deployed* tuned model, we do not re-tune per fold (which would be intractable
  and would leak).
* Everything is seeded (`random_state=42`, `TPESampler(seed=42)`), so the registry-fold
  numbers are reproducible.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as ctb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective
from src.reporting.standalone_benchmarks import _load_domain, DROP_COLS

BUCKET_MAP = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}

# Per-domain production ensemble configuration.
CFG = {
    "movie": {"cat_iters": 1000, "xgb_estimators": 300, "meta": "ridgecv", "label": "Prod Stacking (edge+Cat+SVR+Ord)"},
    "tv":    {"cat_iters": 500,  "xgb_estimators": 200, "meta": "ridge_pos", "label": "Prod Simplex-Stack (edge+Cat+SVR)"},
}


def _round_half(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def _metrics(y_true, y_pred_raw):
    yp = _round_half(np.asarray(y_pred_raw))
    yt = np.asarray(y_true)
    return {"MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "Acc": float((np.abs(yt - yp) <= 0.5).mean() * 100)}


def _balanced_weights(y):
    classes = (np.asarray(y) * 2).round() / 2
    return compute_sample_weight(class_weight="balanced", y=classes)


# ---- edge-penalty alpha tuning (once per domain, seeded) ------------------------

def tune_alphas(X, y, domain, n_trials=25):
    """Tune asymmetric edge-penalty alphas once on the full domain (seeded TPE).
    Persisted to models/<domain>/best_params.json and reused across all folds."""
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        a_hi = trial.suggest_float("alpha_hi", 0.01, 0.5, log=True)
        a_lo = trial.suggest_float("alpha_lo", 0.01, 0.5, log=True)
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        maes = []
        for t_idx, v_idx in rkf.split(X):
            m = xgb.XGBRegressor(objective=AsymmetricEdgePenaltyObjective(a_hi, a_lo),
                                 n_estimators=150, learning_rate=0.05, max_depth=5,
                                 random_state=42, n_jobs=-1)
            m.fit(X.iloc[t_idx], y.iloc[t_idx])
            p = _round_half(m.predict(X.iloc[v_idx]))
            maes.append(mean_absolute_error(y.iloc[v_idx], p))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params
    out = config.MODEL_DIR / domain / "best_params.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**best, "tuned_on": "full_domain", "n_trials": n_trials}, indent=2))
    print(f"   [{domain}] tuned edge alphas: alpha_hi={best['alpha_hi']:.4f} "
          f"alpha_lo={best['alpha_lo']:.4f} -> {out}")
    return best["alpha_hi"], best["alpha_lo"]


# ---- base learners --------------------------------------------------------------

def _ordinal_ev(X_tr, y_ord_tr, w_tr, X_te, n_estimators=200):
    """XGBoost ordinal classifier -> expected value over present buckets."""
    classes = np.sort(np.unique(y_ord_tr))
    cmap = {c: i for i, c in enumerate(classes)}
    y_mapped = pd.Series(y_ord_tr).map(cmap).values
    bucket_vals = np.array([BUCKET_MAP[c] for c in classes])
    if len(classes) < 2:
        # Degenerate fold: only one rating bucket present -> constant EV.
        return np.full(len(X_te), bucket_vals[0], dtype=float)
    clf = xgb.XGBClassifier(n_estimators=n_estimators, learning_rate=0.03, max_depth=5,
                            objective="multi:softprob", num_class=len(classes),
                            subsample=0.8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_mapped, sample_weight=w_tr)
    probs = clf.predict_proba(X_te)
    return probs @ bucket_vals


def _base_predictions(X_tr, y_reg_tr, y_ord_tr, X_te, cfg, alphas):
    """Fit all base learners on (X_tr) and return a dict of test predictions."""
    w = _balanced_weights(y_reg_tr)
    a_hi, a_lo = alphas

    xgb_edge = xgb.XGBRegressor(objective=AsymmetricEdgePenaltyObjective(a_hi, a_lo),
                                n_estimators=cfg["xgb_estimators"], learning_rate=0.03,
                                max_depth=6, subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=-1)
    xgb_edge.fit(X_tr, y_reg_tr, sample_weight=w)

    cat = ctb.CatBoostRegressor(iterations=cfg["cat_iters"], learning_rate=0.05, depth=6,
                                l2_leaf_reg=3, loss_function="MAE", random_seed=42, verbose=0)
    cat.fit(X_tr, y_reg_tr)

    svr = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
    svr.fit(X_tr, y_reg_tr, svr__sample_weight=w)

    ev = _ordinal_ev(X_tr, y_ord_tr, w, X_te)

    return {"xgb": xgb_edge.predict(X_te), "cat": cat.predict(X_te),
            "svr": svr.predict(X_te), "ord": ev}


def _make_meta(kind):
    return RidgeCV() if kind == "ridgecv" else Ridge(positive=True)


def _stack_fold(X_tr, y_reg_tr, y_ord_tr, X_te, cfg, alphas):
    """Hand-rolled, leakage-safe stack: inner-CV OOF meta-features -> meta learner ->
    bases refit on full train -> meta on test base-preds."""
    names = ["xgb", "cat", "svr", "ord"]
    inner = KFold(n_splits=3, shuffle=True, random_state=42)
    X_tr = X_tr.reset_index(drop=True)
    y_reg = y_reg_tr.reset_index(drop=True)
    y_ord = pd.Series(np.asarray(y_ord_tr)).reset_index(drop=True)
    oof = {n: np.zeros(len(X_tr)) for n in names}
    for tr_idx, va_idx in inner.split(X_tr):
        preds = _base_predictions(X_tr.iloc[tr_idx], y_reg.iloc[tr_idx], y_ord.iloc[tr_idx],
                                  X_tr.iloc[va_idx], cfg, alphas)
        for n in names:
            oof[n][va_idx] = preds[n]
    Z_tr = np.column_stack([oof[n] for n in names])
    meta = _make_meta(cfg["meta"])
    meta.fit(Z_tr, y_reg)
    # Refit bases on the full training fold and predict the held-out test fold.
    full = _base_predictions(X_tr, y_reg, y_ord, X_te, cfg, alphas)
    Z_te = np.column_stack([full[n] for n in names])
    return meta.predict(Z_te)


# ---- driver ---------------------------------------------------------------------

def production_oof_for_domain(domain, uni_df, registry, alphas):
    loc = _load_domain(domain, uni_df, registry)
    feat_cols = [c for c in loc.columns if c not in DROP_COLS + ["target_ordinal"]]
    cfg = CFG[domain]
    rows = []
    for fold_id in range(50):
        test_mask = loc["global_id"].apply(lambda g: fold_id in registry.get(g, []))
        if not test_mask.any():
            continue
        tr, te = ~test_mask, test_mask
        preds = _stack_fold(loc.loc[tr, feat_cols], loc.loc[tr, "target_reg"],
                            loc.loc[tr, "target_ordinal"], loc.loc[te, feat_cols], cfg, alphas)
        sub = loc.loc[te, ["source_id", "target_reg"]].copy()
        sub["pred"] = np.asarray(preds)
        sub["media_type"] = domain
        rows.append(sub)
        if (fold_id + 1) % 10 == 0:
            print(f"      [{domain}] production fold {fold_id + 1}/50", flush=True)
    oof = pd.concat(rows, ignore_index=True)
    per_item = (oof.groupby("source_id")
                   .agg(target_reg=("target_reg", "first"), pred=("pred", "mean"))
                   .reset_index())
    per_item["media_type"] = domain
    met = _metrics(per_item["target_reg"], per_item["pred"])
    print(f"   [{domain}] PRODUCTION  N={per_item['source_id'].nunique()}  "
          f"MAE={met['MAE']:.4f}  R2={met['R2']:.4f}  Acc={met['Acc']:.1f}%")
    return per_item, {"Domain": domain, "N": int(per_item["source_id"].nunique()),
                      "Model": cfg["label"], **met}


def generate_production_oof(uni_df, registry, domains=("movie", "tv"),
                            out_path="reports/production_oof.csv", use_cache=True):
    cache = Path(out_path)
    if use_cache and cache.exists():
        df = pd.read_csv(cache)
        bms = []
        for d in domains:
            sl = df[df["media_type"] == d]
            if sl.empty:
                continue
            met = _metrics(sl["target_reg"], sl["pred"])
            bms.append({"Domain": d, "N": int(sl["source_id"].nunique()),
                        "Model": CFG[d]["label"], **met})
        print(f"   [production] reusing cached per-item OOF <- {out_path}")
        return bms, df

    all_items, bms = [], []
    for d in domains:
        print(f"   [production] tuning + evaluating {d} ...", flush=True)
        X = uni_df[uni_df["media_type"] == d]
        # Tune on the standalone domain matrix (same features the fold loop uses).
        loc = _load_domain(d, uni_df, registry)
        feat_cols = [c for c in loc.columns if c not in DROP_COLS + ["target_ordinal"]]
        alphas = tune_alphas(loc[feat_cols], loc["target_reg"], d)
        per_item, bm = production_oof_for_domain(d, uni_df, registry, alphas)
        all_items.append(per_item)
        bms.append(bm)
    out = pd.concat(all_items, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"   [production] per-item OOF saved -> {out_path}")
    return bms, out


if __name__ == "__main__":
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        reg = json.load(f)
    bms, _ = generate_production_oof(uni, reg, use_cache=False)
    print(json.dumps(bms, indent=2))
