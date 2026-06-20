"""
Feature-engineering diagnostics — three checks flagged in err.txt.

(A) Shared-space coverage table: for every domain, the non-zero fraction of each
    shared channel in the unified matrix. Confirms (or refutes) the claim that
    "Books has critic_avg_5 / runtime / genre at 0.00 coverage."

(B) Per-domain SHAP global importances: train a fast XGB on each domain's OWN
    local feature matrix and print the top features by mean|SHAP|. This shows
    WHICH engineered features actually carry signal in each domain (feeds the
    README feature-engineering section) — a signal-importance diagnostic.

(C) Per-base-learner ablation: for the domains whose deployed model is a STACK
    (movie, tv), score each base learner (XGB / CatBoost / SVR) ALONE on the 50
    frozen folds, vs the equal-mean and simplex blends. Answers err.txt's "is one
    member carrying the ensemble, or are they complementary?"

Outputs: reports/feature_diagnostics.json + reports/shap_<domain>.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import catboost as ctb
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.reporting.standalone_benchmarks import _load_domain, DROP_COLS, _round_half

DOMAINS = ["movie", "tv", "game", "book"]
LOCAL_PATHS = {
    "movie": config.TRAINING_DATA_PATH,
    "tv": config.TV_SHOWS_TRAINING_DATA_PATH,
    "game": config.GAMES_TRAINING_DATA_PATH,
    "book": config.BOOKS_TRAINING_DATA_PATH,
}
SHARED_CHANNELS = ["year", "popularity", "critic_avg_5", "runtime", "box_office_log"]


# ---------------------------------------------------------------- (A) coverage
def coverage_table(out):
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    gen = [c for c in uni.columns if c.startswith("gen_")]
    print("=" * 74)
    print("(A) SHARED-SPACE COVERAGE — non-zero fraction per domain (unified matrix)")
    print("=" * 74)
    header = f"{'domain':>7} {'N':>5} " + " ".join(f"{c[:11]:>11}" for c in SHARED_CHANNELS) + f" {'gen_any':>8}"
    print(header)
    cov = {}
    for dom in DOMAINS:
        b = uni[uni.media_type == dom]
        vals = []
        rec = {}
        for c in SHARED_CHANNELS:
            f = float((pd.to_numeric(b[c], errors="coerce").fillna(0) != 0).mean()) if c in uni else float("nan")
            vals.append(f"{f:>11.2f}")
            rec[c] = round(f, 3)
        gen_any = float((b[gen].fillna(0) != 0).any(axis=1).mean()) if gen else 0.0
        rec["gen_any"] = round(gen_any, 3)
        cov[dom] = rec
        print(f"{dom:>7} {len(b):>5} " + " ".join(vals) + f" {gen_any:>8.2f}")
    out["coverage"] = cov
    print("\n  -> 0.00 = that channel is dead for the domain (running blind on it).")


# ---------------------------------------------------------------- (B) SHAP
def shap_per_domain(out):
    print("\n" + "=" * 74)
    print("(B) PER-DOMAIN SHAP — top engineered features by mean|SHAP| (local XGB surrogate)")
    print("=" * 74)
    try:
        import shap
    except Exception as e:
        print(f"  shap unavailable ({e}); skipping.")
        return
    out["shap_top"] = {}
    for dom in DOMAINS:
        df = pd.read_csv(LOCAL_PATHS[dom])
        df = df[df["target_reg"].notna()]
        feat = [c for c in df.columns if c not in DROP_COLS]
        X, y = df[feat], df["target_reg"]
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4,
                             subsample=0.85, colsample_bytree=0.85,
                             objective="reg:absoluteerror", random_state=42)
        m.fit(X, y)
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(X)
        imp = np.abs(sv).mean(axis=0)
        order = np.argsort(imp)[::-1]
        top = [(feat[i], float(imp[i])) for i in order[:12]]
        out["shap_top"][dom] = top
        print(f"\n  {dom.upper()}  (N={len(df)}, {len(feat)} features)")
        for name, v in top:
            bar = "#" * int(round(v / top[0][1] * 30)) if top[0][1] > 0 else ""
            print(f"    {name:<22} {v:7.4f} {bar}")
        # save bar chart
        names = [t[0] for t in top][::-1]
        vals = [t[1] for t in top][::-1]
        plt.figure(figsize=(7, 4.5))
        plt.barh(names, vals, color="#40bcf4")
        plt.xlabel("mean |SHAP| (impact on predicted rating)")
        plt.title(f"{dom.upper()} — top features (local XGB surrogate)")
        plt.tight_layout()
        plt.savefig(f"reports/shap_{dom}.png", dpi=120)
        plt.close()


# ---------------------------------------------------- (C) per-base-learner ablation
def _xgb():
    return xgb.XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=5, random_state=42)


def _cat():
    return ctb.CatBoostRegressor(iterations=200, learning_rate=0.05, depth=5,
                                 loss_function="MAE", verbose=0, random_seed=42)


def _svr():
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))])


def base_learner_ablation(out):
    print("\n" + "=" * 74)
    print("(C) PER-BASE-LEARNER ABLATION — each member ALONE vs blends (50 folds, per-item OOF)")
    print("=" * 74)
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)
    builders = {"xgb": _xgb, "cat": _cat, "svr": _svr}
    out["base_ablation"] = {}
    for dom in ["movie", "tv"]:
        loc = _load_domain(dom, uni, registry)
        feat = [c for c in loc.columns if c not in DROP_COLS]
        # collect per (item, repeat) raw preds for each member + blends
        recs = []
        for fold_id in range(50):
            tmask = loc["global_id"].apply(lambda g: fold_id in registry.get(g, []))
            if not tmask.any():
                continue
            tr, te = loc[~tmask], loc[tmask]
            Xtr, ytr, Xte = tr[feat], tr["target_reg"], te[feat]
            member_pred = {}
            for k, mk in builders.items():
                mdl = mk(); mdl.fit(Xtr, ytr)
                member_pred[k] = np.asarray(mdl.predict(Xte))
            # simplex weights via inner 3-fold OOF MAE
            inner = KFold(n_splits=3, shuffle=True, random_state=42)
            Xa = Xtr.reset_index(drop=True); ya = np.asarray(ytr)
            oof = {k: np.zeros(len(ya)) for k in builders}
            for itr, iva in inner.split(Xa):
                for k, mk in builders.items():
                    mm = mk(); mm.fit(Xa.iloc[itr], ya[itr]); oof[k][iva] = mm.predict(Xa.iloc[iva])
            Z = np.column_stack([oof[k] for k in builders])
            w = minimize(lambda w: mean_absolute_error(ya, Z @ w), np.ones(3) / 3,
                         method="SLSQP", bounds=[(0, 1)] * 3,
                         constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},)).x
            Ztest = np.column_stack([member_pred[k] for k in builders])
            simplex_pred = Ztest @ w
            mean_pred = Ztest.mean(axis=1)
            base = te[["source_id", "target_reg"]].copy()
            for k in builders:
                base[k] = member_pred[k]
            base["mean"] = mean_pred
            base["simplex"] = simplex_pred
            base["w"] = [tuple(np.round(w, 2))] * len(base)
            recs.append(base)
        oofd = pd.concat(recs, ignore_index=True)
        agg = oofd.groupby("source_id").agg(
            target_reg=("target_reg", "first"),
            **{k: (k, "mean") for k in list(builders) + ["mean", "simplex"]}).reset_index()
        wts = oofd["w"].iloc[len(oofd) // 2] if len(oofd) else None
        print(f"\n  {dom.upper()}  (N={agg['source_id'].nunique()} items; example simplex weights "
              f"[xgb,cat,svr]={wts})")
        print(f"    {'member':>10} {'MAE':>7} {'skill':>7}")
        tmean = agg["target_reg"].mean()
        bmae = mean_absolute_error(agg["target_reg"], np.full(len(agg), tmean))
        res = {}
        for k in list(builders) + ["mean", "simplex"]:
            yp = _round_half(agg[k].values)
            mae = mean_absolute_error(agg["target_reg"], yp)
            sk = 1 - mae / bmae
            res[k] = {"MAE": round(float(mae), 4), "skill": round(float(sk), 4)}
            tag = "  <- best member" if False else ""
            print(f"    {k:>10} {mae:7.4f} {sk:7.4f}")
        out["base_ablation"][dom] = res


def main():
    Path("reports").mkdir(exist_ok=True)
    out = {}
    coverage_table(out)
    shap_per_domain(out)
    base_learner_ablation(out)
    Path("reports/feature_diagnostics.json").write_text(json.dumps(out, indent=2))
    print("\nWrote reports/feature_diagnostics.json + reports/shap_<domain>.png")


if __name__ == "__main__":
    main()
