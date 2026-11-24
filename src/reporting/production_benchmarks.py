"""
Deployed-model benchmarks for Movies and TV on the FROZEN registry folds.

`standalone_benchmarks.py` scores each domain with a simpler proxy (movie: XGB alone,
TV: a simplex blend). This module scores the model that actually ships --
`src.common.ensemble.mean_ensemble`, the same constructor the trainers call -- on the
same 50 registry folds, deduplicated to one row per item.

It used to benchmark an Optuna-tuned edge-penalty XGB + CatBoost(1000) + SVR + ordinal
stack with a crowd-anchor residual and a 3-fold inner CV per fold. That stack scored
BELOW the plain mean of its own members (movie skill 0.429 vs 0.458, TV 0.229 vs
0.255) and took ~4 h of the ~6 h pipeline; it was replaced on that evidence.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.ensemble import mean_ensemble
from src.reporting.standalone_benchmarks import (_load_domain, DROP_COLS, load_domain_raw, load_registry,
                                                 reencode_fold, registry_folds, map_folds)

LABEL = "Mean-Ensemble (XGB+Cat+SVR)"


def _round_half(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def _metrics(y_true, y_pred_raw):
    yp = _round_half(np.asarray(y_pred_raw))
    yt = np.asarray(y_true)
    return {"MAE": float(mean_absolute_error(yt, yp)),
            "R2": float(r2_score(yt, yp)),
            "Acc": float((np.abs(yt - yp) <= 0.5).mean() * 100)}


def production_oof_for_domain(domain, uni_df, registry):
    loc = _load_domain(domain, uni_df, registry)
    raw = load_domain_raw(domain, loc)
    feat_cols = [c for c in loc.columns if c not in DROP_COLS + ["target_ordinal"]]

    def one_fold(test_mask):
        X_tr, X_te = reencode_fold(domain, loc, raw, test_mask, feat_cols)
        sub = loc.loc[test_mask, ["source_id", "target_reg"]].copy()
        sub["pred"] = mean_ensemble().fit(X_tr, loc.loc[~test_mask, "target_reg"]).predict(X_te)
        sub["media_type"] = domain
        return sub

    rows = map_folds(one_fold, [(m,) for _, m in registry_folds(loc, registry)])
    oof = pd.concat(rows, ignore_index=True)
    per_item = (oof.groupby("source_id")
                   .agg(target_reg=("target_reg", "first"), pred=("pred", "mean"))
                   .reset_index())
    per_item["media_type"] = domain
    met = _metrics(per_item["target_reg"], per_item["pred"])
    print(f"   [{domain}] PRODUCTION  N={per_item['source_id'].nunique()}  "
          f"MAE={met['MAE']:.4f}  R2={met['R2']:.4f}  Acc={met['Acc']:.1f}%", flush=True)
    return per_item, {"Domain": domain, "N": int(per_item["source_id"].nunique()),
                      "Model": LABEL, **met}


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
            bms.append({"Domain": d, "N": int(sl["source_id"].nunique()), "Model": LABEL, **met})
        print(f"   [production] reusing cached per-item OOF <- {out_path}")
        return bms, df

    all_items, bms = [], []
    for d in domains:
        print(f"   [production] evaluating {d} ...", flush=True)
        per_item, bm = production_oof_for_domain(d, uni_df, registry)
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
    reg = load_registry()
    bms, _ = generate_production_oof(uni, reg, use_cache=False)
    print(json.dumps(bms, indent=2))
