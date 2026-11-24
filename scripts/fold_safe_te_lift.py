"""Leaked vs fold-safe target encodings for the deployed Movies/TV ensemble.

Same 50 registry folds, same items, same process and FOLD_JOBS for both arms:
  leaked     the CSV's whole-library `*_encoded` / `nb_*` columns fed straight into
             each fold (what production_oof.csv held until 2026-10-30)
  fold_safe  `reencode_fold`: encodings refit on each fold's training rows (the
             benchmark path now), via generate_production_oof, which also rewrites
             reports/production_oof.csv

Per arm: MAE (rounded to 0.5, project convention), skill vs predict-the-mean and
Spearman exactly as src/experiments/baseline_sanity.py defines them, plus a paired
Wilcoxon on per-item |error|. The published cache is compared with the leaked arm
re-run here so a thread-count effect cannot pass for an encoding effect.

    python -m scripts.fold_safe_te_lift          -> reports/fold_safe_te_lift.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.common.ensemble import mean_ensemble
from src.reporting import production_benchmarks as pb
from src.reporting.standalone_benchmarks import (_load_domain, DROP_COLS, FOLD_JOBS, load_registry,
                                                 map_folds, registry_folds)

OUT = Path("reports/fold_safe_te_lift.json")
CACHE = Path("reports/production_oof.csv")


def _rh(x):
    return np.round(np.clip(np.asarray(x, float), 0.5, 5.0) * 2) / 2


def metrics(d):
    yt, ypr = np.asarray(d["target_reg"], float), np.asarray(d["pred"], float)
    yp = _rh(ypr)
    mae = mean_absolute_error(yt, yp)
    mae_mean = mean_absolute_error(yt, _rh(np.full_like(yt, yt.mean())))
    rho, p = spearmanr(yt, ypr)
    return {"N": int(len(d)), "MAE": round(float(mae), 4), "MAE_raw": round(float(mean_absolute_error(yt, ypr)), 4),
            "MAE_mean": round(float(mae_mean), 4), "skill_vs_mean": round(float(1 - mae / mae_mean), 4),
            "spearman": round(float(rho), 4), "spearman_p": float(p)}


def leaked_oof(domain, uni, reg):
    loc = _load_domain(domain, uni, reg)
    feat = [c for c in loc.columns if c not in DROP_COLS + ["target_ordinal"]]

    def one_fold(mask):
        sub = loc.loc[mask, ["source_id", "target_reg"]].copy()
        sub["pred"] = mean_ensemble().fit(loc.loc[~mask, feat], loc.loc[~mask, "target_reg"]) \
                                     .predict(loc.loc[mask, feat])
        return sub

    rows = map_folds(one_fold, [(m,) for _, m in registry_folds(loc, reg)])
    return (pd.concat(rows, ignore_index=True).groupby("source_id")
              .agg(target_reg=("target_reg", "first"), pred=("pred", "mean")).reset_index())


def compare(a, b):
    """a, b: per-item frames on the same items. Returns paired stats of b against a."""
    m = a.merge(b, on="source_id", suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "arms do not cover the same items"
    assert np.allclose(m["target_reg_a"], m["target_reg_b"]), "arms disagree on targets"
    ea = np.abs(m["target_reg_a"] - m["pred_a"])
    eb = np.abs(m["target_reg_b"] - m["pred_b"])
    diff = eb - ea
    p = float(wilcoxon(eb, ea).pvalue) if np.any(diff != 0) else 1.0
    return {"n_items": int(len(m)), "mean_abs_pred_diff": round(float(np.abs(m["pred_a"] - m["pred_b"]).mean()), 5),
            "max_abs_pred_diff": round(float(np.abs(m["pred_a"] - m["pred_b"]).max()), 5),
            "delta_MAE_raw": round(float(diff.mean()), 5), "wilcoxon_p": p}


def main():
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    reg = load_registry()
    published = pd.read_csv(CACHE) if CACHE.exists() else None

    out = {"date": time.strftime("%Y-%m-%d"), "model": pb.LABEL, "folds": 50, "fold_jobs": FOLD_JOBS,
           "domains": {}}
    leaked = {}
    for d in ("movie", "tv"):
        t = time.time()
        leaked[d] = leaked_oof(d, uni, reg)
        print(f"[{d}] leaked arm done in {time.time() - t:.0f}s: {metrics(leaked[d])}", flush=True)

    t = time.time()
    _, safe = pb.generate_production_oof(uni, reg, use_cache=False)
    print(f"fold-safe arm done in {time.time() - t:.0f}s", flush=True)

    for d in ("movie", "tv"):
        s = safe[safe["media_type"] == d]
        rec = {"leaked": metrics(leaked[d]), "fold_safe": metrics(s),
               "fold_safe_vs_leaked": compare(leaked[d], s)}
        rec["delta"] = {k: round(rec["fold_safe"][k] - rec["leaked"][k], 4)
                        for k in ("MAE", "MAE_raw", "skill_vs_mean", "spearman")}
        if published is not None:
            pub = published[published["media_type"] == d]
            rec["published_cache_vs_leaked_rerun"] = {"published": metrics(pub), **compare(pub, leaked[d])}
        out["domains"][d] = rec
        print(f"[{d}] {json.dumps(rec, indent=1)}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
