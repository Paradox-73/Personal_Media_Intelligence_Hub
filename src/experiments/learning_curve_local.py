"""
Per-domain learning curves on the PRODUCTION LOCAL models (the rigorous version).

Companion to learning_curve.py (which used the shared-space transfer proxy). This
one uses each domain's *deployed local architecture* on its *own engineered
features* — i.e. the model that actually ships — so it answers the real question:
"should I collect more ratings for THIS domain?" with the features the domain
really has (Games' Metacritic/platforms; Movies' target-encoded directors/awards;
etc.), not the shared-space subset.

Local models (from src/reporting/standalone_benchmarks.py, reused verbatim):
  movie : XGBoost (reg:absoluteerror, balanced weights)
  tv    : Simplex-weighted stack (XGB + CatBoost + SVR)
  game  : local SVR(rbf, C=1, eps=0.1)   -- has Metacritic/platforms/devs
  book  : local SVR(rbf, C=1, eps=0.1)   -- thin: year/pageCount/avgRating/ratings + vibe

Same frozen 50-fold registry, same skill = 1 - MAE_model / MAE_(predict-the-mean).
Train on {25,50,75,100}% of each fold's training rows; read the right-edge slope.

Outputs:
  reports/learning_curve_local_results.csv  (raw)
  reports/learning_curve_local_summary.csv  (aggregated)
  reports/learning_curve_local.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.reporting.standalone_benchmarks import (_load_domain, DOMAIN_FIT, DROP_COLS, _round_half, load_domain_raw,
                                               load_registry, reencode_fold, registry_folds, map_folds,
                                               set_calibration, skill)

DOMAINS = ["movie", "tv", "game", "book"]
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
REPEATS = 3
SEED = 42
STEEP, SHALLOW = 0.04, 0.01


def _verdict(edge, total):
    if edge > STEEP:
        return "DATA-LIMITED — collect aggressively (still climbing at 100%)"
    if edge > SHALLOW:
        return "MODEST RETURNS — collect opportunistically (shallow rise)"
    if total > STEEP:
        return "PLATEAUED — gains came early; near the local ceiling"
    return "SIGNAL/FEATURE-LIMITED — flat throughout; more data won't move it"


def run():
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    registry = load_registry()

    print("LOCAL-MODEL learning curves (deployed architectures, own features)")
    print(f"Fractions {[int(f*100) for f in FRACTIONS]}%  | repeats {REPEATS}  | 50 frozen folds\n")

    rows = []
    for dom in DOMAINS:
        set_calibration(dom)  # same per-domain calibration as the benchmark table
        loc = _load_domain(dom, uni, registry)
        raw = load_domain_raw(dom, loc)
        feat = [c for c in loc.columns if c not in DROP_COLS]
        fit_fn = DOMAIN_FIT[dom]
        y_all = loc["target_reg"].to_numpy(dtype=float)
        print(f"=== {dom.upper()}  (rated N={len(loc)}, local features={len(feat)}) ===")
        def one_fold(fold_id, test_mask):
            out = []
            test_mask = np.asarray(test_mask, dtype=bool)
            tr_pos, te_pos = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
            if len(tr_pos) < 4 or len(te_pos) == 0:
                return out
            y_te = y_all[te_pos]
            tmean = y_all[tr_pos].mean()
            for frac in FRACTIONS:
                reps = 1 if frac >= 1.0 else REPEATS
                for rep in range(reps):
                    if frac >= 1.0:
                        sub_pos = tr_pos
                    else:
                        rng = np.random.RandomState(SEED + fold_id * 100 + rep)
                        n = max(3, int(round(frac * len(tr_pos))))
                        sub_pos = tr_pos[rng.choice(len(tr_pos), size=n, replace=False)]
                    # Encodings refit on the subsample only: with fewer rows the entity
                    # maps are thinner too, which is part of what the curve measures.
                    in_sub = np.zeros(len(loc), dtype=bool)
                    in_sub[sub_pos] = True
                    X_sub, X_rest = reencode_fold(dom, loc, raw, ~in_sub, feat)
                    try:
                        pred = fit_fn(X_sub, y_all[sub_pos], X_rest.loc[te_pos])
                    except Exception as e:
                        print(f"    fold {fold_id} frac {frac} failed: {e}")
                        continue
                    out.append(dict(domain=dom, fraction=frac, fold=fold_id, repeat=rep,
                                    n_train=len(sub_pos), n_test=len(y_te),
                                    MAE=mean_absolute_error(y_te, _round_half(pred)),
                                    skill=skill(y_te, _round_half(np.asarray(pred)), tmean)))
            return out

        for fold_rows in map_folds(one_fold, list(registry_folds(loc, registry))):
            rows.extend(fold_rows)
        print(f"    done ({sum(1 for r in rows if r['domain']==dom)} rows)\n")

    res = pd.DataFrame(rows)
    res.to_csv("reports/learning_curve_local_results.csv", index=False)
    agg = (res.groupby(["domain", "fraction"])
              .agg(n_train=("n_train", "mean"), skill_mean=("skill", "mean"),
                   skill_std=("skill", "std"), mae_mean=("MAE", "mean")).reset_index())
    agg["n_train"] = agg["n_train"].round(0).astype(int)
    agg.to_csv("reports/learning_curve_local_summary.csv", index=False)

    print("=" * 78)
    print("LOCAL-MODEL LEARNING-CURVE SUMMARY")
    print("=" * 78)
    for dom in DOMAINS:
        sub = agg[agg.domain == dom].sort_values("fraction")
        if sub.empty:
            continue
        print(f"\n{dom.upper()}")
        print(f"  {'frac':>5} {'N_train':>8} {'skill':>8} {'±std':>7} {'MAE':>7}")
        for _, r in sub.iterrows():
            print(f"  {int(r.fraction*100):>4}% {r.n_train:>8d} {r.skill_mean:>8.3f} "
                  f"{r.skill_std:>7.3f} {r.mae_mean:>7.3f}")
        s = sub.set_index((sub.fraction * 100).astype(int))["skill_mean"]
        edge = float(s.get(100, np.nan) - s.get(75, np.nan))
        total = float(s.get(100, np.nan) - s.get(25, np.nan))
        print(f"  edge slope (100-75) = {edge:+.3f}   total gain (100-25) = {total:+.3f}")
        print(f"  VERDICT: {_verdict(edge, total)}")

    plt.figure(figsize=(8, 5.5))
    colors = {"movie": "#ff8000", "tv": "#40bcf4", "game": "#00e054", "book": "#b388ff"}
    for dom in DOMAINS:
        sub = agg[agg.domain == dom].sort_values("fraction")
        if sub.empty:
            continue
        plt.errorbar(sub["n_train"], sub["skill_mean"], yerr=sub["skill_std"],
                     marker="o", capsize=3, label=dom, color=colors.get(dom))
    plt.axhline(0, color="#888", lw=0.8, ls="--")
    plt.xlabel("training items (N)")
    plt.ylabel("skill score (vs predict-the-mean)")
    plt.title("Per-domain learning curves — LOCAL models (own features)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig("reports/learning_curve_local.png", dpi=130)
    print("\nWrote reports/learning_curve_local_{results,summary}.csv and .png")


if __name__ == "__main__":
    run()
