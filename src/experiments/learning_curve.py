"""
Per-domain learning curves — "the test before you collect" (err.txt).

For each rated domain, train on {25, 50, 75, 100}% of its CURRENT ratings and
measure skill vs N. The slope at the right edge converts "should I collect more?"
from a guess into a measured decision:

  * Steep climb at 100%          -> data-limited   -> collect aggressively.
  * Flat / plateaued before 100% -> signal-limited -> more data won't help; fix
                                                       features or accept the ceiling.
  * Shallow rising               -> modest returns -> collect opportunistically.

This reuses the EXACT transfer_study.py augmented-fraction machinery, run with the
target's own data as the only source (no transfer): same frozen registry folds,
same shared-space XGB proxy, same CORAL-on-train alignment, same skill score.

Caveat (stated, not hidden): this is the SHARED-SPACE model (vibe PCA + unified
genres + critic_avg + year + log-popularity). It is the right tool for "how much
would more ratings help in the cross-domain space", and is identical to the model
the transfer grid uses. It does NOT include domain-specific columns (e.g. Games'
Metacritic / platforms), so a domain whose signal is largely domain-specific can
look flatter here than its production local model would.

Outputs:
  reports/learning_curve_results.csv  — one row per (domain, fraction, fold, repeat)
  reports/learning_curve_summary.csv  — aggregated per (domain, fraction)
  reports/learning_curve.png          — skill-vs-N curves, one line per domain
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.experiments.transfer_study import (
    RATED_DOMAINS, SEED, select_features, _fit_predict, skill_score,
)
from sklearn.metrics import mean_absolute_error
import src.experiments.transfer_study as ts

FRACTIONS = (0.25, 0.50, 0.75, 1.00)
REPEATS = 3            # subsample repeats for fractions < 1.0 (1.0 is deterministic)
N_ESTIMATORS = 200     # faithful proxy; cheaper than the 300-tree production grid

# Heuristic slope thresholds for the verdict (documented, not magic).
STEEP = 0.04           # edge slope (skill@100 - skill@75) above this  -> data-limited
SHALLOW = 0.01         # between SHALLOW and STEEP                      -> modest returns
                       # at/below SHALLOW (incl. negative)             -> plateaued


def _verdict(edge_slope, total_gain):
    if edge_slope > STEEP:
        return "DATA-LIMITED — collect aggressively (curve still climbing at 100%)"
    if edge_slope > SHALLOW:
        return "MODEST RETURNS — collect opportunistically (shallow rise)"
    if total_gain > STEEP:
        return "PLATEAUED — gains came early; near the ceiling, more data won't rescue it"
    return "SIGNAL-LIMITED — flat throughout; the problem is upstream of N (fix features)"


def run():
    ts.N_ESTIMATORS = N_ESTIMATORS
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)
    print(f"Shared-space features: {len(feat)} (genre one-hots capped at 40)")
    print(f"Model: XGB({N_ESTIMATORS} trees) proxy, CORAL-on-train, skill vs train-mean baseline")
    print(f"Fractions: {[int(f*100) for f in FRACTIONS]}%  | subsample repeats: {REPEATS}  | all 50 frozen folds\n")

    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)

    rows = []
    for target in RATED_DOMAINS:
        tdf = df[df["media_type"] == target].copy()
        n_total = len(tdf)
        n_rated = int(tdf["target_reg"].notna().sum())
        print(f"=== {target.upper()}  (rated items: {n_rated}) ===")
        for fold_id in range(50):
            test_mask = tdf["global_id"].apply(lambda g: fold_id in registry.get(g, []))
            if not test_mask.any():
                continue
            tr_target = tdf[~test_mask]
            te_target = tdf[test_mask]
            # only rated rows carry a usable target
            tr_target = tr_target[tr_target["target_reg"].notna()]
            te_target = te_target[te_target["target_reg"].notna()]
            if len(tr_target) < 4 or len(te_target) == 0:
                continue
            y_te = te_target["target_reg"].values
            train_mean = tr_target["target_reg"].mean()

            for frac in FRACTIONS:
                reps = 1 if frac >= 1.0 else REPEATS
                for rep in range(reps):
                    if frac >= 1.0:
                        sub = tr_target
                    else:
                        rng = np.random.RandomState(SEED + fold_id * 100 + rep)
                        n = max(2, int(round(frac * len(tr_target))))
                        idx = rng.choice(len(tr_target), size=n, replace=False)
                        sub = tr_target.iloc[idx]
                    pred = _fit_predict(sub[feat], sub["target_reg"], sub["media_type"],
                                        te_target[feat], te_target["media_type"])
                    rows.append(dict(
                        domain=target, fraction=frac, fold=fold_id, repeat=rep,
                        n_train=len(sub), n_test=len(y_te),
                        MAE=mean_absolute_error(y_te, pred),
                        skill=skill_score(y_te, pred, train_mean),
                    ))
        print(f"    done ({sum(1 for r in rows if r['domain']==target)} rows)\n")

    res = pd.DataFrame(rows)
    out_raw = Path("reports/learning_curve_results.csv")
    res.to_csv(out_raw, index=False)

    # ---- Aggregate per (domain, fraction): mean skill, std across folds, mean N ----
    agg = (res.groupby(["domain", "fraction"])
              .agg(n_train=("n_train", "mean"),
                   skill_mean=("skill", "mean"),
                   skill_std=("skill", "std"),
                   mae_mean=("MAE", "mean"))
              .reset_index())
    agg["n_train"] = agg["n_train"].round(0).astype(int)
    out_agg = Path("reports/learning_curve_summary.csv")
    agg.to_csv(out_agg, index=False)

    # ---- Print a readable per-domain table + verdict ----
    print("=" * 78)
    print("LEARNING-CURVE SUMMARY  (skill = 1 - MAE_model / MAE_mean-baseline)")
    print("=" * 78)
    verdicts = {}
    for dom in RATED_DOMAINS:
        sub = agg[agg.domain == dom].sort_values("fraction")
        if sub.empty:
            continue
        print(f"\n{dom.upper()}")
        print(f"  {'frac':>5} {'N_train':>8} {'skill':>8} {'±std':>7} {'MAE':>7}")
        for _, r in sub.iterrows():
            print(f"  {int(r.fraction*100):>4}% {r.n_train:>8d} {r.skill_mean:>8.3f} "
                  f"{r.skill_std:>7.3f} {r.mae_mean:>7.3f}")
        s = sub.set_index((sub.fraction * 100).astype(int))["skill_mean"]
        edge_slope = float(s.get(100, np.nan) - s.get(75, np.nan))
        total_gain = float(s.get(100, np.nan) - s.get(25, np.nan))
        v = _verdict(edge_slope, total_gain)
        verdicts[dom] = (edge_slope, total_gain, v)
        print(f"  edge slope (100%-75%) = {edge_slope:+.3f}   total gain (100%-25%) = {total_gain:+.3f}")
        print(f"  VERDICT: {v}")

    # ---- Plot ----
    plt.figure(figsize=(8, 5.5))
    colors = {"movie": "#ff8000", "tv": "#40bcf4", "game": "#00e054", "book": "#b388ff"}
    for dom in RATED_DOMAINS:
        sub = agg[agg.domain == dom].sort_values("fraction")
        if sub.empty:
            continue
        plt.errorbar(sub["n_train"], sub["skill_mean"], yerr=sub["skill_std"],
                     marker="o", capsize=3, label=dom, color=colors.get(dom))
    plt.axhline(0, color="#888", lw=0.8, ls="--")
    plt.xlabel("training items (N)")
    plt.ylabel("skill score (vs predict-the-mean)")
    plt.title("Per-domain learning curves — skill vs N (shared-space proxy)")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_png = Path("reports/learning_curve.png")
    plt.savefig(out_png, dpi=130)
    print(f"\nWrote:\n  {out_raw}\n  {out_agg}\n  {out_png}")
    return agg, verdicts


if __name__ == "__main__":
    run()
