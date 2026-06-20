"""
edge_loss_ablation.py -- Head-to-head ablation of the custom AsymmetricEdgePenalty
XGBoost loss against a standard squared-error baseline, on the FROZEN registry folds.

Motivation
----------
The project ships a custom XGBoost objective (`AsymmetricEdgePenaltyObjective`,
src/movies/custom_objectives.py) that up-weights squared error at the rating
extremes so the model stops regressing every prediction toward the mean (~3).
The tuned alphas (alpha_hi=0.0597, alpha_lo=0.0788) are documented in
reports/PIPELINE_RUN_LOG.md, but the head-to-head "does it actually help?"
comparison was never run. This script runs it.

Design (matches the rest of the project's evaluation discipline)
----------------------------------------------------------------
  * Domain: movie (the domain that ships the loss; N=980 rated items).
  * Same 50 frozen registry folds as every other experiment
    (models/unified/fold_registry.json, RepeatedKFold 10x5, seed 42).
  * Identical XGBoost hyperparameters for BOTH arms -- the ONLY thing that
    changes is the objective. No external sample weights in either arm, so the
    edge penalty is the sole reweighting mechanism (a clean isolation).
        Arm A: objective="reg:squarederror"            (standard squared error)
        Arm B: AsymmetricEdgePenaltyObjective(a_hi,a_lo) (weighted squared error)
  * Per-item OOF: each item lands in 5 test folds; raw predictions are averaged
    across the 5 repeats per item (same dedup as standalone_benchmarks.py).
  * Paired Wilcoxon on per-item |error|, A vs B, computed on the edge items
    (primary) and on all items (secondary).

Metrics
-------
  Primary   : edge-MAE  (true rating <= 1.5 or >= 4.5) ; prediction spread (std).
  Secondary : overall MAE ; exact-match rate at the extremes ; extreme-bin
              coverage (does B use the full 0.5-5 range?).

Outputs
-------
  reports/edge_loss_ablation_results.json
  paper/figures/edge_loss_pred_vs_actual.png   (A vs B scatter, with diagonal)
  paper/figures/edge_loss_calibration.png       (binned predicted vs actual)

Run:
    PYTHONUTF8=1 content_rec/Scripts/python.exe -m src.experiments.edge_loss_ablation
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective
from src.reporting.standalone_benchmarks import _load_domain, DROP_COLS, _round_half

# Tuned alphas as recorded in reports/PIPELINE_RUN_LOG.md (Optuna, movie domain).
ALPHA_HI = 0.0597
ALPHA_LO = 0.0788

# Shared XGBoost hyperparameters -- identical for both arms.
XGB_KW = dict(
    n_estimators=200, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)

EDGE_LO = 1.5   # "hard pass" threshold (true rating <=)
EDGE_HI = 4.5   # "must watch" threshold (true rating >=)

FIG_DIR = config.PROJECT_ROOT / "paper" / "figures" if hasattr(config, "PROJECT_ROOT") \
    else Path(__file__).resolve().parent.parent.parent / "paper" / "figures"
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "reports" / "edge_loss_ablation_results.json"


def _fit_predict(X_tr, y_tr, X_te, objective):
    m = xgb.XGBRegressor(objective=objective, **XGB_KW)
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def _oof_for_arm(loc, feat_cols, registry, objective, label):
    """Run one arm across all 50 frozen folds; return per-item OOF DataFrame."""
    rows = []
    for fold_id in range(50):
        test_mask = loc["global_id"].apply(lambda g: fold_id in registry.get(g, []))
        if not test_mask.any():
            continue
        train_mask = ~test_mask
        X_tr, X_te = loc.loc[train_mask, feat_cols], loc.loc[test_mask, feat_cols]
        y_tr = loc.loc[train_mask, "target_reg"]
        preds = _fit_predict(X_tr, y_tr, X_te, objective)
        sub = loc.loc[test_mask, ["source_id", "target_reg"]].copy()
        sub["pred_raw"] = np.asarray(preds, dtype=float)
        rows.append(sub)
    oof = pd.concat(rows, ignore_index=True)
    per_item = oof.groupby("source_id").agg(
        target_reg=("target_reg", "first"),
        pred_raw=("pred_raw", "mean"),
    ).reset_index()
    # Clip raw to the rating range before any metric (same as the project).
    per_item["pred_clip"] = np.clip(per_item["pred_raw"], 0.5, 5.0)
    per_item["pred_round"] = _round_half(per_item["pred_raw"])
    print(f"   [{label}] {len(per_item)} items OOF  "
          f"mean_pred={per_item['pred_clip'].mean():.3f}  std_pred={per_item['pred_clip'].std():.3f}")
    return per_item


def _arm_metrics(per_item):
    yt = per_item["target_reg"].to_numpy()
    yp_clip = per_item["pred_clip"].to_numpy()
    yp_round = per_item["pred_round"].to_numpy()
    edge = (yt <= EDGE_LO) | (yt >= EDGE_HI)
    return {
        "n": int(len(yt)),
        "n_edge": int(edge.sum()),
        "overall_mae": float(mean_absolute_error(yt, yp_clip)),
        "edge_mae": float(mean_absolute_error(yt[edge], yp_clip[edge])),
        "r2": float(r2_score(yt, yp_clip)),
        "acc_half": float((np.abs(yt - yp_round) <= 0.5).mean() * 100),
        "pred_std": float(np.std(yp_clip)),
        "pred_min": float(np.min(yp_clip)),
        "pred_max": float(np.max(yp_clip)),
        # exact-match at extremes: rounded prediction equals the true extreme rating
        "exact_at_extremes_pct": float((np.abs(yt[edge] - yp_round[edge]) < 1e-9).mean() * 100)
        if edge.sum() else 0.0,
        # how much of the extreme range the arm actually reaches into
        "frac_pred_le_2": float((yp_clip <= 2.0).mean() * 100),
        "frac_pred_ge_4": float((yp_clip >= 4.0).mean() * 100),
    }


def _paired_wilcoxon(a_item, b_item):
    """Paired Wilcoxon on per-item |error|, arm A vs arm B (aligned by source_id)."""
    m = a_item[["source_id", "target_reg", "pred_clip"]].merge(
        b_item[["source_id", "pred_clip"]], on="source_id", suffixes=("_a", "_b"))
    yt = m["target_reg"].to_numpy()
    err_a = np.abs(yt - m["pred_clip_a"].to_numpy())
    err_b = np.abs(yt - m["pred_clip_b"].to_numpy())
    edge = (yt <= EDGE_LO) | (yt >= EDGE_HI)

    def _w(mask):
        ea, eb = err_a[mask], err_b[mask]
        if len(ea) < 5 or np.allclose(ea, eb):
            return {"p": None, "mae_a": float(ea.mean()), "mae_b": float(eb.mean()), "n": int(mask.sum())}
        stat, p = wilcoxon(ea, eb)
        return {"p": float(p), "mae_a": float(ea.mean()), "mae_b": float(eb.mean()), "n": int(mask.sum())}

    return {"all": _w(np.ones_like(edge, dtype=bool)), "edge": _w(edge)}, m


def _plot_pred_vs_actual(merged, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yt = merged["target_reg"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    for ax, col, title in zip(
        axes, ["pred_clip_a", "pred_clip_b"],
        ["Arm A: standard squared-error", "Arm B: edge-penalty loss"],
    ):
        yp = merged[col].to_numpy()
        jitter = (np.random.RandomState(42).rand(len(yt)) - 0.5) * 0.12
        ax.scatter(yt + jitter, yp, s=14, alpha=0.35, edgecolor="none")
        ax.plot([0.5, 5.0], [0.5, 5.0], "k--", lw=1, label="ideal (y=x)")
        # mean prediction per true-rating bucket -- shows the regression-to-mean squash
        buckets = np.round(yt * 2) / 2
        bx = sorted(np.unique(buckets))
        by = [yp[buckets == b].mean() for b in bx]
        ax.plot(bx, by, "o-", color="crimson", lw=1.8, ms=4, label="mean pred / true bucket")
        ax.axvspan(0.5, EDGE_LO, color="grey", alpha=0.08)
        ax.axvspan(EDGE_HI, 5.0, color="grey", alpha=0.08)
        ax.set_title(title)
        ax.set_xlabel("True rating")
        ax.set_xlim(0.3, 5.2)
        ax.set_ylim(0.3, 5.2)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Predicted rating")
    fig.suptitle("Movies: predicted vs. actual -- standard loss (A) vs. edge-penalty loss (B)\n"
                 "(shaded = edge regions where the custom loss is meant to act)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   saved {out}")


def _plot_calibration(merged, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yt = merged["target_reg"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    edges = np.arange(0.5, 5.01, 0.5)
    centers = (edges[:-1] + edges[1:]) / 2
    for col, lab, c in [("pred_clip_a", "Arm A (squared error)", "tab:blue"),
                        ("pred_clip_b", "Arm B (edge penalty)", "tab:red")]:
        yp = merged[col].to_numpy()
        means, lo, hi = [], [], []
        for l, r in zip(edges[:-1], edges[1:]):
            m = (yt >= l) & (yt < r) if r < 5.0 else (yt >= l) & (yt <= r)
            if m.sum() == 0:
                means.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            v = yp[m]
            means.append(v.mean())
            lo.append(v.mean() - v.std()); hi.append(v.mean() + v.std())
        means = np.array(means)
        ax.plot(centers, means, "o-", color=c, label=lab, lw=1.8)
        ax.fill_between(centers, lo, hi, color=c, alpha=0.12)
    ax.plot([0.5, 5.0], [0.5, 5.0], "k--", lw=1, label="perfect calibration")
    ax.axvspan(0.5, EDGE_LO, color="grey", alpha=0.08)
    ax.axvspan(EDGE_HI, 5.0, color="grey", alpha=0.08)
    ax.set_xlabel("True rating (binned)")
    ax.set_ylabel("Mean predicted rating (±1 SD)")
    ax.set_title("Movies calibration: edge-penalty loss vs. standard loss\n"
                 "(closer to the diagonal at the ends = less regression to the mean)", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim(0.3, 5.2); ax.set_ylim(0.3, 5.2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   saved {out}")


# Symmetric alpha-strength sweep, to characterise what the loss does when it
# actually bites (the tuned setting comes out mild, so the head-to-head barely
# moves; the sweep shows the mechanism and the tradeoff it trades against).
SWEEP_ALPHAS = [0.0597, 0.3, 0.6, 1.0]   # alpha_hi == alpha_lo for the sweep rows > tuned
STRONG_ALPHA = 0.8                        # used for the 3rd "un-squash" scatter panel


def _plot_pred_vs_actual_3(merged3, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yt = merged3["target_reg"].to_numpy()
    panels = [
        ("pred_clip_a", "Arm A: standard squared-error"),
        ("pred_clip_b", f"Arm B: edge-penalty (tuned a={ALPHA_HI}/{ALPHA_LO})"),
        ("pred_clip_strong", f"Edge-penalty (strong a={STRONG_ALPHA})"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharex=True, sharey=True)
    rs = np.random.RandomState(42)
    for ax, (col, title) in zip(axes, panels):
        yp = merged3[col].to_numpy()
        jitter = (rs.rand(len(yt)) - 0.5) * 0.12
        ax.scatter(yt + jitter, yp, s=12, alpha=0.30, edgecolor="none")
        ax.plot([0.5, 5.0], [0.5, 5.0], "k--", lw=1, label="ideal (y=x)")
        buckets = np.round(yt * 2) / 2
        bx = sorted(np.unique(buckets))
        by = [yp[buckets == b].mean() for b in bx]
        ax.plot(bx, by, "o-", color="crimson", lw=1.8, ms=4, label="mean pred / true bucket")
        ax.axvspan(0.5, EDGE_LO, color="grey", alpha=0.08)
        ax.axvspan(EDGE_HI, 5.0, color="grey", alpha=0.08)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("True rating")
        ax.set_xlim(0.3, 5.2); ax.set_ylim(0.3, 5.2)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Predicted rating")
    fig.suptitle("Movies predicted vs. actual: the edge-penalty loss only un-squashes the extremes "
                 "when alpha is large;\nat the tuned (mild) setting it is nearly identical to squared error",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   saved {out}")


def _plot_alpha_sweep(sweep, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [s["alpha"] for s in sweep]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(xs, [s["overall_mae"] for s in sweep], "o-", color="tab:blue", label="overall MAE")
    ax1.plot(xs, [s["edge_mae"] for s in sweep], "s-", color="tab:red", label="edge MAE")
    ax1.set_xlabel("edge-penalty strength alpha (symmetric)")
    ax1.set_ylabel("MAE")
    ax2 = ax1.twinx()
    ax2.plot(xs, [s["pred_std"] for s in sweep], "^--", color="tab:green", label="prediction spread (SD)")
    ax2.set_ylabel("prediction spread (SD)", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.set_title("Edge-penalty alpha sweep (movies, 50 folds): stronger alpha widens the\n"
                  "prediction spread and lowers edge-MAE, but raises overall MAE", fontsize=10)
    ax1.axvline(ALPHA_HI, color="grey", ls=":", lw=1)
    ax1.annotate("tuned", (ALPHA_HI, ax1.get_ylim()[1]), fontsize=8, color="grey",
                 ha="left", va="top")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"   saved {out}")


def main():
    print("== Edge-penalty loss ablation (movies, 50 frozen folds) ==")
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)

    loc = _load_domain("movie", uni, registry)
    feat_cols = [c for c in loc.columns if c not in DROP_COLS]
    print(f"   movie items={len(loc)}  features={len(feat_cols)}  "
          f"edge items (<= {EDGE_LO} or >= {EDGE_HI}) = "
          f"{int(((loc['target_reg'] <= EDGE_LO) | (loc['target_reg'] >= EDGE_HI)).sum())}")

    print(f"   Arm A: reg:squarederror")
    a_item = _oof_for_arm(loc, feat_cols, registry, "reg:squarederror", "A std-sq")
    print(f"   Arm B: AsymmetricEdgePenalty(alpha_hi={ALPHA_HI}, alpha_lo={ALPHA_LO})")
    b_item = _oof_for_arm(loc, feat_cols, registry,
                          AsymmetricEdgePenaltyObjective(ALPHA_HI, ALPHA_LO), "B edge")

    metrics_a = _arm_metrics(a_item)
    metrics_b = _arm_metrics(b_item)
    wilcox, merged = _paired_wilcoxon(a_item, b_item)

    # ---- alpha-strength sweep (symmetric) ----
    print("   alpha-strength sweep ...")
    sweep = [{"alpha": 0.0, **{k: metrics_a[k] for k in ("overall_mae", "edge_mae", "pred_std")}}]
    strong_item = None
    for a in SWEEP_ALPHAS:
        s_item = _oof_for_arm(loc, feat_cols, registry,
                              AsymmetricEdgePenaltyObjective(a, a), f"sweep a={a}")
        sm = _arm_metrics(s_item)
        sweep.append({"alpha": a, "overall_mae": sm["overall_mae"],
                      "edge_mae": sm["edge_mae"], "pred_std": sm["pred_std"]})
    # strong arm for the un-squash scatter panel
    strong_item = _oof_for_arm(loc, feat_cols, registry,
                               AsymmetricEdgePenaltyObjective(STRONG_ALPHA, STRONG_ALPHA), "strong")
    merged3 = merged.merge(
        strong_item[["source_id", "pred_clip"]].rename(columns={"pred_clip": "pred_clip_strong"}),
        on="source_id")

    results = {
        "design": {
            "domain": "movie",
            "n_folds": 50,
            "fold_registry": "models/unified/fold_registry.json",
            "xgb_params": XGB_KW,
            "alpha_hi": ALPHA_HI,
            "alpha_lo": ALPHA_LO,
            "alpha_source": "reports/PIPELINE_RUN_LOG.md (Optuna-tuned, movie)",
            "edge_thresholds": {"lo": EDGE_LO, "hi": EDGE_HI},
            "arm_a": "objective=reg:squarederror",
            "arm_b": "objective=AsymmetricEdgePenaltyObjective",
            "note": "Identical hyperparameters and folds; no external sample weights; "
                    "per-item OOF averaged across 5 repeats; predictions clipped to [0.5,5].",
        },
        "arm_a_standard": metrics_a,
        "arm_b_edge": metrics_b,
        "deltas_b_minus_a": {
            "overall_mae": metrics_b["overall_mae"] - metrics_a["overall_mae"],
            "edge_mae": metrics_b["edge_mae"] - metrics_a["edge_mae"],
            "pred_std": metrics_b["pred_std"] - metrics_a["pred_std"],
            "acc_half": metrics_b["acc_half"] - metrics_a["acc_half"],
            "exact_at_extremes_pct": metrics_b["exact_at_extremes_pct"] - metrics_a["exact_at_extremes_pct"],
        },
        "paired_wilcoxon": wilcox,
        "alpha_sweep": sweep,
        "sweep_note": "alpha=0.0 row is Arm A (squared error). Symmetric alpha_hi=alpha_lo.",
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _plot_pred_vs_actual(merged, FIG_DIR / "edge_loss_pred_vs_actual.png")
    _plot_pred_vs_actual_3(merged3, FIG_DIR / "edge_loss_pred_vs_actual_3panel.png")
    _plot_calibration(merged, FIG_DIR / "edge_loss_calibration.png")
    _plot_alpha_sweep(sweep, FIG_DIR / "edge_loss_alpha_sweep.png")

    print("\n== RESULTS ==")
    print(f"  Overall MAE   A={metrics_a['overall_mae']:.4f}  B={metrics_b['overall_mae']:.4f}  "
          f"(B-A={results['deltas_b_minus_a']['overall_mae']:+.4f})  Wilcoxon p={wilcox['all']['p']}")
    print(f"  Edge MAE      A={metrics_a['edge_mae']:.4f}  B={metrics_b['edge_mae']:.4f}  "
          f"(B-A={results['deltas_b_minus_a']['edge_mae']:+.4f})  Wilcoxon p(edge)={wilcox['edge']['p']}")
    print(f"  Pred spread   A={metrics_a['pred_std']:.4f}  B={metrics_b['pred_std']:.4f}")
    print(f"  Acc(+/-0.5)   A={metrics_a['acc_half']:.2f}%  B={metrics_b['acc_half']:.2f}%")
    print(f"  Exact@extreme A={metrics_a['exact_at_extremes_pct']:.1f}%  B={metrics_b['exact_at_extremes_pct']:.1f}%")
    print(f"\n  results -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
