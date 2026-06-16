"""
Phase 2 — Transfer Grid analysis & verdict (Task 2.2).

Reads reports/transfer_grid_results.csv and emits reports/transfer_grid_summary.json
plus a markdown verdict block. Implements the PRE-REGISTERED decision rules (these
are fixed before looking at any result):

  * A cell is a POSITIVE TRANSFER FINDING iff augmented lift > 0 AND paired
    Wilcoxon p < 0.05 versus Protocol A on identical test items, at >= 2 target
    fractions.
  * Project verdict is "taste transfers" iff >= 1 target has a positive finding;
    otherwise the verdict is the null: "in the shared feature space, no domain
    combination beats local models -- transfer must be item-level."

Deliverables:
  - affinity heatmap (zero-shot skill, single-source -> target)
  - best-subset table (per target, highest significant augmented lift at 100%)
  - learning curves (skill vs target-fraction, best-S vs T-alone)
  - similarity<->transfer correlation (aligned-space MMD + centroid distance vs
    zero-shot skill), Spearman.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.experiments.transfer_study import select_features, RATED_DOMAINS

ALPHA = 0.05


def _paired_p(df_cell, df_A):
    """Paired Wilcoxon of per-fold MAE: cell vs Protocol A on identical folds."""
    m = df_cell.merge(df_A, on="fold", suffixes=("_c", "_A"))
    if len(m) < 3:
        return 1.0
    try:
        return float(wilcoxon(m["MAE_c"], m["MAE_A"]).pvalue)
    except ValueError:
        return 1.0


def aligned_space_distances():
    """Pairwise centroid distance + (linear) MMD between domains in shared space."""
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    feat = select_features(df, max_genres=40)
    cent = {d: df[df.media_type == d][feat].mean().values for d in RATED_DOMAINS}
    dist = {}
    for a in RATED_DOMAINS:
        for b in RATED_DOMAINS:
            if a >= b:
                continue
            ca, cb = cent[a], cent[b]
            centroid_d = float(np.linalg.norm(ca - cb))
            # Linear MMD^2 estimate = ||mean_a - mean_b||^2 (same here); keep both names.
            dist[f"{a}|{b}"] = {"centroid_dist": centroid_d, "mmd": centroid_d ** 2}
    return dist


def analyze(results_path="reports/transfer_grid_results.csv"):
    res = pd.read_csv(results_path)
    summary = {"affinity": {}, "best_subset": {}, "learning_curves": {},
               "positive_findings": [], "similarity_corr": {}}

    # ---- Affinity heatmap: zero-shot (Protocol B) single-source skill ----
    singles = res[(res.protocol == "B") & (~res.music) &
                  (~res.source.str.contains(r"\+", na=False)) &
                  (res.source != "(target-only)")]
    aff = (singles.groupby(["source", "target"])["skill"].mean().reset_index())
    for _, r in aff.iterrows():
        summary["affinity"].setdefault(r["target"], {})[r["source"]] = round(float(r["skill"]), 4)

    # ---- Per-target analysis ----
    for target in RATED_DOMAINS:
        A = res[(res.target == target) & (res.protocol == "A")][["fold", "MAE"]]
        a_mae = A["MAE"].mean()
        C = res[(res.target == target) & (res.protocol == "C")]

        best = None
        for (source, music), g in C.groupby(["source", "music"]):
            frac_hits = 0
            lifts = {}
            for frac, gf in g.groupby("fraction"):
                lift = a_mae - gf["MAE"].mean()  # positive => lower MAE than A
                p = _paired_p(gf[["fold", "MAE"]], A)
                lifts[float(frac)] = {"lift": round(float(lift), 4), "p": round(p, 4)}
                if lift > 0 and p < ALPHA:
                    frac_hits += 1
            lift100 = lifts.get(1.0, {}).get("lift", -1)
            positive = frac_hits >= 2
            if positive:
                summary["positive_findings"].append(
                    {"target": target, "source": source, "music": bool(music),
                     "frac_significant": frac_hits, "lift@100": lift100})
            cand = {"source": source, "music": bool(music), "lift@100": lift100,
                    "frac_significant": frac_hits, "lifts": lifts, "positive": positive}
            if best is None or lift100 > best["lift@100"]:
                best = cand
        summary["best_subset"][target] = best if best else "none"

        # Learning curve: best-S skill vs fraction, and T-alone reference.
        if best and best["source"] != "(target-only)":
            srcsel = C[(C.source == best["source"]) & (C.music == best["music"])]
            curve = srcsel.groupby("fraction")["skill"].mean().to_dict()
            a_skill = res[(res.target == target) & (res.protocol == "A")]["skill"].mean()
            summary["learning_curves"][target] = {
                "best_source": best["source"], "music": best["music"],
                "skill_by_fraction": {str(k): round(float(v), 4) for k, v in curve.items()},
                "target_alone_skill": round(float(a_skill), 4)}

    # ---- Similarity <-> transfer correlation ----
    dist = aligned_space_distances()
    pairs, zshot, cdist, mmd = [], [], [], []
    for target in RATED_DOMAINS:
        for source in RATED_DOMAINS:
            if source == target:
                continue
            key = "|".join(sorted([source, target]))
            sk = summary["affinity"].get(target, {}).get(source)
            if sk is None or key not in dist:
                continue
            pairs.append(f"{source}->{target}")
            zshot.append(sk)
            cdist.append(dist[key]["centroid_dist"])
            mmd.append(dist[key]["mmd"])
    if len(zshot) >= 3:
        rho_c = spearmanr(cdist, zshot).correlation
        rho_m = spearmanr(mmd, zshot).correlation
        summary["similarity_corr"] = {
            "spearman_centroid_dist_vs_zeroshot_skill": round(float(rho_c), 4),
            "spearman_mmd_vs_zeroshot_skill": round(float(rho_m), 4),
            "n_pairs": len(zshot)}

    # ---- Verdict (pre-registered) ----
    transfers = len({f["target"] for f in summary["positive_findings"]}) >= 1
    summary["verdict"] = (
        "TASTE TRANSFERS: at least one target shows a positive transfer finding "
        "(augmented lift > 0, paired p < 0.05, at >= 2 fractions)."
        if transfers else
        "NULL: in the shared feature space, no domain combination beats the local "
        "model (Protocol A). Cross-domain taste signal, if it exists, must be "
        "item-level (entity bridges), not feature-level pooling.")
    summary["verdict_is_positive"] = bool(transfers)

    Path("reports/transfer_grid_summary.json").write_text(json.dumps(summary, indent=2))
    print("✅ summary -> reports/transfer_grid_summary.json")
    print("VERDICT:", summary["verdict"])
    return summary


if __name__ == "__main__":
    analyze()
