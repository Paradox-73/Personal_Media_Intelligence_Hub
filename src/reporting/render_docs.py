"""
Single source-of-truth doc renderer (Task 0.2).

Reads reports/latest_metrics.json and regenerates the metric blocks in README.md
(between <!-- BENCHMARKS:BEGIN/END -->) and FINAL_ML_TECHNICAL_REPORT.md (between
<!-- METRICS:BEGIN/END -->). Exits non-zero if the JSON is missing/unparseable.
Idempotent: running twice produces an identical file.
"""
import json
import re
from pathlib import Path

ORDER = ["movie", "unified_rated", "tv", "game", "book"]
LABEL_MAP = {
    "movie": "**Movies**",
    "unified_rated": "**Unified (rated)**",
    "tv": "**TV Shows**",
    "game": "**Games**",
    "book": "**Books**",
}


def format_benchmarks_table(data):
    """Standalone local models + dual unified headline. Standalone != unified slice.

    Skill leads the table. ~60% of films sit in three adjacent half-star bins, so a
    model that always says 3.5 already scores a low MAE and a high +-0.5 accuracy;
    skill = 1 - MAE / MAE(predict the mean) is the share of that free baseline's
    error the model removes, so a constant predictor scores 0 by construction.
    Skill and Spearman come from reports/baseline_sanity.json (same OOF predictions)."""
    lines = [
        "| Domain | N | Model | **Skill vs mean** | Spearman ρ | MAE | MAE (predict mean) | R² | ±0.5★ Acc (mode baseline) |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    benchmarks = {b["Domain"]: b for b in data.get("benchmarks", [])}
    base = (_load_json("reports/baseline_sanity.json") or {}).get("models", {})
    rated_n = benchmarks.get("unified_rated", {}).get("N", 0)
    for domain in ORDER:
        if domain not in benchmarks:
            continue
        b = benchmarks[domain]
        r = base.get("unified" if domain == "unified_rated" else domain, {})
        skill = f"**{r['skill_vs_mean']:+.3f}**" if "skill_vs_mean" in r else "n/a"
        rho = f"{r['spearman']:.3f}" if "spearman" in r else "n/a"
        mae_mean = f"{r['MAE_mean']:.3f}" if "MAE_mean" in r else "n/a"
        acc_mode = f" ({r['acc_mode_baseline']:.0f}%)" if "acc_mode_baseline" in r else ""
        lines.append(
            f"| {LABEL_MAP[domain]} | {b.get('N', 0):,} | {b.get('Model', 'N/A')} | {skill} | {rho} | "
            f"{b.get('MAE', 0):.3f} | {mae_mean} | {b.get('R2', 0):.3f} | {b.get('Acc', 0):.1f}%{acc_mode} |"
        )
    lines.append(
        "\n*Read **Skill vs mean** first: 1 − MAE ÷ MAE of always predicting the average rating. "
        "Ratings cluster in 3–4★, so a constant 3.5★ already earns a respectable MAE and ±0.5★ accuracy "
        "(the bracketed mode baseline is what predicting the single most common rating scores); skill "
        "is 0 for any constant predictor and 1 for a perfect one, so it is the number the clustering "
        "cannot inflate. **Spearman ρ** asks whether the model orders items correctly. "
        "The four domain rows are the **production local models** on the frozen registry "
        "folds (one row per item, N = unique items): **Movies and TV are the deployed tuned "
        "ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) and "
        "**Games and Books are the deployed SVR**. "
        f"The **Unified (rated)** row (N={rated_n:,}) is the cross-domain Mean Ensemble over all four "
        "rated domains. Music is not scored here: it has no star ratings and is handled as an "
        "unsupervised taste-profile / similarity engine.*"
    )
    return "\n".join(lines)


def format_slices_table(data):
    """Unified model's per-domain OOF slice (distinct from the standalone benchmarks)."""
    slices = data.get("slices", [])
    if not slices:
        return ""
    lines = [
        "#### Per-Domain Slice — *Unified model, by domain* (registry OOF)\n",
        "| Domain | N | R² | MAE | ±0.5★ Acc |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    pretty = {"movie": "Movies", "tv": "Shows", "game": "Games", "book": "Books"}
    for s in slices:
        lines.append(
            f"| {pretty.get(s['Domain'], s['Domain'])} | {s.get('N', 0):,} | "
            f"{s.get('R2', 0):.3f} | {s.get('MAE', 0):.3f} | {s.get('Acc', 0):.1f}% |"
        )
    lines.append(
        "\n*Read against the **production benchmarks** above — both measured with the identical "
        "pooled per-item OOF estimator on the frozen folds, on the same items. The verdict per "
        "domain below is computed from the two tables on every render, so it cannot go stale.*"
    )
    bench = {b["Domain"]: b for b in data.get("benchmarks", [])}
    verdicts = []
    for sl in slices:
        d = sl["Domain"]
        if d not in bench:
            continue
        loc, uni = bench[d]["MAE"], sl["MAE"]
        diff = uni - loc
        who = ("tie (within 0.02 MAE, fold-resampling noise)" if abs(diff) < 0.02
               else "local model wins" if diff > 0 else "unified slice wins")
        verdicts.append(f"{pretty.get(d, d)}: local {loc:.3f} vs unified {uni:.3f} → {who}")
    if verdicts:
        lines.append("\n*" + " · ".join(verdicts) + ".*")
    return "\n".join(lines)


def format_ablation_table(data):
    lines = [
        "| Protocol | MAE | R² | Effect Size | p-value |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in data.get("ablation", []):
        lines.append(
            f"| {row['Protocol']} | {row['MAE']:.4f} | {row['R2']:.4f} | "
            f"{row.get('EffectSize', 'N/A')} | {row.get('p', 'N/A')} |"
        )
    rows = {r["Protocol"]: r for r in data.get("ablation", [])}
    base = next((r for k, r in rows.items() if k.startswith("Base XGB")), None)
    ens = next((r for k, r in rows.items() if k.startswith("Mean Ensemble")), None)
    delta = ""
    if base and ens:
        delta = (f"The ensemble's MAE is {ens['MAE'] - base['MAE']:+.4f} vs Base XGB "
                 f"({'better' if ens['MAE'] < base['MAE'] else 'worse'}). ")
    lines.append(
        "\n*Unlike the benchmark table, MAE here is scored **per fold row**: each item is "
        "tested once in each of the 5 CV repeats and counted 5 times, so it differs slightly "
        "from the per-item benchmark number. Effect size is paired Cohen's d and p is a "
        "Wilcoxon test on those same 5×-replicated rows, so p is optimistic (pairs are not "
        f"independent). {delta}Rows 3–4 are removed "
        "paths kept for the record; their numbers come from the legacy 5-fold protocol, not the "
        "frozen registry, and are directionally indicative only.*"
    )
    return "\n".join(lines)


def format_distillation_table(data):
    rows = data.get("distillation", [])
    if not rows:
        return ""
    lines = [
        "#### Distillation-Prior Ablation (Games & Books)\n",
        "| Domain | MAE (no prior) | Skill (no prior) | MAE (with prior) | Skill (with prior) | p-value | Verdict |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    effects = []
    for r in rows:
        lines.append(
            f"| {r['Domain'].title()} | {r['MAE_NoPrior']:.3f} | {r['Skill_NoPrior']:.3f} | "
            f"{r['MAE_WithPrior']:.3f} | {r['Skill_WithPrior']:.3f} | {r['p_value']:.3f} | "
            f"**{r['Verdict']}** |"
        )
        moved = "lowered" if r['MAE_WithPrior'] < r['MAE_NoPrior'] else "raised"
        sig = "significant" if r['p_value'] < 0.05 else "not significant"
        effects.append(f"{r['Domain'].title()}: the prior {moved} MAE "
                       f"{r['MAE_NoPrior']:.3f} → {r['MAE_WithPrior']:.3f}, "
                       f"p = {r['p_value']:.2f} ({sig})")
    kept = [r['Domain'].title() for r in rows if r['Verdict'] == "KEEP"]
    verdict = (f"kept for {', '.join(kept)}" if kept
               else "tested and **dropped** in every domain")
    lines.append("\n*" + "; ".join(effects) + f". The unified prior was {verdict}.*")
    return "\n".join(lines)


SHAP_MODEL_LABEL = {
    "movie": "Movies (stacking ensemble)",
    "tv": "TV Shows (simplex-stack)",
    "game": "Games (local SVR)",
    "book": "Books (local SVR)",
    "unified": "Unified (Mean Ensemble)",
}
SHAP_ORDER = ["movie", "tv", "game", "book", "unified"]
CLS_LABEL = {"movie": "Movies", "tv": "TV Shows", "game": "Games",
             "book": "Books", "unified": "Unified"}
CLS_ORDER = ["movie", "tv", "game", "book", "unified"]
CLASS_NAMES = ["Dislike", "Neutral", "Like"]


def _load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def format_feature_importance():
    """Top predictive features per model, from reports/feature_diagnostics.json
    (mean |SHAP| of a per-model XGB surrogate). Surrogate importance is a proxy
    for the deployed model's driver features — the relative ordering is stable."""
    data = _load_json("reports/feature_diagnostics.json")
    if not data or "shap_top" not in data:
        return None
    shap_top = data["shap_top"]
    lines = [
        "*Top features by mean |SHAP| on a per-model XGB surrogate trained on each "
        "model's own feature matrix (regenerate with `python -m src.experiments.feature_diagnostics`; "
        "per-model bar charts in `reports/shap_<model>.png`). Values are the average absolute "
        "impact on the predicted star rating.*\n",
        "| Model | Most predictive features (mean \\|SHAP\\|, high → low) |",
        "| :--- | :--- |",
    ]
    for m in SHAP_ORDER:
        if m not in shap_top:
            continue
        top5 = shap_top[m][:5]
        cells = ", ".join(f"`{n}` ({v:.3f})" for n, v in top5)
        lines.append(f"| **{SHAP_MODEL_LABEL[m]}** | {cells} |")
    return "\n".join(lines)


def format_classification():
    """Confusion matrices + precision/recall/F1/accuracy per deployed model, from
    reports/classification_diagnostics.json (3-class Dislike/Neutral/Like on the
    cached registry OOF; no retraining)."""
    data = _load_json("reports/classification_diagnostics.json")
    if not data or "models" not in data:
        return None
    models = data["models"]

    out = [
        "*The same registry OOF predictions scored on the regression benchmark table, "
        "re-framed as a 3-class taste decision — **Dislike** (≤2.5★), **Neutral** (2.5–3.5★], "
        "**Like** (>3.5★) — plus a binary **Like** (≥3.5★, the recommend/skip decision). "
        "Predictions are snapped to the 0.5★ grid first. Regenerate with "
        "`python -m src.experiments.classification_diagnostics`; per-model heatmaps in "
        "`reports/confusion_<model>.png`. Music is excluded (no ground-truth rating).*\n",
        "#### Headline classification scores (per deployed model)\n",
        "| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for m in CLS_ORDER:
        if m not in models:
            continue
        r = models[m]
        out.append(
            f"| **{CLS_LABEL[m]}** | {r['N']:,} | {r['accuracy']*100:.1f}% | "
            f"{r['balanced_accuracy']*100:.1f}% | {r['macro_avg']['f1']:.3f} | "
            f"{r['weighted_avg']['f1']:.3f} | {r['cohen_kappa']:.3f} | "
            f"{r['binary_like']['f1']:.3f} |"
        )

    # Per-model confusion matrices + per-class P/R/F1
    out.append("\n#### Confusion matrices & per-class scores\n")
    for m in CLS_ORDER:
        if m not in models:
            continue
        r = models[m]
        cm = r["confusion_matrix"]
        out.append(f"**{CLS_LABEL[m]}** · {r['model']} · N={r['N']} · "
                   f"accuracy {r['accuracy']*100:.1f}%\n")
        out.append("| Actual ↓ / Pred → | Dislike | Neutral | Like |")
        out.append("| :--- | :---: | :---: | :---: |")
        for i, name in enumerate(CLASS_NAMES):
            out.append(f"| **{name}** | {cm[i][0]} | {cm[i][1]} | {cm[i][2]} |")
        out.append("")
        out.append("| Class | Precision | Recall | F1 | Support |")
        out.append("| :--- | :--- | :--- | :--- | :--- |")
        for c in r["per_class"]:
            out.append(
                f"| {CLASS_NAMES[c['class']]} | {c['precision']:.3f} | "
                f"{c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |"
            )
        b = r["binary_like"]
        out.append(
            f"| **Like (≥3.5★, binary)** | {b['precision']:.3f} | {b['recall']:.3f} | "
            f"{b['f1']:.3f} | {b['n_liked_true']} |"
        )
        out.append("")
    return "\n".join(out)


def format_baseline_sanity():
    """Anti-cheating table: does each model beat predict-the-mean, and does it order
    items correctly (Spearman)? From reports/baseline_sanity.json."""
    data = _load_json("reports/baseline_sanity.json")
    if not data or "models" not in data:
        return None
    m = data["models"]
    lines = [
        "*The honest test of \"did it learn taste, or is it just predicting near the "
        "mean/median?\" MAE and ±0.5★ accuracy are gameable in clustered domains — so this "
        "table pits each deployed model against the trivial baselines. Regenerate with "
        "`python -m src.experiments.baseline_sanity`.*\n",
        "| Model | N | MAE | MAE (mean) | Skill vs mean | ±0.5★ acc | ±0.5★ acc (mode) | Acc lift | **Spearman ρ** | p |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for k in CLS_ORDER:
        if k not in m:
            continue
        r = m[k]
        lines.append(
            f"| **{CLS_LABEL[k]}** | {r['N']:,} | {r['MAE_model']:.3f} | {r['MAE_mean']:.3f} | "
            f"{r['skill_vs_mean']:+.3f} | {r['acc_model']:.0f}% | {r['acc_mode_baseline']:.0f}% | "
            f"{r['acc_lift_over_mode']:+.0f}% | **{r['spearman']:.3f}** | {r['spearman_p']:.0e} |"
        )
    vals = [m[k] for k in CLS_ORDER if k in m]
    rng = lambda key, fmt: f"{fmt.format(min(v[key] for v in vals))} to {fmt.format(max(v[key] for v in vals))}"
    small = [m[k] for k in ("tv", "game", "book") if k in m]
    spread = (f"{min(v['spread_ratio'] for v in small):.2f}–{max(v['spread_ratio'] for v in small):.2f}"
              if small else "n/a")
    lines.append(
        "\n*Reading it: **Skill vs mean** > 0 means the model beats a constant "
        f"predict-the-average (here {rng('skill_vs_mean', '{:+.2f}')}). **Acc lift** is the ±0.5★ "
        "accuracy *above* the free mode-baseline (predicting the single most common rating) — the part "
        f"that isn't given away by the clustered rating distribution ({rng('acc_lift_over_mode', '{:+.0f}')} pts). "
        "And the load-bearing one: **Spearman ρ** — a model that just predicts noise around the mean "
        f"has ρ≈0 and cannot order your items; every model here sits at ρ {rng('spearman', '{:.2f}')} "
        f"with p ≤ {max(v['spearman_p'] for v in vals):.0e}, so it genuinely ranks favourites above "
        "passes. **Honest caveat:** the small domains (TV/Games/Books) still compress prediction "
        f"*magnitude* toward the center (`spread_ratio` {spread} in `baseline_sanity.json`), which is "
        "exactly why the pipeline applies rank/quantile calibration and why the Books watchlist queue "
        "ranks by novelty, not by the (mean-hugging) predicted star.*"
    )
    return "\n".join(lines)


def format_bias_variance():
    """Bias/variance decomposition per model + the data-vs-features prescription.
    From reports/bias_variance.json."""
    data = _load_json("reports/bias_variance.json")
    if not data or "models" not in data:
        return None
    m = data["models"]
    lines = [
        "*Bootstrap bias/variance decomposition on each domain's own local features "
        "(`python -m src.experiments.bias_variance`). `bias²` = systematic error of the "
        "average prediction; `variance` = how much predictions wobble as the training set is "
        "resampled. bias² here also absorbs the irreducible 0.5★-self-rating noise.*\n",
        "| Model | N | feats | bias² | variance | **bias share** | test MAE | Diagnosis |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for k in CLS_ORDER:
        if k not in m:
            continue
        r = m[k]
        bs = round((r.get("bias_share") or 0) * 100)
        # The JSON's own prescription, minus its long tail after the arrow.
        diag = str(r.get("prescription", "")).split("→")[0].strip()
        lines.append(
            f"| **{CLS_LABEL[k]}** | {r['N']:,} | {r['n_features']} | {r['bias2']:.3f} | "
            f"{r['variance']:.3f} | {bs}% | {r['test_mae']:.3f} | {diag} |"
        )
    vals = [m[k] for k in CLS_ORDER if k in m]
    pct = lambda key: (f"{min(round((v.get(key) or 0) * 100) for v in vals)}–"
                       f"{max(round((v.get(key) or 0) * 100) for v in vals)}%")
    dominated = all((v.get("bias_share") or 0) > 0.5 for v in vals)
    lines.append(
        f"\n*Bias share spans **{pct('bias_share')}**, variance share {pct('var_share')}. "
        + ("Every model is bias-dominated — stable, not data-starved — so **more rows of the "
           "same features is a weak lever**; the levers are richer *features* (reduce bias) "
           "and the *aleatoric ceiling* you can't cross.*" if dominated else
           "At least one model is variance-dominated; more rows would help it.*")
    )
    return "\n".join(lines)


def format_params(data):
    p = data.get("params", {})
    lam = p.get("lambda", "N/A")
    ah, al = p.get("alpha_hi"), p.get("alpha_lo")
    alpha_txt = ("not applied in the evaluated Mean Ensemble"
                 if ah is None else f"α_hi={ah}, α_lo={al}")
    return (f"*Provenance: λ (temporal decay) = {lam}; asymmetric-objective α: {alpha_txt}. "
            f"Generated at {data.get('generated_at', 'N/A')} · git `{str(data.get('git_hash',''))[:8]}`.*")


def update_file(file_path, markers, new_content):
    path = Path(file_path)
    if not path.exists():
        print(f"⚠️ Warning: {file_path} not found.")
        return
    content = path.read_text(encoding="utf-8")
    pattern = re.compile(rf"{markers['start']}.*?{markers['end']}", re.DOTALL)
    if not pattern.search(content):
        print(f"❌ Error: Markers not found in {file_path}.")
        return
    block = f"{markers['start']}\n{new_content}\n{markers['end']}"
    updated = pattern.sub(lambda _: block, content)  # lambda: no backslash-escape parsing
    path.write_text(updated, encoding="utf-8")
    print(f"✅ Updated {file_path}")


def format_transfer_verdict():
    """Render the pre-registered transfer-grid verdict from its OWN artifact.

    This block carried `<!-- TRANSFER_VERDICT:BEGIN/END -->` markers but nothing
    generated it -- hand-written prose wearing auto-generated clothing, which is
    worse than plain prose: readers and doc-updating agents treat a marker block as
    untouchable AND self-refreshing, so it silently kept asserting a POSITIVE
    transfer finding for a grid that had since been recomputed to NULL.
    """
    data = _load_json("reports/transfer_grid_summary.json")
    if not data:
        return None
    verdict = str(data.get("verdict", "")).strip()
    positive = bool(data.get("verdict_is_positive", False))
    best = data.get("best_subset") or {}
    sim = data.get("similarity_corr") or {}

    label = "TASTE TRANSFERS" if positive else "NULL - no feature-level transfer"
    out = [f"**Realized verdict (full 50-fold grid, domain-blind, quadruple-controlled): "
           f"{label}.** {verdict}", "",
           "| Target | Best source subset | lift @100% | p | Clears the bar? |",
           "| :--- | :--- | ---: | ---: | :--- |"]
    for tgt in ("movie", "tv", "game", "book"):
        b = best.get(tgt)
        if not isinstance(b, dict):
            continue
        p100 = ((b.get("lifts") or {}).get("1.0") or {}).get("p")
        ptxt = f"{p100:.3f}" if isinstance(p100, (int, float)) else "-"
        out.append(f"| {tgt} | `{b.get('source', '?')}` | {b.get('lift@100', 0.0):+.4f} "
                   f"| {ptxt} | {'YES' if b.get('positive') else 'no'} |")

    # Zero-shot, reported as SPEARMAN. The pre-registered rule above scores augmented
    # lift in MAE-skill only, which is why this grid read NULL: with 62% of films in
    # three adjacent half-star bins a constant 3.5* already earns MAE 0.759, so skill
    # barely separates any model from a constant AND conflates ordering with
    # calibration. Rank correlation is the question actually being asked -- does taste
    # in one domain ORDER another. Sign is read from the data, never asserted here.
    zs = data.get("zero_shot") or {}
    if zs:
        # "Target-only XGB" is transfer-grid Protocol A (shared features only), not the
        # deployed model; the deployed column comes from baseline_sanity.json.
        deployed = (_load_json("reports/baseline_sanity.json") or {}).get("models", {})
        out += ["", "**Zero-shot -- can other domains rank a target the model has never "
                "trained on?** (Spearman, mean over folds)", "",
                "| Target | Target-only XGB (same shared features) | Deployed model "
                "| Best source (never sees target) | rho | % of target-only XGB |",
                "| :--- | ---: | ---: | :--- | ---: | ---: |"]
        beats, beats_deployed = [], []
        for tgt in ("movie", "tv", "game", "book"):
            z = zs.get(tgt) or {}
            b = z.get("best") or {}
            if not b:
                continue
            pct = b.get("pct_of_full_model") or 0.0
            cell = f"**{100 * pct:.0f}%**" if pct > 1.0 else f"{100 * pct:.0f}%"
            dep = (deployed.get(tgt) or {}).get("spearman")
            pair = f"`{b.get('source', '?')}->{tgt}`"
            if pct > 1.0:
                beats.append(pair)
                if isinstance(dep, (int, float)) and b.get("rho", 0.0) > dep:
                    beats_deployed.append(pair)
            dep_txt = f"{dep:.3f}" if isinstance(dep, (int, float)) else "n/a"
            out.append(f"| {tgt} | {z.get('rho_full_model', 0.0):.3f} | {dep_txt} "
                       f"| `{b.get('source', '?')}` | {b.get('rho', 0.0):+.3f} | {cell} |")
        if beats:
            out += ["", "Bold rows are targets where a model that has **never seen the "
                    "domain** ranks it better than a target-only XGB on the same shared "
                    "features: " + ", ".join(beats) + ". Against the deployed model "
                    "(pooled per-item Spearman from `baseline_sanity.json`, a different "
                    "estimator than the per-fold mean) it "
                    + ("also wins for " + ", ".join(beats_deployed) if beats_deployed
                       else "wins for none of them") + "."]

    # Mechanism. Why a domain's own data helps, does nothing, or actively hurts.
    coh = _load_json("reports/rating_coherence.json") or {}
    ordered = coh.get("ordered") or []
    if ordered:
        out += ["", "**Why the sign differs by domain.** Local rating coherence -- how "
                "well an item's rating is predicted by its nearest neighbour in the "
                "shared text space -- orders the domains identically to the value of "
                "their own training data:", "",
                "| Domain | n | Coherence (rho) | p | Effect of its own data on a "
                "movie-trained model |",
                "| :--- | ---: | ---: | ---: | :--- |"]
        for r in ordered:
            out.append(f"| {r.get('domain', '?')} | {r.get('n', 0)} "
                       f"| {r.get('coherence_rho', 0.0):+.3f} | {r.get('coherence_p', 0.0):.4f} "
                       f"| {r.get('own_data_effect', '?')} |")
        tv = (coh.get("coherence") or {}).get("tv") or {}
        if tv and not tv.get("neighbour_beats_mean", True):
            out += ["", f"TV is the only domain where the mean gap to the nearest "
                    f"neighbour ({tv.get('mean_abs_gap_to_neighbour', 0):.3f}) EXCEEDS the "
                    f"spread of its own ratings ({tv.get('sd_of_ratings', 0):.3f}) -- "
                    "copying a lookalike is worse than guessing the average. Boosting "
                    "still fits those rows; with no local structure to fit it fits noise, "
                    "into exactly the region where the TV test items live."]

    sc = sim.get("spearman_centroid_dist_vs_zeroshot_skill")
    if isinstance(sc, (int, float)):
        out += ["", f"Aligned-space distance still does not predict transfer "
                f"(Spearman {sc:+.3f} over n={sim.get('n_pairs', '?')} pairs)."]
    return chr(10).join(out)


def format_transfer_grid_full():
    """The whole grid, every target x every source subset, in rank correlation.

    The verdict block above reports only the BEST source per target, which hides the
    shape of the result: Movies is the strongest source into all three other domains and
    Books is the weakest lone source everywhere. Generated by
    src/experiments/transfer_grid_table.py -- nothing here is hand-typed.
    """
    path = Path("reports/transfer_grid_full.md")
    if not path.exists():
        return None
    body = path.read_text(encoding="utf-8")
    # Drop the generated H1 and its lead-in; the host documents supply their own heading.
    lines = body.splitlines()
    out, seen = [], False
    for ln in lines:
        if ln.startswith("## Target:"):
            seen = True
        if seen:
            out.append(ln)
    return chr(10).join(out).strip() or None


def render_docs():
    metrics_path = Path("reports/latest_metrics.json")
    if not metrics_path.exists():
        print("❌ Error: latest_metrics.json missing.")
        raise SystemExit(1)
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("❌ Error: latest_metrics.json is unparseable.")
        raise SystemExit(1)

    benchmarks_md = format_benchmarks_table(data)

    # README: benchmarks block only.
    update_file("README.md",
                {"start": "<!-- BENCHMARKS:BEGIN -->", "end": "<!-- BENCHMARKS:END -->"},
                benchmarks_md)

    # The transfer verdict now regenerates from reports/transfer_grid_summary.json
    # instead of being frozen prose behind auto-generated-looking markers.
    _tv = format_transfer_verdict()
    if _tv:
        for _doc in ("README.md", "reports/FINAL_ML_TECHNICAL_REPORT.md"):
            update_file(_doc,
                        {"start": "<!-- TRANSFER_VERDICT:BEGIN -->",
                         "end": "<!-- TRANSFER_VERDICT:END -->"}, _tv)

    # The whole grid, not just the best cell per target.
    _tg = format_transfer_grid_full()
    if _tg:
        for _doc in ("README.md", "docs/ML_SYSTEM_OVERVIEW.md",
                     "reports/FINAL_ML_TECHNICAL_REPORT.md"):
            update_file(_doc,
                        {"start": "<!-- TRANSFER_GRID_FULL:BEGIN -->",
                         "end": "<!-- TRANSFER_GRID_FULL:END -->"}, _tg)

    # Technical report: full metric block.
    parts = [
        f"### Performance Summary",
        "",
        format_params(data),
        "",
        "#### Benchmarks — *standalone local models* + unified headline\n",
        benchmarks_md,
        "",
        format_slices_table(data),
        "",
        "#### Ablation Study (rated-only, frozen registry folds)\n",
        format_ablation_table(data),
        "",
        format_distillation_table(data),
    ]
    tech_md = "\n".join(p for p in parts if p is not None)
    update_file("reports/FINAL_ML_TECHNICAL_REPORT.md",
                {"start": "<!-- METRICS:BEGIN -->", "end": "<!-- METRICS:END -->"},
                tech_md)

    # Feature-importance + classification blocks (optional — rendered only if the
    # diagnostics JSON exists). Injected into BOTH README and the technical report.
    fi_md = format_feature_importance()
    cls_md = format_classification()
    bs_md = format_baseline_sanity()
    bv_md = format_bias_variance()
    for doc in ["README.md", "reports/FINAL_ML_TECHNICAL_REPORT.md"]:
        if fi_md is not None:
            update_file(doc,
                        {"start": "<!-- FEATURE_IMPORTANCE:BEGIN -->",
                         "end": "<!-- FEATURE_IMPORTANCE:END -->"},
                        fi_md)
        if cls_md is not None:
            update_file(doc,
                        {"start": "<!-- CLASSIFICATION:BEGIN -->",
                         "end": "<!-- CLASSIFICATION:END -->"},
                        cls_md)
        if bs_md is not None:
            update_file(doc,
                        {"start": "<!-- BASELINE_SANITY:BEGIN -->",
                         "end": "<!-- BASELINE_SANITY:END -->"},
                        bs_md)
        if bv_md is not None:
            update_file(doc,
                        {"start": "<!-- BIAS_VARIANCE:BEGIN -->",
                         "end": "<!-- BIAS_VARIANCE:END -->"},
                        bv_md)


if __name__ == "__main__":
    render_docs()
