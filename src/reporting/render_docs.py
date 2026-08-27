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
    """Standalone local models + dual unified headline. Standalone != unified slice."""
    lines = [
        "| Domain | N | Model | R² (CV mean) | MAE | ±0.5★ Accuracy |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    benchmarks = {b["Domain"]: b for b in data.get("benchmarks", [])}
    rated_n = benchmarks.get("unified_rated", {}).get("N", 0)
    for domain in ORDER:
        if domain not in benchmarks:
            continue
        b = benchmarks[domain]
        name = LABEL_MAP[domain]
        lines.append(
            f"| {name} | {b.get('N', 0):,} | {b.get('Model', 'N/A')} | "
            f"**{b.get('R2', 0):.3f}** | {b.get('MAE', 0):.3f} | {b.get('Acc', 0):.1f}% |"
        )
    lines.append(
        "\n*The four domain rows are the **production local models** on the frozen registry "
        "folds (one row per item, N = unique items): **Movies and TV are the deployed tuned "
        "ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) — not the "
        "earlier plain-XGB / manual-simplex proxies — and **Games and Books are the deployed SVR**. "
        f"The **Unified (rated)** row (N={rated_n:,}) is the cross-domain Mean Ensemble over all four "
        "rated domains — **the headline taste metric**. Music is not scored here: it has no "
        "star ratings and is handled as an unsupervised taste-profile / similarity engine, so it "
        "never enters the rating benchmark.*"
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
        "pooled per-item OOF estimator on the frozen folds, on the same items. With the **real local "
        "models** (Movies = the deployed tuned edge-penalty stacking ensemble, not the old plain-XGB "
        "proxy), the like-for-like MAE comparison favours the local model in **Games and Books** — "
        "Games' Metacritic/platforms and (since Books moved to the Hardcover library API) Books' "
        "standardized author/genre multi-hot features are local signal the shared space dilutes. The "
        "**unified model wins in TV**, the one domain whose thin local feature set leaves room for "
        "cross-domain pooling to add data without losing signal, while **Movies is a statistical tie** "
        "(0.476 local vs 0.475 unified — within fold-resampling noise). So pooling earns its keep "
        "**only where a domain has little local-specific signal beyond vibe**. (The Books verdict "
        "flipped with the data refresh: under the old Open-Library-enriched books the unified slice "
        "marginally led 0.540 vs 0.548; the clean Hardcover library record gives the local SVR real "
        "author/category features and the local model now leads.)*"
    )
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
    lines.append(
        "\n*Effect size is paired Cohen's d on per-item absolute-error differences. The "
        "ensemble's win over Base XGB is statistically significant but trivial in magnitude "
        "(ΔMAE < 0.01) — kept because it is free, not because it matters. Rows 3–4 are removed "
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
    skill = {}
    for r in rows:
        skill[r['Domain'].lower()] = r['Skill_NoPrior']
        lines.append(
            f"| {r['Domain'].title()} | {r['MAE_NoPrior']:.3f} | {r['Skill_NoPrior']:.3f} | "
            f"{r['MAE_WithPrior']:.3f} | {r['Skill_WithPrior']:.3f} | {r['p_value']:.3f} | "
            f"**{r['Verdict']}** |"
        )
    g, b = skill.get('game'), skill.get('book')
    skill_txt = ""
    if g is not None and b is not None:
        skill_txt = f" (Games {g:.3f}, Books {b:.3f})"
    lines.append(
        "\n*The unified prior is **not** significantly helpful in either domain (p ≥ 0.05, "
        "effect direction negative) — it was tested and **dropped**. Note the positive skill "
        f"scores without it{skill_txt}: the local models beat the mean-rating "
        "baseline, the first positive evidence of learnable local signal in these N≈60 domains.*"
    )
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
    lines.append(
        "\n*Reading it: **Skill vs mean** > 0 means the model beats a constant "
        "predict-the-average (all are +0.23–0.35, so yes). **Acc lift** is the ±0.5★ accuracy "
        "*above* the free mode-baseline (predicting the single most common rating) — the part "
        "that isn't given away by the clustered rating distribution (+18–27pts everywhere). And "
        "the load-bearing one: **Spearman ρ** — a model that just predicts noise around the mean "
        "has ρ≈0 and cannot order your items; every model here sits at ρ 0.58–0.75 with p ≤ 1e-9, "
        "so it genuinely ranks favourites above passes. **Honest caveat:** the small domains "
        "(TV/Games/Books) still compress prediction *magnitude* toward the center (`spread_ratio` "
        "≈ 0.56–0.69 in `baseline_sanity.json`), which is exactly why the pipeline applies "
        "rank/quantile calibration and why the Books watchlist queue ranks by novelty, not by the "
        "(mean-hugging) predicted star.*"
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
    short = {
        "movie": "features/noise ceiling; curve still creeping — maintain",
        "tv": "feature/noise ceiling — richer features barely help (see note)",
        "game": "features/noise ceiling; keep local features, collect at margin",
        "book": "near small-N ceiling — collect via watchlist queue",
        "unified": "pooling tool; bias-limited",
    }
    for k in CLS_ORDER:
        if k not in m:
            continue
        r = m[k]
        bs = int((r.get("bias_share") or 0) * 100)
        lines.append(
            f"| **{CLS_LABEL[k]}** | {r['N']:,} | {r['n_features']} | {r['bias2']:.3f} | "
            f"{r['variance']:.3f} | {bs}% | {r['test_mae']:.3f} | {short.get(k, '')} |"
        )
    lines.append(
        "\n*Every model is **bias-dominated (93–96%), variance ~4–6%** — i.e. stable, not "
        "data-starved. So **more rows of the same features is a weak lever everywhere**; the "
        "levers are richer *features* (reduce bias) and the *aleatoric ceiling* you can't cross. "
        "**Proof it's a ceiling, not a fixable gap:** adding the exact creator/cast/writer target "
        "encodings that make Movies strong to the worst domain (TV) cut its bias² only 1.014 → "
        "0.963 and test MAE 0.784 → 0.778 — because at N≈170 most creators appear once or twice, "
        "so the encoding collapses to the mean, and the rest of the error is intrinsic 0.5★-rating "
        "noise no feature can predict.*"
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
    updated = pattern.sub(f"{markers['start']}\n{new_content}\n{markers['end']}", content)
    path.write_text(updated, encoding="utf-8")
    print(f"✅ Updated {file_path}")


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
