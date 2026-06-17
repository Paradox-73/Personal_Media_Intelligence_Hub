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

ORDER = ["movie", "unified_rated", "unified_full", "tv", "game", "book"]
LABEL_MAP = {
    "movie": "**Movies**",
    "unified_rated": "**Unified (rated)**",
    "unified_full": "**Unified (full pool)**",
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
    for domain in ORDER:
        if domain not in benchmarks:
            continue
        b = benchmarks[domain]
        name = LABEL_MAP[domain]
        if domain == "unified_full":
            name += "†"
        lines.append(
            f"| {name} | {b.get('N', 0):,} | {b.get('Model', 'N/A')} | "
            f"**{b.get('R2', 0):.3f}** | {b.get('MAE', 0):.3f} | {b.get('Acc', 0):.1f}% |"
        )
    lines.append(
        "\n*The four domain rows are the **production local models** on the frozen registry "
        "folds (one row per item, N = unique items): **Movies and TV are the deployed tuned "
        "ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) — not the "
        "earlier plain-XGB / manual-simplex proxies — and **Games and Books are the deployed SVR**. "
        "The two Unified rows are the cross-domain Mean Ensemble; **the rated row (N=1,264) is the "
        "headline taste metric**.*"
    )
    lines.append(
        "\n*†Unified (full pool) is trained on the rated items **plus 3,688 music PU "
        "pseudo-labels** and evaluated including music via a separate RepeatedKFold(5×1) — "
        "music has no frozen registry. It is **not an actual-taste metric** and is shown only "
        "for transparency; never cite it as the unified result.*"
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
        "proxy), the like-for-like MAE comparison splits cleanly by domain type: the **local model wins "
        "in Movies and Games** — the domains with rich domain-specific features the shared space "
        "discards (Movies' director/actor target-encodings, Games' Metacritic/platforms) — while the "
        "**unified model wins in TV and is level on Books**, the semantic-only domains where pooling "
        "across domains adds data without losing signal. So cross-domain pooling earns its keep **only "
        "where a domain has little local-specific signal beyond vibe**. The earlier 'unified is "
        "competitive or better in movies/TV/books' reading was an artifact of benchmarking against a "
        "**weakened Movies proxy**; with the production model the Movies verdict flips to local.*"
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
    for r in rows:
        lines.append(
            f"| {r['Domain'].title()} | {r['MAE_NoPrior']:.3f} | {r['Skill_NoPrior']:.3f} | "
            f"{r['MAE_WithPrior']:.3f} | {r['Skill_WithPrior']:.3f} | {r['p_value']:.3f} | "
            f"**{r['Verdict']}** |"
        )
    lines.append(
        "\n*The unified prior is **not** significantly helpful in either domain (p ≥ 0.05, "
        "effect direction negative) — it was tested and **dropped**. Note the positive skill "
        "scores without it (Games 0.225, Books 0.108): the local models beat the mean-rating "
        "baseline, the first positive evidence of learnable local signal in these N≈60 domains.*"
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
        "#### Benchmarks — *standalone local models* + dual unified headline\n",
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


if __name__ == "__main__":
    render_docs()
