"""
run_pipeline.py — one command to refresh the whole Personal Media Intelligence Hub.

Run this after you add new ratings/items (e.g. when you hit 1,000 movies). It
merge-ingests new items (keeping your hand-cleaned rows), rebuilds every feature
set, regenerates the frozen fold registry, re-evaluates, retrains the deployed
models, refreshes the README/tech-report metric blocks, and rebuilds the paper
figures/tables. Heavy 50-fold experiments are opt-in via --full.

Each step runs as its own `python -m ...` process (PYTHONUTF8=1) so it behaves
exactly like running it by hand, in the correct dependency order.

USAGE
  python run_pipeline.py                  # core refresh (merge-ingest -> ... -> paper)
  python run_pipeline.py --full           # also run transfer grid / bridges / curves (overnight)
  python run_pipeline.py --no-ingest      # skip ingestion (data already current)
  python run_pipeline.py --domains movies # only re-ingest/feature/train these domain(s)
  python run_pipeline.py --skip-paper     # skip paper figure regeneration
  python run_pipeline.py --dry-run        # print the plan, run nothing

Merge-ingest preserves every existing enriched row and only fetches NEW items:
  movies sync_letterboxd_data · shows run_ingestion · games enrich(merge) · books library(merge).
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = sys.executable
ENV = {**os.environ, "PYTHONUTF8": "1"}

DOMAINS = {
    "movies": {
        "ingest": "src.movies.ingestion",
        "feature": "src.movies.feature_engineering",
        "models": ["src.movies.advanced_movie_model_trainer", "src.movies.model_trainer"],
        "predict": "src.movies.predict_ratings",
    },
    "shows": {
        "ingest": "src.shows.ingestion",
        "feature": "src.shows.feature_engineering",
        "models": ["src.shows.model_trainer"],
        "predict": "src.shows.predict_ratings",
    },
    "games": {
        "ingest": "src.games.ingestion",
        "feature": "src.games.feature_engineering",
        "models": ["src.games.model_trainer"],
        "predict": "src.games.predict_ratings",
    },
    "books": {
        "ingest": "src.books.ingestion",
        "feature": "src.books.feature_engineering",
        "models": ["src.books.model_trainer"],
        "predict": "src.books.predict_ratings",
    },
}

OOF_CACHES = ["reports/oof_predictions.csv", "reports/standalone_oof.csv",
              "reports/production_oof.csv"]


def clear_oof_caches():
    """Delete cached OOF frames so the evaluator recomputes against the new registry."""
    removed = []
    for rel in OOF_CACHES:
        p = REPO / rel
        if p.exists():
            p.unlink()
            removed.append(rel)
    print(f"   cleared caches: {removed or 'none present'}")


def build_plan(args):
    """Return an ordered list of steps: (phase, label, action).
    action is a command list (run via subprocess) or a Python callable."""
    domains = args.domains or list(DOMAINS)
    steps = []

    if not args.no_ingest:
        for d in domains:
            steps.append(("ingest", f"merge-ingest {d}", [PY, "-m", DOMAINS[d]["ingest"]]))

    for d in domains:
        steps.append(("features", f"features {d}", [PY, "-m", DOMAINS[d]["feature"]]))
    steps.append(("features", "unified features", [PY, "-m", "src.unified_model.unified_feature_engineering"]))

    steps.append(("registry", "frozen fold registry", [PY, "-m", "src.unified_model.create_frozen_folds"]))
    steps.append(("registry", "clear OOF caches", clear_oof_caches))

    steps.append(("evaluate", "comprehensive evaluator", [PY, "-m", "src.unified_model.comprehensive_evaluator"]))
    steps.append(("evaluate", "distillation ablation", [PY, "-m", "src.experiments.distillation_ablation"]))
    steps.append(("evaluate", "evaluator (absorb distillation)", [PY, "-m", "src.unified_model.comprehensive_evaluator"]))

    if not args.skip_models:
        steps.append(("models", "unified deployed ensemble", [PY, "-m", "src.unified_model.advanced_unified_model_trainer"]))
        for d in domains:
            for m in DOMAINS[d]["models"]:
                steps.append(("models", f"train {d} ({m.split('.')[-1]})", [PY, "-m", m]))
        for d in domains:
            steps.append(("predict", f"batch predict {d}", [PY, "-m", DOMAINS[d]["predict"]]))

    steps.append(("docs", "render README + tech report", [PY, "-m", "src.reporting.render_docs"]))

    if not args.skip_paper:
        steps.append(("paper", "paper figures + tables", [PY, str(REPO / "paper" / "make_figures.py")]))

    if args.full:
        steps += [
            ("experiments", "transfer grid (full 50-fold)", [PY, "-m", "src.experiments.transfer_study", "full"]),
            ("experiments", "transfer analysis", [PY, "-m", "src.experiments.transfer_analysis"]),
            ("experiments", "entity links (full)", [PY, "-m", "src.linking.build_entity_links", "full"]),
            ("experiments", "entity bridges", [PY, "-m", "src.linking.bridge_features"]),
            ("experiments", "learning curves (unified)", [PY, "-m", "src.experiments.learning_curve"]),
            ("experiments", "learning curves (local)", [PY, "-m", "src.experiments.learning_curve_local"]),
            ("experiments", "active-learning queue", [PY, "-m", "src.evaluation.active_learning_ranker"]),
            ("paper", "paper figures (refresh w/ experiments)", [PY, str(REPO / "paper" / "make_figures.py")]),
        ]
    return steps


def main():
    ap = argparse.ArgumentParser(description="Refresh the whole project after adding data.")
    ap.add_argument("--full", action="store_true", help="also run heavy 50-fold experiments (overnight)")
    ap.add_argument("--no-ingest", action="store_true", help="skip ingestion (data already current)")
    ap.add_argument("--skip-models", action="store_true", help="skip model retraining + batch predictions")
    ap.add_argument("--skip-paper", action="store_true", help="skip paper figure/table regeneration")
    ap.add_argument("--domains", type=lambda s: [d.strip() for d in s.split(",") if d.strip()],
                    help="comma-separated subset of: " + ", ".join(DOMAINS))
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    if args.domains:
        bad = [d for d in args.domains if d not in DOMAINS]
        if bad:
            ap.error(f"unknown domain(s): {bad}. choose from {list(DOMAINS)}")

    plan = build_plan(args)

    print("=" * 70)
    print(f"PIPELINE PLAN ({len(plan)} steps)  full={args.full}  ingest={not args.no_ingest}")
    print("=" * 70)
    for i, (phase, label, action) in enumerate(plan, 1):
        kind = "py()" if callable(action) else " ".join(action[2:]) if action[:2] == [PY, "-m"] else "script"
        print(f"  {i:2}. [{phase:11}] {label:38} {kind}")
    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return 0

    print("\nStarting...\n")
    t0 = time.time()
    for i, (phase, label, action) in enumerate(plan, 1):
        ts = time.time()
        print(f"\n──[{i}/{len(plan)}] {phase}: {label}{'─' * 10}")
        if callable(action):
            action()
        else:
            r = subprocess.run(action, cwd=str(REPO), env=ENV)
            if r.returncode != 0:
                print(f"\n❌ STEP FAILED: {label} (exit {r.returncode}). Stopping.")
                print("   Fix the cause and re-run; earlier steps' outputs are on disk.")
                return r.returncode
        print(f"   ✓ {label} ({time.time() - ts:.0f}s)")

    print("\n" + "=" * 70)
    print(f"✅ PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print("=" * 70)
    print("Auto-updated: enriched data, all features, registry, latest_metrics.json,")
    print("README + tech-report metric blocks, deployed models, paper figures/tables.")
    print("\n⚠️  NOT auto-updated (hand-written prose) — review if numbers shifted materially:")
    print("   • prose paragraphs in reports/FINAL_ML_TECHNICAL_REPORT.md (outside <!--METRICS-->)")
    print("   • prose in paper/PMIH_technical_report.tex (the \\input tables ARE auto)")
    print("   • README narrative outside <!--BENCHMARKS-->/<!--TRANSFER_VERDICT--> markers")
    if not args.full:
        print("   • transfer-grid / entity-bridge / learning-curve numbers (run with --full to refresh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
