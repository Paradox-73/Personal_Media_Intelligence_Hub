# Résumé Bullets — Personal Media Intelligence Hub

Honest, defensible bullets grounded in the project's *actual* findings (not aspirational metrics).

## Headline (ML rigor / evaluation discipline)
- **Designed and pre-registered a paired-significance cross-domain transfer study** (4 targets × all rated-domain source subsets × zero-shot/augmented protocols on a frozen 50-fold registry), with decision rules fixed before results; reported the **null verdict** honestly — no feature-space domain blend beat the local models — turning a negative result into the project's sharpest, most falsifiable contribution.
- **Found and fixed a reporting bug that had inverted a scientific conclusion:** the benchmark table was publishing the unified model's per-domain OOF slice as the standalone models with N inflated 5×. Rebuilding the single-source-of-truth metrics writer (per-item dedup, `assert benchmarks != slices`, `assert N == n_unique_items`) forced standalone and unified onto the *same estimator* — which reversed the apparent result at the time (unified competitive-or-better in 3 of 4 domains, not worse "everywhere"). Once the comparison used the real production local models, the 2026-10-01 run puts the local models ahead in all four domains (skill 0.355 vs 0.315 movies, 0.208 vs 0.186 TV, 0.376 vs 0.202 games, 0.566 vs 0.241 books).
- **Killed features that didn't earn their keep:** paired-Wilcoxon ablations retired Ridge stacking and residual-head correction (signal-destroying) and a unified distillation prior for sparse domains (p = 0.317 / 0.465 in the 2026-10-01 run, non-significant) — keeping only the conservative equal-weight Mean Ensemble.

## Supporting
- **Every deployed model beats predict-the-mean and orders items correctly:** skill (1 − MAE / MAE of predict-the-mean, the primary metric because a constant 3.5★ already scores a decent MAE on 3–4★-clustered ratings) of 0.21–0.57 and Spearman ρ 0.57–0.86 (p ≤ 1e-11) across 1,514 rated items (1,163 movies · 175 TV · 66 games · 110 books; 2026-10-01 run). Books is strongest (skill 0.566, ρ 0.859); TV weakest (skill 0.208).
- Built leakage-safe **frozen-fold registry** + CORAL domain alignment (fit on train folds only) enabling reproducible, paired cross-domain evaluation across Movies/TV/Games/Books/Music.
- Calibrated honest uncertainty: **vanilla split-conformal 80% intervals** with measured OOF coverage within ±5 points of target; surfaced via an active-learning acquisition queue.
- Engineered an **entity-bridge pilot** (Wikidata adaptation / shared-creator / franchise links) and evaluated item-level transfer with paired tests + bootstrap CIs — reported as a non-significant pilot (ΔMAE +0.02, 95% CI [−0.02, +0.06]) rather than overclaiming.

## If asked "what's the one thing you're proudest of?"
Repairing the measurement layer so that *every* number renders from one validated JSON via one script — making it impossible for a hand-edited or mislabeled metric to tell a wrong story with perfect consistency. The bug fix changed the conclusion; the discipline is the deliverable.
