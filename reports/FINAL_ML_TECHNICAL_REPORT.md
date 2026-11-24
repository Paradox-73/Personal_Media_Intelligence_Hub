# Personal Media Intelligence Hub — Final ML Technical Report

This document holds the architecture, the metrics and the analysis for every media domain. It is written in ASD-STE100 Simplified Technical English. Sections 9 to 12 are auto-generated from `reports/latest_metrics.json` and the diagnostic JSON files. Do not edit them by hand.

> **Data snapshot: 1 October 2026** (metrics generated at git `3015c3f`) — N = 1,163 movies · 175 TV shows · 110 books · 72 games (66 rated) · 3,837 music tracks. The rated universe is 1,514 items. The unified frame holds 1,520 rows, because it keeps the 6 unrated "Incomplete" games. Music is unsupervised. The 3,837 tracks feed the taste-profile and similarity engine, not a rating benchmark.

## What changed in the 2026-10-01 run

This run retrained every model on a larger corpus. The corpus changes are:

- **+101 films**, from the Letterboxd export of 2026-10-01. 42 of them are standup comedy specials.
- **+17 books** and **+2 shows**.
- **Movie plot text now uses the full Wikipedia plot** where one exists. That is about 5 embedding windows per film.
- **The fold registry was regenerated.** TV's fold-reshuffle noise (about 0.02 MAE) therefore applies to the TV change.

The primary metric is **skill** = 1 − MAE / MAE of predict-the-mean. Previous run (28 August 2026) → this run:

| Domain | N | Skill | MAE | Spearman ρ | ±0.5★ acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Movies | 1,062 → 1,163 | 0.362 → 0.355 | 0.484 → 0.483 | 0.755 → 0.751 | 77.0% → 77.1% |
| TV | 173 → 175 | 0.258 → 0.208 | 0.656 → 0.694 | 0.583 → 0.565 | 63.6% → 60.6% |
| Games | 66 → 66 | 0.376 → 0.376 | 0.515 → 0.515 | 0.735 → 0.719 | 75.8% → 74.2% |
| Books | 93 → 110 | 0.504 → 0.566 | 0.328 → 0.286 | 0.844 → 0.859 | 88.2% → 88.2% |
| Unified | 1,400 → 1,514 | 0.329 → 0.287 | 0.526 → 0.542 | 0.694 → 0.674 | 75.7% → 73.1% |

**The Movies dip comes from the new mix, not from a worse model.** On the same 1,062 films as before, movie skill is 0.362, which is unchanged. The 101 new films score skill 0.44. The 42 standup specials among them score 0.51.

**The local-against-unified comparison flipped.** The local model now beats the unified slice in all four domains: MAE 0.483 against 0.513 (Movies), 0.694 against 0.714 (TV), 0.515 against 0.659 (Games) and 0.286 against 0.500 (Books). In the previous run the unified slice won Movies and TV by a narrow margin. §4 has the detail.

**TV fell.** Skill dropped from 0.258 to 0.208. Part of this is the regenerated fold registry. The rest is not isolated.

**Books rose.** Skill rose from 0.504 to 0.566 on 17 more books.

**Not re-run in this run:** the transfer grid (2026-08-30), the entity bridges (2026-08-28) and the conformal widths (2026-07-26).

## Artifact provenance

The artifacts do not all share one vintage. Read every number with its date.

| Vintage | Artifacts | Status |
| :--- | :--- | :--- |
| **2026-10-01** | `latest_metrics.json`, `production_oof.csv`, `standalone_oof.csv`, `distillation_ablation_results.json`, `baseline_sanity.json`, `classification_diagnostics.json`, `feature_diagnostics.json`, `bias_variance.json`, `UNIFIED_ABLATION_REPORT.md`, unified training frame | current corpus |
| **2026-08-30** | `transfer_grid_summary.json` | previous corpus (1,062 movies) |
| **2026-08-28** | `entity_bridge_results*.json` | previous corpus |
| **2026-07-26** | conformal widths | earlier corpus |
| **2026-06-26/27** | `edge_loss_ablation_results.json`, `transfer_placebo_control.json` | previous corpus (1,000 movies); both test mechanisms that the refresh does not change |
| **2026-07-11** | `text_vs_crowd.json`, `text_lift.json` | previous corpus |

## Overview of optimizations

1. **Asymmetric edge-penalty loss.** A custom XGBoost objective penalises errors on edge ratings. **Status: retire it.** The head-to-head ablation shows no measurable gain (2026-10-30 re-run on 1,182 movies with fold-safe encodings: overall MAE 0.4268 against 0.4274, p = 0.997; edge MAE 0.654 against 0.644, p = 0.096; the 2026-06-26 run had read +0.0039, p = 0.083). The weight keys on the extremity of the true rating, not on the error, so the loss cannot target the phenomenon it was built for.
2. **Target encoding, refit per benchmark fold.** Movies (m=2) and TV (m=10) use 5-fold out-of-fold, Bayesian-smoothed encodings for creators, cast and writers. Both matrices persist them for training and serving. The stored columns were built once over the whole library, so since 2026-10-30 every benchmark refits the encodings inside each registry fold from training rows only (`reencode_fold` in `standalone_benchmarks.py`); the measured cost of the old protocol is in `reports/fold_safe_te_lift.json`. `actors_encoded` is the fourth-strongest movie feature by SHAP (2026-10-01 run).
3. **Distillation prior.** The unified model's prediction was trialled as a feature for the Games and Books SVRs. **Status: dropped** - both cells null on the current corpus (§3).
4. **Music as clustering and similarity.** Music has no star ratings, and the project no longer synthesizes one. Music uses KMeans personas, cosine similarity and a positive-unlabeled classifier. It is excluded from every rating benchmark.
5. **Temporal taste decay.** A floored exponential weight can down-weight old ratings. **Status: wired but not fitted** (§4).
6. **Frozen-fold registry.** One `fold_registry.json` removes metric drift between runs and makes paired tests valid.
7. **Domain centroid alignment (CORAL).** Per-fold alignment removes the domain-mean offset in the shared latent space.
8. **Semantic music lyrics.** Lyrics reach the music vector in 93% of tracks, so lyrical vibe influences the similarity engine.

---

## How to read these metrics

Every number in the Performance Summary uses the same procedure. Clip the predictions to `[0.5, 5.0]`. Snap them to the nearest 0.5★, which is the grid the user rates on. Average the 5 cross-validation repeats. Score one row per unique item.

| Metric | Definition | What it tells you here |
| :--- | :--- | :--- |
| **Skill Score** | `1 − MAE_model / MAE_baseline`, baseline = predict the training mean | **The primary metric.** Ratings cluster in 3–4★. A constant 3.5★ prediction already scores MAE 0.749 on the films and lands within ±0.5★ 61.9% of the time. So MAE and accuracy look good for a model that learned nothing. Skill is 0 for predict-the-mean and ≈0 for any other constant. It asks the only question that survives low variance: does the model beat guessing the average? Every domain scores +0.21 to +0.57 (2026-10-01 run). |
| **Spearman ρ** | Rank correlation of prediction against truth | **The load-bearing second metric.** A mean-hugging model has ρ ≈ 0 and cannot order items. Every model here reaches ρ 0.57 to 0.86. |
| **MAE** | Mean absolute error, in stars | Secondary. MAE 0.50 means the prediction is half a star off on average. It is easy to read, but a constant earns a respectable MAE in this data. |
| **±0.5★ Accuracy** | Share of items predicted within one grid step | The "is this usable in the Oracle UI" metric. Movies 77.1% means about 3 of 4 predictions land within one half-star notch. Read it against the mode baseline (Movies 51%). |
| **R² (CV mean)** | Fraction of rating variance explained | **Treacherous. Read it only beside skill.** TV, Games and Books ratings cluster in a narrow band, so the variance denominator is small and R² collapses even when MAE is good. A near-zero R² in a low-variance domain is a property of the denominator. |
| **N** | Count of unique rated items | Per item, after de-duplicating the 5 repeats. Never OOF rows. Counting rows once inflated Movies about five times. |

**The one-line takeaway.** Skill score is primary. Spearman ρ proves the model orders items. MAE is secondary. R² is trustworthy only for Movies. Never rank domains by R² alone.

---

## 1. Movies domain

### Deployed model
The deployed Movies regressor is a production stacking ensemble. Its members are an Optuna-tuned asymmetric edge-penalty XGBoost, CatBoost, an SVR and a 10-bucket ordinal expected-value head. A `RidgeCV` meta-learner fuses them. All headline metrics live in the auto-generated Performance Summary below. That table is the single source of truth.

### Analysis
Movies is the strongest domain by a wide margin. It has the most labels (N = 1,163) and the richest features. On the frozen 50-fold registry the deployed model scores **skill 0.355, Spearman 0.751** (MAE 0.483, R² 0.530, 77.1% within half a star) in the 2026-10-01 run.

**Read the numbers.** The model removes about a third of the error of predict-the-mean. The typical prediction is about half a star off. R² 0.530 is trustworthy here, because the wide rating spread gives R² a real denominator. This is the one domain where that holds.

**Calibration explains the gap between two Movies numbers.** Uncalibrated, the simplex blend scores skill 0.410 and MAE 0.455 (`feature_diagnostics.json`). Calibrated, the deployed stack scores skill 0.355 and MAE 0.483. That is a deliberate trade of about four accuracy points to recover the full 0.5–5★ range. Before calibration the model never predicted above 4.5★, because 62% of ratings fall in [3, 4] and hugging the mode minimises MAE.

**The ensemble gain is gone on this corpus (changed in the 2026-10-01 run).** XGBoost alone scores skill 0.414 (MAE 0.451). The mean blend scores 0.411 (MAE 0.454) and the simplex blend 0.410 (MAE 0.455). In the previous run the simplex blend beat every single member (skill 0.3849 against 0.3799 for XGBoost). The gap is 0.004 skill in either direction, which is inside noise. The deployed stack is not changed.

**A retired myth.** An earlier single 80/20-split run quoted R² ≈ 0.61 for Movies. The honest 50-fold registry figure for the same architecture is far lower. A single split is optimistic.

---

## 2. TV Shows domain

### Deployed model
The deployed TV model is a simplex stack. It is a constrained non-negative weighted average of XGBoost, CatBoost and SVR base learners. The weights are non-negative and sum to one. At N = 175 a learned `RidgeCV` meta-learner overfits, so the constrained average is the safer choice.

### Analysis
TV now scores **skill 0.208, Spearman 0.565** (MAE 0.694, R² 0.271, 60.6%) on the registry folds in the 2026-10-01 run. It is the weakest domain. The previous run scored skill 0.258 (MAE 0.656). The fold registry was regenerated in this run, and TV's fold-reshuffle noise is about 0.02 MAE, so part of the drop is noise.

**This is the largest per-domain gain in the project's history.** The earlier TV model scored MAE 0.802 and R² 0.082. Adding leakage-safe creator, cast and writer encodings, IMDb multi-critic fields, runtime, country and awards features produced the lift. The lesson is diagnostic. **TV's earlier weakness was missing features, not an aleatoric ceiling.** Once the model could see who made a show and who starred in it, the explained variance more than tripled.

**A reversed recommendation.** An earlier revision of this report recommended replacing the TV stack with a plain SVR, because the per-base-learner ablation then placed SVR-alone (skill 0.110) above the blend (0.095). **That recommendation is withdrawn.** On the current corpus (2026-10-01) the TV SVR is still the weakest member (skill 0.164). The mean blend is the best (skill 0.232). It beats the best single member (CatBoost, 0.229) by +0.003 and SVR-alone by +0.068. The simplex blend (0.216) falls below CatBoost alone in this run; in the previous run it was the best.

**TV is the one domain the pipeline does not calibrate.** Its rank correlation is too weak (ρ = 0.565 in the 2026-10-01 run). Quantile calibration would place 5★ on the wrong shows and cost about ten accuracy points.

---

## 3. Games and Books (niche domains)

### Deployed models
Both domains are sparse (N = 66 and 110) and both use a local `SVR(rbf)` with frozen hyperparameters (C = 1.0, ε = 0.1). A high-variance learner overfits at this N, so boosted trees were rejected.

Books additionally ships a **text-similarity predictor**. A `KNeighborsRegressor` (k = 8, cosine metric, distance weights) over the book vibe-PCA reaches skill **0.426**, against **0.244** for the SVR that mixes in crowd features. A blend of the two is worse (≈0.39). For Books, text alone is the honest best model, and it is explainable: it names the neighbour books that drove the prediction.

**N provenance.** The Games file keeps 6 titles with the "Incomplete" status as rows for ranking, but they carry no rating. The local Games model therefore scores N = 66, while the corpus and the unified frame hold 72. Both counts are correct in their own context.

### Analysis
The headline result is the **skill score**. Without any prior, the deployed local SVRs beat the mean-rating baseline by **0.376 (Games, Spearman 0.719)** and **0.566 (Books, Spearman 0.859)** in the 2026-10-01 run. This is genuine local signal at N = 66 and 110, in domains where R² is treacherous.

**Games is the clearest case for a local model in the whole project.** The Games SVR leans on RAWG community columns — `ratings_count` (SHAP 0.268), `rating` (0.110) and `playtime` (0.091). The shared cross-domain space discards those columns. When they are gone, the unified Games slice falls to skill 0.202 (MAE 0.659) against the local skill 0.376 (MAE 0.515). That contrast is the cleanest evidence that not all signal survives pooling.

**Books now tells the same story, more strongly.** Since ingestion moved to the Hardcover library API, the local Books SVR reads author, category, publisher, mood and tag fields as five target encodings in a 29-feature matrix. In the 2026-10-01 run the local model beats the unified slice on every metric: skill 0.566 against 0.241, MAE 0.286 against 0.500, R² 0.715 against 0.388, and accuracy 88.2% against 75.5%. The same refresh populated the three previously-dead unified Books channels (`critic_avg_5`, `runtime`, `gen_*`), which are now at 100%.

### Distillation prior — dropped; both cells now null
The test adds the unified model's OOF prediction to the local SVRs as a feature. In the 2026-10-30 run (deployed local SVRs with balanced weights and calibration, target encodings refit inside each fold, one row per item) both cells are null.

- **Games: DROP.** MAE rises from 0.5227 to 0.5455 with the prior; skill falls from 0.338 to 0.310. p = 0.317.
- **Books: DROP.** MAE 0.3423 with and without the prior; p = 1.0. The prior moves the raw SVR output by up to 0.5 stars on some items, but the quantile calibration snaps outputs onto the training rating grid and the per-item mean of five repeats rounds to the same half-star for all 111 books. The earlier p = 0.0343 'KEEP' never came back.
- **The no-prior MAEs equal the benchmark Local SVR rows exactly** (0.5227 and 0.3423), so the two harnesses now agree. The 2026-10-01 cells (Games 0.5894 → 0.5818, Books 0.3491 → 0.3527) came from a stand-in `SVR(C=1)` with no balanced weights and no calibration, scored over 5× repeated fold rows.

**The project verdict stays DROP.** Three reasons support that decision. Neither cell is near significance. The folds are correlated at N = 66 and 111, so even these p-values are optimistic. The Books cell has read p = 0.705, 0.034, 0.446, 0.336, 0.248, 0.465 and now 1.0 across refreshes, with the sign of the effect flipping, so it is not stable. The deployed Books SVR attaches the prior only when a unified prediction file is present, so the shipped behaviour is conditional. Re-test at the next corpus refresh.

---

## 4. Unified model

### Deployed model
The unified model is a cross-domain **Mean Ensemble** of XGBoost and CatBoost. It uses leak-free domain centroid alignment, fitted on the training folds only, and multi-modal missingness masks. It has a single headline row in the Performance Summary.

- **Unified (rated)** is the headline taste metric. It trains and evaluates on real ratings across the four rated domains, on the frozen registry folds. It scores **skill 0.287, Spearman 0.674** (MAE 0.542, R² 0.445, 73.1%) over 1,514 rated items in the 2026-10-01 run.
- **Music is excluded.** Music has no star ratings, and the project no longer synthesizes one. The former "Unified (full pool, +music PU pseudo-labels)" row has been **removed**, not footnoted. It scored the model partly against another model's fabricated affinities.

### Analysis — the local model wins in every rated domain

Both sides of this comparison use the identical pooled per-item OOF estimator on the same frozen folds, on the same items. Both skill columns use the same per-domain mean baseline (`baseline_sanity.json`: movies 0.744, TV 0.877, games 0.826, books 0.667). Numbers are from the 2026-10-30 run, the first in which every local benchmark refits its target encodings inside each fold.

| Domain | Local skill | Unified slice skill | Local MAE | Unified slice MAE | Winner |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Movies (N=1,182) | **0.437** | 0.332 | **0.419** | 0.497 | local, by 0.078 MAE |
| TV (N=175) | **0.225** | 0.179 | **0.680** | 0.720 | local, by 0.040 MAE; TV's fold-reshuffle noise is about 0.02 MAE |
| Games (N=66) | **0.367** | 0.239 | **0.523** | 0.629 | local, by 0.106 MAE |
| Books (N=111) | **0.487** | 0.324 | **0.342** | 0.450 | local, by 0.108 MAE |

**This verdict first appeared in the 2026-10-01 run and held on 2026-10-30.** In the run before 2026-10-01 the unified slice won Movies (0.475 against 0.486 MAE) and TV (0.666 against 0.680) by narrow margins. In the 2026-10-01 run the entity encodings on both sides still leaked across the fold boundary. Since 2026-10-14 (unified) and 2026-10-30 (local models) every benchmark refits its encodings inside each registry fold; the measured cost of the old protocol on the local Movies/TV ensemble is +0.0026 and +0.0257 MAE (paired p = 0.95 and 0.34, `reports/fold_safe_te_lift.json`), and the local model still wins all four.

**The earlier "pooling rescues the data-poor domain" result is gone.** The TV feature enrichment and the Books library refresh left no weak rated domain for pooling to rescue. This is why the deployed system keeps **local models for production scoring** and uses the unified model only for the transfer study.

This does **not** dissolve the transfer grid (§5). "Does the pooled model beat local on average" is a blunt question. The grid asks the sharp one: which specific source combinations transfer into which targets.

### Architecture ablation
The project grew the unified model and then pruned it. Adding the mean ensemble over the aligned base XGBoost improves MAE from 0.5521 to 0.5445 (2026-10-01 run). The improvement is statistically significant but trivial in size (Cohen's d = 0.068, p < 10⁻⁴). It is kept because it is free, not because it matters. Ridge stacking (MAE 0.5609) and residual heads (MAE 0.6088) both made performance worse and were removed. Complexity did not help. That is itself the result.

### Temporal decay — wired but not fitted
The ensemble can down-weight old ratings by `w_i = max(exp(−λ·Δt_i), 0.2)` with λ = 0.000429. The rating dates are genuinely populated: there are 253 distinct dates across the rated items. But the frozen-fold evaluation does **not** apply the weighting, so every reported metric is decay-free. The quoted "taste half-life ≈ 1,615 days" is only `ln 2 / λ` for a hardcoded constant. It is an **assumed** rate, not a fitted one, and it carries no empirical content until λ is tuned over the real timestamps. An Optuna path exists in `advanced_unified_model_trainer.py`.

---

## 5. Cross-domain transfer

### 5a. Legacy music transfer matrix — deprecated

> ⚠️ **Deprecated.** This early transfer matrix treated music as a rated domain through synthesized PU pseudo-labels. Music is now unsupervised with **no rating target**, so the four `Music → …` rows below predict one model's fabricated affinities. They are kept only for historical continuity. The load-bearing transfer result (§5c) involves only the four rated domains.

| Source | Target | Raw R² (Exp B) | Baseline R² | Combined R² | Lift (Exp A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Music | Movie | -0.051 | 0.544 | 0.549 | +0.005 |
| Music | TV | -0.393 | 0.325 | 0.320 | -0.005 |
| Music | Game | -0.124 | 0.005 | 0.015 | +0.010 |
| Music | Book | -0.239 | -0.293 | -0.266 | +0.027 |

### 5b. The transfer grid — pre-registration

*Written before any grid result was inspected.* The grid (`src/experiments/transfer_study.py`) tests every non-empty subset of the non-target rated domains against four targets, in the shared feature space only. It uses three protocols: **A** target-only baseline, **B** zero-shot (train on the source, evaluate on the target) and **C** augmented (source plus {0, 25, 50, 100}% of the target training data). Every cell runs on the frozen registry folds, with CORAL fitted on the training folds only.

**Pre-registered decision rules.**
1. A cell is a **positive transfer finding** if the augmented lift is greater than 0 **and** the paired Wilcoxon p is below 0.05 against Protocol A on identical test items, **at two or more target fractions.**
2. The project verdict is "taste transfers" if at least one target has a positive finding. Otherwise the verdict is the null: in the shared feature space, no domain combination beats the local models, and transfer must be item-level (§6).

### 5c. The transfer grid — realized verdict

<!-- TRANSFER_VERDICT:BEGIN -->
**Realized verdict (full 50-fold grid, domain-blind, quadruple-controlled): NULL - no feature-level transfer.** NULL: in the shared feature space, no domain combination beats the local model (Protocol A). Cross-domain taste signal, if it exists, must be item-level (entity bridges), not feature-level pooling.

| Target | Best source subset | lift @100% | p | Clears the bar? |
| :--- | :--- | ---: | ---: | :--- |
| movie | `game` | +0.0030 | 0.394 | no |
| tv | `movie+book` | -0.0180 | 0.200 | no |
| game | `tv` | +0.0195 | 0.328 | no |
| book | `game` | +0.0442 | 0.009 | no |

**Zero-shot -- can other domains rank a target the model has never trained on?** (Spearman, mean over folds)

| Target | Target-only XGB (same shared features) | Deployed model | Best source (never sees target) | rho | % of target-only XGB |
| :--- | ---: | ---: | :--- | ---: | ---: |
| movie | 0.669 | 0.796 | `tv+game+book` | +0.496 | 74% |
| tv | 0.417 | 0.611 | `movie` | +0.448 | **108%** |
| game | 0.401 | 0.683 | `movie+book` | +0.553 | **138%** |
| book | 0.593 | 0.857 | `movie+tv+game` | +0.599 | **101%** |

Bold rows are targets where a model that has **never seen the domain** ranks it better than a target-only XGB on the same shared features: `movie->tv`, `movie+book->game`, `movie+tv+game->book`. Against the deployed model (pooled per-item Spearman from `baseline_sanity.json`, a different estimator than the per-fold mean) it wins for none of them.

**Why the sign differs by domain.** Local rating coherence -- how well an item's rating is predicted by its nearest neighbour in the shared text space -- orders the domains identically to the value of their own training data:

| Domain | n | Coherence (rho) | p | Effect of its own data on a movie-trained model |
| :--- | ---: | ---: | ---: | :--- |
| tv | 175 | +0.102 | 0.1805 | hurts |
| movie | 1182 | +0.258 | 0.0000 | n/a (is the source) |
| game | 67 | +0.378 | 0.0016 | hurts |
| book | 111 | +0.705 | 0.0000 | hurts |

TV is the only domain where the mean gap to the nearest neighbour (1.097) EXCEEDS the spread of its own ratings (1.071) -- copying a lookalike is worse than guessing the average. Boosting still fits those rows; with no local structure to fit it fits noise, into exactly the region where the TV test items live.

Aligned-space distance still does not predict transfer (Spearman -0.424 over n=12 pairs).
<!-- TRANSFER_VERDICT:END -->

### The complete transfer grid
<!-- TRANSFER_GRID_FULL:BEGIN -->
## Target: Movies (119 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.669** (SE 0.007). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `tv+game+book` | +0.496 | +0.633 | +0.643 | +0.661 |
| `tv+book` | +0.492 | +0.626 | +0.652 | **+0.672** |
| `tv` | +0.445 | +0.621 | +0.645 | +0.665 |
| `game+book` | +0.323 | +0.622 | +0.651 | +0.664 |
| `tv+game` | +0.311 | +0.616 | +0.647 | +0.666 |
| `book` | +0.189 | +0.631 | +0.652 | +0.668 |
| `game` | +0.088 | +0.622 | +0.642 | +0.669 |

## Target: TV (18 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.417** (SE 0.024). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.448** | +0.347 | +0.363 | +0.394 |
| `movie+book` | **+0.430** | +0.353 | +0.397 | +0.410 |
| `movie+game+book` | +0.380 | +0.334 | +0.374 | +0.369 |
| `movie+game` | +0.338 | +0.343 | +0.370 | +0.399 |
| `game+book` | +0.337 | +0.127 | +0.191 | +0.242 |
| `book` | +0.122 | +0.165 | +0.212 | +0.249 |
| `game` | +0.114 | +0.079 | +0.117 | +0.114 |

## Target: Games (7 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.401** (SE 0.042). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+book` | **+0.553** | **+0.493** | **+0.481** | **+0.452** |
| `movie` | **+0.516** | +0.396 | **+0.421** | **+0.454** |
| `movie+tv+book` | **+0.512** | **+0.481** | **+0.501** | **+0.498** |
| `movie+tv` | **+0.473** | +0.378 | +0.395 | **+0.462** |
| `tv` | **+0.412** | **+0.519** | **+0.528** | **+0.522** |
| `tv+book` | **+0.408** | +0.173 | +0.253 | +0.250 |
| `book` | +0.255 | +0.261 | +0.298 | +0.310 |

## Target: Books (12 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.593** (SE 0.031). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+tv+game` | **+0.599** | +0.474 | +0.479 | +0.523 |
| `movie` | +0.592 | +0.483 | +0.460 | +0.487 |
| `movie+game` | +0.539 | +0.455 | +0.470 | +0.501 |
| `movie+tv` | +0.532 | +0.477 | +0.471 | +0.482 |
| `tv` | +0.452 | +0.483 | +0.530 | +0.585 |
| `tv+game` | +0.387 | +0.502 | +0.555 | +0.583 |
| `game` | +0.226 | +0.393 | +0.552 | **+0.652** |
<!-- TRANSFER_GRID_FULL:END -->

### Four adversarial controls

A "data-starved target benefits from any larger pool" result is easy to fake. Four controls test for it.

1. **Domain-identity leak, removed.** The shared-space feature set once included `has_{domain}_feats` masks, so a model could read the domain label. Removing them makes the features domain-blind. The Games transfer finding **survives**. The same fix demoted a false-positive bridge result (§6).
2. **Prior against transfer.** Is the lift only a better constant than the games mean? No. The transfer model reaches MAE 0.99 against 1.05 (game mean), 1.17 (source mean) and 1.05 (global mean). It also rank-tracks true game ratings at Spearman **+0.47**. A prior is a near-constant with ρ ≈ 0, so this is content.
3. **Text-length normalization — necessary.** Game descriptions ran about five times longer than movie plots (median 1,074 against 214 characters). On the raw-length vibe space the transfer measured only +0.05 and was **not significant** (0/4). Length-normalizing the embeddings (genre template, first 40 words, median ≈232 characters) **unmasks** the effect. Every Games-source configuration then clears 4/4 on this control's protocol.
4. **Label-permutation placebo — decisive.** The project re-ran `movie → game` with the source ratings shuffled and the features intact, on the same 50 folds. The real lift is +0.132 / +0.101 / +0.122 / +0.134 across the four fractions. Under permutation it collapses to −0.004 / −0.025 / −0.015 / +0.053. The real-against-placebo gap is significant at every fraction (p ≤ 0.011; p < 0.0002 at 0–50% target data). At full target data the placebo recovers +0.053, which is pure variance reduction from the added rows. At low target data, where transfer matters, the placebo is zero. **The lift is movie-rating content, not added structure.**

### Multiple comparisons and fold dependence

The grid tests about 30 source configurations against 4 targets with no family-wise correction. Two honesty notes bound the headline.

First, the paired Wilcoxon runs over 50 repeated-CV folds. Those folds are **not independent**. With only 66 unique Games items, each item recurs across folds, so the p-values bound significance optimistically. The trustworthy signal is the **consistency** of the lift across all four fractions, not any single p.

Second, the strongest cell holds up. `movie+tv+book → game` clears p < 0.05 at all four fractions, and its 50% fraction (p = 0.0003) clears a Bonferroni threshold over about 120 cells (0.05/120 ≈ 4×10⁻⁴). The other fractions do not. So the broad column is suggestive, and the single strongest cell carries the finding.

### Reading it correctly (reconciling with §4)

Games' rating signal has two parts. **(a) Domain-specific quality** — Metacritic, platforms, developers. Only the local SVR sees it, and it is why local beats unified in §4. **(b) Genre and vibe taste** — this lives in the shared space and transfers from movies. The grid operates purely in space (b), where a games-only model is nearly useless. Both results are true at once. Keep the local model for its own features, and borrow cross-domain taste for the shared component.

### Similarity does not predict transfer

Aligned-space domain distance does **not** predict zero-shot skill. The Spearman correlation is **+0.191** over n = 12 pairs (2026-08-30 grid). The sign has moved between runs (−0.18, then +0.17, then +0.08, then −0.014, now +0.19), so at n = 12 this is noise, not a clean null. Geometry is not the transfer oracle. Target data-starvation is the better predictor.

---

## 6. Entity bridges — "does knowing I loved the book predict I'll love the show?"

The grid found feature-level transfer, but only into Games and only for the shared genre and vibe component. Entity bridges ask the complementary item-level question. If two items are linked across domains — an adaptation, a shared creator, a shared franchise — does the rating of one predict the rating of the other?

`src/linking/build_entity_links.py` resolves library items to Wikidata QIDs. It emits links through P144 and P4969 (based-on, derivative), P50, P57 and P86 (author, director, composer) and P179 (series). The current pass finds **2,691 cross-domain links**: 2,214 same-franchise, 456 shared-creator and 21 adaptation.

The evaluation adds OOF-safe bridge features to a base model over the linked and rated items. The features are the linked partner's rating, a franchise mean rating and a has-link indicator. A partner rating counts only when the partner sits in the training fold.

| Metric (2026-10-13 run, `reports/entity_bridge_results.json`) | Pooled (all link types) | Adaptation only |
| :--- | ---: | ---: |
| Linked rated items (N) | **700** | 57 |
| MAE without bridge features | 0.5343 | 0.7368 |
| MAE with bridge features | 0.5186 | 0.7368 |
| ΔMAE | +0.0157 | 0.0 |
| 95% CI | [−0.0029, +0.0350] | [0.0, 0.0] |
| Paired Wilcoxon p | 0.12 | n/a (no item moved) |

**Verdict: no significant bridge effect at this N.** ΔMAE +0.0157 with a 95% CI that includes zero and p = 0.12 over 700 linked items. The adaptation-only slice (57 items) did not move at all: no adaptation partner sat in a training fold with a usable rating. The result file calls itself a pilot.

**This result has now reversed three times, and the history matters.** ΔMAE +0.032 at p = 0.004 (artifact of the domain-identity leak, §8) → +0.019 at p = 0.077 → +0.0038 at p = 0.928 (null) → +0.0361 at p = 0.0014 (2026-08-28 corpus, 664 items) → **+0.0157 at p = 0.12** on the present corpus. The 2026-08-28 positive appeared after the split `lang_en` / `lang_English` column (a partial domain indicator) was merged and six unrated games with `target_reg = 0.0` left the registry, so it was not a returned leak. The 2026-10-13 run added 36 linked items and the Wikipedia plot text to every embedding; which of those moved the result has not been separated.

The mechanism is visible in the link graph rather than in the model. `resolve_qid` was rewritten (§8): it previously fell back to the first search hit whenever nothing matched the type hint, which mapped the film *Batman* to **Batman Province, Turkey**, *Ghostbusters* to the 1984 song and *Toy Story* to the franchise entity. Adaptation links rose 21 → 32, and **movie↔book adaptations rose from 2 to 18** — the highest-signal link type, and the one this test depends on. `n_linked` barely moved (665 → 664), so this is not a coverage effect: the same number of linked items, with edges that point at the right entities.

**Attribution caveat.** The entity graph and the corpus changed in the same run (the no-bridge baseline itself moved 0.5180 → 0.5301), so these cannot be separated from these artifacts alone. Evaluating the new graph on the old corpus would settle it and **has not been run**.

**Informative censoring — an open direction.** If you dislike a book, you may decline to watch its film adaptation. That refusal is itself a signal, and the missingness is not at random. The data to test it does not exist. **18** of the 32 adaptation links now join a movie to a book, up from 2 of 21 before the resolver rewrite. That is within reach of the 30-to-40 verified pairs this test needs to be informative, and it is the first time the censoring question has been close to answerable.

---

## 7. Uncertainty and active learning

The Oracle's prediction intervals are **vanilla split-conformal at 80% coverage**, with one constant width per domain. The width is the 80th percentile of the OOF absolute residuals.

**The widths were recomputed on the current corpus** (2026-07-26). The earlier published set (±0.78 / ±1.10 / ±1.23 / ±0.96) predated the refresh and no longer matched the residuals. The current widths and their measured coverage are below.

| Domain | Width | Measured OOF coverage | Against the 80% target |
| :--- | ---: | ---: | :--- |
| Movies | ±0.90 | 81.6% | on target |
| TV | ±0.99 | 80.1% | on target |
| Books | ±0.71 | 79.8% | on target |
| Games | ±1.00 | 87.9% | conservative (over-covers) |

Games over-covers. At N = 66 with ratings on a 0.5★ grid, the residuals take few distinct values, so the band quantizes coarsely. The band is honest but conservative there.

**One caution.** This is an in-sample OOF approximation, and it is optimistically biased. True coverage needs a held-out calibration set that the project does not have at this N. Treat the widths as approximate, not certified.

**The active-learning queue** ranks unrated items by `Z(uncertainty) + Z(novelty)`. Uncertainty is the standard deviation across the raw base-model predictions. Novelty is the mean kNN cosine distance to the training items in the aligned PCA space. The queue is dominated by Games, which is the sparsest and most uncertain domain. That is the correct thing for the system to want next.

---

## 8. Latent-space and transfer-wiring diagnostics

A sharp user question motivated this audit: *"I mostly play action games and watch action movies. Why don't they overlap in the latent space?"* The audit asked whether the then-null transfer result was a real finding or a pipeline artifact. It paid off twice. It caught a domain-identity leak, and — with the grid run at full 50-fold power — it helped flip the verdict to positive for the Games target.

`src/experiments/latent_space_diagnostics.py` runs six checks plus a music check.

| # | Check | Verdict | Finding |
| :--- | :--- | :--- | :--- |
| 1 | Embeddings populated? | ✅ PASS | No silent zeroing. Every domain has real, varied vibe vectors. The "empty embeddings" hypothesis is false. |
| 2 | Text-shape parity? | ⚠️ WARN | The embedded text differs in length by domain: movies 214 chars, TV 252, books 763, games 1,074. MiniLM maps different text lengths to different regions, even when the genre is shared. |
| 3 | CORAL applied? | ✅ PASS | The raw embedding is domain-stratified (separability ratio 1.25). Per-fold centroid/CORAL alignment reduces it to ≈0. |
| 4 | Domain-identity leak? | ❌ **FAIL → FIXED** | The shared-space feature set included `has_{domain}_feats` masks, so the model could read the domain label. The feature set is now domain-blind. |
| 5 | Unified genre vocabulary? | ✅ PASS | `gen_Action` is a single shared column that fires for movie, TV and game. 16 genres span two or more domains. Movie-action and game-action **are** the same feature. |
| 6 | Taste consistency? | INFO | Per-domain rating means are similar (spread 0.23★). A per-domain scale mismatch is **not** the transfer ceiling. |
| M | Music lyrics embedded? | ✅ PASS | `lyric_embed_text` is non-trivial in 93% of tracks. Lyrics genuinely reach the music vector. |

**What this means for the user's intuition.** The latent space does stratify by domain, so the intuition is correct. But the cause is not empty embeddings (Check 1) and not an unshared genre vocabulary (Check 5). The cause is **Check 2**. The four domains feed MiniLM different shapes of text, so the raw vibe geometry separates by writing style before it separates by content. Genre transfers as an explicit shared feature. The free-text vibe is what stratifies.

**The one real bug, and why it mattered.** Check 4's leak was live in both the transfer grid and the entity-bridge features. Removing it **changed a conclusion**: the bridge effect looked significant with the leak (p = 0.004) and is null without it (§6). The leak let the base model split on domain identity instead of transferring content. Audit the wiring before you trust a result. **Update (2026-08-28):** the bridge effect was significant again (+0.0361, p = 0.0014) — on a corpus with *fewer* domain handles, not more, and after a link-graph rewrite that took movie-to-book adaptations from 2 to 18. **Update (2026-10-13):** null again, +0.0157 at p = 0.12 over 700 items. See §6.

**Check 2, now controlled.** The project did not leave the text-length mismatch as an untested caveat. `text_norm_transfer_control.py` re-embeds every domain with length-matched, identically-templated text and re-runs the Games grid. That control was run when the grid was positive. **On the 2026-08-28 corpus the grid was NULL**, so the control then confirmed a null rather than a positive. The 2026-08-30 grid was positive for the Games target; the 2026-10-13 grid (the latest, run with the CORAL alignment active) is NULL again under the pre-registered rule (§5c), with zero-shot ranking as the one positive result. The **production** pipeline still uses the original embeddings. Re-templating the whole pipeline is the recommended next engineering pass. It would not change the transfer verdict, which is already controlled, but it would tidy the latent-space explorer and the slice metrics.

---

# Final conclusion

The hub is a reproducible, domain-aligned recommender pipeline. Two disciplines carry it: the frozen fold registry, and a reporting spine where every standalone and unified number is measured with the same estimator on the same folds.

The central question — does taste transfer across domains? — has a two-part answer on the 2026-10-13 grid (§5c). Under the pre-registered rule, **no**: no source subset lifts any target at two or more fractions, and pooling hurts every target that has no data of its own. As ranking, **partly**: a model that never saw the target orders it at 74% to 138% of a target-only XGB on the same shared features (`movie+book -> game` Spearman +0.553 against 0.401), though never as well as the deployed local model. The 2026-08-30 grid's positive Games finding did not survive the corpus refresh and the alignment fix, and those two changes have not been separated.

## What the previous revision changed (2026-08-28)

*Historical record. For the 2026-10-01 run, see "What changed in the 2026-10-01 run" at the top.* This revision was a measurement refresh, not a re-render of cached numbers. Five conclusions moved.

1. **Local models now win, or tie, everywhere (§4).** The earlier claim that the unified model wins in TV is gone: local 0.646 against pooled 0.664. Books moved from a pooled win to a tie on MAE and a local win on every other metric. Pooling no longer rescues any rated domain.
2. **The TV ensemble recommendation reversed (§2).** The earlier report recommended replacing the TV stack with a plain SVR. On the current corpus the SVR is the weakest member and the blend is the best. The recommendation is withdrawn.
3. **Entity bridges are significant again (§6).** +0.0038 (p = 0.928) → **+0.0361 (p = 0.0014)**, after `resolve_qid` was rewritten and the corrected corpus removed a partial domain indicator. This is the third reversal for this result; §6 gives the full history and the attribution caveat.
4. **The transfer headline is withdrawn entirely (§5c).** The pre-registered grid returned NULL on the 2026-08-28 corpus: no source subset clears the bar for any target, and zero-shot lifts are significantly negative. Every positive cell in the previous grid had Games as its target, and Games is the domain that lost six impossible `target_reg = 0.0` labels in this run — the leading suspect, though it has not been isolated. The similarity correlation is −0.014 at n = 12, still noise.
5. **The edge-penalty loss is retired (§Overview).** The head-to-head ablation found a small net harm. The deeper problem is the formulation: the weight keys on the label's extremity, not on the error.

The deterministic parts — the unified rated headline, the per-domain slices, and the ablation effect sizes — reproduced exactly against the prior run. That is the frozen-fold registry doing its job. The numbers that moved, moved because the corpus grew or the measurement improved. Each change is documented above, not buried.

---

<!-- METRICS:BEGIN -->
### Performance Summary

*Provenance: λ (temporal decay) = 0.000429; asymmetric-objective α: not applied in the evaluated Mean Ensemble. Generated at 2026-11-18T09:12:55 · git `777b4bea`.*

#### Benchmarks — *standalone local models* + unified headline

| Domain | N | Model | **Skill vs mean** | Spearman ρ | MAE | MAE (predict mean) | R² | ±0.5★ Acc (mode baseline) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,234 | Mean-Ensemble (XGB+Cat+SVR) | **+0.435** | 0.796 | 0.421 | 0.744 | 0.640 | 83.5% (52%) |
| **Unified (rated)** | 1,587 | Mean Ensemble | **+0.307** | 0.670 | 0.525 | 0.757 | 0.456 | 74.0% (51%) |
| **TV Shows** | 175 | Mean-Ensemble (XGB+Cat+SVR) | **+0.231** | 0.611 | 0.674 | 0.877 | 0.285 | 63.4% (47%) |
| **Games** | 66 | Local SVR | **+0.339** | 0.683 | 0.545 | 0.826 | 0.442 | 69.7% (47%) |
| **Books** | 112 | Local SVR | **+0.510** | 0.857 | 0.330 | 0.674 | 0.675 | 88.4% (54%) |

*Read **Skill vs mean** first: 1 − MAE ÷ MAE of always predicting the average rating. Ratings cluster in 3–4★, so a constant 3.5★ already earns a respectable MAE and ±0.5★ accuracy (the bracketed mode baseline is what predicting the single most common rating scores); skill is 0 for any constant predictor and 1 for a perfect one, so it is the number the clustering cannot inflate. **Spearman ρ** asks whether the model orders items correctly. The four domain rows are the **production local models** on the frozen registry folds (one row per item, N = unique items): **Movies and TV are the deployed mean ensembles** (equal-weight average of XGBoost, CatBoost and SVR) and **Games and Books are the deployed SVR**. The **Unified (rated)** row (N=1,587) is the cross-domain Mean Ensemble over all four rated domains. Music is not scored here: it has no star ratings and is handled as an unsupervised taste-profile / similarity engine.*

#### Per-Domain Slice — *Unified model, by domain* (registry OOF)

| Domain | N | R² | MAE | ±0.5★ Acc |
| :--- | :--- | :--- | :--- | :--- |
| Movies | 1,234 | 0.505 | 0.500 | 75.8% |
| Shows | 175 | 0.164 | 0.700 | 60.6% |
| Games | 66 | 0.363 | 0.636 | 65.2% |
| Books | 112 | 0.458 | 0.460 | 80.4% |

*Read against the **production benchmarks** above — both measured with the identical pooled per-item OOF estimator on the frozen folds, on the same items. The verdict per domain below is computed from the two tables on every render, so it cannot go stale.*

*Movies: local 0.421 vs unified 0.500 → local model wins · Shows: local 0.674 vs unified 0.700 → local model wins · Games: local 0.545 vs unified 0.636 → local model wins · Books: local 0.330 vs unified 0.460 → local model wins.*

#### Ablation Study (rated-only, frozen registry folds)

| Protocol | MAE | R² | Effect Size | p-value |
| :--- | :--- | :--- | :--- | :--- |
| Base XGB (Aligned) | 0.5375 | 0.4363 | ref | ref |
| Mean Ensemble (XGB+Cat) | 0.5246 | 0.4563 | d=0.076 (trivial) | 0.0026 |
| Ridge Stack (removed — legacy 5-fold protocol) | 0.5609 | 0.3837 | n/a (legacy) | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088 | 0.2688 | n/a (legacy) | n/a (legacy) |

*Unlike the benchmark table, MAE here is scored **per fold row**: each item is tested once in each of the 5 CV repeats and counted 5 times, so it differs slightly from the per-item benchmark number. Effect size is paired Cohen's d and p is a Wilcoxon test on those same 5×-replicated rows, so p is optimistic (pairs are not independent). The ensemble's MAE is -0.0129 vs Base XGB (better). Rows 3–4 are removed paths kept for the record; their numbers come from the legacy 5-fold protocol, not the frozen registry, and are directionally indicative only.*

#### Distillation-Prior Ablation (Games & Books)

| Domain | MAE (no prior) | Skill (no prior) | MAE (with prior) | Skill (with prior) | p-value | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game | 0.545 | 0.310 | 0.553 | 0.300 | 0.705 | **DROP** |
| Book | 0.330 | 0.511 | 0.321 | 0.524 | 0.480 | **DROP** |

*Game: the prior raised MAE 0.545 → 0.553, p = 0.71 (not significant); Book: the prior lowered MAE 0.330 → 0.321, p = 0.48 (not significant). The unified prior was tested and **dropped** in every domain.*
<!-- METRICS:END -->

*(The Distillation-Prior Ablation table above is auto-generated from `latest_metrics.json`; the earlier hand-written copy here was removed to avoid drift. Raw terminal outputs of every pipeline run — master evaluation, renderer, distillation ablation, transfer grid, transfer verdict, entity links, bridge eval, active-learning queue — are captured in `reports/PIPELINE_RUN_LOG.md`.)*

---

## 9. Most Predictive Features (per model)

Auto-rendered from `reports/feature_diagnostics.json` (regenerate with `python -m src.experiments.feature_diagnostics`). Each row is the top features by mean |SHAP| of a per-model XGB surrogate fit on that model's own training matrix — a proxy for the deployed model's driver features whose *ordering* is stable across reruns. Per-model bar charts are saved to `reports/shap_<model>.png`, and the full field→feature provenance is in the README's Feature Engineering Deep-Dive.

<!-- FEATURE_IMPORTANCE:BEGIN -->
*Top features by mean |SHAP| on a per-model XGB surrogate trained on each model's own feature matrix (regenerate with `python -m src.experiments.feature_diagnostics`; per-model bar charts in `reports/shap_<model>.png`). Values are the average absolute impact on the predicted star rating.*

| Model | Most predictive features (mean \|SHAP\|, high → low) |
| :--- | :--- |
| **Movies (XGB surrogate of the mean ensemble)** | `lb_avg_rating` (0.447), `critic_avg_5` (0.048), `age_at_rating` (0.038), `lb_gap` (0.035), `actors_encoded` (0.031) |
| **TV Shows (XGB surrogate of the mean ensemble)** | `imdb_rating` (0.363), `pca_1` (0.084), `imdb_votes_log` (0.076), `vote_average` (0.066), `vote_count_log` (0.060) |
| **Games (local SVR)** | `ratings_count` (0.268), `pca_1` (0.159), `rating` (0.110), `playtime` (0.091), `reviews_count` (0.063) |
| **Books (local SVR)** | `pca_1` (0.117), `authors_encoded` (0.112), `pca_0` (0.097), `ratingsCount` (0.094), `pca_2` (0.080) |
| **Unified (Mean Ensemble)** | `critic_avg_5` (0.422), `creator_encoded` (0.089), `emb_334` (0.020), `emb_192` (0.018), `emb_184` (0.014) |
<!-- FEATURE_IMPORTANCE:END -->

---

## 10. Classification Performance — Confusion Matrices & Scores

The regression benchmark (§Performance Summary) answers *"how many stars off?"*; this section answers *"did the model make the right taste call?"*. Both use the **identical cached registry OOF predictions** — no retraining — re-scored as a discrete decision. The 3-class scheme applies the deployed movies `target_class` bins uniformly (**Dislike** ≤2.5★ / **Neutral** 2.5–3.5★] / **Like** >3.5★); the binary **Like** view (≥3.5★) is the recommend/skip decision the Oracle actually surfaces. Music is excluded — it has no ground-truth rating (PU pseudo-labels only), so there is no "truth" to build a confusion matrix against.

<!-- CLASSIFICATION:BEGIN -->
*The same registry OOF predictions scored on the regression benchmark table, re-framed as a 3-class taste decision — **Dislike** (≤2.5★), **Neutral** (2.5–3.5★], **Like** (>3.5★) — plus a binary **Like** (≥3.5★, the recommend/skip decision). Predictions are snapped to the 0.5★ grid first. Regenerate with `python -m src.experiments.classification_diagnostics`; per-model heatmaps in `reports/confusion_<model>.png`. Music is excluded (no ground-truth rating).*

#### Headline classification scores (per deployed model)

| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,234 | 68.4% | 68.8% | 0.695 | 0.687 | 0.514 | 0.857 |
| **TV Shows** | 175 | 56.6% | 54.5% | 0.551 | 0.574 | 0.313 | 0.805 |
| **Games** | 66 | 59.1% | 52.6% | 0.526 | 0.599 | 0.316 | 0.854 |
| **Books** | 112 | 75.0% | 70.3% | 0.730 | 0.745 | 0.588 | 0.864 |
| **Unified** | 1,587 | 60.6% | 60.1% | 0.612 | 0.610 | 0.389 | 0.813 |

#### Confusion matrices & per-class scores

**Movies** · Mean-Ensemble (XGB+Cat+SVR) · N=1234 · accuracy 68.4%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 212 | 77 | 7 |
| **Neutral** | 54 | 317 | 105 |
| **Like** | 1 | 146 | 315 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.794 | 0.716 | 0.753 | 296 |
| Neutral | 0.587 | 0.666 | 0.624 | 476 |
| Like | 0.738 | 0.682 | 0.709 | 462 |
| **Like (≥3.5★, binary)** | 0.813 | 0.906 | 0.857 | 695 |

**TV Shows** · Mean-Ensemble (XGB+Cat+SVR) · N=175 · accuracy 56.6%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 12 | 17 | 3 |
| **Neutral** | 4 | 39 | 11 |
| **Like** | 0 | 41 | 48 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.750 | 0.375 | 0.500 | 32 |
| Neutral | 0.402 | 0.722 | 0.517 | 54 |
| Like | 0.774 | 0.539 | 0.636 | 89 |
| **Like (≥3.5★, binary)** | 0.728 | 0.901 | 0.805 | 101 |

**Games** · Local SVR · N=66 · accuracy 59.1%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 2 | 6 | 0 |
| **Neutral** | 1 | 15 | 4 |
| **Like** | 0 | 16 | 22 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.667 | 0.250 | 0.364 | 8 |
| Neutral | 0.405 | 0.750 | 0.526 | 20 |
| Like | 0.846 | 0.579 | 0.688 | 38 |
| **Like (≥3.5★, binary)** | 0.759 | 0.978 | 0.854 | 45 |

**Books** · Local SVR · N=112 · accuracy 75.0%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 13 | 13 | 0 |
| **Neutral** | 4 | 47 | 1 |
| **Like** | 0 | 10 | 24 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.765 | 0.500 | 0.605 | 26 |
| Neutral | 0.671 | 0.904 | 0.770 | 52 |
| Like | 0.960 | 0.706 | 0.814 | 34 |
| **Like (≥3.5★, binary)** | 0.891 | 0.838 | 0.864 | 68 |

**Unified** · Mean Ensemble (XGB+Cat) · N=1587 · accuracy 60.6%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 207 | 137 | 18 |
| **Neutral** | 80 | 366 | 156 |
| **Like** | 8 | 226 | 389 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.702 | 0.572 | 0.630 | 362 |
| Neutral | 0.502 | 0.608 | 0.550 | 602 |
| Like | 0.691 | 0.624 | 0.656 | 623 |
| **Like (≥3.5★, binary)** | 0.765 | 0.868 | 0.813 | 909 |

<!-- CLASSIFICATION:END -->

*Reading these: **accuracy** is the raw fraction of correct 3-class calls; **balanced accuracy** and **macro-F1** weight the rare Dislike/Like classes equally with the common Neutral one (so they punish a model that just predicts the majority bucket); **Cohen κ** is agreement above chance; the **binary-Like F1** is the single most decision-relevant number — how well the model separates "would recommend" from "would skip". These are honest OOF numbers on the 0.5★-snapped grid, so they read a little below the continuous-MAE story, exactly as expected once predictions are bucketed.*

---

## 11. Is the Model Cheating? — Beating the Mean/Median (baseline sanity)

The most important honesty check in a clustered-rating project. Because most ratings sit in a narrow 3.0–4.5★ band, a model that simply predicts *near the mean/median* — never risking a 5★ or a 1★ — earns a respectable MAE and ±0.5★ accuracy **for free**. MAE and accuracy therefore cannot, by themselves, prove the model learned taste. This section pits every deployed model against the trivial baselines and — decisively — measures whether it **orders** items correctly. Auto-rendered from `reports/baseline_sanity.json` (`python -m src.experiments.baseline_sanity`), computed on the same cached registry OOF as the benchmark table; music is excluded (no ground-truth rating).

<!-- BASELINE_SANITY:BEGIN -->
*The honest test of "did it learn taste, or is it just predicting near the mean/median?" MAE and ±0.5★ accuracy are gameable in clustered domains — so this table pits each deployed model against the trivial baselines. Regenerate with `python -m src.experiments.baseline_sanity`.*

| Model | N | MAE | MAE (mean) | Skill vs mean | ±0.5★ acc | ±0.5★ acc (mode) | Acc lift | **Spearman ρ** | p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,234 | 0.421 | 0.744 | +0.435 | 84% | 52% | +31% | **0.796** | 2e-271 |
| **TV Shows** | 175 | 0.674 | 0.877 | +0.231 | 63% | 47% | +17% | **0.611** | 3e-19 |
| **Games** | 66 | 0.545 | 0.826 | +0.339 | 70% | 47% | +23% | **0.683** | 3e-10 |
| **Books** | 112 | 0.330 | 0.674 | +0.510 | 88% | 54% | +34% | **0.857** | 2e-33 |
| **Unified** | 1,587 | 0.525 | 0.757 | +0.307 | 74% | 51% | +23% | **0.670** | 2e-207 |

*Reading it: **Skill vs mean** > 0 means the model beats a constant predict-the-average (here +0.23 to +0.51). **Acc lift** is the ±0.5★ accuracy *above* the free mode-baseline (predicting the single most common rating) — the part that isn't given away by the clustered rating distribution (+17 to +34 pts). And the load-bearing one: **Spearman ρ** — a model that just predicts noise around the mean has ρ≈0 and cannot order your items; every model here sits at ρ 0.61 to 0.86 with p ≤ 3e-10, so it genuinely ranks favourites above passes. **Honest caveat:** the small domains (TV/Games/Books) still compress prediction *magnitude* toward the center (`spread_ratio` 0.47–0.76 in `baseline_sanity.json`), which is exactly why the pipeline applies rank/quantile calibration and why the Books watchlist queue ranks by novelty, not by the (mean-hugging) predicted star.*
<!-- BASELINE_SANITY:END -->

**Why Spearman ρ is the argument.** A constant / near-constant predictor (predict-the-mean ± noise) has, by construction, **skill = 0** and **Spearman ρ ≈ 0** — it cannot rank your favourites above your passes. Every deployed model instead posts **skill 0.21–0.57 over predict-the-mean** and **Spearman ρ 0.57–0.86 at p ≤ 1e-11**, and lifts ±0.5★ accuracy **+14–33 points above the mode-baseline** (predicting the single most common rating) in the 2026-10-01 run. That trio is the evidence the models track real taste, not the histogram. **The honest half:** the small domains still *compress the magnitude* of their predictions toward the center (`spread_ratio` ≈ 0.58–0.81 — predictions vary about 20–40% less than the truth, and 66–74% land within ±0.5 of the mean), so they *rank* well but under-commit to extremes. That specific, measured failure mode is precisely what the **rank/quantile calibration** (Engineering Highlights) corrects, and why the Books watchlist active-learning queue ranks by **novelty**, not by the (mean-hugging) predicted star. Movies, by contrast, predicts with essentially the full spread of the truth (`spread_ratio` ≈ 1.02) — no compression, no cheating.

---

## 12. Bias / Variance — and Why the Metrics Are Where They Are

The strategic "what do I actually need — more data, more features, or a better model?" question, answered with a bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`).

<!-- BIAS_VARIANCE:BEGIN -->
*Bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`). `bias²` = systematic error of the average prediction; `variance` = how much predictions wobble as the training set is resampled. bias² here also absorbs the irreducible 0.5★-self-rating noise.*

| Model | N | feats | bias² | variance | **bias share** | test MAE | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,234 | 111 | 0.343 | 0.022 | 94% | 0.444 | BIAS-limited and the learning curve is flat |
| **TV Shows** | 175 | 56 | 0.945 | 0.050 | 95% | 0.776 | BIAS-limited, but the learning curve is still rising |
| **Games** | 66 | 51 | 0.705 | 0.048 | 94% | 0.612 | BIAS-limited, but the learning curve is still rising |
| **Books** | 112 | 29 | 0.309 | 0.028 | 92% | 0.389 | BIAS-limited, but the learning curve is still rising |
| **Unified** | 1,587 | 406 | 0.495 | 0.028 | 95% | 0.551 | BIAS-limited and the learning curve is flat |

*Bias share spans **92–95%**, variance share 5–8%. Every model is bias-dominated — stable, not data-starved — so **more rows of the same features is a weak lever**; the levers are richer *features* (reduce bias) and the *aleatoric ceiling* you can't cross.*
<!-- BIAS_VARIANCE:END -->

**Why the bias is high (and R²/MAE look mediocre) — three compounding reasons.** (1) **Aleatoric ceiling:** ratings are self-reported on a coarse 0.5★ grid, so the same item wobbles ±a grid step by mood/recency/rounding — irreducible noise that lands inside `bias²`; an MAE near 0.5★ is therefore close to the floor. (2) **Metadata ≠ personal taste:** the features are crowd-quality signals (IMDb/critic/votes/genre), which capture "is this generally good," not "will *this user* like it" — the personal residual is unobservable, so bias is lowest where features are richest (Movies) and highest where they're thin (TV). (3) **Small N** (66–175) for the niche domains caps person-specific learning.

**The decisive experiment (§ this revision).** To test whether reason (2) is fixable, we added the *exact* feature that makes Movies strong — leakage-safe, Bayesian-smoothed, out-of-fold **creator/cast/writer target encodings** — to the worst domain, TV. The bias moved only **1.014 → 0.963** (test MAE 0.784 → 0.778; historical one-off run, not regenerated by the pipeline): at N≈170 most creators appear once or twice, so the smoothed encoding reverts to the global mean and the residual is dominated by reason (1). **This is the honest conclusion: the metrics are near the achievable ceiling for predicting one person's coarse, noisy self-ratings from crowd metadata — not a modelling deficiency a bigger ensemble or more of the same rows would fix.** It is precisely why the project leads with MAE-skill and Spearman ρ (which show real, significant signal) rather than R².
