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

1. **Asymmetric edge-penalty loss.** A custom XGBoost objective penalises errors on edge ratings. **Status: retire it.** The head-to-head ablation shows a small net harm (overall MAE +0.0039, p = 0.083). The weight keys on the extremity of the true rating, not on the error, so the loss cannot target the phenomenon it was built for.
2. **Leakage-safe target encoding.** Movies and TV use 5-fold out-of-fold, Bayesian-smoothed (m=10) encodings for creators, cast and writers. Both matrices persist them, and `actors_encoded` is the fourth-strongest movie feature by SHAP (2026-10-01 run).
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
The test adds the unified model's OOF prediction to the local SVRs as a feature. In the 2026-10-01 run both cells are null.

- **Games: DROP.** MAE falls from 0.5894 to 0.5818; skill rises from 0.254 to 0.264. p = 0.317.
- **Books: DROP.** MAE rises from 0.3491 to 0.3527; skill falls from 0.474 to 0.468. p = 0.465. The earlier p = 0.0343 'KEEP' never came back.
- **Caveat on the two cells above.** They came from a stand-in `SVR(C=1)` with no balanced weights and no calibration, scored over 5× repeated fold rows. The ablation now runs the deployed local model and tests one row per item. A 2026-10-11 in-memory rerun with that setup gave Games MAE 0.515 → 0.538 with the prior (worse, p = 0.26) and Books 0.286 → 0.282 (p = 0.71). Both stay DROP. The no-prior MAEs equal the benchmark Local SVR rows exactly. `reports/distillation_ablation_results.json` and the table below update at the next `--full` run.

**The project verdict stays DROP.** Three reasons support that decision. Neither cell is near significance. The folds are correlated at N = 66 and 110, so even these p-values are optimistic. The Books cell has read p = 0.705, 0.034, 0.446, 0.336, 0.248 and now 0.465 across refreshes, with the sign of the effect flipping, so it is not stable. The deployed Books SVR attaches the prior only when a unified prediction file is present, so the shipped behaviour is conditional. Re-test at the next corpus refresh.

---

## 4. Unified model

### Deployed model
The unified model is a cross-domain **Mean Ensemble** of XGBoost and CatBoost. It uses leak-free domain centroid alignment, fitted on the training folds only, and multi-modal missingness masks. It has a single headline row in the Performance Summary.

- **Unified (rated)** is the headline taste metric. It trains and evaluates on real ratings across the four rated domains, on the frozen registry folds. It scores **skill 0.287, Spearman 0.674** (MAE 0.542, R² 0.445, 73.1%) over 1,514 rated items in the 2026-10-01 run.
- **Music is excluded.** Music has no star ratings, and the project no longer synthesizes one. The former "Unified (full pool, +music PU pseudo-labels)" row has been **removed**, not footnoted. It scored the model partly against another model's fabricated affinities.

### Analysis — the local model wins in every rated domain

Both sides of this comparison use the identical pooled per-item OOF estimator on the same frozen folds, on the same items. Both skill columns use the same per-domain mean baseline (`baseline_sanity.json`). Numbers are from the 2026-10-01 run.

| Domain | Local skill | Unified slice skill | Local MAE | Unified slice MAE | Winner |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Movies | **0.355** | 0.315 | **0.483** | 0.513 | local |
| TV | **0.208** | 0.186 | **0.694** | 0.714 | local, but by 0.020 MAE, which is inside TV's fold-reshuffle noise |
| Games | **0.376** | 0.202 | **0.515** | 0.659 | local, by a wide margin |
| Books | **0.566** | 0.241 | **0.286** | 0.500 | local, by a wide margin |

**This verdict changed in the 2026-10-01 run.** In the previous run the unified slice won Movies (0.475 against 0.486 MAE) and TV (0.666 against 0.680) by narrow margins. Now the local model wins all four. The unified numbers are optimistic, because the unified entity encodings leak across the fold boundary, so the local wins are conservative.

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
**Realized verdict (full 50-fold grid, domain-blind, quadruple-controlled): TASTE TRANSFERS.** TASTE TRANSFERS: at least one target shows a positive transfer finding (augmented lift > 0, paired p < 0.05, at >= 2 fractions).

| Target | Best source subset | lift @100% | p | Clears the bar? |
| :--- | :--- | ---: | ---: | :--- |
| movie | `book` | +0.0018 | 0.555 | no |
| tv | `book` | +0.0265 | 0.021 | no |
| game | `movie` | +0.1452 | 0.000 | YES |
| book | `tv+game` | +0.0220 | 0.244 | no |

**Zero-shot -- can other domains rank a target the model has never trained on?** (Spearman, mean over folds)

| Target | Target-only XGB (same shared features) | Deployed model | Best source (never sees target) | rho | % of target-only XGB |
| :--- | ---: | ---: | :--- | ---: | ---: |
| movie | 0.677 | 0.751 | `tv+game+book` | +0.490 | 72% |
| tv | 0.426 | 0.574 | `movie` | +0.524 | **123%** |
| game | 0.307 | 0.719 | `movie` | +0.599 | **195%** |
| book | 0.635 | 0.859 | `movie+tv` | +0.600 | 95% |

Bold rows are targets where a model that has **never seen the domain** ranks it better than a target-only XGB on the same shared features: `movie->tv`, `movie->game`. Against the deployed model (pooled per-item Spearman from `baseline_sanity.json`, a different estimator than the per-fold mean) it wins for none of them.

**Why the sign differs by domain.** Local rating coherence -- how well an item's rating is predicted by its nearest neighbour in the shared text space -- orders the domains identically to the value of their own training data:

| Domain | n | Coherence (rho) | p | Effect of its own data on a movie-trained model |
| :--- | ---: | ---: | ---: | :--- |
| tv | 175 | +0.102 | 0.1805 | hurts |
| movie | 1062 | +0.245 | 0.0000 | n/a (is the source) |
| game | 66 | +0.361 | 0.0029 | null |
| book | 104 | +0.716 | 0.0000 | helps |

TV is the only domain where the mean gap to the nearest neighbour (1.097) EXCEEDS the spread of its own ratings (1.071) -- copying a lookalike is worse than guessing the average. Boosting still fits those rows; with no local structure to fit it fits noise, into exactly the region where the TV test items live.

Aligned-space distance still does not predict transfer (Spearman +0.191 over n=12 pairs).
<!-- TRANSFER_VERDICT:END -->

### The complete transfer grid
<!-- TRANSFER_GRID_FULL:BEGIN -->
## Target: Movies (107 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.677** (SE 0.008). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `tv+game+book` | +0.490 | +0.631 | +0.653 | **+0.683** |
| `tv` | +0.430 | +0.624 | +0.650 | **+0.680** |
| `tv+book` | +0.420 | +0.637 | +0.656 | +0.677 |
| `tv+game` | +0.373 | +0.621 | +0.649 | +0.674 |
| `game+book` | +0.320 | +0.626 | +0.650 | +0.676 |
| `book` | +0.137 | +0.628 | +0.658 | **+0.685** |
| `game` | +0.091 | +0.617 | +0.648 | **+0.677** |

## Target: TV (18 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.426** (SE 0.029). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.524** | **+0.433** | +0.424 | **+0.465** |
| `movie+game+book` | **+0.471** | **+0.433** | **+0.464** | **+0.493** |
| `movie+book` | **+0.453** | **+0.433** | **+0.470** | **+0.518** |
| `movie+game` | **+0.435** | **+0.468** | **+0.437** | **+0.493** |
| `game+book` | +0.331 | +0.322 | +0.367 | **+0.468** |
| `book` | +0.150 | +0.331 | +0.379 | **+0.483** |
| `game` | +0.149 | +0.246 | +0.316 | +0.419 |

## Target: Games (7 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.307** (SE 0.059). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.599** | **+0.586** | **+0.612** | **+0.602** |
| `movie+book` | **+0.584** | **+0.591** | **+0.601** | **+0.619** |
| `movie+tv` | **+0.541** | **+0.543** | **+0.569** | **+0.538** |
| `movie+tv+book` | **+0.508** | **+0.551** | **+0.535** | **+0.554** |
| `tv+book` | **+0.468** | **+0.474** | **+0.481** | **+0.507** |
| `tv` | **+0.404** | **+0.431** | **+0.487** | **+0.551** |
| `book` | **+0.333** | **+0.426** | **+0.442** | **+0.490** |

## Target: Books (11 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.635** (SE 0.029). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+tv` | +0.600 | +0.626 | **+0.642** | **+0.676** |
| `movie+tv+game` | +0.575 | +0.621 | **+0.658** | **+0.689** |
| `movie` | +0.570 | +0.617 | **+0.649** | **+0.684** |
| `movie+game` | +0.562 | **+0.636** | **+0.653** | **+0.683** |
| `tv` | +0.514 | +0.550 | +0.629 | +0.628 |
| `tv+game` | +0.510 | +0.576 | +0.612 | **+0.655** |
| `game` | +0.223 | +0.481 | +0.578 | **+0.636** |
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

| Metric | Pooled (all link types) | Adaptation only |
| :--- | ---: | ---: |
| Linked rated items (N) | **664** | 57 |
| MAE without bridge features | 0.5301 | 0.5526 |
| MAE with bridge features | 0.4940 | 0.5351 |
| ΔMAE | **+0.0361** | +0.0175 |
| 95% CI | **[+0.0143, +0.0572]** | [−0.0175, +0.0526] |
| Paired Wilcoxon p | **0.0014** | 0.317 |

**Verdict: the pooled bridge effect is significant; the adaptation-only slice is not.** ΔMAE +0.0361 with a 95% CI excluding zero and p = 0.0014 over 664 linked items. The adaptation slice grew from 38 to 57 items and is directionally positive (+0.0175) but still underpowered (p = 0.317).

**This result has now reversed twice, and the history matters.** ΔMAE +0.032 at p = 0.004 (artifact of the domain-identity leak, §8) → +0.019 at p = 0.077 → +0.0038 at p = 0.928 (null) → **+0.0361 at p = 0.0014** on the present corpus. A reader is right to ask whether the leak has returned. It has not, and the evidence points the other way: this corpus has *fewer* domain-identity handles than the null one, not more. The split `lang_en` / `lang_English` column — a partial domain indicator, since movies and TV carried ISO codes while books carried names — was merged in this run, and six unrated games carrying an impossible `target_reg = 0.0` were dropped from the registry. The effect appeared *after* removing a leak, not by adding one.

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

**The one real bug, and why it mattered.** Check 4's leak was live in both the transfer grid and the entity-bridge features. Removing it **changed a conclusion**: the bridge effect looked significant with the leak (p = 0.004) and is null without it (§6). The leak let the base model split on domain identity instead of transferring content. Audit the wiring before you trust a result. **Update (2026-08-28):** the bridge effect is significant again (+0.0361, p = 0.0014) — but on a corpus with *fewer* domain handles, not more, and after a link-graph rewrite that took movie-to-book adaptations from 2 to 18. See §6.

**Check 2, now controlled.** The project did not leave the text-length mismatch as an untested caveat. `text_norm_transfer_control.py` re-embeds every domain with length-matched, identically-templated text and re-runs the Games grid. That control was run when the grid was positive. **On the 2026-08-28 corpus the grid was NULL**, so the control then confirmed a null rather than a positive. The 2026-08-30 grid (the latest; not re-run on 2026-10-01) is positive again for the Games target (§5c). The **production** pipeline still uses the original embeddings. Re-templating the whole pipeline is the recommended next engineering pass. It would not change the transfer verdict, which is already controlled, but it would tidy the latent-space explorer and the slice metrics.

---

# Final conclusion

The hub is a reproducible, domain-aligned recommender pipeline. Two disciplines carry it: the frozen fold registry, and a reporting spine where every standalone and unified number is measured with the same estimator on the same folds.

The central question — does taste transfer across domains? — has an answer. **Yes, at the feature level, into the data-starved Games domain, from movies (§5c).** The finding needed full statistical power, domain-blind features, and four adversarial controls.

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

*Provenance: λ (temporal decay) = 0.000429; asymmetric-objective α: not applied in the evaluated Mean Ensemble. Generated at 2026-10-02T15:31:08 · git `3015c3f5`.*

#### Benchmarks — *standalone local models* + unified headline

| Domain | N | Model | **Skill vs mean** | Spearman ρ | MAE | MAE (predict mean) | R² | ±0.5★ Acc (mode baseline) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,163 | Prod Mean-Ensemble (edge+Cat+SVR+Ord, residual) | **+0.355** | 0.751 | 0.483 | 0.749 | 0.531 | 77.4% (51%) |
| **Unified (rated)** | 1,514 | Mean Ensemble | **+0.303** | 0.679 | 0.530 | 0.761 | 0.460 | 73.4% (50%) |
| **TV Shows** | 175 | Prod Simplex-Stack (edge+Cat+SVR) | **+0.241** | 0.574 | 0.666 | 0.877 | 0.306 | 63.4% (47%) |
| **Games** | 66 | Local SVR | **+0.376** | 0.719 | 0.515 | 0.826 | 0.497 | 74.2% (47%) |
| **Books** | 110 | Local SVR | **+0.566** | 0.859 | 0.286 | 0.659 | 0.715 | 88.2% (56%) |

*Read **Skill vs mean** first: 1 − MAE ÷ MAE of always predicting the average rating. Ratings cluster in 3–4★, so a constant 3.5★ already earns a respectable MAE and ±0.5★ accuracy (the bracketed mode baseline is what predicting the single most common rating scores); skill is 0 for any constant predictor and 1 for a perfect one, so it is the number the clustering cannot inflate. **Spearman ρ** asks whether the model orders items correctly. The four domain rows are the **production local models** on the frozen registry folds (one row per item, N = unique items): **Movies and TV are the deployed tuned ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) and **Games and Books are the deployed SVR**. The **Unified (rated)** row (N=1,514) is the cross-domain Mean Ensemble over all four rated domains. Music is not scored here: it has no star ratings and is handled as an unsupervised taste-profile / similarity engine.*

#### Per-Domain Slice — *Unified model, by domain* (registry OOF)

| Domain | N | R² | MAE | ±0.5★ Acc |
| :--- | :--- | :--- | :--- | :--- |
| Movies | 1,163 | 0.516 | 0.500 | 75.8% |
| Shows | 175 | 0.149 | 0.711 | 60.0% |
| Games | 66 | 0.355 | 0.667 | 62.1% |
| Books | 110 | 0.452 | 0.477 | 77.3% |

*Read against the **production benchmarks** above — both measured with the identical pooled per-item OOF estimator on the frozen folds, on the same items. The verdict per domain below is computed from the two tables on every render, so it cannot go stale.*

*Movies: local 0.483 vs unified 0.500 → tie (within 0.02 MAE, fold-resampling noise) · Shows: local 0.666 vs unified 0.711 → local model wins · Games: local 0.515 vs unified 0.667 → local model wins · Books: local 0.286 vs unified 0.477 → local model wins.*

#### Ablation Study (rated-only, frozen registry folds)

| Protocol | MAE | R² | Effect Size | p-value |
| :--- | :--- | :--- | :--- | :--- |
| Base XGB (Aligned) | 0.5430 | 0.4364 | ref | ref |
| Mean Ensemble (XGB+Cat) | 0.5333 | 0.4526 | d=0.131 (trivial) | 0.0000 |
| Ridge Stack (removed — legacy 5-fold protocol) | 0.5609 | 0.3837 | n/a (legacy) | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088 | 0.2688 | n/a (legacy) | n/a (legacy) |

*Unlike the benchmark table, MAE here is scored **per fold row**: each item is tested once in each of the 5 CV repeats and counted 5 times, so it differs slightly from the per-item benchmark number. Effect size is paired Cohen's d and p is a Wilcoxon test on those same 5×-replicated rows, so p is optimistic (pairs are not independent). The ensemble's MAE is -0.0097 vs Base XGB (better). Rows 3–4 are removed paths kept for the record; their numbers come from the legacy 5-fold protocol, not the frozen registry, and are directionally indicative only.*

#### Distillation-Prior Ablation (Games & Books)

| Domain | MAE (no prior) | Skill (no prior) | MAE (with prior) | Skill (with prior) | p-value | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game | 0.589 | 0.254 | 0.588 | 0.256 | 0.827 | **DROP** |
| Book | 0.349 | 0.474 | 0.345 | 0.481 | 0.384 | **DROP** |

*Game: the prior lowered MAE 0.589 → 0.588, p = 0.83 (not significant); Book: the prior lowered MAE 0.349 → 0.345, p = 0.38 (not significant). The unified prior was tested and **dropped** in every domain.*
<!-- METRICS:END -->

*(The Distillation-Prior Ablation table above is auto-generated from `latest_metrics.json`; the earlier hand-written copy here was removed to avoid drift. Raw terminal outputs of every pipeline run — master evaluation, renderer, distillation ablation, transfer grid, transfer verdict, entity links, bridge eval, active-learning queue — are captured in `reports/PIPELINE_RUN_LOG.md`.)*

---

## 9. Most Predictive Features (per model)

Auto-rendered from `reports/feature_diagnostics.json` (regenerate with `python -m src.experiments.feature_diagnostics`). Each row is the top features by mean |SHAP| of a per-model XGB surrogate fit on that model's own training matrix — a proxy for the deployed model's driver features whose *ordering* is stable across reruns. Per-model bar charts are saved to `reports/shap_<model>.png`, and the full field→feature provenance is in the README's Feature Engineering Deep-Dive.

<!-- FEATURE_IMPORTANCE:BEGIN -->
*Top features by mean |SHAP| on a per-model XGB surrogate trained on each model's own feature matrix (regenerate with `python -m src.experiments.feature_diagnostics`; per-model bar charts in `reports/shap_<model>.png`). Values are the average absolute impact on the predicted star rating.*

| Model | Most predictive features (mean \|SHAP\|, high → low) |
| :--- | :--- |
| **Movies (stacking ensemble)** | `critic_avg_5` (0.261), `imdb_rating` (0.178), `vote_average` (0.062), `actors_encoded` (0.058), `age_at_rating` (0.044) |
| **TV Shows (simplex-stack)** | `imdb_rating` (0.364), `imdb_votes_log` (0.092), `vote_average` (0.080), `pca_4` (0.064), `year` (0.059) |
| **Games (local SVR)** | `ratings_count` (0.268), `pca_1` (0.159), `rating` (0.110), `playtime` (0.091), `reviews_count` (0.063) |
| **Books (local SVR)** | `pca_1` (0.227), `pca_5` (0.084), `pca_2` (0.066), `authors_encoded` (0.060), `pca_3` (0.053) |
| **Unified (Mean Ensemble)** | `critic_avg_5` (0.426), `creator_encoded` (0.094), `emb_334` (0.022), `emb_192` (0.014), `emb_93` (0.014) |
<!-- FEATURE_IMPORTANCE:END -->

---

## 10. Classification Performance — Confusion Matrices & Scores

The regression benchmark (§Performance Summary) answers *"how many stars off?"*; this section answers *"did the model make the right taste call?"*. Both use the **identical cached registry OOF predictions** — no retraining — re-scored as a discrete decision. The 3-class scheme applies the deployed movies `target_class` bins uniformly (**Dislike** ≤2.5★ / **Neutral** 2.5–3.5★] / **Like** >3.5★); the binary **Like** view (≥3.5★) is the recommend/skip decision the Oracle actually surfaces. Music is excluded — it has no ground-truth rating (PU pseudo-labels only), so there is no "truth" to build a confusion matrix against.

<!-- CLASSIFICATION:BEGIN -->
*The same registry OOF predictions scored on the regression benchmark table, re-framed as a 3-class taste decision — **Dislike** (≤2.5★), **Neutral** (2.5–3.5★], **Like** (>3.5★) — plus a binary **Like** (≥3.5★, the recommend/skip decision). Predictions are snapped to the 0.5★ grid first. Regenerate with `python -m src.experiments.classification_diagnostics`; per-model heatmaps in `reports/confusion_<model>.png`. Music is excluded (no ground-truth rating).*

#### Headline classification scores (per deployed model)

| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,163 | 65.6% | 66.5% | 0.665 | 0.656 | 0.474 | 0.823 |
| **TV Shows** | 175 | 54.3% | 53.5% | 0.539 | 0.557 | 0.281 | 0.824 |
| **Games** | 66 | 62.1% | 56.1% | 0.569 | 0.634 | 0.354 | 0.880 |
| **Books** | 110 | 77.3% | 75.6% | 0.769 | 0.773 | 0.635 | 0.915 |
| **Unified** | 1,514 | 60.0% | 59.5% | 0.606 | 0.604 | 0.380 | 0.810 |

#### Confusion matrices & per-class scores

**Movies** · Prod Stacking (edge+Cat+SVR+Ord) · N=1163 · accuracy 65.6%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 203 | 68 | 10 |
| **Neutral** | 77 | 257 | 120 |
| **Like** | 3 | 122 | 303 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.717 | 0.722 | 0.720 | 281 |
| Neutral | 0.575 | 0.566 | 0.571 | 454 |
| Like | 0.700 | 0.708 | 0.704 | 428 |
| **Like (≥3.5★, binary)** | 0.818 | 0.828 | 0.823 | 645 |

**TV Shows** · Prod Simplex-Stack (edge+Cat+SVR) · N=175 · accuracy 54.3%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 16 | 12 | 4 |
| **Neutral** | 11 | 30 | 13 |
| **Like** | 0 | 40 | 49 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.593 | 0.500 | 0.542 | 32 |
| Neutral | 0.366 | 0.556 | 0.441 | 54 |
| Like | 0.742 | 0.551 | 0.632 | 89 |
| **Like (≥3.5★, binary)** | 0.758 | 0.901 | 0.824 | 101 |

**Games** · Local SVR · N=66 · accuracy 62.1%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 3 | 5 | 0 |
| **Neutral** | 2 | 13 | 5 |
| **Like** | 0 | 13 | 25 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.600 | 0.375 | 0.462 | 8 |
| Neutral | 0.419 | 0.650 | 0.510 | 20 |
| Like | 0.833 | 0.658 | 0.735 | 38 |
| **Like (≥3.5★, binary)** | 0.800 | 0.978 | 0.880 | 45 |

**Books** · Local SVR · N=110 · accuracy 77.3%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 18 | 8 | 0 |
| **Neutral** | 6 | 43 | 3 |
| **Like** | 0 | 8 | 24 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.750 | 0.692 | 0.720 | 26 |
| Neutral | 0.729 | 0.827 | 0.775 | 52 |
| Like | 0.889 | 0.750 | 0.814 | 32 |
| **Like (≥3.5★, binary)** | 0.936 | 0.894 | 0.915 | 66 |

**Unified** · Mean Ensemble (XGB+Cat) · N=1514 · accuracy 60.0%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 194 | 137 | 16 |
| **Neutral** | 82 | 355 | 143 |
| **Like** | 7 | 220 | 360 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.685 | 0.559 | 0.616 | 347 |
| Neutral | 0.499 | 0.612 | 0.549 | 580 |
| Like | 0.694 | 0.613 | 0.651 | 587 |
| **Like (≥3.5★, binary)** | 0.765 | 0.860 | 0.810 | 857 |

<!-- CLASSIFICATION:END -->

*Reading these: **accuracy** is the raw fraction of correct 3-class calls; **balanced accuracy** and **macro-F1** weight the rare Dislike/Like classes equally with the common Neutral one (so they punish a model that just predicts the majority bucket); **Cohen κ** is agreement above chance; the **binary-Like F1** is the single most decision-relevant number — how well the model separates "would recommend" from "would skip". These are honest OOF numbers on the 0.5★-snapped grid, so they read a little below the continuous-MAE story, exactly as expected once predictions are bucketed.*

---

## 11. Is the Model Cheating? — Beating the Mean/Median (baseline sanity)

The most important honesty check in a clustered-rating project. Because most ratings sit in a narrow 3.0–4.5★ band, a model that simply predicts *near the mean/median* — never risking a 5★ or a 1★ — earns a respectable MAE and ±0.5★ accuracy **for free**. MAE and accuracy therefore cannot, by themselves, prove the model learned taste. This section pits every deployed model against the trivial baselines and — decisively — measures whether it **orders** items correctly. Auto-rendered from `reports/baseline_sanity.json` (`python -m src.experiments.baseline_sanity`), computed on the same cached registry OOF as the benchmark table; music is excluded (no ground-truth rating).

<!-- BASELINE_SANITY:BEGIN -->
*The honest test of "did it learn taste, or is it just predicting near the mean/median?" MAE and ±0.5★ accuracy are gameable in clustered domains — so this table pits each deployed model against the trivial baselines. Regenerate with `python -m src.experiments.baseline_sanity`.*

| Model | N | MAE | MAE (mean) | Skill vs mean | ±0.5★ acc | ±0.5★ acc (mode) | Acc lift | **Spearman ρ** | p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,163 | 0.483 | 0.749 | +0.355 | 77% | 51% | +26% | **0.751** | 2e-211 |
| **TV Shows** | 175 | 0.666 | 0.877 | +0.241 | 63% | 47% | +17% | **0.574** | 1e-16 |
| **Games** | 66 | 0.515 | 0.826 | +0.376 | 74% | 47% | +27% | **0.719** | 1e-11 |
| **Books** | 110 | 0.286 | 0.659 | +0.566 | 88% | 56% | +33% | **0.859** | 3e-33 |
| **Unified** | 1,514 | 0.530 | 0.761 | +0.303 | 73% | 50% | +23% | **0.679** | 2e-205 |

*Reading it: **Skill vs mean** > 0 means the model beats a constant predict-the-average (here +0.24 to +0.57). **Acc lift** is the ±0.5★ accuracy *above* the free mode-baseline (predicting the single most common rating) — the part that isn't given away by the clustered rating distribution (+17 to +33 pts). And the load-bearing one: **Spearman ρ** — a model that just predicts noise around the mean has ρ≈0 and cannot order your items; every model here sits at ρ 0.57 to 0.86 with p ≤ 1e-11, so it genuinely ranks favourites above passes. **Honest caveat:** the small domains (TV/Games/Books) still compress prediction *magnitude* toward the center (`spread_ratio` 0.57–0.81 in `baseline_sanity.json`), which is exactly why the pipeline applies rank/quantile calibration and why the Books watchlist queue ranks by novelty, not by the (mean-hugging) predicted star.*
<!-- BASELINE_SANITY:END -->

**Why Spearman ρ is the argument.** A constant / near-constant predictor (predict-the-mean ± noise) has, by construction, **skill = 0** and **Spearman ρ ≈ 0** — it cannot rank your favourites above your passes. Every deployed model instead posts **skill 0.21–0.57 over predict-the-mean** and **Spearman ρ 0.57–0.86 at p ≤ 1e-11**, and lifts ±0.5★ accuracy **+14–33 points above the mode-baseline** (predicting the single most common rating) in the 2026-10-01 run. That trio is the evidence the models track real taste, not the histogram. **The honest half:** the small domains still *compress the magnitude* of their predictions toward the center (`spread_ratio` ≈ 0.58–0.81 — predictions vary about 20–40% less than the truth, and 66–74% land within ±0.5 of the mean), so they *rank* well but under-commit to extremes. That specific, measured failure mode is precisely what the **rank/quantile calibration** (Engineering Highlights) corrects, and why the Books watchlist active-learning queue ranks by **novelty**, not by the (mean-hugging) predicted star. Movies, by contrast, predicts with essentially the full spread of the truth (`spread_ratio` ≈ 1.02) — no compression, no cheating.

---

## 12. Bias / Variance — and Why the Metrics Are Where They Are

The strategic "what do I actually need — more data, more features, or a better model?" question, answered with a bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`).

<!-- BIAS_VARIANCE:BEGIN -->
*Bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`). `bias²` = systematic error of the average prediction; `variance` = how much predictions wobble as the training set is resampled. bias² here also absorbs the irreducible 0.5★-self-rating noise.*

| Model | N | feats | bias² | variance | **bias share** | test MAE | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,163 | 106 | 0.398 | 0.024 | 94% | 0.486 | BIAS-limited and the learning curve is flat |
| **TV Shows** | 175 | 56 | 0.928 | 0.052 | 95% | 0.771 | BIAS-limited, but the learning curve is still rising |
| **Games** | 66 | 51 | 0.663 | 0.051 | 93% | 0.604 | BIAS-limited, but the learning curve is still rising |
| **Books** | 110 | 29 | 0.216 | 0.030 | 88% | 0.321 | BIAS-limited, but the learning curve is still rising |
| **Unified** | 1,514 | 406 | 0.523 | 0.028 | 95% | 0.571 | BIAS-limited and the learning curve is flat |

*Bias share spans **88–95%**, variance share 5–12%. Every model is bias-dominated — stable, not data-starved — so **more rows of the same features is a weak lever**; the levers are richer *features* (reduce bias) and the *aleatoric ceiling* you can't cross.*
<!-- BIAS_VARIANCE:END -->

**Why the bias is high (and R²/MAE look mediocre) — three compounding reasons.** (1) **Aleatoric ceiling:** ratings are self-reported on a coarse 0.5★ grid, so the same item wobbles ±a grid step by mood/recency/rounding — irreducible noise that lands inside `bias²`; an MAE near 0.5★ is therefore close to the floor. (2) **Metadata ≠ personal taste:** the features are crowd-quality signals (IMDb/critic/votes/genre), which capture "is this generally good," not "will *this user* like it" — the personal residual is unobservable, so bias is lowest where features are richest (Movies) and highest where they're thin (TV). (3) **Small N** (66–175) for the niche domains caps person-specific learning.

**The decisive experiment (§ this revision).** To test whether reason (2) is fixable, we added the *exact* feature that makes Movies strong — leakage-safe, Bayesian-smoothed, out-of-fold **creator/cast/writer target encodings** — to the worst domain, TV. The bias moved only **1.014 → 0.963** (test MAE 0.784 → 0.778; historical one-off run, not regenerated by the pipeline): at N≈170 most creators appear once or twice, so the smoothed encoding reverts to the global mean and the residual is dominated by reason (1). **This is the honest conclusion: the metrics are near the achievable ceiling for predicting one person's coarse, noisy self-ratings from crowd metadata — not a modelling deficiency a bigger ensemble or more of the same rows would fix.** It is precisely why the project leads with MAE-skill and Spearman ρ (which show real, significant signal) rather than R².
