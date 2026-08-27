# Personal Media Intelligence Hub — Final ML Technical Report

This document holds the architecture, the metrics and the analysis for every media domain. It is written in ASD-STE100 Simplified Technical English. Sections 9 to 12 are auto-generated from `reports/latest_metrics.json` and the diagnostic JSON files. Do not edit them by hand.

> **Data snapshot: 25 July 2026** (git `4972d15`) — N = 1,039 movies · 171 TV shows · 89 books · 72 games (66 rated) · 3,837 music tracks. The rated universe is 1,365 items. The unified frame holds 1,371 rows, because it keeps the 6 unrated "Incomplete" games. Music is unsupervised. The 3,837 tracks feed the taste-profile and similarity engine, not a rating benchmark.

## Artifact provenance

The artifacts do not all share one vintage. Read every number with its date.

| Vintage | Artifacts | Status |
| :--- | :--- | :--- |
| **2026-07-25** | `latest_metrics.json`, `production_oof.csv`, `standalone_oof.csv`, `distillation_ablation_results.json`, `transfer_grid_summary.json`, `entity_bridge_results*.json`, unified training frame | current corpus |
| **2026-07-26** | `baseline_sanity.json`, `classification_diagnostics.json`, `feature_diagnostics.json`, `bias_variance.json`, conformal widths | regenerated for this revision |
| **2026-06-26/27** | `edge_loss_ablation_results.json`, `transfer_placebo_control.json` | previous corpus (1,000 movies); both test mechanisms that the refresh does not change |
| **2026-07-11** | `text_vs_crowd.json`, `text_lift.json` | previous corpus |

## Overview of optimizations

1. **Asymmetric edge-penalty loss.** A custom XGBoost objective penalises errors on edge ratings. **Status: retire it.** The head-to-head ablation shows a small net harm (overall MAE +0.0039, p = 0.083). The weight keys on the extremity of the true rating, not on the error, so the loss cannot target the phenomenon it was built for.
2. **Leakage-safe target encoding.** Movies and TV use 5-fold out-of-fold, Bayesian-smoothed (m=10) encodings for creators, cast and writers. Both matrices persist them, and `actors_encoded` is the third-strongest movie feature by SHAP.
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
| **MAE** | Mean absolute error, in stars | **The primary yardstick.** MAE 0.50 means the prediction is half a star off on average. It resists the low-variance problem, so it is trustworthy in every domain. |
| **±0.5★ Accuracy** | Share of items predicted within one grid step | The "is this usable in the Oracle UI" metric. Movies 75.8% means about 3 of 4 predictions land within one half-star notch. |
| **R² (CV mean)** | Fraction of rating variance explained | **Treacherous. Read it only beside MAE.** TV, Games and Books ratings cluster in a narrow band, so the variance denominator is small and R² collapses even when MAE is good. A near-zero R² in a low-variance domain is a property of the denominator. |
| **Skill Score** | `1 − MAE_model / MAE_baseline`, baseline = predict the training mean | The honest accuracy metric for the small domains. It asks the only question that survives low variance: does the model beat guessing the average? Every domain scores +0.24 to +0.33. |
| **Spearman ρ** | Rank correlation of prediction against truth | **The load-bearing metric.** A mean-hugging model has ρ ≈ 0 and cannot order items. Every model here reaches ρ 0.59 to 0.74. |
| **N** | Count of unique rated items | Per item, after de-duplicating the 5 repeats. Never OOF rows. Counting rows once inflated Movies about five times. |

**The one-line takeaway.** MAE and skill score are load-bearing. Spearman ρ proves the model orders items. R² is trustworthy only for Movies. Never rank domains by R² alone.

---

## 1. Movies domain

### Deployed model
The deployed Movies regressor is a production stacking ensemble. Its members are an Optuna-tuned asymmetric edge-penalty XGBoost, CatBoost, an SVR and a 10-bucket ordinal expected-value head. A `RidgeCV` meta-learner fuses them. All headline metrics live in the auto-generated Performance Summary below. That table is the single source of truth.

### Analysis
Movies is the strongest domain by a wide margin. It has the most labels (N = 1,039) and the richest features. On the frozen 50-fold registry the deployed model scores **MAE 0.509, R² 0.481, 75.8%** within half a star.

**Read the numbers.** The typical prediction is about half a star off. R² 0.481 is trustworthy here, because the wide rating spread gives R² a real denominator. This is the one domain where that holds.

**Calibration explains the gap between two Movies numbers.** Uncalibrated, the same stack scores MAE 0.474 and skill 0.385 (`feature_diagnostics.json`). Calibrated, it scores MAE 0.509. That is a deliberate trade of about four accuracy points to recover the full 0.5–5★ range. Before calibration the model never predicted above 4.5★, because 62% of ratings fall in [3, 4] and hugging the mode minimises MAE.

**The ensemble earns its place.** The simplex blend beats every single member: MAE 0.4735 and skill 0.3849, against the best member (XGBoost) at MAE 0.4774 and skill 0.3799. No member is dead weight.

**A retired myth.** An earlier single 80/20-split run quoted R² ≈ 0.61 for Movies. The honest 50-fold registry figure for the same architecture is far lower. A single split is optimistic.

---

## 2. TV Shows domain

### Deployed model
The deployed TV model is a simplex stack. It is a constrained non-negative weighted average of XGBoost, CatBoost and SVR base learners. The weights are non-negative and sum to one. At N = 171 a learned `RidgeCV` meta-learner overfits, so the constrained average is the safer choice.

### Analysis
TV now scores **MAE 0.646, R² 0.332, 67.8%** on the registry folds.

**This is the largest per-domain gain in the project's history.** The earlier TV model scored MAE 0.802 and R² 0.082. Adding leakage-safe creator, cast and writer encodings, IMDb multi-critic fields, runtime, country and awards features produced the lift. The lesson is diagnostic. **TV's earlier weakness was missing features, not an aleatoric ceiling.** Once the model could see who made a show and who starred in it, the explained variance more than tripled.

**A reversed recommendation.** An earlier revision of this report recommended replacing the TV stack with a plain SVR, because the per-base-learner ablation then placed SVR-alone (skill 0.110) above the blend (0.095). **That recommendation is withdrawn.** On the current corpus the TV SVR is the weakest member (skill 0.1426) and the simplex blend is the best (skill 0.2356). The blend beats the best single member by +0.021 and beats SVR-alone by +0.093.

**TV is the one domain the pipeline does not calibrate.** Its rank correlation is too weak (ρ = 0.587). Quantile calibration would place 5★ on the wrong shows and cost about ten accuracy points.

---

## 3. Games and Books (niche domains)

### Deployed models
Both domains are sparse (N < 100) and both use a local `SVR(rbf)` with frozen hyperparameters (C = 1.0, ε = 0.1). A high-variance learner overfits at this N, so boosted trees were rejected.

Books additionally ships a **text-similarity predictor**. A `KNeighborsRegressor` (k = 8, cosine metric, distance weights) over the book vibe-PCA reaches skill **0.426**, against **0.244** for the SVR that mixes in crowd features. A blend of the two is worse (≈0.39). For Books, text alone is the honest best model, and it is explainable: it names the neighbour books that drove the prediction.

**N provenance.** The Games file keeps 6 titles with the "Incomplete" status as rows for ranking, but they carry no rating. The local Games model therefore scores N = 66, while the corpus and the unified frame hold 72. Both counts are correct in their own context.

### Analysis
The headline result is the **skill score**. Without any prior, the local SVRs beat the mean-rating baseline by **0.275 (Games)** and **0.282 (Books)**. This is genuine local signal at N ≈ 60, in domains where R² is treacherous.

**Games is the clearest case for a local model in the whole project.** The Games SVR leans on RAWG community columns — `ratings_count` (SHAP 0.196), `rating` (0.169) and `metacritic` (0.075). The shared cross-domain space discards those columns. When they are gone, the unified Games slice degrades to MAE 0.931 against the local 0.598. That contrast is the cleanest evidence that not all signal survives pooling.

**Books now tells the same story.** Since ingestion moved to the Hardcover library API, the local Books SVR gained 9 author and 152 standardized-genre columns. The local model now matches the unified slice on MAE (0.472 against 0.472) and beats it on R² (0.332 against 0.315) and on accuracy (79.8% against 74.2%). The same refresh populated the three previously-dead unified Books channels (`critic_avg_5`, `runtime`, `gen_*`), which are now at 100%.

### Distillation prior — dropped; both cells now null
The test adds the unified model's OOF prediction to the local SVRs as a feature. The current result splits.

- **Games: DROP.** The prior changes nothing. MAE is 0.5970 with it and without it. p = 1.000.
- **Books: DROP.** MAE rises from 0.3634 to 0.3699; skill falls from 0.4568 to 0.4472. p = 0.4461. The earlier p = 0.0343 'KEEP' did not survive the corpus refresh — the third flip for this cell.

**The project verdict stays DROP.** Three reasons support that decision. The effect is not reliable across domains, because one cell is exactly null. The favourable cell sits just under p = 0.05 at N = 89 with correlated folds, so it is fragile. The Books result flipped from p = 0.705 on the previous corpus, so it is not stable across refreshes. The deployed Books SVR attaches the prior only when a unified prediction file is present, so the shipped behaviour is conditional. Re-test at the next corpus refresh.

---

## 4. Unified model

### Deployed model
The unified model is a cross-domain **Mean Ensemble** of XGBoost and CatBoost. It uses leak-free domain centroid alignment, fitted on the training folds only, and multi-modal missingness masks. It has a single headline row in the Performance Summary.

- **Unified (rated)** is the headline taste metric. It trains and evaluates on real ratings across the four rated domains, on the frozen registry folds. It scores **MAE 0.526, R² 0.476, 75.5%** over 1,371 rows.
- **Music is excluded.** Music has no star ratings, and the project no longer synthesizes one. The former "Unified (full pool, +music PU pseudo-labels)" row has been **removed**, not footnoted. It scored the model partly against another model's fabricated affinities.

### Analysis — the local model wins, or ties, in every rated domain

Both sides of this comparison use the identical pooled per-item OOF estimator on the same frozen folds, on the same items.

| Domain | Production local MAE | Unified slice MAE | Winner |
| :--- | ---: | ---: | :--- |
| Movies | 0.509 | **0.480** | pooled, **but only against the calibrated local model**; the uncalibrated local stack scores 0.474 and wins |
| TV | **0.646** | 0.664 | local |
| Games | **0.598** | 0.931 | local, by a wide margin |
| Books | 0.472 | 0.472 | tie on MAE; local on R² and accuracy |

**The earlier "pooling rescues the data-poor domain" result is gone.** The TV feature enrichment and the Books library refresh left no weak rated domain for pooling to rescue. This is why the deployed system keeps **local models for production scoring** and uses the unified model only for the transfer study.

This does **not** dissolve the transfer grid (§5). "Does the pooled model beat local on average" is a blunt question. The grid asks the sharp one: which specific source combinations transfer into which targets.

### Architecture ablation
The project grew the unified model and then pruned it. Adding the mean ensemble over the aligned base XGBoost improves MAE from 0.5400 to 0.5345. The improvement is statistically significant but trivial in size (Cohen's d = 0.069, p < 10⁻⁴). It is kept because it is free, not because it matters. Ridge stacking (MAE 0.5609) and residual heads (MAE 0.6088) both made performance worse and were removed. Complexity did not help. That is itself the result.

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

**Verdict: TASTE TRANSFERS — into Games.** Every positive cell in the grid has Games as the target. This is a broad positive **column**, not one lucky configuration. Five source subsets clear the pre-registered bar.

| Source → Games | lift @100% | significant fractions | per-fraction detail |
| :--- | ---: | :--- | :--- |
| **`movie + tv + book`** | **+0.121** | **4 / 4** | +0.085 (p=0.031) · +0.081 (p=0.030) · +0.130 (p=0.0003) · +0.121 (p=0.0014) |
| `movie + tv` | +0.120 | 2 / 4 | — |
| `movie` (most parsimonious) | +0.111 | 3 / 4 | — |
| `movie + book` | +0.104 | 3 / 4 | — |
| `tv` | +0.021 | 3 / 4 | small lift |

**No other target transfers.** The best cells are +0.021 into TV (0/4 fractions), +0.023 into Books (0/4) and −0.0001 into Movies (0/4).

**Zero-shot affinity is now positive into Games.** A model trained only on a source, with no game data at all, scores +0.067 (movie), +0.088 (TV) and +0.082 (book) skill on games. The reverse direction is strongly negative: Games as a **source** scores −0.297 into movies, −0.458 into TV and −0.362 into books. Games is a good target and a bad source.

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

Aligned-space domain distance does **not** predict zero-shot skill. The Spearman correlation is **+0.085** over n = 12 pairs. The sign has moved between runs (−0.18, then +0.17, then +0.08), so at n = 12 this is noise, not a clean null. Geometry is not the transfer oracle. Target data-starvation is the better predictor.

---

## 6. Entity bridges — "does knowing I loved the book predict I'll love the show?"

The grid found feature-level transfer, but only into Games and only for the shared genre and vibe component. Entity bridges ask the complementary item-level question. If two items are linked across domains — an adaptation, a shared creator, a shared franchise — does the rating of one predict the rating of the other?

`src/linking/build_entity_links.py` resolves library items to Wikidata QIDs. It emits links through P144 and P4969 (based-on, derivative), P50, P57 and P86 (author, director, composer) and P179 (series). The current pass finds **2,691 cross-domain links**: 2,214 same-franchise, 456 shared-creator and 21 adaptation.

The evaluation adds OOF-safe bridge features to a base model over the linked and rated items. The features are the linked partner's rating, a franchise mean rating and a has-link indicator. A partner rating counts only when the partner sits in the training fold.

| Metric | Pooled (all link types) | Adaptation only |
| :--- | ---: | ---: |
| Linked rated items (N) | **665** | 38 |
| MAE without bridge features | 0.5180 | 0.7500 |
| MAE with bridge features | 0.5143 | 0.7237 |
| ΔMAE | +0.0038 | +0.0263 |
| 95% CI | [−0.0188, +0.0278] | [−0.0395, +0.0924] |
| Paired Wilcoxon p | **0.928** | **0.480** |

**Verdict: null in both slices.** The pooled effect is essentially zero and the confidence interval straddles zero. The adaptation slice is directionally larger but far from significant at N = 38.

**This is a downgrade, and the report states it.** The previous run reported +0.019 with p = 0.077 over 646 items, which read as a suggestive nudge. The corpus refresh moved the effect toward zero. An even earlier run reported ΔMAE +0.032 at p = 0.004, but that significance was **an artifact of the domain-identity leak** (§8). Item-level cross-domain linking is **not established** at this N.

**Informative censoring — an open direction.** If you dislike a book, you may decline to watch its film adaptation. That refusal is itself a signal, and the missingness is not at random. The data to test it does not exist. Only **2** of the 21 adaptation links join a movie to a book, so the figure generator skips the censoring-asymmetry plot. About 30 to 40 verified movie-to-book pairs would make the test informative.

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

**The one real bug, and why it mattered.** Check 4's leak was live in both the transfer grid and the entity-bridge features. Removing it **changed a conclusion**: the bridge effect looked significant with the leak (p = 0.004) and is null without it (§6). The leak let the base model split on domain identity instead of transferring content. Audit the wiring before you trust a result.

**Check 2, now controlled.** The project did not leave the text-length mismatch as an untested caveat. `text_norm_transfer_control.py` re-embeds every domain with length-matched, identically-templated text and re-runs the Games grid. The transfer **survives and strengthens** (§5c). Two scopes stay distinct. The control confirms the verdict on a clean space. The **production** pipeline still uses the original embeddings. Re-templating the whole pipeline is the recommended next engineering pass. It would not change the transfer verdict, which is already controlled, but it would tidy the latent-space explorer and the slice metrics.

---

# Final conclusion

The hub is a reproducible, domain-aligned recommender pipeline. Two disciplines carry it: the frozen fold registry, and a reporting spine where every standalone and unified number is measured with the same estimator on the same folds.

The central question — does taste transfer across domains? — has an answer. **Yes, at the feature level, into the data-starved Games domain, from movies (§5c).** The finding needed full statistical power, domain-blind features, and four adversarial controls.

## What this revision changed

This revision was a measurement refresh, not a re-render of cached numbers. Five conclusions moved.

1. **Local models now win, or tie, everywhere (§4).** The earlier claim that the unified model wins in TV is gone: local 0.646 against pooled 0.664. Books moved from a pooled win to a tie on MAE and a local win on every other metric. Pooling no longer rescues any rated domain.
2. **The TV ensemble recommendation reversed (§2).** The earlier report recommended replacing the TV stack with a plain SVR. On the current corpus the SVR is the weakest member and the blend is the best. The recommendation is withdrawn.
3. **Entity bridges fell to null (§6).** The pooled effect moved from +0.019 (p = 0.077) to +0.0038 (p = 0.928) after the corpus refresh. The suggestive nudge did not survive.
4. **The transfer headline moved from `movie+book` to `movie+tv+book` (§5c).** The strongest cell is now the three-source subset at +0.121 with 4/4 significant fractions. `movie` alone remains the parsimonious sufficient source at +0.111 (3/4). The similarity correlation moved from −0.18 to +0.085, which confirms it is noise at n = 12.
5. **The edge-penalty loss is retired (§Overview).** The head-to-head ablation found a small net harm. The deeper problem is the formulation: the weight keys on the label's extremity, not on the error.

The deterministic parts — the unified rated headline, the per-domain slices, and the ablation effect sizes — reproduced exactly against the prior run. That is the frozen-fold registry doing its job. The numbers that moved, moved because the corpus grew or the measurement improved. Each change is documented above, not buried.

---

<!-- METRICS:BEGIN -->
### Performance Summary

*Provenance: λ (temporal decay) = 0.000429; asymmetric-objective α: not applied in the evaluated Mean Ensemble. Generated at 2026-08-22T18:15:34 · git `4972d154`.*

#### Benchmarks — *standalone local models* + unified headline

| Domain | N | Model | R² (CV mean) | MAE | ±0.5★ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,062 | Prod Mean-Ensemble (edge+Cat+SVR+Ord, residual) | **0.520** | 0.492 | 76.7% |
| **Unified (rated)** | 1,407 | Mean Ensemble | **0.468** | 0.531 | 74.4% |
| **TV Shows** | 173 | Prod Simplex-Stack (edge+Cat+SVR) | **0.288** | 0.668 | 64.7% |
| **Games** | 66 | Local SVR | **0.473** | 0.515 | 74.2% |
| **Books** | 100 | Local SVR | **0.733** | 0.305 | 89.0% |

*The four domain rows are the **production local models** on the frozen registry folds (one row per item, N = unique items): **Movies and TV are the deployed tuned ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) — not the earlier plain-XGB / manual-simplex proxies — and **Games and Books are the deployed SVR**. The **Unified (rated)** row (N=1,407) is the cross-domain Mean Ensemble over all four rated domains — **the headline taste metric**. Music is not scored here: it has no star ratings and is handled as an unsupervised taste-profile / similarity engine, so it never enters the rating benchmark.*

#### Per-Domain Slice — *Unified model, by domain* (registry OOF)

| Domain | N | R² | MAE | ±0.5★ Acc |
| :--- | :--- | :--- | :--- | :--- |
| Movies | 1,062 | 0.563 | 0.479 | 78.5% |
| Shows | 173 | 0.216 | 0.708 | 57.2% |
| Games | 72 | 0.132 | 0.924 | 54.2% |
| Books | 100 | 0.426 | 0.495 | 75.0% |

*Read against the **production benchmarks** above — both measured with the identical pooled per-item OOF estimator on the frozen folds, on the same items. With the **real local models** (Movies = the deployed tuned edge-penalty stacking ensemble, not the old plain-XGB proxy), the like-for-like MAE comparison favours the local model in **Games and Books** — Games' Metacritic/platforms and (since Books moved to the Hardcover library API) Books' standardized author/genre multi-hot features are local signal the shared space dilutes. The **unified model wins in TV**, the one domain whose thin local feature set leaves room for cross-domain pooling to add data without losing signal, while **Movies is a statistical tie** (0.476 local vs 0.475 unified — within fold-resampling noise). So pooling earns its keep **only where a domain has little local-specific signal beyond vibe**. (The Books verdict flipped with the data refresh: under the old Open-Library-enriched books the unified slice marginally led 0.540 vs 0.548; the clean Hardcover library record gives the local SVR real author/category features and the local model now leads.)*

#### Ablation Study (rated-only, frozen registry folds)

| Protocol | MAE | R² | Effect Size | p-value |
| :--- | :--- | :--- | :--- | :--- |
| Base XGB (Aligned) | 0.5428 | 0.4569 | ref | ref |
| Mean Ensemble (XGB+Cat) | 0.5345 | 0.4666 | d=0.056 (trivial) | 0.0000 |
| Ridge Stack (removed — legacy 5-fold protocol) | 0.5609 | 0.3837 | n/a (legacy) | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088 | 0.2688 | n/a (legacy) | n/a (legacy) |

*Effect size is paired Cohen's d on per-item absolute-error differences. The ensemble's win over Base XGB is statistically significant but trivial in magnitude (ΔMAE < 0.01) — kept because it is free, not because it matters. Rows 3–4 are removed paths kept for the record; their numbers come from the legacy 5-fold protocol, not the frozen registry, and are directionally indicative only.*

#### Distillation-Prior Ablation (Games & Books)

| Domain | MAE (no prior) | Skill (no prior) | MAE (with prior) | Skill (with prior) | p-value | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game | 0.595 | 0.246 | 0.603 | 0.237 | 0.317 | **DROP** |
| Book | 0.353 | 0.512 | 0.347 | 0.520 | 0.355 | **DROP** |

*The unified prior is **not** significantly helpful in either domain (p ≥ 0.05, effect direction negative) — it was tested and **dropped**. Note the positive skill scores without it (Games 0.246, Books 0.512): the local models beat the mean-rating baseline, the first positive evidence of learnable local signal in these N≈60 domains.*
<!-- METRICS:END -->

*(The Distillation-Prior Ablation table above is auto-generated from `latest_metrics.json`; the earlier hand-written copy here was removed to avoid drift. Raw terminal outputs of every pipeline run — master evaluation, renderer, distillation ablation, transfer grid, transfer verdict, entity links, bridge eval, active-learning queue — are captured in `reports/PIPELINE_RUN_LOG.md`.)*

---

## 9. Most Predictive Features (per model)

Auto-rendered from `reports/feature_diagnostics.json` (regenerate with `python -m src.experiments.feature_diagnostics`). Each row is the top features by mean |SHAP| of a per-model XGB surrogate fit on that model's own training matrix — a proxy for the deployed model's driver features whose *ordering* is stable across reruns. Per-model bar charts are saved to `reports/shap_<model>.png`, and the full field→feature provenance is in the README's Feature Engineering Deep-Dive.

<!-- FEATURE_IMPORTANCE:BEGIN -->
*Top features by mean |SHAP| on a per-model XGB surrogate trained on each model's own feature matrix (regenerate with `python -m src.experiments.feature_diagnostics`; per-model bar charts in `reports/shap_<model>.png`). Values are the average absolute impact on the predicted star rating.*

| Model | Most predictive features (mean \|SHAP\|, high → low) |
| :--- | :--- |
| **Movies (stacking ensemble)** | `critic_avg_5` (0.231), `imdb_rating` (0.187), `age_at_rating` (0.049), `vote_average` (0.046), `actors_encoded` (0.042) |
| **TV Shows (simplex-stack)** | `imdb_rating` (0.332), `vote_average` (0.071), `imdb_votes_log` (0.063), `pca_10` (0.050), `total_wins` (0.049) |
| **Games (local SVR)** | `ratings_count` (0.189), `rating` (0.163), `metacritic` (0.074), `pca_14` (0.063), `genres_encoded` (0.055) |
| **Books (local SVR)** | `pca_4` (0.135), `pca_1` (0.123), `ratingsCount` (0.111), `averageRating` (0.070), `authors_encoded` (0.048) |
| **Unified (Mean Ensemble)** | `critic_avg_5` (0.284), `imdb_rating` (0.155), `year` (0.086), `vote_average` (0.050), `pca_2` (0.040) |
<!-- FEATURE_IMPORTANCE:END -->

---

## 10. Classification Performance — Confusion Matrices & Scores

The regression benchmark (§Performance Summary) answers *"how many stars off?"*; this section answers *"did the model make the right taste call?"*. Both use the **identical cached registry OOF predictions** — no retraining — re-scored as a discrete decision. The 3-class scheme applies the deployed movies `target_class` bins uniformly (**Dislike** ≤2.5★ / **Neutral** 2.5–3.5★] / **Like** >3.5★); the binary **Like** view (≥3.5★) is the recommend/skip decision the Oracle actually surfaces. Music is excluded — it has no ground-truth rating (PU pseudo-labels only), so there is no "truth" to build a confusion matrix against.

<!-- CLASSIFICATION:BEGIN -->
*The same registry OOF predictions scored on the regression benchmark table, re-framed as a 3-class taste decision — **Dislike** (≤2.5★), **Neutral** (2.5–3.5★], **Like** (>3.5★) — plus a binary **Like** (≥3.5★, the recommend/skip decision). Predictions are snapped to the 0.5★ grid first. Regenerate with `python -m src.experiments.classification_diagnostics`; per-model heatmaps in `reports/confusion_<model>.png`. Music is excluded (no ground-truth rating).*

#### Headline classification scores (per deployed model)

| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,062 | 65.6% | 66.7% | 0.667 | 0.656 | 0.476 | 0.816 |
| **TV Shows** | 173 | 57.2% | 56.5% | 0.570 | 0.585 | 0.324 | 0.817 |
| **Games** | 66 | 60.6% | 54.3% | 0.538 | 0.613 | 0.345 | 0.900 |
| **Books** | 93 | 77.4% | 74.7% | 0.769 | 0.773 | 0.635 | 0.849 |
| **Unified** | 1,400 | 61.9% | 61.8% | 0.629 | 0.623 | 0.412 | 0.812 |

#### Confusion matrices & per-class scores

**Movies** · Prod Stacking (edge+Cat+SVR+Ord) · N=1062 · accuracy 65.6%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 197 | 65 | 6 |
| **Neutral** | 69 | 245 | 110 |
| **Like** | 3 | 112 | 255 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.732 | 0.735 | 0.734 | 268 |
| Neutral | 0.581 | 0.578 | 0.579 | 424 |
| Like | 0.687 | 0.689 | 0.688 | 370 |
| **Like (≥3.5★, binary)** | 0.812 | 0.821 | 0.816 | 563 |

**TV Shows** · Prod Simplex-Stack (edge+Cat+SVR) · N=173 · accuracy 57.2%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 17 | 12 | 3 |
| **Neutral** | 8 | 31 | 14 |
| **Like** | 1 | 36 | 51 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.654 | 0.531 | 0.586 | 32 |
| Neutral | 0.392 | 0.585 | 0.470 | 53 |
| Like | 0.750 | 0.580 | 0.654 | 88 |
| **Like (≥3.5★, binary)** | 0.763 | 0.879 | 0.817 | 99 |

**Games** · Local SVR · N=66 · accuracy 60.6%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 2 | 6 | 0 |
| **Neutral** | 1 | 16 | 3 |
| **Like** | 0 | 16 | 22 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.667 | 0.250 | 0.364 | 8 |
| Neutral | 0.421 | 0.800 | 0.552 | 20 |
| Like | 0.880 | 0.579 | 0.698 | 38 |
| **Like (≥3.5★, binary)** | 0.818 | 1.000 | 0.900 | 45 |

**Books** · Local SVR · N=93 · accuracy 77.4%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 19 | 7 | 0 |
| **Neutral** | 2 | 38 | 3 |
| **Like** | 0 | 9 | 15 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.905 | 0.731 | 0.808 | 26 |
| Neutral | 0.704 | 0.884 | 0.783 | 43 |
| Like | 0.833 | 0.625 | 0.714 | 24 |
| **Like (≥3.5★, binary)** | 0.840 | 0.857 | 0.849 | 49 |

**Unified** · Mean Ensemble (XGB+Cat) · N=1400 · accuracy 61.9%

| Actual ↓ / Pred → | Dislike | Neutral | Like |
| :--- | :---: | :---: | :---: |
| **Dislike** | 210 | 121 | 9 |
| **Neutral** | 70 | 364 | 106 |
| **Like** | 4 | 223 | 293 |

| Class | Precision | Recall | F1 | Support |
| :--- | :--- | :--- | :--- | :--- |
| Dislike | 0.739 | 0.618 | 0.673 | 340 |
| Neutral | 0.514 | 0.674 | 0.583 | 540 |
| Like | 0.718 | 0.564 | 0.631 | 520 |
| **Like (≥3.5★, binary)** | 0.761 | 0.870 | 0.812 | 756 |

<!-- CLASSIFICATION:END -->

*Reading these: **accuracy** is the raw fraction of correct 3-class calls; **balanced accuracy** and **macro-F1** weight the rare Dislike/Like classes equally with the common Neutral one (so they punish a model that just predicts the majority bucket); **Cohen κ** is agreement above chance; the **binary-Like F1** is the single most decision-relevant number — how well the model separates "would recommend" from "would skip". These are honest OOF numbers on the 0.5★-snapped grid, so they read a little below the continuous-MAE story, exactly as expected once predictions are bucketed.*

---

## 11. Is the Model Cheating? — Beating the Mean/Median (baseline sanity)

The most important honesty check in a clustered-rating project. Because most ratings sit in a narrow 3.0–4.5★ band, a model that simply predicts *near the mean/median* — never risking a 5★ or a 1★ — earns a respectable MAE and ±0.5★ accuracy **for free**. MAE and accuracy therefore cannot, by themselves, prove the model learned taste. This section pits every deployed model against the trivial baselines and — decisively — measures whether it **orders** items correctly. Auto-rendered from `reports/baseline_sanity.json` (`python -m src.experiments.baseline_sanity`), computed on the same cached registry OOF as the benchmark table; music is excluded (no ground-truth rating).

<!-- BASELINE_SANITY:BEGIN -->
*The honest test of "did it learn taste, or is it just predicting near the mean/median?" MAE and ±0.5★ accuracy are gameable in clustered domains — so this table pits each deployed model against the trivial baselines. Regenerate with `python -m src.experiments.baseline_sanity`.*

| Model | N | MAE | MAE (mean) | Skill vs mean | ±0.5★ acc | ±0.5★ acc (mode) | Acc lift | **Spearman ρ** | p |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,062 | 0.484 | 0.759 | +0.362 | 77% | 51% | +26% | **0.755** | 2e-196 |
| **TV Shows** | 173 | 0.656 | 0.884 | +0.258 | 64% | 46% | +17% | **0.583** | 4e-17 |
| **Games** | 66 | 0.515 | 0.826 | +0.376 | 76% | 47% | +29% | **0.735** | 2e-12 |
| **Books** | 93 | 0.328 | 0.661 | +0.504 | 88% | 55% | +33% | **0.844** | 2e-26 |
| **Unified** | 1,400 | 0.526 | 0.783 | +0.329 | 76% | 49% | +26% | **0.694** | 1e-201 |

*Reading it: **Skill vs mean** > 0 means the model beats a constant predict-the-average (all are +0.23–0.35, so yes). **Acc lift** is the ±0.5★ accuracy *above* the free mode-baseline (predicting the single most common rating) — the part that isn't given away by the clustered rating distribution (+18–27pts everywhere). And the load-bearing one: **Spearman ρ** — a model that just predicts noise around the mean has ρ≈0 and cannot order your items; every model here sits at ρ 0.58–0.75 with p ≤ 1e-9, so it genuinely ranks favourites above passes. **Honest caveat:** the small domains (TV/Games/Books) still compress prediction *magnitude* toward the center (`spread_ratio` ≈ 0.56–0.69 in `baseline_sanity.json`), which is exactly why the pipeline applies rank/quantile calibration and why the Books watchlist queue ranks by novelty, not by the (mean-hugging) predicted star.*
<!-- BASELINE_SANITY:END -->

**Why Spearman ρ is the argument.** A constant / near-constant predictor (predict-the-mean ± noise) has, by construction, **skill = 0** and **Spearman ρ ≈ 0** — it cannot rank your favourites above your passes. Every deployed model instead posts **skill 0.23–0.35 over predict-the-mean** and **Spearman ρ 0.58–0.75 at p ≤ 1e-9**, and lifts ±0.5★ accuracy **+18–27 points above the mode-baseline** (predicting the single most common rating). That trio is the evidence the models track real taste, not the histogram. **The honest half:** the small domains still *compress the magnitude* of their predictions toward the center (`spread_ratio` ≈ 0.56–0.69 — predictions vary about 40% less than the truth, and 75–83% land within ±0.5 of the mean), so they *rank* well but under-commit to extremes. That specific, measured failure mode is precisely what the **rank/quantile calibration** (Engineering Highlights) corrects, and why the Books watchlist active-learning queue ranks by **novelty**, not by the (mean-hugging) predicted star. Movies, by contrast, predicts with essentially the full spread of the truth (`spread_ratio` ≈ 1.03) — no compression, no cheating.

---

## 12. Bias / Variance — and Why the Metrics Are Where They Are

The strategic "what do I actually need — more data, more features, or a better model?" question, answered with a bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`).

<!-- BIAS_VARIANCE:BEGIN -->
*Bootstrap bias/variance decomposition on each domain's own local features (`python -m src.experiments.bias_variance`). `bias²` = systematic error of the average prediction; `variance` = how much predictions wobble as the training set is resampled. bias² here also absorbs the irreducible 0.5★-self-rating noise.*

| Model | N | feats | bias² | variance | **bias share** | test MAE | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 1,062 | 108 | 0.399 | 0.020 | 95% | 0.480 | features/noise ceiling; curve still creeping — maintain |
| **TV Shows** | 173 | 56 | 0.928 | 0.053 | 94% | 0.752 | feature/noise ceiling — richer features barely help (see note) |
| **Games** | 66 | 51 | 0.687 | 0.040 | 94% | 0.618 | features/noise ceiling; keep local features, collect at margin |
| **Books** | 93 | 29 | 0.278 | 0.029 | 90% | 0.372 | near small-N ceiling — collect via watchlist queue |
| **Unified** | 1,400 | 235 | 0.597 | 0.027 | 95% | 0.579 | pooling tool; bias-limited |

*Every model is **bias-dominated (93–96%), variance ~4–6%** — i.e. stable, not data-starved. So **more rows of the same features is a weak lever everywhere**; the levers are richer *features* (reduce bias) and the *aleatoric ceiling* you can't cross. **Proof it's a ceiling, not a fixable gap:** adding the exact creator/cast/writer target encodings that make Movies strong to the worst domain (TV) cut its bias² only 1.014 → 0.963 and test MAE 0.784 → 0.778 — because at N≈170 most creators appear once or twice, so the encoding collapses to the mean, and the rest of the error is intrinsic 0.5★-rating noise no feature can predict.*
<!-- BIAS_VARIANCE:END -->

**Why the bias is high (and R²/MAE look mediocre) — three compounding reasons.** (1) **Aleatoric ceiling:** ratings are self-reported on a coarse 0.5★ grid, so the same item wobbles ±a grid step by mood/recency/rounding — irreducible noise that lands inside `bias²`; an MAE near 0.5★ is therefore close to the floor. (2) **Metadata ≠ personal taste:** the features are crowd-quality signals (IMDb/critic/votes/genre), which capture "is this generally good," not "will *this user* like it" — the personal residual is unobservable, so bias is lowest where features are richest (Movies) and highest where they're thin (TV). (3) **Small N** (66–171) for the niche domains caps person-specific learning.

**The decisive experiment (§ this revision).** To test whether reason (2) is fixable, we added the *exact* feature that makes Movies strong — leakage-safe, Bayesian-smoothed, out-of-fold **creator/cast/writer target encodings** — to the worst domain, TV. The bias moved only **1.014 → 0.963** (test MAE 0.784 → 0.778): at N≈170 most creators appear once or twice, so the smoothed encoding reverts to the global mean and the residual is dominated by reason (1). **This is the honest conclusion: the metrics are near the achievable ceiling for predicting one person's coarse, noisy self-ratings from crowd metadata — not a modelling deficiency a bigger ensemble or more of the same rows would fix.** It is precisely why the project leads with MAE-skill and Spearman ρ (which show real, significant signal) rather than R².
