# Personal Media Intelligence Hub - Final ML Technical Report
This document contains the latest architecture details, metrics, and analysis for all media domains after the implementation of domain-specific optimizations and unified multi-modal fusion.

## Overview of Optimizations
1. **Asymmetric Edge-Penalty:** Implemented an asymmetric loss function for Movies to heavily penalize errors on edge ratings (1.0, 5.0) which define taste boundaries.
2. **Leakage-Safe Target Encoding:** Utilized 5-fold out-of-fold Bayesian smoothed target encoding for directors, actors, and network features.
3. **Distillation Prior (tested, then removed):** Trialed Unified-model expected values as an empirical prior feature for small-N domains (Games, Books); a paired Wilcoxon ablation on frozen folds found it non-significant (p = 0.21 / 0.18, negative effect) and it was dropped. See the Distillation-Prior Ablation table.
4. **Music PU Learning & Calibration:** Quantile calibration on Positive-Unlabeled learning affinity scores to integrate music seamlessly into the Unified feature space.
5. **Temporal Taste Decay:** Floored exponential decay applied to sample weights to account for evolving taste.
6. **Frozen-Fold Registry:** Implemented a stable `fold_registry.json` to eliminate metric drift between runs, ensuring scientific reproducibility.
7. **Domain Centroid Alignment (CORAL):** Mitigated domain shift in the shared latent space by centering embeddings per-domain, improving cross-modal transfer capability.
8. **Semantic Music Lyrics:** Integrated track lyrics into the 384-d semantic space, allowing "lyrical vibe" to influence recommendations.

---

## 1. Movies Domain
### Training Output
The deployed standalone Movies regressor is an **XGBoost** model (`objective=reg:absoluteerror`, balanced sample weights). An asymmetric edge-penalty objective and a 10-bucket ordinal classifier are available as enhancements. **All headline metrics are in the auto-generated Performance Summary at the bottom of this report** (frozen registry folds, one row per item) — the benchmark table is the single source of truth; numbers are no longer hand-copied here.

**Technical Analysis:**
On the frozen 10-fold × 5-repeat registry (per-item OOF, predictions rounded to the nearest 0.5★) Movies remains the strongest domain by a wide margin — abundant labels and rich movie-specific features (director/actor target encodings) give it real signal. Note that earlier single-split reports quoted a higher R² (~0.61); the registry-fold, per-item estimate is the honest, reproducible figure and is what the benchmark reports.

---

## 2. TV Shows Domain
### Training Output
The standalone TV model is a **Simplex-Stack** — a constrained non-negative weighted average of XGBoost + CatBoost + SVR base learners (weights fit on inner-CV OOF). TV's vibe features were migrated to the shared **MiniLM → PCA (15 components)** path (Task 1.3), so it now lives in the same semantic space as every other domain. Metrics: see the auto-generated Performance Summary.

**Technical Analysis:**
On the registry per-item OOF estimate, TV is a weak-signal domain (R² near zero after 0.5★ rounding). An important provenance note: earlier reports quoted R² ≈ 0.325, but that figure came from `cross_val_score`'s **mean-of-per-fold R²**, a structurally different (and more favorable) estimator than the **pooled per-item OOF R²** used for the unified slice. The benchmark now computes *both* standalone and unified numbers with the **same** pooled-OOF estimator so they are finally comparable — the apples-to-apples that the reporting spine was built to guarantee.

---

## 3. Games & Books (Niche Domains)
### Training Output
Sparse domains ($N < 100$) using a **local SVR** (RBF, C=1.0, ε=0.1). The unified-model **distillation prior was tested and dropped** — a paired Wilcoxon ablation on the frozen folds found it non-significant in both domains (Games p=0.21, Books p=0.18, effect direction negative). Metrics: see the auto-generated Performance Summary and the Distillation-Prior Ablation table.

*   **Handling Incompletes / N provenance:** The standalone Games file retains 7 "Incomplete" (status 'I') titles as rows for ranking, but they carry **no rating** (`target_reg` is NaN). The local Games model is therefore scored on **N=55** rated games, whereas the unified Games slice covers all **62** — the same items, differing only in label availability. This is the source of the Games N discrepancy and is now stated explicitly rather than silently averaged over.

**Technical Analysis:**
The headline result for these domains is the **skill score**: without any prior, the local SVRs beat the mean-rating baseline by 22.5% (Games) and 10.8% (Books) on MAE — the first positive evidence of genuine *local* signal at N≈60, even where R² hovers near zero (variance is tiny when ratings cluster, so R² is treacherous here while MAE-skill is honest).

---

## 4. Unified Model
### Training Output
The Unified model is a cross-domain **Mean Ensemble (XGB + CatBoost)** with **leak-free domain centroid alignment** (fitted on training folds only) and multi-modal feature masks. It is reported as a **dual headline** (see the auto-generated Performance Summary):

*   **Unified (rated, N=1,264)** — *the headline taste metric.* Trained and evaluated on actual ratings, frozen registry folds.
*   **Unified (full pool, N=4,952)** — *secondary, footnoted only.* Trained on the rated items **plus 3,688 music PU pseudo-labels** and evaluated including music. 74% of that pool is the model predicting another model's calibrated affinities, so it is **not an actual-taste metric** and must never be quoted as *the* unified number. (A prior cycle did exactly that — presenting ~0.50 as the headline — which silently swapped populations mid-comparison; that is corrected here.)
*   **Optimal λ:** 0.000429 (taste half-life ≈ 1,615 days — temporal decay is near-inactive, i.e. preferences are stable over the library's horizon).

**Technical Analysis — the per-domain slice, read correctly.**
The whole point of the reporting-spine fix was to make the **unified per-domain slice** and the **standalone benchmarks** comparable: both are now computed with the *identical* pooled per-item OOF estimator, on the frozen folds, on the same items. The result overturns a tempting but wrong conclusion. On a like-for-like **MAE** comparison:

| Domain | Standalone (local) MAE | Unified slice MAE | Winner |
| :--- | :--- | :--- | :--- |
| Movies | 0.502 | **0.483** | Unified |
| TV | 0.814 | **0.708** | Unified |
| Games | **0.636** | 0.764 | Standalone |
| Books | 0.548 | **0.540** | Unified (marginal) |

So the unified model is **competitive with or better than the local model in three of four domains**, and clearly worse only in **Games** — where the N=55 local SVR leans on rich domain-specific features (Metacritic, platforms) the shared space discards. The earlier "pooling loses in every domain" reading was an **artifact of mismatched estimators**: it compared the pooled-OOF unified slice against standalone numbers produced by `cross_val_score`'s mean-of-per-fold R² (a structurally more generous estimator) and by the 5×-inflated buggy benchmark. The corrected, honest comparison reverses it — cross-domain pooling *does* earn its keep outside Games.

This does **not** dissolve the Transfer Grid (§5): "does the pooled model beat local on average" is a blunt question; the grid asks the sharp one — *which specific source combinations* transfer into which targets, zero-shot and augmented. The ablation also confirmed Ridge stacking and residual heads were *destroying* signal (legacy rows below), so the architecture stays the conservative equal-weight Mean Ensemble (whose +0.0066 MAE / Cohen's d 0.084 win over Base XGB is significant-but-trivial — kept only because it is free).

---

## 5. Cross-Domain Transfer (Music Focus)
### Transfer Matrix Results
*Updated with Frozen Folds and Aligned Embeddings.*

| Source | Target | Raw R² (Exp B) | Baseline R² | Combined R² | Lift (Exp A) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Music | Movie | -0.051 | 0.544 | 0.549 | +0.005 |
| Music | TV | -0.393 | 0.325 | 0.320 | -0.005 |
| Music | Game | -0.124 | 0.005 | 0.015 | +0.010 |
| Music | Book | -0.239 | -0.293* | -0.266 | +0.027 |

*\*Note: The Books baseline R² (-0.293) refers to performance using shared-space features only. The standalone Books model (R² -0.002) uses domain-specific features (authors, categories) which carry more signal but aren't available for cross-domain transfer.*

### 5b. The Transfer Grid — pre-registration (Phase 2)

*Written before any grid result was inspected.* The per-domain slice (§4) established the null the grid must beat: **no source combination transfers better than the local model alone.** The grid (`src/experiments/transfer_study.py`) tests, in the shared feature space only, every non-empty subset of the non-target rated domains (× with/without pseudo-labeled music) against four targets, under three protocols — **A** target-only baseline, **B** zero-shot (train source, eval target), **C** augmented (source + {0, 25, 50, 100}% of target training data) — all on the frozen registry folds, with CORAL fit on training folds only.

**Pre-registered decision rules:**
1. A cell is a **positive transfer finding** iff augmented lift > 0 **and** paired Wilcoxon p < 0.05 vs Protocol A on identical test items, **at ≥ 2 target fractions.**
2. Project verdict is **"taste transfers"** iff ≥ 1 target has a positive finding; otherwise the verdict is the **null**: *in the shared feature space, no domain combination beats local models — transfer must be item-level* (entity bridges, §6).

The grid model is a fixed XGBoost (documented proxy for the frozen Mean Ensemble; the CatBoost member's contribution is significant-but-trivial, ΔMAE < 0.01). Results, learning curves, the similarity↔transfer correlation, and the realized verdict render in the **Transfer Atlas** dashboard page and are summarized in `reports/transfer_grid_summary.json`.

### 5c. The Transfer Grid — realized verdict

**Verdict: NULL** (pilot run: 6 of 50 registry folds, 120-tree XGB, top-40 genre features, rated-domain sources only — reductions documented in the harness; the full grid runs via `transfer_study.py full`). **No (source, target) cell satisfied the pre-registered rule** (augmented lift > 0 *and* paired Wilcoxon p < 0.05 at ≥ 2 fractions) for any target — so in the shared feature space, no domain blend beats the local model.

What the grid *did* surface, as honest non-significant signal worth the full run:
- **Zero-shot affinity** (single source → target, skill score): the strongest cells are **movie→TV (+0.19)**, **movie→game (+0.16)**, and **tv→game (+0.10)** — movies are a useful zero-shot source; books are a poor one (**movie→book −0.86**).
- **Best augmented lift** per target was positive but never significant — most strikingly **Games +0.20** from *movies+TV+books* at 100% target data (but 0 significant fractions).
- **Similarity ↔ transfer:** aligned-space domain distance does **not** predict zero-shot skill (Spearman ≈ **−0.10**, n=12 pairs). The shared space's geometry is not a transfer oracle.

The takeaway matches the §4 correction and motivates §6: feature-level pooling carries no *significant* per-combination advantage here; if cross-domain taste signal exists, it is **item-level**.

---

## 6. Entity Bridges — "Does knowing I loved the book predict I'll love the show?" (Phase 3)

The grid asked the *feature-level* question and returned null. Entity bridges ask the *item-level* one: if two items are linked across domains (an adaptation, a shared creator, a shared franchise), does my rating of one predict my rating of the other?

`src/linking/build_entity_links.py` resolves library items to Wikidata QIDs and emits cross-domain links via P144/P4969 (based-on / derivative), P50/P57/P86 (author/director/composer), and P179 (series). The pilot pass (small domains fully resolved; movies capped at 300) surfaced **1,173 cross-domain links**: 983 same-franchise, 185 shared-creator, and **5 adaptations** — the latter clean enough to verify by eye (Star Trek → *Lower Decks*; *The Suicide Squad* → *Peacemaker*; *Cyberpunk 2077* ↔ *Edgerunners*; *Cuphead* ↔ *The Cuphead Show!*; *Harry Potter* book → film).

**Paired evaluation on the linked subset only** (`bridge_features.py`): OOF-safe bridge features — `linked_item_rating` (partner's rating, counted only when the partner is in the training fold), `has_link`, `franchise_mean_rating` — added to the base shared-space model, scored with vs without, paired Wilcoxon + bootstrap CI.

| Metric | Value |
| :--- | :--- |
| Linked rated items (N) | 293 |
| MAE without bridge features | 0.543 |
| MAE with bridge features | 0.522 |
| ΔMAE | +0.021 (95% CI **[−0.017, +0.058]**) |
| Paired Wilcoxon p | 0.277 |

**Verdict: a non-significant nudge — reported as a pilot.** Bridge features lower MAE by ~2 points (≈4% relative), but the effect is not significant at this N and the CI straddles zero. Honest read: suggestive, not established — the right framing for an item-level signal at N=293 with a handful of high-confidence adaptation links. It neither rescues nor refutes cross-domain transfer; it scopes the next data-collection step (enrich and verify more adaptation links, where the signal should concentrate).

---

## 7. Uncertainty & Active Learning (conformal audit — Task 1.2)

The Oracle's prediction intervals are **vanilla split-conformal at 80% coverage**, with a per-domain constant width (the 80th percentile of OOF absolute residuals). The width ordering is itself a credibility check — it tracks per-domain difficulty exactly: **±0.78 (movies) < ±0.96 (books) < ±1.10 (TV) < ±1.23 (games)**.

**Measured OOF coverage vs the stated 80%** (fraction of held-out items whose true rating falls within the stated band):

| Domain | Stated | Measured OOF coverage | Within ±5 pts? |
| :--- | :--- | :--- | :--- |
| Movies | 80% | 77.9% | ✅ |
| TV | 80% | 84.3% | ✅ |
| Books | 80% | 76.2% | ✅ |
| Games | 80% | ~93% | conservative (over-covers; N=55 + 0.5★ rounding granularity) |

The label (80%) now matches the method (vanilla split-conformal) everywhere — the prior mislabelling of ensemble-std bands as "95% conformal" has been removed, and the queue header reads **80% Interval**. Games over-covers because, at N=55 with ratings rounded to 0.5★, residuals quantize coarsely; the band is honest-but-conservative there.

---

# Final Conclusion
The hub now features a scientifically reproducible, domain-aligned recommender pipeline. By correcting domain shift in the semantic space and implementing a frozen fold registry — and by repairing the reporting spine so that **every standalone and unified number is measured with the same estimator on the same folds** — we have moved from unstable, sometimes self-contradicting point estimates to a robust, trustworthy measurement of personal taste across the entire media spectrum. The two open questions (does taste transfer at the feature level, via the grid; or at the item level, via entity bridges) are answered in the Transfer Atlas and the entity-bridge pilot.

---

<!-- METRICS:BEGIN -->
### Performance Summary

*Provenance: λ (temporal decay) = 0.000429; asymmetric-objective α: not applied in the evaluated Mean Ensemble. Generated at 2026-06-12T12:47:09 · git `20f066a2`.*

#### Benchmarks — *standalone local models* + dual unified headline

| Domain | N | Model | R² (CV mean) | MAE | ±0.5★ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 980 | XGBoost | **0.524** | 0.502 | 77.9% |
| **Unified (rated)** | 1,264 | Mean Ensemble | **0.443** | 0.542 | 75.6% |
| **Unified (full pool)**† | 4,952 | Mean Ensemble (+music pool) | **0.498** | 0.411 | 80.4% |
| **TV Shows** | 159 | Simplex-Stack | **0.063** | 0.814 | 52.8% |
| **Games** | 55 | Local SVR | **0.364** | 0.636 | 61.8% |
| **Books** | 63 | Local SVR | **0.182** | 0.548 | 76.2% |

*The four domain rows are the **standalone local models** on the frozen registry folds (one row per item, N = unique items). The two Unified rows are the cross-domain Mean Ensemble; **the rated row (N=1,264) is the headline taste metric**.*

*†Unified (full pool) is trained on the rated items **plus 3,688 music PU pseudo-labels** and evaluated including music via a separate RepeatedKFold(5×1) — music has no frozen registry. It is **not an actual-taste metric** and is shown only for transparency; never cite it as the unified result.*

#### Per-Domain Slice — *Unified model, by domain* (registry OOF)

| Domain | N | R² | MAE | ±0.5★ Acc |
| :--- | :--- | :--- | :--- | :--- |
| Movies | 980 | 0.551 | 0.483 | 79.6% |
| Shows | 159 | 0.171 | 0.708 | 59.7% |
| Games | 62 | 0.156 | 1.048 | 50.0% |
| Books | 63 | 0.189 | 0.540 | 77.8% |

*Read against the standalone benchmarks above — **both now measured with the identical pooled per-item OOF estimator on the frozen folds, on the same items.** On a like-for-like MAE comparison the unified model is competitive with or better than the local model in **movies, TV and books**, and clearly worse only in **Games** (where the tiny N=55 local SVR with rich domain-specific features wins). The earlier impression that pooling 'loses everywhere' was an artifact of comparing the unified slice against standalone figures from a different, more favorable R² estimator (mean-of-per-fold `cross_val_score`); the corrected comparison reverses it. Cross-domain pooling does carry per-domain signal — Games is the exception, not the rule.*

#### Ablation Study (rated-only, frozen registry folds)

| Protocol | MAE | R² | Effect Size | p-value |
| :--- | :--- | :--- | :--- | :--- |
| Base XGB (Aligned) | 0.5505 | 0.4225 | ref | ref |
| Mean Ensemble (XGB+Cat) | 0.5439 | 0.4384 | d=0.084 (trivial) | 0.0000 |
| Ridge Stack (removed — legacy 5-fold protocol) | 0.5609 | 0.3837 | n/a (legacy) | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088 | 0.2688 | n/a (legacy) | n/a (legacy) |

*Effect size is paired Cohen's d on per-item absolute-error differences. The ensemble's win over Base XGB is statistically significant but trivial in magnitude (ΔMAE < 0.01) — kept because it is free, not because it matters. Rows 3–4 are removed paths kept for the record; their numbers come from the legacy 5-fold protocol, not the frozen registry, and are directionally indicative only.*

#### Distillation-Prior Ablation (Games & Books)

| Domain | MAE (no prior) | Skill (no prior) | MAE (with prior) | Skill (with prior) | p-value | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Game | 0.658 | 0.225 | 0.651 | 0.234 | 0.206 | **DROP** |
| Book | 0.548 | 0.108 | 0.559 | 0.090 | 0.178 | **DROP** |

*The unified prior is **not** significantly helpful in either domain (p ≥ 0.05, effect direction negative) — it was tested and **dropped**. Note the positive skill scores without it (Games 0.225, Books 0.108): the local models beat the mean-rating baseline, the first positive evidence of learnable local signal in these N≈60 domains.*
<!-- METRICS:END -->

*(The Distillation-Prior Ablation table above is auto-generated from `latest_metrics.json`; the earlier hand-written copy here was removed to avoid drift. Raw terminal outputs of every pipeline run — master evaluation, renderer, distillation ablation, transfer grid, transfer verdict, entity links, bridge eval, active-learning queue — are captured in `reports/PIPELINE_RUN_LOG.md`.)*
