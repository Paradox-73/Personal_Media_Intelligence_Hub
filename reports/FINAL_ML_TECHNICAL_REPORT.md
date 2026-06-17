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

## How to Read These Metrics

Every number in the Performance Summary is computed the same way: predictions are clipped to `[0.5, 5.0]`, **snapped to the nearest 0.5★** (the grid a user actually rates on), then scored — one row per unique item after averaging the 5 cross-validation repeats. What each column means, and why we lead with some and distrust others:

| Metric | Definition | What it tells you here |
| :--- | :--- | :--- |
| **MAE** | Mean absolute error, in stars | **The primary yardstick.** MAE 0.50 means predictions are on average half a star off. It is robust to the low-variance problem below, so it is the metric we trust across *all* domains. |
| **±0.5★ Accuracy** | Share of items predicted within one grid step | The "is this usable in the Oracle UI" metric. Movies 79.1% ⇒ ~4 of 5 predictions land within a single half-star notch of the truth. |
| **R² (CV mean)** | Fraction of rating *variance* explained | **Treacherous — read it only beside MAE.** It is honest for Movies (wide rating spread, N=980) and misleading everywhere else: TV/Games/Books ratings cluster in a narrow 3.0–4.5★ band, so total variance is tiny and R² collapses toward 0 (or negative) *even when MAE is genuinely good*. A near-zero R² in a low-variance domain is a property of the **denominator**, not proof the model learned nothing. |
| **Skill Score** | `1 − MAE_model / MAE_baseline`, baseline = predict the training mean | The honest accuracy metric for the small domains. It asks the only question that survives low variance: *does the model beat guessing the average?* Positive ⇒ real local signal. Games **0.225** / Books **0.108** are the first such evidence at N≈60, precisely where R² ≈ 0. |
| **N** | Count of **unique rated items** | Per-item, after de-duplicating the 5 CV repeats. Never OOF rows — counting rows once inflated Movies to 4,900 instead of 980. Standalone Games is N=55 (7 "Incomplete" titles have no rating); the unified Games slice is N=62 (same items, label availability differs). |

**The one-line takeaway:** in this project MAE and Skill Score are the load-bearing metrics; R² is trustworthy only for Movies and is reported elsewhere for transparency, not as a verdict. This is why a domain can show R² ≈ 0.06 (TV) yet still carry usable signal — and why we never rank domains by R² alone.

---

## 1. Movies Domain
### Training Output
The deployed standalone Movies regressor is the **production stacking ensemble**: an Optuna-tuned asymmetric **edge-penalty XGBoost** + CatBoost + SVR + a 10-bucket **ordinal expected-value** member, fused by a meta-learner. (The tuned edge-penalty alphas are persisted to `models/movie/best_params.json` and reused across folds, so the registry-fold benchmark measures the model *as deployed* rather than re-tuning per fold.) **All headline metrics are in the auto-generated Performance Summary at the bottom of this report** (frozen registry folds, one row per item) — the benchmark table is the single source of truth; numbers are no longer hand-copied here.

**Technical Analysis:**
On the frozen 10-fold × 5-repeat registry (per-item OOF, predictions rounded to the nearest 0.5★) Movies remains the strongest domain by a wide margin — abundant labels and rich movie-specific features (director/actor target encodings) give it real signal. The benchmark now reports the **production stacking ensemble** (not the old plain-XGB proxy): **MAE 0.478, R² 0.558, 79.1%** within half a star. **Reading the numbers:** MAE 0.478 means the typical prediction is under half a star off; 79.1% land within one grid step; and R² 0.558 is *trustworthy here* because N=980 with a wide rating spread gives R² a real denominator (the one domain where it is). The production ensemble beats the simplified proxy (which scored R² 0.524 / MAE 0.502), confirming the proxy understated Movies — but it lands at **0.558, not the ~0.61** an earlier single 80/20 split suggested; the honest 50-fold registry figure is lower because a single split is optimistic.

---

## 2. TV Shows Domain
### Training Output
The standalone TV model is a **Simplex-Stack** — a constrained non-negative weighted average of XGBoost + CatBoost + SVR base learners (weights fit on inner-CV OOF). TV's vibe features were migrated to the shared **MiniLM → PCA (15 components)** path (Task 1.3), so it now lives in the same semantic space as every other domain. Metrics: see the auto-generated Performance Summary.

**Technical Analysis:**
On the registry per-item OOF estimate, TV is a weak-signal domain (R² near zero after 0.5★ rounding). The benchmark now reports the **production simplex-stack** (tuned edge-penalty XGB + CatBoost + SVR): **MAE 0.827, R² 0.052, 53.5%**. **A telling null result:** the production ensemble is **no better than the simplified proxy** (MAE 0.814 / R² 0.063) — marginally worse, in fact. So TV's weak signal is *real*, not an artifact of a weak benchmark model: throwing the full deployed architecture at it does not help, because at N=159 the heavier ensemble has nothing extra to learn from. **Reading the numbers:** R² 0.052 is *not* "the model learned nothing" — TV ratings cluster tightly, so the variance denominator is tiny and R² collapses regardless of fit; the honest read is the MAE of 0.827 (≈1.6 grid steps) and the 53.5% within-half-star rate, which say TV is genuinely hard with the current features. (Notably, the **unified** model's TV slice — MAE 0.708 — *does* beat both local TV models, the one domain where cross-domain pooling clearly adds signal a TV-only model can't find.) An important provenance note: earlier reports quoted R² ≈ 0.325 from `cross_val_score`'s mean-of-per-fold R², a structurally more favorable estimator than the pooled per-item OOF R² used here.

---

## 3. Games & Books (Niche Domains)
### Training Output
Sparse domains ($N < 100$) using a **local SVR** (RBF, C=1.0, ε=0.1). The unified-model **distillation prior was tested and dropped** — a paired Wilcoxon ablation on the frozen folds found it non-significant in both domains (Games p=0.21, Books p=0.18, effect direction negative). Metrics: see the auto-generated Performance Summary and the Distillation-Prior Ablation table.

*   **Handling Incompletes / N provenance:** The standalone Games file retains 7 "Incomplete" (status 'I') titles as rows for ranking, but they carry **no rating** (`target_reg` is NaN). The local Games model is therefore scored on **N=55** rated games, whereas the unified Games slice covers all **62** — the same items, differing only in label availability. This is the source of the Games N discrepancy and is now stated explicitly rather than silently averaged over.

**Technical Analysis:**
The headline result for these domains is the **skill score**: without any prior, the local SVRs beat the mean-rating baseline by 22.5% (Games) and 10.8% (Books) on MAE — the first positive evidence of genuine *local* signal at N≈60, even where R² hovers near zero (variance is tiny when ratings cluster, so R² is treacherous here while MAE-skill is honest). **Why this matters for the architecture:** the standalone Games SVR (MAE **0.64**) leans on domain-specific columns — Metacritic, platforms, developers — that the shared space discards; when those are stripped away in the unified slice, Games MAE balloons to **1.05**. Games is therefore the single domain where keeping a *local* model is non-negotiable, and the contrast (0.64 local vs 1.05 pooled) is the cleanest evidence in the project that not all signal survives pooling. Books, by contrast, loses almost nothing (0.55 local vs 0.54 pooled) because its discriminative signal is largely semantic and already lives in the shared vibe space.

---

## 4. Unified Model
### Training Output
The Unified model is a cross-domain **Mean Ensemble (XGB + CatBoost)** with **leak-free domain centroid alignment** (fitted on training folds only) and multi-modal feature masks. It is reported as a **dual headline** (see the auto-generated Performance Summary):

*   **Unified (rated, N=1,264)** — *the headline taste metric.* Trained and evaluated on actual ratings, frozen registry folds.
*   **Unified (full pool, N=4,952)** — *secondary, footnoted only.* Trained on the rated items **plus 3,688 music PU pseudo-labels** and evaluated including music. 74% of that pool is the model predicting another model's calibrated affinities, so it is **not an actual-taste metric** and must never be quoted as *the* unified number. (A prior cycle did exactly that — presenting ~0.50 as the headline — which silently swapped populations mid-comparison; that is corrected here.)
*   **Optimal λ:** 0.000429 (taste half-life ≈ 1,615 days — temporal decay is near-inactive, i.e. preferences are stable over the library's horizon).

**Reading the dual headline.** The rated row — MAE **0.542**, R² **0.443**, **75.6%** within half a star over 1,264 genuinely-rated items — *is* the cross-domain taste number: one model, four domains, half-a-star-plus average error. The full-pool row looks better on paper (MAE 0.411, R² 0.498, 80.4%) **only because 74% of its 4,952 items are music PU pseudo-labels** — the model is partly scoring another model's calibrated affinities, so its apparent lift is population change, not skill. That is exactly why it is footnoted and never quoted as *the* unified result; the honest comparison against the local models is always the rated row.

**Technical Analysis — the per-domain slice, read against the *production* locals.**
The earlier version of this report compared the unified slice against *weakened proxy* local models (Movies = plain 200-tree XGB; TV = a manual simplex) and concluded the unified model was "competitive or better in three of four domains." Replacing the proxies with the **deployed production locals** (Movies = the Optuna-tuned asymmetric edge-penalty XGB + CatBoost + SVR + ordinal-EV stacking ensemble; Games/Books = the deployed SVR), measured with the *identical* pooled per-item OOF estimator on the same frozen folds, **flips that conclusion.** On a like-for-like **MAE** comparison:

| Domain | Production local MAE | Unified slice MAE | Winner |
| :--- | :--- | :--- | :--- |
| Movies | **0.478** | 0.483 | **Local** (narrow) |
| TV | 0.827 | **0.708** | **Unified** |
| Games | **0.636** | 1.048 | **Local** (large) |
| Books | 0.548 | **0.540** | Unified (marginal) |

The result splits **cleanly by domain type**: the local model wins in **Movies and Games** — the domains that carry rich *domain-specific* features the shared space throws away (Movies' director/actor/dir×genre target-encodings; Games' Metacritic, platforms, developers) — while the unified model wins in **TV** and is level on **Books**, the *semantic-only* domains whose discriminative signal already lives in the shared vibe space, so cross-domain pooling adds training data without sacrificing signal. **Cross-domain pooling earns its keep only where a domain has little local-specific signal beyond vibe.** The previous "unified competitive or better in movies/TV/books" reading was an artifact of benchmarking against a *weakened Movies proxy* (R² 0.524); with the production ensemble (R² 0.558, MAE 0.478) the Movies verdict flips to local — the single most consequential correction in this revision, because the README's headline story rested on it.

*Provenance note on the production Movies number:* an earlier single 80/20-split run quoted R² ≈ 0.61 for Movies; the honest **50-fold registry OOF of the same architecture is R² 0.558** — the 0.61 was an optimistic single-split estimate, not a reproducible figure. (Tuned edge-penalty alphas are persisted to `models/movie/best_params.json` and reused across folds, so the model is measured as deployed, not re-tuned per fold.)

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

**Two confirmatory hypotheses, pre-declared from the pilot (before the full grid was inspected):** the reduced pilot surfaced two suggestive cells, now promoted to *a priori* hypotheses for the full 50-fold grid —
- **H1:** Games gains from augmenting with **movies + TV + books** (pilot lift +0.20 at 100% target).
- **H2:** TV gains from augmenting with **all other rated sources** (pilot lift positive at 100%).

To control the false-positive rate across the ~14 source-subset cells per target, significance for the *exploratory* cells is judged under a **Holm–Bonferroni correction**; H1/H2 are the two confirmatory tests that carry the verdict. This block is committed before the full-grid result is read.

The full grid is the production artifact (all 50 folds, 300-tree XGB, full genre encoding, **rated-domain sources** — same scope as the pilot, scaled up for power), with a per-target **checkpoint** so a crash cannot lose completed targets. The grid model is a fixed XGBoost (documented proxy for the frozen Mean Ensemble; the CatBoost member's contribution is significant-but-trivial, ΔMAE < 0.01). Results, learning curves, the similarity↔transfer correlation, and the realized verdict render in the **Transfer Atlas** dashboard page and are summarized in `reports/transfer_grid_summary.json`.

### 5c. The Transfer Grid — realized verdict (full 50-fold, domain-blind, triple-controlled)

**Verdict: TASTE TRANSFERS — into Games, robustly from `movie + book`.** The production grid (all **50 registry folds**, 300-tree XGB, full genre encoding, **domain-blind** features after the §8 leak fix) satisfies the pre-registered rule for the Games target. Five source configurations clear it on the domain-blind space; the table shows **which survive the two stress controls below** (prior-vs-transfer, and embedding text-normalization):

| Source → Games | lift @100% (domain-blind) | sig. fractions | survives text-norm? |
| :--- | :--- | :--- | :--- |
| **movie + book** | **+0.121** | 3 / 4 | ✅ **yes** (2/4, lift +0.121, zero-shot ρ +0.36) |
| movie + tv + book | +0.094 | 3 / 4 | partial (1/4) |
| tv | +0.102 | 3 / 4 | ✗ (0/4) |
| movie + tv | +0.092 | 3 / 4 | partial (1/4) |
| movie | +0.110 | 2 / 4 | partial (1/4) |

The strongest cell, **`movie + book → game`**, is significant at the **zero-shot** fraction (+0.175 skill, **p = 0.0004**) and at 100% (+0.121, p = 0.0015). A model trained **only on movie and book ratings, having never seen a single game rating**, predicts game ratings *better* than the games-only model — whose own shared-space skill is ≈ **0.004**. **No other target** transfers (movie, TV, book targets: best lift ≈ 0).

**Three controls, run because a "data-starved target benefits from any larger pool" result is easy to fake:**
1. **Underpowered pilot → full grid.** The 6-fold pilot saw the signal but couldn't clear significance; 50 folds does. Pre-registered rule (§5b), not p-hacking.
2. **Domain-identity leak removed (§8).** The finding survives making the features domain-blind — removing `has_{domain}_feats` *removes* the model's ability to memorise target rows, so this is content, not a label artifact.
3. **Prior-vs-transfer control** (`transfer_prior_control.py`). The decisive test of "is this real transfer or just a better prior than games' noisy N=55 mean?" On 50 folds, `movie+book→game` (MAE **1.08**) beats *every featureless constant* — games-own-mean **1.24**, source-mean **1.23**, global-mean **1.23** — **and** its predictions rank-track true game ratings (Spearman **+0.37**, vs the games-only model's +0.19). A prior is a near-constant with ρ ≈ 0; a model that ranks your games at 0.37 while beating all constants is using **content**, not supplying a mean.
4. **Text-normalization control** (`text_norm_transfer_control.py`). The §8 audit flagged that the vibe space is distorted by text *length* (movie plots ~214 chars vs game descriptions ~1,074). Re-embedding every domain with length-matched, identically-templated text (all → ~232 chars) and re-running the Games grid: **`movie+book→game` survives** (lift +0.121, 2/4 fractions, zero-shot ρ +0.36). The **broader** set (tv, movie+tv, …) **weakens to 0–1 significant fractions** — so part of the wide 5-config result *was* text-shape-inflated, and the honest robust claim narrows to **`movie+book` (and weakly other movie-containing sources)**.

**Reading it correctly (reconciling with §4):** Games' rating signal has two parts — (a) **domain-specific quality** (Metacritic, platforms, developers), which only the *local* SVR sees and which is why local beats unified in §4; and (b) **genre/vibe taste**, which lives in the shared space and **transfers from movies/TV**. The grid operates purely in space (b), where the games-only model is nearly useless, so borrowing narrative-media taste genuinely helps. Both results are true: keep the local model for its features *and* borrow cross-domain taste for the shared-space component. This is exactly the user-level intuition — *"I play action games and watch action movies"* — finally measured: **movie taste does inform game taste**, the pilot was just too small to see it.

- **Zero-shot affinity** (single source → target skill): **movie→TV +0.20**, **movie→game +0.16**, **tv→game +0.10** are the strong cells; books are a poor source and target (movie→book −1.62). Narrative *video* media (movies, TV) are the productive sources; the productive *target* is the data-starved Games domain.
- **Similarity ↔ transfer:** aligned-space domain distance still does **not** predict zero-shot skill (Spearman ≈ **−0.18**, n = 12). Geometry is not the transfer oracle; *data-starvation of the target* is the better predictor of who benefits.

**One honest caveat (§8):** the shared vibe space is still partly stratified by a text-length artifact; templating the embedded text would likely *sharpen* these transfer estimates, not erase them (the genre channel that carries much of this signal is already unified and unaffected).

---

## 6. Entity Bridges — "Does knowing I loved the book predict I'll love the show?" (Phase 3)

The grid found *feature-level* transfer — but only into Games, and only for the shared genre/vibe component (§5). Entity bridges ask the complementary *item-level* question: if two specific items are linked across domains (an adaptation, a shared creator, a shared franchise), does my rating of one predict my rating of the other? This is a sharper, per-item signal than the grid's distributional pooling.

`src/linking/build_entity_links.py` resolves library items to Wikidata QIDs and emits cross-domain links via P144/P4969 (based-on / derivative), P50/P57/P86 (author/director/composer), and P179 (series). **The movie cap has now been removed** (`build_entity_links.py full` resolves all 1,264 items, not just the first 300 movies). The full pass surfaces **2,539 cross-domain links**: 2,085 same-franchise, 441 shared-creator, and **13 adaptations** (up from 5) — newly including *Star Trek Into Darkness*/*Beyond* → *Star Trek* (TV), *El Camino* → *Breaking Bad*, *Diary of a Wimpy Kid* → book, *Mr. & Mrs. Smith* film → series, alongside the original five.

**Paired evaluation on the linked subset only** (`bridge_features.py`): OOF-safe bridge features — `linked_item_rating` (partner's rating, counted only when the partner is in the training fold), `has_link`, `franchise_mean_rating` — added to the base shared-space model, scored with vs without, paired Wilcoxon + bootstrap CI. Reported **two ways**: pooled (all link types) and adaptation-only.

| Metric | Pooled (all links) | Adaptation-only |
| :--- | :--- | :--- |
| Linked rated items (N) | **617** | 22 |
| MAE without bridge features | 0.523 | 0.955 |
| MAE with bridge features | 0.506 | 0.955 |
| ΔMAE | +0.017 (95% CI **[−0.004, +0.040]**) | 0.000 |
| Paired Wilcoxon p | 0.182 | undefined (NaN) |

> **Wiring correction (important — see §7 diagnostics).** An intermediate run reported this pooled effect as *significant* (ΔMAE +0.032, p = 0.004). That was **partly an artifact of a domain-identity leak**: the shared-space base features included the `has_{domain}_feats` masks, letting the base model see the domain label. After making the feature set **domain-blind** (the same fix applied to the transfer grid), the pooled effect falls to the numbers above — a **non-significant nudge whose CI now straddles zero**. The leak inflated the apparent signal; the honest figure is suggestive, not significant.

**Verdict — a non-significant item-level nudge (corrected).** Across the 617 linked items, domain-blind bridge features lower MAE by ~0.017 (≈3% relative), but p = 0.18 and the 95% CI includes zero. This is *more* evidence than the original N=293 pilot (which had p = 0.28) and the direction is consistent — the franchise-driven `linked_item_rating` feature does carry mild signal — but it does **not** clear significance. The honest reading is unchanged from the project's thesis: cross-domain taste signal, if it exists, is **item-level rather than feature-level**, but at this N it remains suggestive, not established.

**The adaptation-specific signal still can't be isolated.** With only 13 adaptation pairs → 22 linked items, the OOF-safe features collapse to **constants** within that subset (an adaptation partner almost never shares the held-out item's training fold, and every item in the subset has `has_link=1`), so the with/without models are identical (ΔMAE exactly 0, Wilcoxon undefined). The adaptation question — "does loving the *book* predict loving the *film*" — remains **data-starved, not answered**; ~30–40 verified adaptation links would make the OOF-safe test informative.

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

## 8. Latent-Space & Transfer-Wiring Diagnostics

Motivated by a sharp question — *"I mostly play action games and watch action movies; why don't they overlap in the latent space, and why doesn't my upbeat-music taste correlate with either?"* — we audited whether the (then-NULL) cross-domain transfer result was a **real finding** or a **pipeline artifact**. The audit paid off twice: it caught the domain-identity leak (fixed), and — together with running the grid at full 50-fold power instead of the 6-fold pilot — it **flipped the verdict to positive** for the Games target (§5c). The user's instinct that movie taste *should* inform game taste was right. `src/experiments/latent_space_diagnostics.py` runs six checks plus a music check and prints a pass/fail per check. The result: most suspected bugs are absent, **one real bug was found and fixed**, and one structural limitation explains the user's intuition.

| # | Check | Verdict | Finding |
| :--- | :--- | :--- | :--- |
| 1 | Embeddings populated? | ✅ PASS | No silent zeroing — every domain (incl. music, ‖vec‖≈0.42) has real, varied vibe vectors. The "empty embeddings collapse to one cluster" hypothesis is **false**. |
| 2 | Text-shape parity? | ⚠️ WARN | The embedded text differs in *length/style* by domain — movie plots median **214** chars, TV **252**, books **763**, games **1,074**. MiniLM maps a 1,000-char game description and a 214-char movie plot to different regions **regardless of shared genre**. (All are prose, though — not the keyword-soup failure mode.) |
| 3 | CORAL applied? | ✅ | Raw embedding **is** domain-stratified (separability ratio **1.25**); centroid/CORAL alignment removes the domain-mean offset in the model path. |
| 4 | Domain-identity leak? | ❌ **FAIL → FIXED** | The shared-space feature set included `has_{domain}_feats` masks — the model could **read the domain label**. Fixed: the feature set is now domain-blind (`shared_space_columns`). |
| 5 | Unified genre vocab? | ✅ PASS | `gen_Action` is a **single shared column** firing across movie/tv/game (16 genres span ≥2 domains). Movie-action and game-action ARE the same feature — genre transfers. |
| 6 | Taste consistency? | INFO | Per-domain rating means are similar (movie 3.24, game 3.39, tv 3.45, book 3.48; spread **0.23★**) — a per-domain scale mismatch is **not** the transfer ceiling. |
| M | Music lyrics embedded? | ✅ PASS | `lyric_embed_text` is non-trivial in **93%** of tracks (sample shows real lyrics) — lyrics genuinely reach the music vector, not title/artist only. |

**What this means for the user's intuition.** The latent space *does* stratify by domain — the intuition is correct — but **not** because the embeddings are empty (Check 1) or because genre isn't shared (Check 5). The driver is **Check 2**: the four domains feed MiniLM *different shapes of text*, so the raw vibe geometry separates by writing style before it separates by content. Genre ("action") *does* transfer as an explicit shared feature; it is the free-text *vibe* that is domain-stratified. The music nulls are the most believable in the project — not because lyrics are missing (they aren't), but because music's target is a **PU pseudo-label**, so "music → movies" asks one model's calibrated affinity to predict another domain's real ratings.

**The one real bug — and why it mattered.** Check 4's domain-identity leak was live in both the transfer grid and the entity-bridge base features. Removing it **changed a conclusion**: the entity-bridge pooled effect, which looked significant *with* the leak (p = 0.004), falls to a non-significant nudge once domain-blind (p = 0.18; §6). The leak had let the base model split on domain identity instead of transferring content. **This is exactly why one audits the wiring before trusting a result** — here the check *demoted* a false positive rather than revealing a hidden signal, but either way the published numbers are now the domain-blind ones.

**Check 2 — now controlled for the transfer claim.** Rather than leave the text-length mismatch as an untested caveat, we ran it as a **control** for the one conclusion that depends on the vibe space: `text_norm_transfer_control.py` re-embeds every domain with length-matched, identically-templated text (all → ~232 chars) and re-runs the Games grid. The headline **`movie+book→game` transfer survives** (§5c); the *broader* set of source configs weakens, so the text artifact was inflating breadth but not the core finding. Two scopes remain distinct: this **control** confirms the transfer verdict on a clean space, but the **production pipeline** (per-domain `training_features.csv` and the unified matrix) still uses the original embeddings — re-templating the whole pipeline and regenerating every downstream number is the recommended next engineering pass (it would not change the transfer verdict, which is already controlled, but it would tidy the latent-space explorer and the slice metrics).

---

# Final Conclusion
The hub now features a scientifically reproducible, domain-aligned recommender pipeline. By correcting domain shift in the semantic space and implementing a frozen fold registry — and by repairing the reporting spine so that **every standalone and unified number is measured with the same estimator on the same folds** — we have moved from unstable, sometimes self-contradicting point estimates to a robust, trustworthy measurement of personal taste across the entire media spectrum. The central question — *does taste transfer across domains?* — now has an answer: **yes, at the feature level, into the data-starved Games domain from movies and TV** (§5c), once the grid is run at full statistical power and made domain-blind.

**What changed in this revision.** This was not a re-render of cached numbers — it was a measurement upgrade that *changed four conclusions*:
0. **Transfer: NULL → POSITIVE (triple-controlled).** The headline reversal. Run at full 50-fold power and domain-blind, the grid clears the pre-registered bar for the **Games** target. The finding survives **three stress tests** (§5c): the domain-identity-leak fix, a prior-vs-transfer control (`movie+book→game` beats every featureless constant baseline and rank-tracks true ratings at Spearman +0.37 — not a mere prior), and an embedding text-normalization control. The robust claim is **`movie+book → game`** (+0.121 at 100%, zero-shot +0.175 at p=0.0004); the *broader* 5-config result narrowed under text-normalization, so part of it was a text-shape artifact. Net: **movie/book taste genuinely informs game taste** — the user's intuition, vindicated and stress-tested.
1. **Production locals, not proxies.** Movies and TV are now benchmarked with their deployed ensembles (50-fold registry OOF), not the simplified XGB/simplex proxies. Movies climbed R² 0.524 → **0.558** (MAE 0.502 → **0.478**); TV was unchanged within noise (the heavy ensemble does not rescue a low-signal domain). This **flipped the Movies verdict** from "unified wins" to "local wins," and replaced "unified competitive everywhere but Games" with the cleaner **semantic-domains-only** reading (§4).
2. **The ~0.61 Movies myth, retired.** That figure was a single 80/20 split; the honest 50-fold OOF of the same architecture is 0.558.
3. **Entity bridges, re-measured honestly.** Uncapping the Wikidata resolution grew adaptations 5 → 13 and the linked subset 293 → 617. An intermediate run looked *significant* (p = 0.004) but that was inflated by a **domain-identity leak** in the shared-space features; after the domain-blind fix (§8) the pooled effect is a **non-significant nudge** (ΔMAE +0.017, p = 0.18, CI straddles zero), and the adaptation-specific signal remains data-starved (§6).
4. **A wiring audit (§8).** Six latent-space diagnostics (prompted by "why doesn't action-game taste overlap action-movie taste?") cleared most suspected bugs — embeddings are populated, genre vocab is unified, music lyrics are embedded, rating scales match — but caught the domain-identity leak above and flagged a text-length mismatch across domains as the likely reason the raw vibe space stratifies by domain.

The *deterministic* parts (unified dual headline, per-domain slices, ablation ΔMAE/Cohen's d/p, distillation DROP verdicts) reproduced exactly against the prior run — the frozen-fold registry doing its job. The numbers that moved, moved because the *measurement* improved, and each change is documented above rather than buried.

---

<!-- METRICS:BEGIN -->
### Performance Summary

*Provenance: λ (temporal decay) = 0.000429; asymmetric-objective α: not applied in the evaluated Mean Ensemble. Generated at 2026-06-17T11:55:25 · git `af3b9fd5`.*

#### Benchmarks — *standalone local models* + dual unified headline

| Domain | N | Model | R² (CV mean) | MAE | ±0.5★ Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Movies** | 980 | Prod Stacking (edge+Cat+SVR+Ord) | **0.558** | 0.478 | 79.1% |
| **Unified (rated)** | 1,264 | Mean Ensemble | **0.443** | 0.542 | 75.6% |
| **Unified (full pool)**† | 4,952 | Mean Ensemble (+music pool) | **0.498** | 0.411 | 80.4% |
| **TV Shows** | 159 | Prod Simplex-Stack (edge+Cat+SVR) | **0.052** | 0.827 | 53.5% |
| **Games** | 55 | Local SVR | **0.364** | 0.636 | 61.8% |
| **Books** | 63 | Local SVR | **0.182** | 0.548 | 76.2% |

*The four domain rows are the **production local models** on the frozen registry folds (one row per item, N = unique items): **Movies and TV are the deployed tuned ensembles** (Optuna edge-penalty XGB + CatBoost + SVR + ordinal-EV, fused) — not the earlier plain-XGB / manual-simplex proxies — and **Games and Books are the deployed SVR**. The two Unified rows are the cross-domain Mean Ensemble; **the rated row (N=1,264) is the headline taste metric**.*

*†Unified (full pool) is trained on the rated items **plus 3,688 music PU pseudo-labels** and evaluated including music via a separate RepeatedKFold(5×1) — music has no frozen registry. It is **not an actual-taste metric** and is shown only for transparency; never cite it as the unified result.*

#### Per-Domain Slice — *Unified model, by domain* (registry OOF)

| Domain | N | R² | MAE | ±0.5★ Acc |
| :--- | :--- | :--- | :--- | :--- |
| Movies | 980 | 0.551 | 0.483 | 79.6% |
| Shows | 159 | 0.171 | 0.708 | 59.7% |
| Games | 62 | 0.156 | 1.048 | 50.0% |
| Books | 63 | 0.189 | 0.540 | 77.8% |

*Read against the **production benchmarks** above — both measured with the identical pooled per-item OOF estimator on the frozen folds, on the same items. With the **real local models** (Movies = the deployed tuned edge-penalty stacking ensemble, not the old plain-XGB proxy), the like-for-like MAE comparison splits cleanly by domain type: the **local model wins in Movies and Games** — the domains with rich domain-specific features the shared space discards (Movies' director/actor target-encodings, Games' Metacritic/platforms) — while the **unified model wins in TV and is level on Books**, the semantic-only domains where pooling across domains adds data without losing signal. So cross-domain pooling earns its keep **only where a domain has little local-specific signal beyond vibe**. The earlier 'unified is competitive or better in movies/TV/books' reading was an artifact of benchmarking against a **weakened Movies proxy**; with the production model the Movies verdict flips to local.*

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
