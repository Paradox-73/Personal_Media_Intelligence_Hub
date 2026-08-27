# Personal Media Intelligence Hub — ML System Overview

Audience: ML scientists. Scope: the full modelling chain, from raw data to served predictions.
Written in ASD-STE100 Simplified Technical English. All numbers come from the live repository
(`reports/latest_metrics.json`, generated 2026-07-25, git `4972d15`) and from the on-disk feature matrices.

---

## 1. Problem definition

### 1.1 The task
- The system learns **one person's taste** across five media domains: movies, TV shows, games, books and music.
- The primary task is **regression**. The model predicts the user's star rating on a 0.5★ grid (0.5 to 5.0).
- The secondary task is **classification**. The system re-scores the same predictions as a taste decision.
- The system is **single-user**. It does not generalize to other users. This is a design limit, not a defect.

### 1.2 Targets
| Column | Type | Definition |
| :--- | :--- | :--- |
| `target_reg` | float | The user rating on the 0.5★ grid. This is the regression label. |
| `target_ordinal` | int 0–9 | The same rating as 10 ordinal buckets (0.5★→0 … 5.0★→9). |
| `target_class` | int | A coarse low/medium/high bin. It is kept for legacy dashboards. |

### 1.3 Classification frames
- **3-class taste decision:** Dislike (≤2.5★), Neutral (2.5–3.5★], Like (>3.5★).
- **Binary recommend decision:** Like (≥3.5★) against Skip.
- The system does **not** train separate classifiers for the benchmark. It snaps the regression output to the
  0.5★ grid, then bins it. This keeps one model and one set of predictions.

### 1.4 Secondary research questions
- **Cross-domain transfer:** Does taste learned in one domain predict ratings in another domain?
- **Pooling:** Does a single cross-domain model beat four local models?
- **Acquisition:** Which unrated item gives the most information if the user rates it next?

### 1.5 Explicit non-targets
- **Music has no rating label.** The project removed the earlier synthetic "affinity rating" because it was
  misleading. Music uses unsupervised clustering, cosine similarity and a positive-unlabeled (PU) classifier.
  Music is excluded from every rating metric and confusion matrix.
- **YouTube is ingestion only.** There is no YouTube rating model.

---

## 2. Data collection

### 2.1 Rating sources (the labels)
| Domain | Label source |
| :--- | :--- |
| Movies | Letterboxd export, plus a Letterboxd RSS diary sync for new entries. |
| TV shows | Serializd watched list, with user-supplied ratings. |
| Games | A manual rating list. The sentinel `'I'` marks incomplete games. It maps to `NaN`. |
| Books | The authenticated user's Hardcover library (`user_books` with a rating). |
| Music | None. The library membership is the only positive signal. |

### 2.2 Enrichment APIs (verified against `src/<domain>/ingestion.py`)
| Domain | APIs | Key fields fetched |
| :--- | :--- | :--- |
| Movies | TMDB, OMDb | Year, runtime, genres, director, writer, actors, plot, overview, tagline, language, country, awards text, IMDb rating and votes, Metascore, Rotten Tomatoes, box office, popularity, TMDB vote average and count, keywords, franchise, budget, revenue. |
| TV | TMDB, TVMaze, OMDb, TVDB (fill only) | Name, overview, tagline, creators, genres, network, language, runtime, first and last air date, season and episode counts, status, age rating, awards, actors, writer, IMDb rating and votes, TMDB vote average and count, per-season episode ratings and guest stars. |
| Games | RAWG | Name, released date, age rating, genres, developers, publishers, Metacritic, RAWG global rating, ratings count, reviews count, tags, description, playtime, platforms. |
| Books | Hardcover GraphQL (library); Open Library (Oracle lookups only) | Title, ISBN, authors, publisher, published date, page count, categories, average rating, ratings count, description, cover. |
| Music | Spotify, ReccoBeats, MusicBrainz, Genius, Deezer, Last.fm, TheAudioDB, Wikidata | Track and artist metadata, popularity, followers, album position, explicit flag; 9 audio metrics; MusicBrainz genres, tags, length, country and language; lyrics; tempo; moods. |
| YouTube | YouTube Data API v3 | Video ID, duration, tags, description, views, likes, comments; channel title, subscriber and video counts. |
| Linking | Wikidata | Cross-domain entity links (people and franchises) for the entity-bridge study. |

**Notes on the API layer.**
- Spotify removed `/v1/audio-features` on 2024-11-27. ReccoBeats supplies the audio metrics, and Deezer supplies
  tempo. Treat these values as approximations of the old Spotify features.
- TVMaze and Open Library need no key. Books needs `HARDCOVER_API_KEY` as a full `Bearer <jwt>` string.
- All responses are cached under `data/cache/`. Re-runs only call the API for missing cells.
- Ingestion is **merge-only**. It keeps every existing hand-cleaned row and appends new items only.

### 2.3 Data volumes (live on disk)
| Domain | Enriched rows | Rated rows (training) |
| :--- | ---: | ---: |
| Movies | 1,039 | 1,039 |
| TV shows | 171 | 171 |
| Games | 650 | 72 |
| Books | 329 | 89 |
| Music | 3,837 | not applicable |
| Unified (rated) | — | 1,371 |

The enriched row count is larger than the rated count for games and books. The extra rows are the backlog and
the watchlist. The pipeline scores them, but it excludes them from supervised metrics.

### 2.4 Schema layers
Each domain writes two files:
1. `data/processed/<domain>/enriched_data.csv` — the ingested schema. This is what the APIs returned.
2. `data/processed/<domain>/training_features.csv` — the training schema. This is the numeric matrix.

Pydantic models enforce the music record contract. The other domains rely on explicit column checks.

---

## 3. Feature engineering

### 3.1 The universal recipe
Every domain applies the same five steps:
1. **Clean the numerics.** Parse money and count strings. Apply `log1p` to skewed money and vote counts.
2. **Encode the categoricals.** Apply one-hot or multi-hot encoding. Gate rare levels by frequency to control width.
3. **Build the vibe text.** Concatenate title, creator and description. Embed the text with
   `all-MiniLM-L6-v2` (384-d). Reduce the embedding with PCA.
4. **Add relational encodings** where the domain has enough repeated people (movies and TV only).
5. **Save the state.** Write medians, the fitted PCA, the fitted encoders and the column list to a `joblib`
   state file. Inference reloads this state and rebuilds the identical matrix. This gives train-serve parity.

### 3.2 Missing values, normalization and skew
| Problem | Treatment |
| :--- | :--- |
| Missing numeric field | Fill with the **training median**. Store the median in the preprocessor state. |
| Missing text field | Fill with an empty string before concatenation. |
| Missing award string | Regex-extract wins and nominations. Fill a failed match with 0. |
| Missing rating (the label) | Drop the row from supervised training. Keep the row for batch scoring. |
| Missing audio metric (music) | Fill with the median **and** add a `<feature>_missing` flag column. |
| Missing domain channel (unified) | Add `has_<domain>_feats` masks. The model can then separate "absent" from "real zero". |
| Skewed money and counts | Apply `log1p` (`box_office_log`, `vote_count_log`, `artist_followers_log`). |
| Scale differences | Apply `StandardScaler` inside the SVR and kNN pipelines. Tree models need no scaling. |
| Different critic scales | Blend IMDb, Metascore, Rotten Tomatoes and TMDB or RAWG into `critic_avg_5` on one 5★ scale. |
| Rare categorical levels | Gate by frequency: movies language top-3, TV network top-10, TV genres ≥5% of N, games developers ≥2, books authors ≥2, music genres top-60 and artists top-40. |

### 3.3 Per-domain matrices (live column counts)
| Domain | Features | PCA vibe components | Relational encodings |
| :--- | ---: | ---: | :--- |
| Movies | 85 | 40 | `director_encoded`, `actors_encoded`, `dir_genre_encoded` |
| TV shows | 57 | 15 | `creator_encoded`, `actors_encoded`, `writer_encoded` |
| Games | 59 | 15 | none |
| Books | 180 | 15 | none |
| Unified | 231 | 10 | none |
| Music | 173 | 32 | none (unsupervised) |

> These are the live on-disk widths, verified 2026-07-26. Earlier drafts of the README and the technical report
> quoted stale design-time widths (66 movie features, 397 unified features, 25 movie PCA components). Those
> documents have since been corrected to match this table.

### 3.4 Field-to-feature mapping by domain

**Movies (85 features)**
- Numeric: `year`, `runtime`, `imdb_rating`, `imdb_votes`, `metascore`, `rotten_tomatoes_rating`,
  `vote_average`, `popularity`, `box_office_log`, `total_wins`, `total_nominations`, `critic_avg_5`.
- Categorical: `lang_*` (top-3 + Other), `gen_*` genre multi-hot, `rated_Adult/Teen/General`.
- Relational: Director, Actors and Director×Genre **target encodings**. They use 5-fold out-of-fold splits and
  Bayesian smoothing (m=10). Unseen levels fall back to the global mean. This design blocks label leakage.
- Vibe: `pca_0…39` from the MiniLM embedding of Title + Director + Cast + Genre + Tagline + Plot + Overview.

**TV shows (57 features)**
- Numeric: `year`, `vote_average`, `vote_count_log`, `imdb_rating`, `imdb_votes_log`, `season_count`, `is_adult`.
- Categorical: `net_*` network one-hot (top-10 + Other), `g_*` genre multi-hot (≥5% of N).
- Relational: creator, actors and writer target encodings, built the same way as movies.
- Vibe: `pca_0…14` from Title + Network + Overview.

**Games (59 features)**
- Numeric: `year`, `metacritic`, `rating` (RAWG global), `ratings_count`, `reviews_count`.
- Categorical: `plat_*` platform one-hot, `gen_*` genre multi-hot, `dev_*` developer multi-hot (≥2).
- Vibe: `pca_0…14` from Name + Genres + Tags + Description.

**Books (180 features)**
- Numeric: `year`, `pageCount`, `averageRating`, `ratingsCount`.
- Categorical: `aut_*` author multi-hot (≥2), `cat_*` category multi-hot (Hardcover standardized genres).
- Vibe: `pca_0…14` from Title + Authors + Categories + Description.

**Music (173 features, unsupervised)**
- Numeric: `duration_min`, `release_year`, `track_age`, `popularity`, `artist_popularity`,
  `artist_followers_log`, `album_total_tracks`, `track_number`, `explicit`.
- Audio: the 9 ReccoBeats metrics, each median-imputed with a `_missing` flag.
- Lyrical: VADER sentiment (`pos/neu/neg/compound`), word count, line count, unique-word ratio, `lyrics_found`.
- Categorical: `g_*` genres and tags (top-60), `a_*` artists (top-40).
- Vibe: `emb_0…31` from Name + Genres + MusicBrainz tags + a lyric snippet.
- The whole matrix passes through `StandardScaler` before clustering.

**Unified (231 features)**
- **Schema harmonization:** rename heterogeneous fields onto one backbone. Example: game `developers` → `director`,
  book `pageCount` → `runtime`.
- **Identity and missingness:** `is_tv_show`, `is_game`, `is_book`, `is_music`, plus `has_<domain>_feats` masks.
- **Shared vibe:** one 10-d PCA over unified text (Title + Lead + first 40 words of the description).
  The text is **length-normalized**. Raw text length differs strongly across domains (movie plots ≈214 chars,
  game descriptions ≈1,074 chars). Without normalization the embedding encodes text shape, not content.
- **Domain alignment:** a `DomainAligner` (CORAL or centroid) centres each domain's PCA block.
  The aligner fits **inside each training fold only**. This keeps the evaluation leakage-safe.
- **Shared genre, quality and commercial channels:** cross-domain `gen_*` multi-hot, `critic_avg_5`,
  `box_office_log`, `popularity`.
- Music is **not** pooled here, because music has no rating target.

### 3.5 Transfer-grid feature subset
The transfer experiments use a **domain-blind** subset of the unified space:
- Keep `pca_*` (aligned), `gen_*` (top-40), `critic_avg_5`, `year`, `popularity`.
- **Drop** every `is_*` and `has_*_feats` column. Without this, a "transfer" model can split on domain identity
  and report a false transfer effect.

---

## 4. Model selection

### 4.1 Selection principle
- Model capacity follows N. Movies (N≈1,062) support a tuned four-member stack. Games (N=72) and books (N=93)
  support only a single regularized learner.
- Every added component must win a **paired Wilcoxon test** on the frozen folds. If it does not win, remove it.

### 4.2 Deployed models
| Domain | Deployed model | Reason |
| :--- | :--- | :--- |
| Movies | Stacking ensemble: Optuna-tuned edge-penalty XGBoost + CatBoost (MAE loss) + RBF-SVR + ordinal expected value, fused by `RidgeCV` over 5-fold internal CV. | N is large enough to fit a meta-learner. The members are complementary (see §6.6). |
| TV shows | Simplex-weighted stack: edge-penalty XGBoost + CatBoost + SVR. Weights are non-negative and sum to 1. | At N=171 a learned `RidgeCV` meta-learner overfits. A simplex average is more robust. |
| Games | `StandardScaler` + RBF-SVR, plus split-conformal intervals. | N=72. SVR is stable at small N and needs few hyperparameters. |
| Books | Same SVR pipeline for the benchmark; the **deployed predictor** is a text kNN (`KNeighborsRegressor`, k=8, cosine metric, distance weights) over the vibe PCA. | Book taste lives in the description text. The text kNN scores skill 0.43 against 0.24 for the crowd-mixing SVR. Blending the two hurts (≈0.39). |
| Unified | Simplex-weighted average of edge-penalty XGBoost and CatBoost, with per-fold CORAL alignment and temporal sample weights. | Simple, stable and leakage-safe across four heterogeneous domains. |
| Music | KMeans personas (k chosen by silhouette) + cosine similarity index + a PU XGBoost classifier (library=1, mined pool=0, hard negatives weighted ×2). | Music has no label. Clustering and PU learning are the honest formulation. |

### 4.3 Custom components
- **Asymmetric edge-penalty objective** (`src/movies/custom_objectives.py`):
  `L(e, r) = ½·e²·exp(α_hi·max(0, r−4.0) + α_lo·max(0, 1.5−r))`.
  The loss penalizes errors on favourites and hard passes more heavily. Optuna tunes `α_hi` and `α_lo`.
- **Ordinal expected-value regressor:** a 10-class XGBoost classifier returns class probabilities.
  The wrapper returns `Σ p(bucket)·bucket_value`. It refits a clone inside each CV fold, so stacking stays safe.
- **Simplex weighted averager:** SLSQP minimizes MAE (or MSE for the unified model) under the constraints
  `w ≥ 0` and `Σw = 1`.
- **Temporal decay weights:** `w_i = max(exp(−λ·Δt_i), w_min)`. Optuna tunes λ. The tuned value is
  λ = 0.000429, which is a half-life of about 1,615 days (≈4.4 years). Taste is therefore near-stationary
  over the library horizon.
- **Rank/quantile calibration** (`src/utils/deployment_calibration.py`): map each raw prediction's rank among
  the reference predictions onto the empirical rating quantiles. The map is monotone, so it preserves ranking.

### 4.4 Rejected components (with the evidence)
| Component | Test | Verdict |
| :--- | :--- | :--- |
| Distillation prior (unified prediction as a local feature) | Paired Wilcoxon on frozen folds. Games p=0.513 (MAE 0.5985→0.6030), books p=0.446 (0.3634→0.3699) — both null, prior dropped | **DROP** overall. Both cells are null and the prior moves MAE the wrong way in each. The books cell has now read p=0.71, 0.034, then 0.446 across three refreshes - unstable and underpowered, not a real effect. |
| Ridge stack for the unified model | Legacy 5-fold protocol. MAE 0.561, R² 0.384 against the mean ensemble 0.535 / 0.461. | **REMOVED.** |
| Residual domain-correction heads | Legacy 5-fold protocol. MAE 0.609, R² 0.269. | **REMOVED.** |
| Custom edge-penalty XGBoost loss | Head-to-head on 50 folds, all else fixed. Overall MAE +0.0039 (p=0.083), edge MAE +0.0002 (p=0.61). | **RETIRE.** A small net harm. The weight keys on the label's extremity, not the error, so it is ordinary importance weighting. |
| Entity-bridge features (Wikidata links) | Paired Wilcoxon + bootstrap CI on 665 linked items. ΔMAE +0.0038, CI [−0.019, +0.028], p=0.928. Adaptation slice: N=38, +0.0263, p=0.480. | **NULL.** Not established at this N. |
| Rank calibration for TV | Spearman ρ=0.587 is still too weak; calibration costs ~10 accuracy points. | **SKIPPED for TV.** Applied to movies, games and books only. |

---

## 5. Training

### 5.1 Two distinct protocols
The project separates **model fitting** from **model measurement**. Do not confuse them.
1. **Deployment fit** (`*_model_trainer.py`): an 80/20 `train_test_split` with `random_state=42`.
   This produces the serialized artifacts in `models/`.
2. **Measurement fit** (`src/reporting/production_benchmarks.py`, `src/unified_model/comprehensive_evaluator.py`):
   the same model recipe refits inside every frozen fold. This produces the reported metrics.

### 5.2 The frozen fold registry
- `src/unified_model/create_frozen_folds.py` writes `models/unified/fold_registry.json`.
- Protocol: `RepeatedStratifiedKFold`, 10 splits × 5 repeats = **50 folds**, `random_state=42`.
- Stratification key: `media_type | rating_bin`, where bins are low (≤2.5★), mid, high (>4.5★).
  Within each domain, the code merges a rare bin into the largest bin until every bin holds ≥10 items.
  This keeps the domain mix **and** the rating curve stable in every fold.
- The registry covers the rated universe only. Music rows are excluded.
- Every experiment reads the same registry. Identical splits make **paired** statistical tests valid and
  remove metric drift between runs.

### 5.3 Sample weighting
- `compute_sample_weight(class_weight='balanced')` over the rating buckets counters the clustered
  rating distribution (about 62% of movies sit between 3★ and 4★).
- An **edge boost** of ×5.0 multiplies the weight of extreme items (≤1.5★ or ≥4.5★).
  Without it, the MAE-optimal action is to predict the mode, and the model never predicts 5★.
- The unified model adds the temporal decay weight from §4.3.

### 5.4 Hyperparameter tuning
| Model | Method | Search space |
| :--- | :--- | :--- |
| Movies XGBoost | Optuna, 30 trials, inner 5×2 `RepeatedKFold`, objective = mean MAE | `alpha_hi`, `alpha_lo` (0.01–0.5, log), `max_depth` 3–8, `min_child_weight` 1–10, `reg_lambda` 1e-3–10 (log) |
| TV XGBoost | Optuna, 20 trials, inner 5-fold, objective = negative MAE | `alpha_hi`, `alpha_lo` (0.01–0.4) |
| Unified | Optuna, 10 trials, objective = MAE on a held-out split | `lambda_decay` 1e-5–0.01 (log), `alpha_hi`, `alpha_lo` (0.05–0.3) |
| CatBoost | Fixed | 1,000 iterations, lr 0.05, depth 6, `l2_leaf_reg` 3, MAE loss, early stopping 50 |
| SVR | Fixed | RBF kernel, C=1.0, ε=0.1, inside a `StandardScaler` pipeline |
| Music KMeans | Silhouette sweep | k selected by the best silhouette score |

After tuning, the movies trainer refits the final XGBoost with 300 trees at learning rate 0.03.

### 5.5 Leakage controls
- Target encodings use out-of-fold means with Bayesian smoothing.
- The `DomainAligner` fits on the training fold only.
- The rank calibration reference uses training-fold predictions only.
- The evaluator asserts `benchmarks != slices` and `N == n_unique_items`. These assertions exist because two
  earlier measurement bugs reversed a scientific conclusion.

### 5.6 One-command refresh
Run `python run_pipeline.py`. The script executes each stage as its own `python -m` process, in order:
merge-ingest → enrich → per-domain features → unified features → frozen folds → clear OOF caches →
evaluate → train → batch predict → render docs → paper figures.
Add `--full` to run the 50-fold transfer grid, the entity bridges and the learning curves.
The script stops at the first failing step and prints the cause.

---

## 6. Evaluation

### 6.1 Protocol
- 50 frozen folds (10-fold × 5 repeats), identical for every model.
- **Per-item deduplication:** average the raw predictions of the 5 repeats, then round to the 0.5★ grid.
  Therefore `N` equals the number of unique items, not the number of out-of-fold rows.
- Reported metrics: R², MAE, ±0.5★ accuracy, skill score, Spearman ρ, and the classification scores.

### 6.2 Headline benchmark (production models, frozen folds)
| Domain | N | Model | R² | MAE | ±0.5★ accuracy |
| :--- | ---: | :--- | ---: | ---: | ---: |
| Movies | 1,039 | Mean ensemble (edge+Cat+SVR+Ordinal) | 0.481 | 0.509 | 75.8% |
| **Unified (rated)** | **1,371** | **Mean ensemble** | **0.476** | **0.526** | **75.5%** |
| TV shows | 171 | Simplex stack (edge+Cat+SVR) | 0.332 | 0.646 | 67.8% |
| Games | 66 | Local SVR | 0.391 | 0.598 | 63.6% |
| Books | 89 | Local SVR | 0.332 | 0.472 | 79.8% |

**How to read this table.**
- **R²** is unreliable in the small domains. The rating variance there is tiny, so a few noisy items move R²
  from 0.35 to near zero. Do not rank domains by R².
- **MAE** is the stable yardstick. An MAE near 0.5★ equals one grid step. The label itself wobbles by about
  one grid step, so this is close to the noise floor.
- **±0.5★ accuracy** looks high in books (79.8%) mainly because book ratings cluster tightly. Compare it with
  the mode baseline in §6.3 before you read it as skill.
- The unified model reaches almost the movie-level MAE across four domains. That is the headline result.

### 6.3 Baseline sanity — is the model only predicting the mean?
| Model | N | MAE | MAE (mean) | Skill | ±0.5★ acc | acc (mode) | Acc lift | Spearman ρ | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 1,039 | 0.509 | 0.757 | +0.328 | 76% | 51% | +25 pts | 0.743 | 8e-183 |
| TV shows | 171 | 0.646 | 0.848 | +0.238 | 68% | 39% | +29 pts | 0.587 | 3e-17 |
| Games | 66 | 0.598 | 0.826 | +0.275 | 64% | 47% | +17 pts | 0.706 | 4e-11 |
| Books | 89 | 0.472 | 0.657 | +0.282 | 80% | 55% | +25 pts | 0.696 | 4e-14 |
| Unified | 1,371 | 0.526 | 0.778 | +0.323 | 75% | 50% | +26 pts | 0.697 | 3e-200 |

**Interpretation of each metric.**
- **Skill score** = `1 − MAE_model / MAE_mean`. A value above 0 means the model beats a constant predictor.
  All five models score +0.23 to +0.35. This is the project's primary metric, because it stays meaningful
  when the rating variance is small.
- **Acc lift** is the ±0.5★ accuracy above the mode baseline. It removes the accuracy that the clustered
  rating distribution gives away for free. The lift is +18 to +27 points everywhere.
- **Spearman ρ** is the load-bearing metric. A mean-hugging model has ρ≈0 and cannot order items.
  Every model reaches ρ 0.58–0.75 with p ≤ 1e-9. The models therefore rank favourites above passes.
- **Caveat:** the small domains still compress the prediction magnitude. The spread ratio (predicted SD over
  true SD) is 0.59 for TV and games, 0.61 for books and 0.69 for the unified model. Movies is the exception at
  1.05, because calibration already restored its range. Rank calibration (§4.3) corrects the rest at serve time.

### 6.4 Classification performance
| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 1,039 | 64.1% | 65.3% | 0.651 | 0.640 | 0.453 | 0.816 |
| TV shows | 171 | 56.7% | 54.2% | 0.557 | 0.576 | 0.308 | 0.823 |
| Games | 66 | 62.1% | 52.0% | 0.535 | 0.622 | 0.324 | 0.860 |
| Books | 89 | 62.9% | 57.1% | 0.589 | 0.611 | 0.373 | 0.729 |
| Unified | 1,371 | 61.6% | 61.7% | 0.627 | 0.620 | 0.408 | 0.809 |

**Interpretation.**
- **Cohen κ** 0.31–0.45 means fair to moderate agreement above chance. The 3-class task is genuinely hard,
  because the Neutral band is only one star wide.
- **The Neutral class is the weak class everywhere.** Movie Neutral F1 is 0.558, against 0.719 for Dislike and
  0.677 for Like. The model pushes middle items toward the edges of the band.
- **Balanced accuracy below plain accuracy** (games 52.0% against 62.1%) shows class imbalance. Games hold
  38 Like items and 8 Dislike items. Games Dislike recall is only 0.250, with precision 0.667.
  The model rarely calls a game bad.
- **Binary-Like F1 is 0.73–0.86 everywhere.** This is the metric that matters in production, because the
  user's real decision is "watch or skip". The binary decision is reliable even where the 3-class decision is not.
- **Books Like recall is 0.318.** The books model under-predicts enthusiasm. The deployed text-kNN model and
  the novelty-ranked queue exist partly to work around this.

### 6.5 Bias-variance decomposition
| Model | N | Features | bias² | variance | Bias share | Test MAE | Diagnosis |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Movies | 1,039 | 85 | 0.418 | 0.022 | 94.9% | 0.495 | Feature and noise ceiling; curve still creeping. |
| TV shows | 171 | 57 | 0.963 | 0.049 | 95.1% | 0.778 | Feature and noise ceiling; curve flat. |
| Games | 66 | 59 | 0.755 | 0.038 | 95.2% | 0.677 | Keep the local features. |
| Books | 89 | 180 | 0.386 | 0.025 | 93.9% | 0.450 | Near the small-N ceiling. |
| Unified | 1,371 | 231 | 0.584 | 0.028 | 95.5% | 0.573 | Bias-limited pooling tool. |

**Interpretation.** Every model is bias-dominated (94–96%) with variance near 4–6%. The models are stable,
not data-starved. More rows of the same features is therefore a weak lever. Richer features and the
irreducible label noise set the limit.

**Direct proof that this is a ceiling.** The team added the creator, cast and writer target encodings that
make movies strong to the weakest domain, TV. The bias² fell only from 1.014 to 0.963, and the test MAE fell
only from 0.784 to 0.778. At N≈171 most creators appear once, so the smoothed encoding collapses to the
global mean. The rest of the error is intrinsic 0.5★ rating noise.

**Why the metrics look modest — three compounding causes, largest first.**
1. **Aleatoric ceiling.** The ratings are self-reported on a coarse grid. The same item can receive 3.5★ or
   4.0★ on a different day. The decomposition counts this noise inside bias².
2. **The features measure crowd quality, not personal taste.** IMDb scores and vote counts capture "is this
   generally good". The user's deviation from the crowd is the residual the features cannot see.
3. **Small N** in TV, games and books blocks person-specific patterns.

### 6.6 Per-member ablation (are the ensembles complementary?)
Skill vs the mean baseline, 50 frozen folds, all rows uncalibrated.

| Domain | XGB alone | CatBoost alone | SVR alone | Mean blend | Simplex blend | Reading |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Movies | 0.380 | 0.372 | 0.371 | 0.381 | **0.385** | Complementary. The simplex blend beats every member. |
| TV shows | 0.215 | 0.205 | 0.143 | 0.229 | **0.236** | Complementary. The blend beats the best member by +0.021 and SVR-alone by +0.093. |

**This reverses an earlier finding.** On the previous corpus the same harness put TV's SVR-alone (0.110) ahead
of the blend (0.095), and the docs recommended replacing the TV stack with a plain SVR. That recommendation is
withdrawn: on the current corpus the TV SVR is the weakest member. Both ensembles now earn their place.

### 6.7 Where does taste live — text or crowd?
| Domain | Text-only skill | Crowd-only skill | Verdict |
| :--- | ---: | ---: | :--- |
| Movies / TV / Unified | 0.05–0.06 | 0.26–0.33 | The user tracks critic consensus. Text cannot rescue these domains. |
| Books | **0.39** | 0.17 | Book taste lives in the description and author text. |

This finding drove the deployed books text-kNN model (skill 0.43 against 0.24 for the SVR).

### 6.8 SHAP feature attribution (mean |SHAP| on a per-model XGB surrogate)
| Model | Top features |
| :--- | :--- |
| Movies | `critic_avg_5` (0.231), `imdb_rating` (0.175), `actors_encoded` (0.064), `vote_average` (0.062), `year` (0.061) |
| TV shows | `imdb_rating` (0.358), `vote_average` (0.063), `vote_count_log` (0.057), `imdb_votes_log` (0.054), `pca_2` (0.054) |
| Games | `ratings_count` (0.196), `rating` (0.169), `pca_14` (0.078), `metacritic` (0.075), `pca_2` (0.072) |
| Books | `pca_1` (0.129), `pca_4` (0.123), `ratingsCount` (0.100), `pca_8` (0.082), `averageRating` (0.071) |
| Unified | `critic_avg_5` (0.287), `imdb_rating` (0.131), `year` (0.081), `vote_average` (0.072), `pca_2` (0.047) |

**Interpretation.** Movies and TV are critic-consensus models. Games depend on RAWG community columns that
live **outside** the shared space; this explains why games need a local model. Books are embedding-led,
which agrees with §6.7.

### 6.9 Cross-domain transfer result
- **Design.** Pre-registered decision rule, fixed before any result was seen: a finding is positive only if the
  augmented lift is > 0 with paired p < 0.05 at **at least 2** of the 4 target fractions.
- **Result.** Taste transfers **into games**, and only into games. Five source subsets clear the bar.

  | Source → Games | lift @100% | significant fractions |
  | :--- | ---: | :--- |
  | `movie+tv+book` | **+0.121** | **4/4** (+0.085 p=0.031 · +0.081 p=0.030 · +0.130 p=0.0003 · +0.121 p=0.0014) |
  | `movie+tv` | +0.120 | 2/4 |
  | `movie` | +0.111 | 3/4 |
  | `movie+book` | +0.104 | 3/4 |
  | `tv` | +0.021 | 3/4 |

- **Negative targets.** Movies, TV and books show no significant positive transfer (0 of 4 fractions each).
- **Zero-shot affinity is asymmetric.** Every source is mildly positive *into* games (+0.067 movie, +0.088 TV,
  +0.082 book). Games as a *source* is strongly negative into everything else (−0.297 movies, −0.458 TV,
  −0.362 books). Games is a good target and a bad source.
- **Four stress controls passed.** (1) Domain-identity columns dropped. (2) The transfer model beats every
  featureless constant baseline (MAE 0.99 vs 1.05–1.17) and rank-tracks true ratings at Spearman +0.47, so it
  carries content, not a better prior. (3) Text-length normalization was necessary; on raw-length embeddings
  the effect measured only +0.05 and was **not** significant. (4) A label-permutation placebo is decisive:
  shuffling the source ratings collapses the lift from +0.132/+0.101/+0.122/+0.134 to
  −0.004/−0.025/−0.015/+0.053, with the real-vs-placebo gap significant at every fraction (p ≤ 0.011;
  p < 0.0002 at 0–50% target data).
- **What predicts transfer.** Aligned-space distance does **not** predict transfer (Spearman +0.085 over n=12
  pairs; the sign has moved between runs, so this is noise). Target data-starvation does.
- **Caveat on the p-values.** The 50 repeated-CV folds are not independent — with 66 unique games, items recur
  across folds. The p-values are optimistic. The trustworthy signal is consistency across all four fractions.
- **Reconciliation with §6.2.** The game rating has a domain-specific part (Metacritic, platforms) that only
  the local SVR sees, and a genre/vibe part that lives in the shared space and transfers. Both facts hold.

### 6.10 Known open items
- **The Books distillation cell dissents.** The artifact says KEEP at p=0.034; the project verdict says DROP
  (§4.4). Re-test at the next corpus refresh.
- **The Games uncalibrated baseline is not settled.** The benchmark harness reports MAE 0.652 and the
  distillation harness reports 0.5970. This changes the measured size of the calibration win, not the decision.
- **The edge-loss ablation predates the current corpus** (2026-06-26, 1,000 movies). Re-run it.
- Games and books remain pilots at N=66–89. Report skill and conformal intervals, not R².

### 6.11 Artifact provenance
Not every artifact shares one vintage. Read each number with its date.

| Vintage | Artifacts |
| :--- | :--- |
| **2026-07-25** (current corpus) | `latest_metrics.json`, `production_oof.csv`, `standalone_oof.csv`, `distillation_ablation_results.json`, `transfer_grid_summary.json`, `entity_bridge_results*.json`, unified training frame |
| **2026-07-26** (regenerated) | `baseline_sanity.json`, `classification_diagnostics.json`, `feature_diagnostics.json`, `bias_variance.json`, conformal widths |
| **2026-06-26/27** (previous corpus) | `edge_loss_ablation_results.json`, `transfer_placebo_control.json` |
| **2026-07-11** (previous corpus) | `text_vs_crowd.json`, `text_lift.json` |

---

## 7. Deployment and monitoring

### 7.1 Serving path
- The interface is a multi-page **Streamlit** app. Start it with `streamlit run app/main.py`.
- Each domain has a **Dashboard** page (analytics) and an **Oracle** page (prediction).
- Inference sequence for one item:
  1. Fetch the item metadata from the domain API.
  2. Load the `joblib` preprocessor state. Rebuild the identical feature row, and fill any gap with the
     stored training median.
  3. Predict with the deployed model artifact from `models/<domain>/`.
  4. Apply rank/quantile calibration for movies, games and books. Skip it for TV.
  5. Clip to [0.5, 5.0] and round to the 0.5★ grid.
  6. Show the SHAP waterfall explanation, the nearest neighbours and the uncertainty interval.
- Batch scoring runs through `src/<domain>/predict_ratings.py` and
  `src/unified_model/predict_unified_ratings.py`. The outputs land in `results/` and `data/predictions/`.

### 7.2 Serve-time calibration
- The calibration maps each raw prediction's rank among the reference predictions onto the empirical quantiles
  of the user's true ratings. It needs at least 20 rated reference items, otherwise it returns the raw value.
- The map is monotone, so it never changes the ranking. It only restores the output distribution.
- Measured effect: games MAE 0.652 → 0.598 (ρ=0.71); books near-neutral, buying 5★ coverage (ρ=0.70); movies
  trade about 4 points of ±0.5★ accuracy for 45 genuine 5★ predictions (MAE 0.474 → 0.509, ρ=0.74); TV skipped
  (ρ=0.59 is still too weak, and calibration costs ~10 accuracy points).
- Two conditions must hold, not one: the model must **rank** well *and* have **compressed** its range. Movies
  ranks well but was already spread, so calibration there is a deliberate accuracy-for-coverage trade.

### 7.3 Uncertainty quantification
- Every domain has an 80% **split-conformal** interval width: the 80th percentile of the absolute OOF
  residuals. The Oracle shows it directly, for example "3.5 ± 1.0".
- Widths and measured coverage on the frozen-fold OOF (recomputed 2026-07-26):

  | Domain | Width | Measured coverage (target 80%) |
  | :--- | ---: | ---: |
  | Movies | ±0.90 | 81.6% |
  | TV shows | ±0.99 | 80.1% |
  | Books | ±0.71 | 79.8% |
  | Games | ±1.00 | 87.9% (over-covers) |

- Games over-covers because at N=66 on a 0.5★ grid the residuals take few distinct values.
- The width is constant inside a domain. That is a property of vanilla split conformal, which guarantees
  marginal — not conditional — coverage.
- **Caveat:** this is an in-sample OOF approximation and is optimistically biased. Certifying coverage needs a
  held-out calibration set that does not exist at this N. Treat the widths as approximate.
- The games and books trainers also save `conformal_width_80`, `skill_score`, `mae`, `r2` and `train_mean` in
  `model_meta.joblib`, computed on their own 5-fold partition. Those are the reference point for the next
  retraining cycle, and they differ slightly from the registry-OOF widths above.

### 7.4 Monitoring
| Signal | Where | What it detects |
| :--- | :--- | :--- |
| **Taste drift** | `app/pages/14_Taste_Drift.py` | A 90-day rolling Shannon entropy over the genre distribution. Rising entropy means the taste is broadening. A step change means a regime change. |
| **Calibration reliability** | `app/pages/13_Model_Calibration.py` | A reliability diagram binned by predicted rating. In a calibrated model, items predicted at 4.5★ average 4.5★ in truth. |
| **Ensemble disagreement** | `src/evaluation/active_learning_ranker.py` | The standard deviation across the raw base-model predictions. High disagreement marks an item the model does not understand. |
| **Feature-space novelty** | Same module | The mean cosine distance to the 5 nearest training items in the aligned PCA space. High novelty marks an out-of-distribution item. |
| **Coverage of shared channels** | `src/experiments/feature_diagnostics.py` | The populated fraction of each unified channel per domain. A channel that falls to zero means an upstream ingestion break. |
| **Metric provenance** | `reports/latest_metrics.json` | Every rendered number carries a `generated_at` timestamp and a `git_hash`. |

### 7.5 The metrics spine (single source of truth)
1. The evaluators write **only** to `reports/latest_metrics.json`.
2. `src/reporting/render_docs.py` injects those numbers into the README and the technical report, between
   `<!-- NAME:BEGIN -->` and `<!-- NAME:END -->` markers.
3. Never hand-edit a number inside a marker block. Re-run the renderer instead.
4. Invariants enforced in code: `benchmarks != slices`; benchmarks use the production local models;
   slices come from the unified model; `N == n_unique_items`.
   Two earlier violations of these invariants each reversed a published conclusion.

### 7.6 Retraining and data refresh
- **Trigger.** Retrain after you add ratings, for example at a round number of movies.
- **Command.** Run `python run_pipeline.py`. Use `--full` for the heavy 50-fold experiments.
- **Order matters.** The pipeline rebuilds the fold registry, then deletes the cached OOF frames
  (`reports/*_oof.csv`), then re-evaluates. A stale cache against a new registry is a silent metric bug.
- **Not auto-updated.** Hand-written prose in the README and the technical report. Review it if the numbers
  moved materially.

### 7.7 Active learning (closing the data gap)
- The cross-domain ranker scores each unrated item by `z(uncertainty) + z(novelty)`, per domain.
- Domain queues exist for movies, TV shows and books. Each queue excludes items the user already consumed.
- The books queue ranks by **novelty**, not by the predicted star, because the small-N models compress the
  predicted magnitude toward the centre.
- Outputs: `reports/ACTIVE_LEARNING_QUEUE.md`, `reports/MOVIE_...`, `reports/SHOW_...`, `reports/BOOKS_...`.

### 7.8 Automated production job
- `.github/workflows/daily-playlist.yml` runs `daily_playlist/daily_playlist.py` every day at 06:30 UTC.
- The job builds one 200-track Spotify playlist from three buckets: heavy rotation (Last.fm 7-day top tracks),
  older liked tracks (saved tracks minus recent scrobbles), and discovery (Last.fm similar tracks resolved on
  Spotify). It replaces the same playlist in place.
- Credentials come from GitHub Actions secrets. The job uses a stored Spotify refresh token.
- Discovery uses Last.fm because Spotify deprecated `/v1/recommendations` in November 2024.

---

## 8. Limitations
- **Single user.** Nothing here generalizes across users. Multi-user generalization is out of scope.
- **Aleatoric ceiling.** Self-reported 0.5★ ratings cap the achievable R² and MAE.
- **Small-N domains.** Games and books are pilots. Lead with skill score and conformal intervals.
- **Music is unrated by design.** It is excluded from every rating metric and every confusion matrix.
- **Music transfer is gated off.** Music-to-other transfer still measures ≤0.
- **The TV stack is mis-weighted** and the movie stack needs a retune (§6.10).
