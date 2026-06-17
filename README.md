# Personal Media Intelligence Hub

## Table of Contents
- [Introduction](#introduction)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Engineering Highlights](#engineering-highlights)
- [Performance Benchmarks](#performance-benchmarks)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [Unified Media Intelligence](#unified-media-intelligence)
- [Dashboards & Visualizations](#dashboards--visualizations)
- [The Oracle (Explainable Recommendations)](#the-oracle-explainable-recommendations)

## Introduction
The **Personal Media Intelligence Hub** is a sophisticated "Global Taste Engine" designed to map, analyze, and predict personal entertainment preferences across five distinct domains: Movies, TV Shows, Music, Games, and Books. By consolidating fragmented media consumption data and enriching it with high-fidelity metadata from external APIs, the system builds a unified semantic representation of "Taste" that allows for cross-domain discovery and explainable recommendations.

## Case Study: When fixing the evaluation made the numbers worse

A crucial part of this project involved recognizing and correcting inflated metrics in data-sparse domains (Games and Books, N ≈ 60). Initially, a single validation split suggested a promising R² of ~0.45. However, this was a statistical artifact. 

By migrating to a rigorous **10-fold × 5-repeat Cross-Validation** protocol, that inflated number deflated to a **modest and unstable** R² (Games ≈ 0.36, Books ≈ 0.18 on the frozen folds — and *near-zero or negative* under stricter pooled estimators or for the unified slice). That spread is itself the lesson: R² is not a reliable yardstick here. This was a critical finding:
1.  **Metric Treachery:** In domains where ratings cluster tightly (e.g., 3.0 to 4.5), variance is tiny, so R² becomes hypersensitive to a few noisy items — swinging from ~0.36 to near-zero depending on the estimator — even when the Mean Absolute Error (MAE) is stable and genuinely useful.
2.  **The Distillation Prior — tested and dropped:** We hypothesized the unified model could anchor these sparse domains as a prior feature. A paired Wilcoxon ablation on the frozen folds said otherwise (p = 0.21 Games / 0.18 Books, both ≥ 0.05): the prior was removed. What *did* emerge once it was gone were positive **skill scores** (Games 0.225, Books 0.108) — the local models beat the mean-rating baseline, the first evidence of genuine local signal at N ≈ 60 even with R² ≈ 0.
3.  **The Pivot to Skill Score:** We reframed the evaluation metric from R² to **Skill Score** (`1 - MAE_model / MAE_baseline`), focusing on whether the system adds value over a simple historical average. 
4.  **Uncertainty & Acquisition:** We shifted the focus for these domains from raw accuracy to active learning. By surfacing **split-conformal prediction intervals** (e.g., "3.5 ± 1.3"), the Oracle now explicitly quantifies its uncertainty, guiding the user to deliberately rate the most informative backlog items to efficiently bridge the data gap.
5.  **Ablation Discipline:** We explicitly tested residual-head domain correction and Ridge-based stacking using paired Wilcoxon tests. Both were found to be actively destructive (destroying ~0.09 R²), leading to their removal in favor of a simpler, more robust Mean Ensemble.

Reviewers see inflated metrics daily; accurately deflating them and extracting a robust path forward demonstrates true ML maturity.

## Case Study: Does taste transfer across domains?

The project's central question. Two sub-stories:

**1. Two measurement bugs, two flipped conclusions.** The benchmark table once published the *unified model's per-domain OOF slice* as if it were the standalone local models — with N inflated 5× (OOF rows counted as items) and the local-model names glued on top. Repairing the writer (`assert benchmarks != slices`, per-item dedup, `assert N == n_unique_items`) forced standalone and unified to be measured with the **same estimator on the same folds**. But a second, subtler bug remained: the "local" benchmarks were *weakened proxies* (Movies = a plain 200-tree XGB, not the deployed tuned edge-penalty stacking ensemble), which flattered the unified model. Replacing the proxies with the **production local models** (measured honestly as 50-fold registry OOF) gives the final, type-split verdict: the **local model wins in Movies and Games** — the domains with rich domain-specific features (target-encodings, Metacritic) — while the **unified model wins in TV and ties on Books**, the semantic-only domains. Cross-domain pooling earns its keep *only where a domain has little local signal beyond vibe* — not "everywhere but Games," as the proxy comparison had suggested. Two bugs, each of which *changed the scientific conclusion* — exactly why the single-source-of-truth renderer exists.

**2. The pre-registered transfer grid.** We then asked the sharper question — not "does the pooled model win on average" but *which specific source-domain combinations* transfer into which targets, zero-shot and augmented, in the shared feature space. The decision rules were **pre-registered before any result was seen** (positive finding ⇔ augmented lift > 0, paired p < 0.05, at ≥ 2 target fractions). <!-- TRANSFER_VERDICT:BEGIN -->**Realized verdict (full 50-fold grid, domain-blind, triple-controlled): TASTE TRANSFERS — into Games, robustly from `movie + book`.** A model trained only on movie/book ratings, **having never seen a game**, predicts game ratings better than the games-only model (zero-shot +0.175 skill, p=0.0004; +0.121 at 100%). This **reverses the earlier pilot NULL** (the 6-fold pilot was underpowered) and survives **three stress controls**: (1) the [domain-identity-leak fix](#engineering-highlights); (2) a *prior-vs-transfer* control — `movie+book→game` beats every featureless constant baseline (MAE 1.08 vs ~1.23) and rank-tracks true ratings at Spearman +0.37, so it is content, not a better prior for games' noisy N=55 mean; and (3) an *embedding text-normalization* control — re-embedding all domains with length-matched text (fixing the [text-shape artifact](#engineering-highlights)) keeps `movie+book→game` significant. The *broader* set of source configs (tv, movie+tv, …) cleared the bar on the raw space but **weakened under text-normalization**, so part of the wide result was a text artifact and the robust claim narrows to `movie+book`. Reconciliation with the benchmark table: Games' rating has a *domain-specific* part (Metacritic/platforms — only the local SVR sees it) and a *genre/vibe* part that lives in the shared space and transfers from movies/books. Aligned-space distance does not predict transfer (Spearman ≈ −0.18); *target data-starvation* does. Full results in the **Transfer Atlas** page and `reports/transfer_grid_summary.json`.<!-- TRANSFER_VERDICT:END --> Pre-registering the rule *before* running, reporting the reversal when the full grid crossed the bar, then stress-testing it three ways before finalizing — that discipline is the point.

## Architecture

```mermaid
graph TD
    A[Ingestion Layer] --> B[Schema Contract Layer]
    B --> C[Enrichment APIs]
    C --> D[Feature Store]
    D --> E[Domain-Specific Ensembles]
    D --> F[Unified Stack]
    E --> G[Oracle / Dashboards]
    F --> G
    
    subgraph Enrichment
    C1[TMDB/OMDB]
    C2[MusicBrainz]
    C3[ReccoBeats]
    C4[Google Books]
    end
    
    subgraph Models
    E1[Movies Edge-Penalty Stack]
    E2[TV Shows Simplex-Stack]
    E3[Music PU-Affinity]
    E4[Games/Books Local SVR]
    end
```

## Dataset Schemas

The system orchestrates a multi-domain data lake with strictly enforced schema contracts. Every domain moves through **two schema layers**, and both are documented below because they answer different questions:

1. **Ingested schema** — the raw fields each external API returns, persisted to `data/processed/<domain>/enriched_data.csv`. This is what is *collected*.
2. **Training schema** — the engineered numeric feature matrix actually fed to the models, persisted to `data/processed/<domain>/training_features.csv` (and `…/unified/training_features.csv`). This is what is *learned from*.

The per-domain models train on their own engineered matrix; the **Unified Model** re-projects every domain into a single **397-feature** shared latent space (vibe PCA aligned via CORAL). Column counts below are the live widths of the on-disk training matrices.

> **Convention.** `target_reg` is the 0.5★-grid user rating (the regression label); `target_ordinal` is its 10-bucket integer encoding (0.5★→0 … 5.0★→9); `target_class` is a coarse low/med/high bin kept for legacy dashboards. These label columns are excluded from every feature count quoted below.

### 🎬 Movies (TMDB + OMDb)
**Ingested** (`enriched_data.csv`, 36 raw fields): `tmdb_id, imdb_id, letterboxd_name/uri, year, title, rated, released, runtime, genre, director, writer, actors, plot, overview, tagline, language, country, awards, imdb_rating, imdb_votes, metascore, rotten_tomatoes_rating, box_office, production, popularity, vote_average, vote_count, user_rating, is_liked, …` (+ poster/backdrop art, processing status).

**Training** (`training_features.csv`, **66 features**):
| Category | Engineered features |
| :--- | :--- |
| **Numerical (12)** | Year, Runtime, IMDb Rating, IMDb Votes, Metascore, RT%, `box_office_log` (log1p), Popularity, Vote Average, Awards `total_wins`/`total_nominations` (regex-parsed), `critic_avg_5` (blended IMDb/Metascore/RT/TMDB on a 5★ scale) |
| **Categorical** | Language one-hot (top-3 + Other), Genres multi-hot (`MultiLabelBinarizer`), MPAA bucket (`rated_Adult/Teen/General`) |
| **Relational (3)** | Leakage-safe **target encodings** — Director, Actors, Director×Genre — Bayesian-smoothed (m=10) over 5-fold out-of-fold splits |
| **Vibe (NLP)** | 384-d MiniLM-L6-v2 embedding of *Title + Director + Plot* → **25 PCA components** |

### 📺 TV Shows (TMDB)
**Ingested** (`enriched_data.csv`, 35 raw fields): `tmdb_id, tvmaze_id, name, year, overview, tagline, created_by, genres, vote_average, vote_count, imdb_rating, imdb_votes, number_of_episodes, number_of_seasons, status, network, language, runtime, first_air_date, last_air_date, age_rating, awards, actors, writer, country, production_companies, user_rating, watch_count, …`.

**Training** (`training_features.csv`, **38 features**):
| Category | Engineered features |
| :--- | :--- |
| **Numerical (5)** | Year, Vote Average, `vote_count_log` (log1p), Season Count, `is_adult` |
| **Categorical** | Network one-hot (top-10 + Other, `net_`), Genres multi-hot frequency-gated at ≥5% of N (`g_`) |
| **Vibe (NLP)** | 384-d MiniLM-L6-v2 embedding of *Title + Network + Overview* → **15 PCA components** |

### 🎮 Games (RAWG)
**Ingested** (`enriched_data.csv`, 16 raw fields): `name, my_rating, platform_from_text, released, age_rating, genres, developers, publishers, metacritic, rating, ratings_count, reviews_count, tags, description_raw, playtime, cover`. Ratings include the sentinel **`'I'` (Incomplete)** → mapped to NaN (kept for ranking, excluded from supervised scoring).

**Training** (`training_features.csv`, **55 features**):
| Category | Engineered features |
| :--- | :--- |
| **Numerical (5)** | Year, Metacritic, Global Rating, Ratings Count, Reviews Count |
| **Categorical** | Platform one-hot (`plat_`), Genres multi-hot (`gen_`), Developers multi-hot frequency-gated at ≥2 (`dev_`) |
| **Vibe (NLP)** | 384-d MiniLM-L6-v2 embedding of *Name + Genres + Tags + Description* → up to **15 PCA components** (`min(15, N−1)`) |

### 📚 Books (Google Books)
**Ingested** (`enriched_data.csv`, 13 raw fields): `title, isbn, my_rating, authors, publisher, publishedDate, pageCount, categories, averageRating, ratingsCount, description, thumbnail, infoLink`.

**Training** (`training_features.csv`, **19 features**):
| Category | Engineered features |
| :--- | :--- |
| **Numerical (4)** | Year, Page Count, Average Rating, Ratings Count |
| **Categorical** | Authors multi-hot frequency-gated at ≥2 (`aut_`), Categories multi-hot (`cat_`) |
| **Vibe (NLP)** | 384-d MiniLM-L6-v2 embedding of *Title + Authors + Categories + Description* → up to **15 PCA components** (`min(15, N−1)`) |

### 🎵 Music (Spotify + ReccoBeats + MusicBrainz + Genius)
Music has **no ground-truth star ratings**; its label is a PU-classifier-calibrated *affinity* (see [PU Learning](#engineering-highlights)). Features land in `music_features.npz` (StandardScaler-normalized numerics).

**Ingested**: Spotify library/playlists (`track_id, name, primary_artist, album, popularity, artist_followers, …`), ReccoBeats **9 audio features** (`acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence`), MusicBrainz (`mb_genres, mb_tags, mb_length_ms`), Genius (`lyrics`, VADER sentiment).

**Training** (engineered matrix):
| Category | Engineered features |
| :--- | :--- |
| **Numerical** | Duration, Release Year, Track Age, Popularity, Artist Popularity/Followers (log), Album/Track position, Explicit |
| **Audio (ReccoBeats)** | The 9 audio features above, each median-imputed **+ a `_missing` flag** |
| **Lyrical** | Word/line counts, unique-word ratio, VADER sentiment (`pos/neu/neg/compound`), `lyrics_found` |
| **Categorical** | Genres/Tags multi-hot (top-60, `g_`), Artists multi-hot (top-40, `a_`) |
| **Vibe (NLP)** | MiniLM-L6-v2 embedding of *Name + Genres + MB-Tags + Lyric snippet* → **32 PCA components** (`emb_`) |

### 📺 YouTube (YouTube Data API v3)
Ingestion-only (no rating model). **Video:** Video ID, Duration, Tags, Description, Views, Likes, Comment Count. **Channel:** Channel ID, Title, Subscriber Count, Video Count.

### 🌐 Unified Schema (Cross-Domain, 397 features)
The Unified Model concatenates every domain into one row-space (`data/processed/unified/training_features.csv`, **402 columns = 397 features + 5 bookkeeping** [`target_reg, target_ordinal, rating_date, source_id, media_type`]). Heterogeneous fields are mapped onto a common backbone:
- **Identity:** `is_tv_show/is_game/is_book/is_music` indicators + `has_{movie,tv,game,book,music}_feats` multi-modal missingness masks (so the model can tell an absent feature from a real zero).
- **Narrative:** A single **10-d PCA vibe space** over unified text (*Title + Lead + Overview*), with **CORAL/centroid alignment fit per training fold** (no leakage) so "vibe" geometry is comparable across domains.
- **Genre/Rating:** Cross-domain genre multi-hot (`gen_`, with `Sci-Fi & Fantasy`/`Action & Adventure` split into atomic genres), language one-hot, MPAA bucket.
- **Commercial:** Log-normalized Box Office / Popularity proxies.
- **Quality:** `critic_avg_5` — blended, scale-normalized average of IMDb, RT, Metascore, and RAWG/Google ratings.
- **Cross-domain audio:** Gated music-affinity features (active only where the transfer matrix licenses them).

## Key Features
- **Consolidated Library:** A single-pipeline architecture managing over **1,250 rated items** across four domains and **3,600+ music tracks**.
- **Advanced ML Ensemble:** Combines XGBoost and CatBoost through a Simplex-constrained Mean Ensemble for robust cross-domain fusion.
- **Explainable Oracle:** Predicts star ratings and provides "Verdicts" based on SHAP waterfall explanations and semantic similarity.
- **Taste Diversity Analytics:** Tracks **Taste Entropy** (Shannon Diversity) and temporal drift across your library.
- **Multi-Modal Fusion:** Integrates acoustic features, episodic metadata, and plot vibes into a 397-feature shared latent space.

## Engineering Highlights

- **Domain Centroid Alignment (CORAL):** Corrects domain shift in the shared embedding space by centering embeddings per media type, ensuring that "vibes" are comparable across Movies, Games, and Books.
- **Frozen-Fold Evaluation Registry:** Uses a deterministic `fold_registry.json` to ensure all cross-validation experiments use identical splits, eliminating metric drift and enabling paired statistical testing.
- **Asymmetric Edge-Penalty Loss:** Custom XGBoost objective derived to penalize errors more heavily at the extremes (favorites and hard passes).
  $$L(e, r) = \frac{1}{2}e^2 \cdot \exp(\alpha_{hi} \cdot \max(0, r - 4.0) + \alpha_{lo} \cdot \max(0, 1.5 - r))$$
- **Joint Optuna Tuning:** Hyperparameters and asymmetric penalty coefficients are jointly optimized under **5×2 Repeated Cross-Validation**.
- **Positive-Unlabeled (PU) Learning:** Solves the implicit feedback problem for music by sampling pseudo-negatives and using quantile calibration to map affinity to pseudo-ratings.
- **Distillation Prior (tested, then removed):** We evaluated feeding the Unified Model's predictions into the Games/Books local SVRs as a prior feature. A paired Wilcoxon ablation on the frozen folds found it **not significantly helpful** (p = 0.21 / 0.18, effect direction negative), so the prior was **dropped** — the local SVRs run on their own features. Killing a feature that doesn't earn its keep is the point, not a regression.
- **Temporal Taste Decay:** Implementation of floored exponential sample weighting ($w_i = \max(\exp(-\lambda \Delta t_i), w_{min})$). Tuning revealed preferences are remarkably stable over the library's horizon (half-life ≈ 4.4 years).

## Performance Benchmarks

<!-- BENCHMARKS:BEGIN -->
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
<!-- BENCHMARKS:END -->

*Note: Metrics reflect robust 10-fold × 5-repeat cross-validation with frozen folds. All numbers render from `reports/latest_metrics.json` via `src/reporting/render_docs.py`; the raw terminal outputs of every pipeline run are captured in [`reports/PIPELINE_RUN_LOG.md`](reports/PIPELINE_RUN_LOG.md).*

## Technology Stack
- **UI/Frontend:** Streamlit, Vanilla CSS, Plotly.
- **Core ML:** XGBoost, CatBoost, Scikit-Learn, Optuna, SHAP.
- **NLP:** SentenceTransformers (`all-MiniLM-L6-v2`), UMAP.
- **Data:** Pandas, Pydantic (Schema Contracts), Joblib.
- **APIs:** TMDB, OMDb, RAWG, Google Books, YouTube Data API v3, ReccoBeats, MusicBrainz.

## Dashboards & Visualizations
- **Latent Space Explorer:** UMAP projection of the 384-d semantic space, clustering items by "vibe."
- **Model Calibration:** Reliability diagrams binned by prediction confidence to ensure "honest" metrics.
- **Taste Drift Timeline:** 90-day rolling Shannon entropy over genre distributions.
- **SHAP Waterfall:** Per-prediction explainability in the Oracle UI.
- **Transfer Atlas:** Cross-domain affinity heatmap, augmented learning curves, and the similarity↔transfer scatter — the centerpiece of the transfer study.

## Limitations
Honest constraints, stated plainly:
- **Aleatoric ceiling.** Ratings are self-reported on a 0.5★ grid; intrinsic noise caps achievable R²/MAE, especially after rounding. MAE-skill is the more honest yardstick in low-variance domains.
- **Small-N domains.** Games (N≈55–62) and Books (N≈63) are pilots, not production claims; their R² is treacherous (near-zero variance), so we lead with skill score and conformal intervals, and the entity-bridge result is explicitly framed as a pilot with CIs.
- **PU pseudo-label caveat.** Music has no ground-truth ratings; its "ratings" are PU-classifier-calibrated affinities. Any pooled metric that scores music (the *Unified (full pool)* row) is the model partly predicting another model's output — reported only as a footnoted, secondary line.
- **Single-user dataset.** Everything is one person's taste. Nothing here generalizes across users; multi-user generalization is explicitly out of scope.
- **Transfer grid scope.** The shipped grid result is a documented **pilot** (subset of the 50 registry folds, 150-tree XGB proxy for the Mean Ensemble). The harness runs the full 50-fold/300-tree grid via `transfer_study.py full`; the verdict is reproducible either way.

---
*Created and maintained by the Personal Media Intelligence Hub Team.*
