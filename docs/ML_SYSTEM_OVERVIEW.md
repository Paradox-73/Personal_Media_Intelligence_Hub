# Personal Media Intelligence Hub — ML System Overview

Audience: ML scientists. Scope: the full modelling chain, from raw data to served predictions.
Written in ASD-STE100 Simplified Technical English. All numbers come from the live repository
(`reports/latest_metrics.json`, `generated_at` 2026-10-01, git `3015c3f`; the four diagnostics JSONs
regenerated 2026-10-01) and from the on-disk feature matrices. **The primary metric is skill**
(1 − MAE / MAE of predict-the-mean). Ratings cluster in 3–4★. A constant 3.5★ prediction already scores
MAE 0.749 on the films and lands within ±0.5★ 61.9% of the time, so MAE and ±0.5★ accuracy look good for a
model that learned nothing. Skill is 0 for predict-the-mean and ≈0 for any other constant. Spearman ρ is the
second metric. MAE and R² are secondary. Each table says which artifact it reads.
**Read section 8 before you quote a metric.** One known defect makes every reported MAE and R² in section 6
optimistic by an amount that has not been measured.

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

### 2.3 Data volumes (live on disk, 2026-10-01)
| Domain | Enriched rows | Rated rows (training) |
| :--- | ---: | ---: |
| Movies | 1,163 | 1,163 |
| TV shows | 175 | 175 |
| Games | 72 | 66 |
| Books | 110 | 110 |
| Music | 3,837 | not applicable |
| Unified | 1,520 | **1,514** |

The 2026-10-01 run added 101 films (Letterboxd export of 2026-10-01, 42 of them standup comedy specials)
refreshed the books (110 rated) and shows (175), and regenerated the fold registry.

Only games differ. Six games carry the sentinel `'I'` (incomplete), so they have no rating. They stay in
`enriched_data.csv` and in the unified frame, and the batch scorer still scores them, but the frozen fold
registry, the unified trainer and the evaluator all exclude them.

**Why that matters.** Until 2026-08-27 those six rows carried `target_reg = 0.0` — the output of a `fillna`,
not a `NaN` — which is off the bottom of the 0.5★ grid, and no `notna()` guard removed them. They trained and
were scored in all 50 folds. Removing them alone moved the headline unified MAE from 0.5153 to 0.5029 and the
unified game slice from 0.8681 to 0.6364 on the previous OOF. The registry then held exactly 1,407 items; after
the 2026-10-01 regeneration it holds 1,514.

The unified per-domain slice and the standalone game model are therefore measured on the **same 66 items**
for the first time. Earlier documents compared a 72-item slice against a 66-item local model while claiming
both were measured on the same items.

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
3. **Build the vibe text.** Build a short header (title, creator, genre, tags) and a body (plot,
   overview, description). Embed both with `all-MiniLM-L6-v2` (384-d) through the **chunk-pooled**
   encoder in §3.6, never with a single truncated `model.encode` call. Reduce the embedding with PCA.
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
| Domain | Features | PCA vibe components | Entity target encodings |
| :--- | ---: | ---: | :--- |
| Movies | 107 | 40 | 8 crew columns + `dir_genre_encoded` + 2 TMDB-neighbour columns |
| TV shows | 56 | 15 | `creator_encoded`, `actors_encoded`, `writer_encoded` |
| Games | 51 | 15 | `developers_`, `publishers_`, `tags_`, `genres_encoded` |
| Books | 29 | 15 | `authors_`, `categories_`, `publisher_`, `moods_`, `hc_tags_encoded` |
| Unified | 239 | 10 | `creator_encoded`, `studio_encoded` |
| Music | 173 | 32 | none (unsupervised) |

> These are the live on-disk widths, counted from `data/processed/*/training_features.csv` on 2026-08-28
> (total columns minus the `target_*` / `source_id` / `media_type` / `rating_date` bookkeeping columns).
>
> Books and games got **narrower**, not wider. Both replaced a wide, sparse multi-hot block (one column per
> author, one per developer) with a smoothed target encoding plus a support count. A 180-column author
> multi-hot over 104 books is one column per two rows; the encoding carries the same information in two.
>
> Movies grew from 85 to 107. `enrich_extended.py` had already fetched composer, cinematographer, editor,
> producer and production company, and nothing read them; `writer` was never encoded at all, despite being
> the strongest entity signal in the domain (leave-one-out ρ 0.563 against 0.460 for director).

### 3.4 Field-to-feature mapping by domain

**Movies (107 features)**
- Numeric: `year`, `runtime`, `imdb_rating`, `imdb_votes`, `metascore`, `rotten_tomatoes_rating`,
  `vote_average`, `popularity`, `box_office_log`, `total_wins`, `total_nominations`, `critic_avg_5`,
  plus `budget_log`, `revenue_log`, `vote_count_log`, `rel_month` and `age_at_rating`.
- Categorical: `lang_*` (top-3 + Other), `gen_*` genre multi-hot, `rated_Adult/Teen/General`.
- Relational: eight crew **target encodings** — director, actors, writer, composer, cinematographer, editor,
  producer, production company — each with an `_n` support count, plus `dir_genre_encoded`.
  Smoothing is **m=2**, not m=10. A sweep on 10 folds gave m=0 0.4769 · m=1 0.4732 · **m=2 0.4694** ·
  m=3 0.4765 · m=5 0.4713 · m=10 0.4802: at m=10, on a library where the median director has one or two
  credits, the encoding collapses onto the global mean and is indistinguishable from having no feature.
- Collaborative: `nb_similar_*` and `nb_recommendations_*` — the mean of the user's own ratings over the TMDB
  item-item neighbours the user has already rated. This is the only collaborative signal in the project.
  Everything else restates how good the crowd thinks the item is.
- Vibe: `pca_0…39` from the MiniLM embedding of Title + Genre + Director + Writer + Cast + Tagline, then
  Plot + Overview as the body. See §8 for a measured defect in this header.

**Three parsing bugs were fixed in this block on 2026-08-27.** None of them moved an offline metric, because
the metric was computed on the same corrupted frame:
- `imdb_votes` arrives from OMDb Indian-grouped (`'7,13,701'`); 747 of 1,062 rows contain a comma. A
  `dtype == 'object'` guard stopped firing under pandas' Arrow-backed `str` dtype, so 756 rows were
  median-filled. Distinct values went **305 → 1,051**, mean 148,566 → 382,697. The guard is now gone: the
  clean is unconditional, and a no-op on an already-numeric column.
- `runtime` arrives as `"87 min"` for 95 films. Parseable rows **967 → 1,062**.
- `rotten_tomatoes_rating` is stored as `"92%"`. Under the same dead guard it crashed the movie feature build
  outright. 949 real RT scores now parse.

**TV shows (56 features)**
- Numeric: `year`, `vote_average`, `vote_count_log`, `imdb_rating`, `imdb_votes_log`, `season_count`,
  `episode_count`, `runtime`, `is_adult`, `total_wins`, `total_nominations`.
- Categorical: `net_*` network one-hot (top-10 + Other), `g_*` genre multi-hot (≥5% of N),
  `ctry_*` origin-country one-hot.
- Relational: creator, actors and writer target encodings, built the same way as movies.
- Vibe: `pca_0…14` from Title + Network + Overview **+ the season narrative** (§3.7).

**Games (51 features)**
- Numeric: `year`, `metacritic`, `rating` (RAWG global), `ratings_count`, `reviews_count`, `playtime`.
- Categorical: `plat_*` platform one-hot, `gen_*` genre multi-hot.
- Relational: `developers_`, `publishers_`, `tags_` and `genres_encoded`, each with an `_n` support count.
  These replace the old `dev_*` multi-hot block.
- Vibe: `pca_0…14` from Name + Genres + Tags + Description.
- **The games text was corrupt until 2026-08-27.** 40 of 72 rows carried double-encoded UTF-8 in
  `description_raw` — which *is* the whole embedding body for games — and `God of War: RagnarÃƒÂ¶k` in the
  title. The cause was a bare `except: → encoding='latin1'` CSV fallback that writes the frame back as UTF-8,
  so each run that hits the fallback encodes one level deeper. Enrichment never repaired it, because
  enrichment only refills cells that are **empty**, and a mojibaked cell is not empty. 44 corrupt cells → 0.
  `src/common/parsing.py::read_media_csv` now repairs on the fallback path instead of propagating it.

**Books (29 features)**
- Numeric: `year`, `pageCount`, `averageRating`, `ratingsCount`.
- Relational: `authors_`, `categories_`, `publisher_`, `moods_` and `hc_tags_encoded`, each with an `_n`
  support count. `moods` and `hc_tags` come from Hardcover's `cached_tags`, which carries Mood
  ("reflective, dark, tense") and Tag ("Plot driven") alongside Genre; ingestion used to read only Genre.
- Vibe: `pca_0…14` from Title + Authors + Categories + Description.
- Books is the one domain where the text block carries most of the signal (§6.7), so the chunk-pooled
  encoder matters most here: 22% of book descriptions overran the encoder's 256-token window.

**Music (173 features, unsupervised)**
- Numeric: `duration_min`, `release_year`, `track_age`, `popularity`, `artist_popularity`,
  `artist_followers_log`, `album_total_tracks`, `track_number`, `explicit`.
- Audio: the 9 ReccoBeats metrics, each median-imputed with a `_missing` flag.
- Lyrical: VADER sentiment (`pos/neu/neg/compound`), word count, line count, unique-word ratio, `lyrics_found`.
- Categorical: `g_*` genres and tags (top-60), `a_*` artists (top-40).
- Vibe: `emb_0…31` from Name + Genres + MusicBrainz tags + a lyric snippet.
- The whole matrix passes through `StandardScaler` before clustering.

**Unified (239 features)**
- **Schema harmonization:** rename heterogeneous fields onto one backbone. Example: game `developers` →
  `director`, book `pageCount` → `runtime`, TV `created_by` → `director`, book `publisher` /
  TV `network` / game `publishers` → `studio`.
- **Identity and missingness:** `is_tv_show`, `is_game`, `is_book`, `is_music`, plus `has_<domain>_feats` masks.
- **Shared vibe:** one 10-d PCA over the chunk-pooled embedding of a fixed header
  (`Title: … Genre: … Tags: …`) plus the full description. Every domain fills the Tags slot from its own
  vocabulary — movie keywords, TMDB keywords + TVDB tags for TV, RAWG tags for games, Hardcover moods +
  tags for books — so the slot has the same shape everywhere. Filling it for only some domains would let the
  embedding separate domains by text *shape*, which is the confound the whole text pipeline exists to remove.
- **Shared creator/studio axis:** `creator_encoded` and `studio_encoded`, out-of-fold target encodings over
  the union of everyone credited (director **and** writer, so Bryan Lee O'Malley connects the Scott Pilgrim
  books, show and film even though he directed none of them). This block is new. It is the third-strongest
  feature in the unified SHAP table (§6.8) and it is also the source of the leakage in §8.
- **Shared geography axis:** `ctry_*` one-hot over the top six origin countries, plus `ctry_Other`. Every
  training row has exactly one `ctry_*` column set.
- **Domain alignment:** a `DomainAligner` (centroid, `movie` as reference) centres each domain's PCA block.
  The aligner fits **inside each training fold only**. This keeps the evaluation leakage-safe — and it is
  the reason `pca_*` is stored unaligned on disk. See §3.8; the ordering there is load-bearing.
- **Shared genre, quality and commercial channels:** cross-domain `gen_*` multi-hot, `critic_avg_5`,
  `box_office_log`, `popularity`.
- **Absent must mean NaN, not 0.** `build_universal_dataset` median-imputes every numeric column, so a book —
  which has no Metascore, no Rotten Tomatoes score, no `vote_average` and no `popularity` — trains with those
  columns at the *global* median. Reading them as 0 at inference handed the model a vector no book in training
  ever had, and moved book and game predictions by about 1.7 stars.
- Music is **not** pooled here, because music has no rating target. Turning it on would put 3,837 unrated rows
  in the build and refit the shared text PCA over them, which moves `pca_*` for every other domain and
  invalidates the benchmarks, the entity bridges and the pre-registered transfer verdict.

### 3.5 Transfer-grid feature subset
The transfer experiments use a **domain-blind** subset of the unified space:
- Keep `emb_*` (the full 384-dimensional sentence embedding), `gen_*` (top-40), `critic_avg_5`, `year`,
  `popularity`. The space was a 10-component PCA (`pca_*`) until 2026-08-29; it is now the whole embedding.
- **Drop** every `is_*` and `has_*_feats` column. Without this, a "transfer" model can split on domain identity
  and report a false transfer effect.
- **Known gap:** `_fit_predict` applies CORAL domain alignment only to columns named `pca_*`. Those no longer
  exist, so the alignment step selects an empty column set and silently does nothing. It is off identically for
  every cell, so comparisons between cells hold, but the grid below measures **unaligned** transfer.

### 3.5.1 Transfer-grid results — the whole grid
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

### 3.6 Text representation — chunk-pooled sentence windows
`src/common/text_embedding.py::encode_pooled` is now the **only** way any code in this project turns text
into a vector. This is the single largest representation change of the 2026-08 refresh.

**The problem.** `all-MiniLM-L6-v2` reads at most 256 tokens (`SentenceTransformer.max_seq_length == 256`).
Descriptions do not respect that, and they do not overrun it evenly. Measured against the cap, before the fix:

| Domain | Mean tokens | Over the 256-token cap |
| :--- | ---: | ---: |
| Games | 293 | **48.6%** |
| Books | 208 | 22.0% |
| Movies | 90 | 0% |
| TV | 90 | 0% |

Truncation was therefore discarding about half of the games corpus and none of the movies. That is not noise:
it is a **length confound that falls entirely on the smallest domain**, and it lands on exactly the axis the
cross-domain model is supposed to compare. Whatever the model learned about games, it learned from the front
of the blurb — which is where each source puts its boilerplate. The earlier mitigation, cutting every
description to its first 40 words, made this worse, not better: it is a positional cut that kept ~100% of a
movie plot and ~23% of a game blurb.

**The fix.** Split the description on sentence boundaries into windows of at most 180 tokens (leaving headroom
for the header), embed every window with the same per-item header prefixed, then average the window vectors
and re-normalize. The whole text is read. Because each window vector is unit length, the mean of one window
and the mean of eight are both unit vectors in the same space, so a short movie plot and a long game blurb
stay comparable **by construction** rather than by cutting them to a common length.

**And the movies row of the token table above was measuring the wrong thing.** It counts *body* tokens only. The per-item
header prefixed to every window carried the full billed cast — a median of 41 names, a median of 230 tokens
on its own — so 42.7% of movie headers overflowed the 256-token cap before a single word of plot was
appended, and 69.0% of movie header+window strings were truncated. Chunk-pooling was a no-op on the largest
domain until the header was capped in the 2026-08-28 run (§8.1). With people lists capped to 6 actors /
3 directors / 3 writers / 5 genres those rates are 0.0% and 0.1%, and the movie benchmark moved
MAE 0.4925 → 0.4863 on identical items.

**Measured, 5 seeds, text block alone, skill = 1 − MAE/MAE_mean**
(`src/experiments/text_norm_variants.py`, `reports/text_norm_variants.csv`):

| Variant | book skill | movie→book | movie→game |
| :--- | ---: | ---: | ---: |
| `prefix40` (the old positional cut) | 0.045 ±.006 | 0.060 ±.001 | 0.002 ±.001 |
| stopwords removed | 0.057 ±.007 | 0.035 ±.002 | 0.005 ±.000 |
| TF-IDF theme terms | 0.035 ±.004 | 0.012 ±.001 | −0.007 ±.001 |
| **chunk-pool mean (shipped)** | **0.098 ±.008** | **0.070 ±.002** | **0.007 ±.001** |

Books more than double, and *both* cross-domain transfers improve. Keyword extraction is worse, not better:
stripping a description to its most distinctive terms destroys the word order a sentence transformer encodes,
and `movie→game` goes negative. Reading the whole text is what makes domains comparable, not compressing it.
`chunk-pool max` scores marginally higher on books alone (0.110) but loses `movie→game` transfer, so the plain
mean ships.

**Where it is used.** Every path, without exception — this is the point of the change. The four domain feature
builders, all four `predict_ratings.py` batch scorers, the unified builder, the unified batch scorer,
`transform_single_media`, all five Oracles and the active-learning queues. A single-item form,
`encode_pooled_single`, exists so the Oracle path and the batch builder call literally the same function and
cannot drift apart; `scripts/audit_inference_parity.py` diffs the two paths on the real library and fails the
pipeline if they do.

### 3.7 TV season narrative
TV is the worst-served domain in the text pipeline. A TMDB show `overview` runs a **median of one sentence**,
so there is no narrative shape in it to read; an earlier experiment that embedded thirds of the description
looking for an arc found nothing, because there was nothing there.

`data/processed/shows/season_features.csv` already carries a per-season `season_overview`.
`src/shows/season_data.py::season_overview_by_show` concatenates them **in season order**
("Season 1: … Season 2: …", capped at 12 seasons so a 20-season procedural does not dwarf every other row) and
folds the result into `overview`. Seasons are labelled and joined in order rather than blended, so the
sentence windows the encoder builds land on act boundaries instead of straddling them.

Coverage is **119 of 175 shows (68%)**. Effect on the TV text, measured 2026-08-28:

| | before | after |
| :--- | ---: | ---: |
| Median words | 33 | **109** |
| Median sentences | 1 | **5** |
| Shows with ≥3 sentences | 22% | **74%** |

This is the only beat-ordered plot text the project owns that needs no new API and no generation step. It is
wired into every path that builds show text — the feature builder, the batch scorer, the TV Oracle (inference
and similarity), the show active-learning queue, the unified builder, the unified batch scorer and the unified
Oracle backbone. A path that skipped it would embed a one-line logline while the model was trained on the full
multi-season text, which is the exact train/serve skew class `audit_inference_parity.py` exists to catch.

### 3.8 The DomainAligner ordering trap
This is a property of the system, not a bug report, and anyone who rebuilds features must know it.

1. `build_universal_dataset` fits the shared PCA and writes `pca_*` to disk **unaligned, on purpose**.
   Fitting the aligner in the builder would fit domain means over the whole library, including the rows that
   later become test folds. That is leakage, so the builder must not do it.
2. The aligner is therefore fit **per fold**, on the training side only — in the evaluator
   (`comprehensive_evaluator.unified_oof_on_registry`) and again in the deployment trainer
   (`advanced_unified_model_trainer`).
3. The deployment trainer then writes its fitted aligner back into the preprocessor state
   (`state['aligner'] = aligner`) so that inference can apply it.
4. **And `build_universal_dataset` writes a fresh state dict with no `aligner` key at all.**

So: any feature rebuild that is not followed by a retrain silently disables alignment for every served
prediction. Nothing crashes. The model is fed `pca_*` in a space it was not fitted on.

Measured cost, mean |Δ prediction| on the deployed XGB, aligned against unaligned:

| Domain | mean \|Δpred\| |
| :--- | ---: |
| Movie | **0.000** |
| TV | 0.071 |
| Book | 0.127 |
| Game | 0.150 |

Movie is exactly zero because movie is the reference domain — the centroid map is the identity there. That is
why this hid for so long: the domain with 75% of the rows showed no symptom at all.

Three defences are now in place. The pipeline orders the unified batch scorer **after** the trainer (§5.6);
`predict_unified_ratings` prints a loud `WARNING: state has no 'aligner'` instead of scoring quietly; and
`scripts/data_sanity_check.py` has an aligner-presence check that fails the run.

**Corollary, and it is not obvious:** a constant per-domain offset between a stored `pca_*` column and a
re-derived one is the aligner, not skew. Only row-varying differences are real train/serve skew.

---

## 4. Model selection

### 4.1 Selection principle
- Model capacity follows N. Movies (N=1,163) support a tuned four-member stack. Games (N=66) and books (N=110)
  support only a single regularized learner.
- Every added component must win a **paired Wilcoxon test** on the frozen folds. If it does not win, remove it.

### 4.2 Deployed models
| Domain | Deployed model | Reason |
| :--- | :--- | :--- |
| Movies | Stacking ensemble: Optuna-tuned edge-penalty XGBoost + CatBoost (MAE loss) + RBF-SVR + ordinal expected value, fused by `RidgeCV` over 5-fold internal CV. | N is large enough to fit a meta-learner. In the 2026-10-01 run the blends no longer beat XGBoost alone (skill 0.410–0.411 against 0.414, see §6.6); the stack is kept because the gap is inside noise. |
| TV shows | Simplex-weighted stack: edge-penalty XGBoost + CatBoost + SVR. Weights are non-negative and sum to 1. | At N=175 a learned `RidgeCV` meta-learner overfits. A simplex average is more robust. |
| Games | `StandardScaler` + RBF-SVR (C=3.0), plus split-conformal intervals. | N=66. SVR is stable at small N and needs few hyperparameters. |
| Books | Same SVR pipeline (C=3.0) for the benchmark; the **Oracle predictor** is a text kNN (`KNeighborsRegressor`, k=8, cosine metric, distance weights) over the vibe PCA. | Book taste lives in the description text. The text kNN scores skill 0.43 against 0.24 for the crowd-mixing SVR. Blending the two hurts (≈0.39). The kNN artifact is **not** rebuilt by the pipeline — see §8. |
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
| Distillation prior (unified prediction as a local feature) | Paired Wilcoxon on frozen folds, 2026-10-01. Games p=0.317 (MAE 0.5894→0.5818), books p=0.465 (0.3491→0.3527). | **DROP.** Both cells are null. The prior moves games MAE very slightly the *right* way and books MAE the *wrong* way; in the 2026-08-28 refresh both moved the right way. That is the point: the books cell has read p=0.71, 0.034, 0.446, 0.336, 0.248 and now 0.465 across six refreshes. That is an unstable, underpowered cell, not an effect. **The feature is also switched off in code** — see §8. |
| Ridge stack for the unified model | Legacy 5-fold protocol. MAE 0.561, R² 0.384 against the mean ensemble 0.544 / 0.438 (frozen folds, 2026-10-01). | **REMOVED.** |
| Residual domain-correction heads | Legacy 5-fold protocol. MAE 0.609, R² 0.269. | **REMOVED.** |
| Custom edge-penalty XGBoost loss | Head-to-head on 50 folds, all else fixed. Overall MAE +0.0039 (p=0.083), edge MAE +0.0002 (p=0.61). | **RETIRE.** A small net harm. The weight keys on the label's extremity, not the error, so it is ordinary importance weighting. |
| Entity-bridge features (Wikidata links) | Paired Wilcoxon + bootstrap CI on 664 linked items. ΔMAE **+0.0361**, CI [+0.014, +0.057], **p=0.0014**. Adaptation slice: N=57, +0.0175, p=0.317. | **SIGNIFICANT** (2026-08-28, after the entity-link rebuild this row previously asked for). The QIDs were indeed wrong often enough to matter: movie↔book adaptations went 2 → 18 after the `resolve_qid` rewrite while `n_linked` held at ~664. Adaptation-only slice still underpowered. Graph and corpus changed in the same run and have not been separated. |
| Rank calibration for TV | Spearman ρ=0.565 (2026-10-01) is still too weak; calibration costs ~10 accuracy points. | **SKIPPED for TV.** Applied to movies, games and books only. |

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
- The registry covers the rated universe only: **1,514 of the 1,520 unified rows** (2026-10-01). Music rows are excluded
  because music has no label, and `create_frozen_folds` now also filters `target_reg > 0`, which removes the
  six unrated games that a `fillna(0)` had disguised as genuine 0★ items (§2.3).
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
| SVR | Fixed | RBF kernel, ε=0.1, inside a `StandardScaler` pipeline. C=1.0 as a stack member in movies and TV; C=3.0 as the standalone learner in games and books (nested-CV tuned) |
| Music KMeans | Silhouette sweep | k selected by the best silhouette score |

After tuning, the movies trainer refits the final XGBoost with 300 trees at learning rate 0.03.

### 5.5 Leakage controls
- Target encodings use out-of-fold means with Bayesian smoothing. The inner OOF is load-bearing: a control
  that dropped it scored MAE 0.6243 against 0.5075 for simply deleting the columns. **But that inner OOF is
  not the evaluation fold — read §8 before you trust any encoded-feature model's reported score.**
- The `DomainAligner` fits on the training fold only (§3.8).
- The rank calibration reference uses training-fold predictions only.
- The evaluator asserts `benchmarks != slices` and `N == n_unique_items`, and it derives the expected slice
  counts from the data rather than hard-coding them, so the check survives library growth. These assertions
  exist because two earlier measurement bugs each reversed a scientific conclusion.
- `scripts/data_sanity_check.py` gates the data itself (exit 1 on failure): row-count agreement across
  enriched → training → unified, persisted imputation, parse health *after* the pipeline's own cleaner,
  mojibake, rating-grid sanity and aligner presence. It took the repo from 11 FAIL to 1 on 2026-08-28, and
  the surviving failure is the aligner check correctly saying "do not score until the trainer runs". It is
  **not** in the pipeline plan; run it by hand.

### 5.6 One-command refresh
Run `python run_pipeline.py`. The script executes each stage as its own `python -m` process. The plan is
deterministic for a given set of flags, and `--dry-run` prints it without running anything. With
`--no-ingest --full` it is **36 numbered steps**; the same run with ingestion is 44.

Phase order: merge-ingest → enrich → per-domain features → unified features → frozen fold registry →
clear OOF caches → evaluate → distillation ablation → re-evaluate (absorb distillation) → unified trainer →
**unified batch predictions** → per-domain trainers → per-domain batch predictions → **four diagnostics** →
render docs → paper figures → (with `--full`) transfer grid, entity links and bridges, learning curves,
active-learning queue.

Two orderings in that list are not cosmetic, and both were wrong until 2026-08-27.

**Unified batch predictions is step 12** (`--no-ingest --full` numbering): after the unified trainer and
before the per-domain trainers. It has to be *after* the trainer because the trainer is the only thing that
restores the `DomainAligner` into the preprocessor state (§3.8) — scoring before it serves unaligned `pca_*`.
It has to be *before* the per-domain trainers because those read
`data/predictions/unified/*.csv` for the distillation prior. `src.unified_model.predict_unified_ratings` was
**never in the plan at all**: the predictions file on disk was two months older than the features, which is
why games/books distillation silently never ran on fresh numbers.

**The four diagnostics are steps 23–26**, after the batch predictions and before `render_docs`.
`feature_diagnostics`, `classification_diagnostics`, `baseline_sanity` and `bias_variance` were *read* by
`render_docs` but *produced* by nothing in the plan, so the README and the technical report were republishing
2026-08-16 SHAP tables and confusion matrices next to freshly computed benchmarks, with nothing to say they
came from a different library. They read the OOF CSVs, so they must run after the evaluator.

`--full` adds the 50-fold transfer grid, the entity links and bridges, the learning curves and the
active-learning queue. Note that the entity-link rebuild (step 31) is where a `resolve_qid` fix lands: the
old resolver fell back to `cands[0]` whenever nothing matched the type hint and returned **Batman Province,
Turkey** for the film *Batman*, the 1984 *song* for *Ghostbusters*, a 1986 *video game* for *Top Gun* and the
*franchise* entity for *Toy Story*. The rewrite widens the candidate list to 20 (the film is not in the top 5
for "Batman" at all), matches type words positionally, rejects franchise and song phrases, disambiguates by
year, and returns `None` rather than guessing. QID resolution fell 96.8% → 92.6%, and the fall is the point.
`entity_links.csv` is built on those QIDs, so the entity-bridge result in §6.9 predates the fix.

The script stops at the first failing step and prints the cause. `--from-step N` resumes, but the numbering
shifts with the flags, so pass the same flags.

---

## 6. Evaluation

### 6.1 Protocol
- 50 frozen folds (10-fold × 5 repeats), identical for every model.
- **Per-item deduplication:** average the raw predictions of the 5 repeats, then round to the 0.5★ grid.
  Therefore `N` equals the number of unique items, not the number of out-of-fold rows.
- Reported metrics: R², MAE, ±0.5★ accuracy, skill score, Spearman ρ, and the classification scores.

### 6.2 Headline benchmark (production models, frozen folds)
`reports/latest_metrics.json` and `reports/baseline_sanity.json`, 2026-10-01. Skill = 1 − MAE / MAE of
predict-the-mean; it is the primary metric (see the header of this document).

| Domain | N | Model | Skill | Spearman ρ | MAE | R² | ±0.5★ accuracy |
| :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| Movies | 1,163 | Prod Mean-Ensemble (edge+Cat+SVR+Ord, residual) | 0.355 | 0.751 | 0.483 | 0.530 | 77.1% |
| **Unified (rated)** | **1,514** | **Mean Ensemble** | **0.287** | **0.674** | **0.542** | **0.445** | **73.1%** |
| TV shows | 175 | Prod Simplex-Stack (edge+Cat+SVR) | 0.208 | 0.565 | 0.694 | 0.271 | 60.6% |
| Games | 66 | Local SVR | 0.376 | 0.719 | 0.515 | 0.497 | 74.2% |
| Books | 110 | Local SVR | 0.566 | 0.859 | 0.286 | 0.715 | 88.2% |

**Unified per-domain slices** — the same unified ensemble, cut by domain. These are *not* the rows above.
Skill uses the same per-domain mean baseline as the local rows, so the two tables compare directly:

| Domain | N | Skill | MAE | R² | ±0.5★ acc |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Movie | 1,163 | 0.315 | 0.513 | 0.498 | 75.6% |
| TV | 175 | 0.186 | 0.714 | 0.161 | 59.4% |
| Game | 66 | 0.202 | 0.659 | 0.359 | 62.1% |
| Book | 110 | 0.241 | 0.500 | 0.388 | 75.5% |

**How to read this table.**
- **Skill** is the primary yardstick. It is 0 for predict-the-mean, so the clustered rating curve cannot
  inflate it. **Spearman ρ** shows whether the model orders items.
- **R²** is unreliable in the small domains. The rating variance there is tiny, so a few noisy items move R²
  a long way. Do not rank domains by R².
- **MAE** is secondary. An MAE near 0.5★ equals one grid step, and the label itself wobbles by about one grid
  step. But a constant 3.5★ already scores MAE 0.749 on the films, so MAE alone does not prove learning.
- **±0.5★ accuracy** looks high in books (88.2%) partly because book ratings cluster tightly. Compare it with
  the mode baseline in §6.3 before you read it as skill — the mode baseline for books is 55.5%, so the lift
  is a genuine +32.7 points.
- **Pooling now loses in every domain (changed in the 2026-10-01 run).** The local model beats the unified
  slice on movies (0.483 against 0.513), TV (0.694 against 0.714), games (0.515 against 0.659) and books
  (0.286 against 0.500). In the previous run the unified slice won movies (0.475 against 0.486) and TV
  (0.666 against 0.680) by narrow margins. The TV margin (0.020) is still inside TV's fold-reshuffle noise.
  The books gap is long-standing and large; the standalone books model has had `authors_encoded` since the
  beginning, and it reads a text block that carries most of the book signal.

**What changed in the 2026-10-01 run.** The corpus grew by 101 films (Letterboxd export of 2026-10-01, 42 of
them standup comedy specials), 17 books and 2 shows; movie plot text now uses the full Wikipedia plot (~5
embedding windows per film); and the fold registry was regenerated. Against the last committed run
(2026-08-28):

| Domain | N | Skill | MAE | Spearman ρ | ±0.5★ acc |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Movies | 1,062 → 1,163 | 0.362 → 0.355 | 0.484 → 0.483 | 0.755 → 0.751 | 77.0% → 77.1% |
| TV | 173 → 175 | 0.258 → 0.208 | 0.656 → 0.694 | 0.583 → 0.565 | 63.6% → 60.6% |
| Games | 66 → 66 | 0.376 → 0.376 | 0.515 → 0.515 | 0.735 → 0.719 | 75.8% → 74.2% |
| Books | 93 → 110 | 0.504 → 0.566 | 0.328 → 0.286 | 0.844 → 0.859 | 88.2% → 88.2% |
| Unified | 1,400 → 1,514 | 0.329 → 0.287 | 0.526 → 0.542 | 0.694 → 0.674 | 75.7% → 73.1% |

- **The movie dip is the new mix, not a worse model.** On the same 1,062 films as before, movie skill is
  0.362, unchanged. The 101 new films score skill 0.44; the 42 standup specials among them score 0.51.
- **TV fell by 0.05 skill.** The regenerated registry adds about 0.02 MAE of fold-reshuffle noise to TV. The
  rest is not isolated.

**Historical: what changed in the 2026-08-28 snapshot, and what may not be attributed to it.** Against
`reports/latest_metrics.BEFORE_creator.json` (2026-08-22):

| Domain | before MAE (N) | after MAE (N) | Δ | like-for-like? |
| :--- | ---: | ---: | ---: | :--- |
| Movie | 0.4920 (1062) | 0.4863 (1062) | −0.0057 | **yes** — the same 1,062 items |
| Game | 0.5152 (66) | 0.5303 (66) | +0.0151 | **yes** — the same 66 items |
| Unified | 0.5313 (1407) | 0.5025 (1407) | −0.0288 | **no** — see below |
| TV | 0.6676 (173) | 0.6800 (175) | +0.0124 | no — 2 new shows |
| Book | 0.3050 (100) | 0.2981 (104) | −0.0069 | no — 4 new books |

- **The parsing fixes were flat; the header cap was not.** The movie move splits in two. The five parsing
  fixes — `categorize_rating`, `"87 min"` runtimes, comma-grouped `imdb_votes`, the `lang_en`/`lang_English`
  split, 121 embeddings containing the literal token `nan` — took MAE 0.4920 → 0.4925 on identical items.
  They made the features **correct** without making the model measurably better, which is itself a real and
  reportable result: correctness and predictive lift are different things, and several of those columns are
  low-weight for this model. Capping the movie embedding header then took the same 1,062 films
  **0.4925 → 0.4863**, R² 0.5133 → **0.5226**. That is the one clean, attributable win of the cycle, and it
  means chunk-pooling had been a no-op on the largest domain until now (§3.6, §8.1).
- **Games are the same 66 items and slightly worse** (0.5152 → 0.5303). At N=66 a 0.015 MAE move is well
  inside noise. Do not claim a direction.
- **Normalising the unified language axis cost +0.0046 MAE** (0.4979 → 0.5025 on identical items), and that
  is the honest outcome, not a regression to revert. `lang_en` carried movies and TV (TMDB ISO codes) while
  `lang_English` carried mostly books (OMDb/Hardcover names), so the split column was a partial **domain
  indicator**: a model whose thesis is cross-domain generalisation was reading domain identity off a feature
  meant to bridge domains. Merging them removes a confound and gives up 0.0046 MAE for it.
- **The unified 1,407 → 1,407 match is a coincidence, not a controlled comparison.** Before: 1,062 + 173 + 72
  + 100. After: 1,062 + 175 + 66 + 104. Six unrated games left and eight new items arrived. Same total,
  different item set. Part of the −0.029 is certainly the six zero-target games leaving the registry
  (measured separately on the old OOF: 0.5153 → 0.5029). The rest is confounded with the changed item set.
- **TV, game and book MAE are byte-identical across the two builds of this run, and that is not a caching
  bug.** Calibrated predictions are snapped to the half-star grid, so MAE is quantised to multiples of
  0.5/N. Equal counts of half-star errors give an identical MAE with a different continuous R² — which is
  exactly what TV does here: MAE unchanged at 0.6800, R² 0.2802 → 0.2826.

### 6.3 Baseline sanity — is the model only predicting the mean?
`reports/baseline_sanity.json`, regenerated 2026-10-01.

| Model | N | MAE | MAE (mean) | Skill | ±0.5★ acc | acc (mode) | Acc lift | Spearman ρ | p |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 1,163 | 0.483 | 0.749 | +0.355 | 77.1% | 51.1% | +26.1 pts | 0.751 | 1e-211 |
| TV shows | 175 | 0.694 | 0.877 | +0.208 | 60.6% | 46.9% | +13.7 pts | 0.565 | 4e-16 |
| Games | 66 | 0.515 | 0.826 | +0.376 | 74.2% | 47.0% | +27.3 pts | 0.719 | 1e-11 |
| Books | 110 | 0.286 | 0.659 | +0.566 | 88.2% | 55.5% | +32.7 pts | 0.859 | 3e-33 |
| Unified | 1,514 | 0.542 | 0.761 | +0.287 | 73.1% | 50.1% | +23.1 pts | 0.674 | 6e-201 |

**Interpretation of each metric.**
- **Skill score** = `1 − MAE_model / MAE_mean`. A value above 0 means the model beats a constant predictor.
  All five models score +0.21 to +0.57. This is the project's primary metric, because it stays meaningful
  when the rating variance is small.
- **Acc lift** is the ±0.5★ accuracy above the mode baseline. It removes the accuracy that the clustered
  rating distribution gives away for free. The lift is +14 to +33 points everywhere.
- **Spearman ρ** is the load-bearing second metric. A mean-hugging model has ρ≈0 and cannot order items.
  Every model reaches ρ 0.57–0.86 with p ≤ 1e-11. The models therefore rank favourites above passes.
- **Books is the strongest model on every one of these columns** (skill +0.57, ρ 0.86, 88.2% within half a
  star) on N=110. Treat it as a strong pilot, not a settled result: the same cell has moved a long way between
  refreshes, and the entity-encoding defect in §8 applies to `authors_encoded` here as much as anywhere.
- **Caveat:** the small domains still compress the prediction magnitude. The spread ratio (predicted SD over
  true SD) is 0.58 for TV, 0.69 for games, 0.81 for books and 0.66 for the unified model. Movies is the
  exception at 1.02, because calibration already restored its range. Rank calibration (§4.3) corrects the rest
  at serve time.

### 6.4 Classification performance
`reports/classification_diagnostics.json`, regenerated 2026-10-01.

| Model | N | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 | Cohen κ | Binary-Like F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Movies | 1,163 | 65.8% | 66.6% | 0.666 | 0.658 | 0.477 | 0.821 |
| TV shows | 175 | 50.9% | 50.3% | 0.508 | 0.524 | 0.226 | 0.820 |
| Games | 66 | 62.1% | 56.1% | 0.569 | 0.634 | 0.355 | 0.880 |
| Books | 110 | 77.3% | 75.6% | 0.769 | 0.773 | 0.636 | 0.915 |
| Unified | 1,514 | 60.0% | 59.6% | 0.605 | 0.604 | 0.381 | 0.806 |

**Interpretation.**
- **Cohen κ** 0.23–0.48 for movies, TV, games and the unified model means fair to moderate agreement above
  chance. The 3-class task is genuinely hard, because the Neutral band is only one star wide. TV is the
  weakest at κ 0.23.
- **The Neutral class is the weakest class for movies, TV and the unified model.** Movie Neutral F1 is 0.575,
  against 0.717 for Dislike and 0.707 for Like. The model pushes middle items toward the edges of the band.
  For games and books the Dislike class is weaker (F1 0.462 and 0.720).
- **Balanced accuracy below plain accuracy** (games 56.1% against 62.1%) shows class imbalance. Games hold 38
  Like items and 8 Dislike items. Games Dislike recall is 0.375 at precision 0.600 — the model called five
  games bad, and three of them were. It is not a reliable Dislike detector for games.
- **Books is the exception and the outlier** (κ 0.636, macro-F1 0.769, Neutral F1 0.775). At N=110 with 26
  Dislike items, read it as a pilot result.
- **Binary-Like F1 is 0.81–0.92 everywhere.** This is the metric that matters in production, because the
  user's real decision is "watch or skip". The binary decision is reliable even where the 3-class decision
  is not.

### 6.5 Bias-variance decomposition
`reports/bias_variance.json`, regenerated 2026-10-01. This harness used to score the unified row on every
unified row, because it filtered on `target_reg.notna()` while the six unrated games carry
`target_reg == 0.0` rather than `NaN`. That is fixed: it now excludes them like the registry, the unified
trainer and the evaluator do, so the unified row below is the same **1,514** rated items as everywhere else.

| Model | N | Features | bias² | variance | Bias share | Test MAE | Diagnosis |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Movies | 1,163 | 106 | 0.397 | 0.025 | 94.2% | 0.485 | Feature and noise ceiling; curve flat. |
| TV shows | 175 | 56 | 0.929 | 0.052 | 94.7% | 0.766 | Bias-limited, curve still rising. |
| Games | 66 | 51 | 0.663 | 0.051 | 92.9% | 0.604 | Bias-limited, curve still rising. |
| Books | 110 | 29 | 0.216 | 0.030 | 87.7% | 0.321 | Bias-limited, curve rising fastest of the five. |
| Unified | 1,514 | 406 | 0.523 | 0.028 | 95.0% | 0.571 | Bias-limited pooling tool; curve flat. |

**Interpretation.** Every model is bias-dominated (88–95%) with variance near 5–12%. The models are stable,
not data-starved. More rows of the same features is a weak lever in movies and the unified model, where the
learning-curve edge slope is 0.0025 and 0.0068. It is a real lever in books (0.061), TV (0.024) and games
(0.018), where the curve has not flattened. Richer features and the irreducible label noise set the limit
everywhere else.

**Direct evidence that TV is feature-limited, not data-limited.** The creator, cast and writer target
encodings that make movies strong were added to the weakest domain, TV. The bias² fell from 1.014 to 0.963 on
the corpus of the time, stood at 0.928 on the 2026-08-28 corpus and 0.929 on 2026-10-01; the test MAE went
0.784 → 0.778 → 0.771 → 0.766. At N=175 most creators appear once, so the smoothed encoding collapses to the
global mean. The season narrative (§3.7) is the first genuinely new *feature* TV has received; its effect is
not separable from the other changes in its cycle, so no attribution is claimed here.

**Why the metrics look modest — three compounding causes, largest first.**
1. **Aleatoric ceiling.** The ratings are self-reported on a coarse grid. The same item can receive 3.5★ or
   4.0★ on a different day. The decomposition counts this noise inside bias².
2. **The features measure crowd quality, not personal taste.** IMDb scores and vote counts capture "is this
   generally good". The user's deviation from the crowd is the residual the features cannot see.
3. **Small N** in TV, games and books blocks person-specific patterns.

### 6.6 Per-member ablation (are the ensembles complementary?)
Skill vs the mean baseline, 50 frozen folds, all rows uncalibrated. `reports/feature_diagnostics.json`,
2026-10-01.

| Domain | XGB alone | CatBoost alone | SVR alone | Mean blend | Simplex blend | Reading |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Movies | **0.414** | 0.395 | 0.382 | 0.411 | 0.410 | No blend gain. XGB alone beats the best blend by +0.003. |
| TV shows | 0.219 | 0.229 | 0.164 | **0.232** | 0.216 | Barely complementary. The mean blend beats the best member by +0.003 and SVR-alone by +0.068; the simplex blend is below CatBoost alone. |

**Three readings have now been withdrawn here, so treat this cell as unstable.** An older corpus put TV's
SVR-alone (0.110) ahead of the blend (0.095) and the docs recommended replacing the TV stack with a plain SVR;
that was withdrawn. One refresh then put the *simplex* blend ahead of the *mean* blend in both domains, the
next put the mean blend ahead in both, the 2026-08-28 corpus put the simplex blend ahead again, and the
2026-10-01 corpus puts the mean blend ahead in TV and XGB alone ahead in movies. The ordering
**blend > best single member**, which had survived every earlier refresh, broke for movies in the 2026-10-01
run. Differences of this size (≤0.004 skill) are not resolvable at these N. What still survives every refresh
is that SVR is the weakest member in both domains. Do not read the blend-versus-blend column.

### 6.7 Where does taste live — text or crowd?
`reports/text_vs_crowd.json`. **Vintage 2026-07-11** — this predates the chunk-pooled encoder, the season
narrative and the current corpus, and it has not been re-run. The direction is almost certainly still right;
the magnitudes are not current.

| Domain | Text-only skill | Crowd-only skill | Verdict |
| :--- | ---: | ---: | :--- |
| Movies / TV / Unified | 0.05–0.07 | 0.26–0.32 | The user tracks critic consensus. Text cannot rescue these domains. |
| Games | 0.11 | 0.12 | Neither block alone; the full matrix wins (0.20). |
| Books | **0.39** | 0.17 | Book taste lives in the description and author text. |

This finding drove the books text-kNN Oracle model (skill 0.43 against 0.24 for the SVR). §3.6 measured the
same effect from the other side: chunk-pooling more than doubles the book text-block skill, 0.045 → 0.098.

### 6.8 SHAP feature attribution (mean |SHAP| on a per-model XGB surrogate)
`reports/feature_diagnostics.json`, regenerated 2026-10-01.

| Model | Top features |
| :--- | :--- |
| Movies | `critic_avg_5` (0.261), `imdb_rating` (0.178), `vote_average` (0.062), `actors_encoded` (0.058), `age_at_rating` (0.044) |
| TV shows | `imdb_rating` (0.364), `imdb_votes_log` (0.092), `vote_average` (0.080), `pca_4` (0.064), `year` (0.059) |
| Games | `ratings_count` (0.268), `pca_1` (0.159), `rating` (0.110), `playtime` (0.091), `reviews_count` (0.063) |
| Books | `pca_1` (0.227), `pca_5` (0.084), `pca_2` (0.066), `authors_encoded` (0.060), `pca_3` (0.053) |
| Unified | `critic_avg_5` (0.426), `creator_encoded` (0.094), `emb_334` (0.022), `emb_192` (0.014), `emb_93` (0.014) |

**Interpretation.** Movies and TV are critic-consensus models. Games depend on RAWG community columns that
live **outside** the shared space — `ratings_count`, `rating`, `playtime` and `reviews_count` are four of the
top five — which is the clearest explanation of why games still need a local model. Books are embedding-led:
four of the top five are PCA components, which agrees with §6.7.

Two changes are worth naming. `creator_encoded` entered the unified table at rank 3 on its first run
(2026-08-28) and is rank 2 in the 2026-10-01 run. And `playtime` entered the games table having previously
been fetched and never read.

**Caveat that applies to the unified row specifically.** `creator_encoded` is the column with the fold-boundary
leak described in §8. Its SHAP weight is therefore an upper bound on what it would be worth under a clean
protocol.

### 6.9 Cross-domain transfer result
`reports/transfer_grid_summary.json`. **Vintage 2026-08-30.** This predates the 2026-10-01 corpus (101 more
films, Wikipedia plot text). Re-run it with `--full` before quoting it as current. The pre-registered rule below was fixed before any result was
seen and must not be re-tuned when it is re-run.

- **Design.** Pre-registered decision rule, fixed before any result was seen: a finding is positive only if the
  augmented lift is > 0 with paired p < 0.05 at **at least 2** of the 4 target fractions.
- **Result (2026-08-30 grid, the latest).** **TASTE TRANSFERS**, into games only. Six source subsets clear the
  pre-registered bar for the games target; no subset clears it for movies, TV or books (TV `book` reaches
  significance at 1 of 4 fractions only).

  | Source → Games | lift @100% | significant fractions |
  | :--- | ---: | :--- |
  | `movie` | **+0.145** | **4/4** (+0.102 p=0.0002 · +0.101 p=0.0005 · +0.124 p<0.0001 · +0.145 p<0.0001) |
  | `movie+book` | +0.136 | 4/4 |
  | `movie+tv+book` | +0.109 | 4/4 |
  | `movie+tv` | +0.106 | 4/4 |
  | `tv` | +0.095 | 2/4 |
  | `tv+book` | +0.088 | 2/4 |

- **Previous result (2026-08-28 corpus).** **NULL** - no source subset clears the pre-registered bar for any target. Best cells: movie `tv+game+book` +0.0040 (p=0.117), tv `book` +0.0068 (p=0.309), game `book` +0.0395 (p=0.250), book `tv+game` +0.0197 (p=0.183). Zero-shot lifts are significantly NEGATIVE (movie -0.108, book -0.103, both p<0.001): pooling hurts a target with no data of its own. This REVERSES the previous run, which reported taste transferring into games via five subsets. The corpus changed in three ways that could each bear on it - six unrated games with `target_reg = 0.0` left the registry (Games was the target of every old positive cell), the split language column was merged, and the movie header was capped - and these have **not** been separated experimentally.

  The positive table from the run before that (2026-07-25) is superseded by the 2026-08-30 table above.

- **Negative targets.** Movies, TV and books show no significant positive transfer (0 of 4 fractions each).
- **Zero-shot affinity is asymmetric** (2026-08-30 grid). Every source is positive *into* games (+0.239 movie,
  +0.103 TV, +0.177 book). Games as a *source* is mildly negative into movies (−0.069) and books (−0.110) and
  about zero into TV (+0.014). Games is a good target and a poor source. (The 2026-07-25 grid had games as a
  strongly negative source: −0.297 movies, −0.458 TV, −0.362 books.)
- **Four stress controls passed.** (1) Domain-identity columns dropped. (2) The transfer model beats every
  featureless constant baseline (MAE 0.99 vs 1.05–1.17) and rank-tracks true ratings at Spearman +0.47, so it
  carries content, not a better prior. (3) Text-length normalization was necessary; on raw-length embeddings
  the effect measured only +0.05 and was **not** significant. (4) A label-permutation placebo is decisive:
  shuffling the source ratings collapses the lift from +0.132/+0.101/+0.122/+0.134 to
  −0.004/−0.025/−0.015/+0.053, with the real-vs-placebo gap significant at every fraction (p ≤ 0.011;
  p < 0.0002 at 0–50% target data).
- **What predicts transfer.** Aligned-space distance does **not** predict transfer (Spearman +0.191 over n=12
  pairs in the 2026-08-30 grid; the sign has moved between runs, so this is noise). Target data-starvation does.
- **Caveat on the p-values.** The 50 repeated-CV folds are not independent — with 66 unique games, items recur
  across folds. The p-values are optimistic. The trustworthy signal is consistency across all four fractions.
- **Reconciliation with §6.2.** The game rating has a domain-specific part (Metacritic, platforms) that only
  the local SVR sees, and a genre/vibe part that lives in the shared space and transfers. Both facts hold.

### 6.10 Known open items
- **The Books distillation dissent is resolved, in the direction of DROP.** The artifact and the project
  verdict now agree: p=0.465 (2026-10-01), verdict DROP (§4.4). The cell remains underpowered; do not read the agreement
  as evidence.
- **The Games baseline still differs between harnesses.** The benchmark harness reports MAE 0.515 for the
  local SVR; the distillation harness reports 0.589 for the same nominal model without the prior (2026-10-01). The two
  differ in preprocessing and in the calibration step, not in the fold registry. This changes the measured
  size of the calibration win, not the decision.
- **The edge-loss ablation predates the current corpus** (2026-06-26, 1,000 movies). Re-run it.
- **The transfer grid, the entity bridges and the text-vs-crowd study all predate this corpus.** See §6.11.
- Games and books remain pilots at N=66–110. Report skill and conformal intervals, not R².

### 6.11 Artifact provenance
Not every artifact shares one vintage. Read each number with its date.

| Vintage | Artifacts |
| :--- | :--- |
| **2026-10-01** (current corpus) | `latest_metrics.json`, `distillation_ablation_results.json`, `baseline_sanity.json`, `classification_diagnostics.json`, `feature_diagnostics.json`, `bias_variance.json`, `production_oof.csv`, `standalone_oof.csv`, `oof_predictions.csv`, all four training frames, the unified training frame |
| **2026-08-30** (previous corpus) | `transfer_grid_summary.json`, `models/books/text_knn.joblib` |
| **2026-08-28** (previous corpus) | `entity_bridge_results*.json`, conformal widths (see §7.3) |
| **2026-07-11** (previous corpus) | `text_vs_crowd.json`, `text_lift.json` |
| **2026-06-26/27** (previous corpus) | `edge_loss_ablation_results.json`, `transfer_placebo_control.json`, `edge_boost_ablation.json` |

The four diagnostics were, until 2026-08-27, produced by nothing in the pipeline plan while `render_docs`
kept publishing them. That is why the vintage of the second row used to be older than the first. §5.6.

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

**The batch scorers are a distinct code path from the feature builders, and that is the project's main source
of production bugs.** A scorer that forgets to build a training column does not crash; it emits a zero, which
is indistinguishable from a feature that is legitimately zero. `predict_unified_ratings.py` was doing this to
the whole creator/studio block and the whole `ctry_*` block: `creator_encoded` and `studio_encoded` sit on a
~3.3★ scale and are never 0 in any of the 1,407 training rows, and every training row has exactly one `ctry_*`
column set. Measured on the deployed XGB, the zero-fill cost **mean |Δpred| 0.52 stars, max 1.83**. Fixed in
the same pass: `tags` was never populated, so 173 of 173 TV rows and 71 of 104 books were embedded as
`"Tags: ."`; the `DomainAligner` was never applied (§3.8); comma-grouped numbers were not stripped before the
median fill; genre one-hots were built without `canon_genre`, so 115 rows lost bits; and a stale local
`categorize_rating` put 151 rows (10.7%) in the wrong bucket.

Two defences now exist. `predict_unified_ratings` scores every row through the Oracle's `transform_single_media`,
which **raises** on any training column it did not build. And `scripts/audit_inference_parity.py` diffs the batch builder against the
single-item inference path on the real library.

The equivalent defence does **not** yet exist in `movies/predict_ratings.py`. See §8.

### 7.2 Serve-time calibration
- The calibration maps each raw prediction's rank among the reference predictions onto the empirical quantiles
  of the user's true ratings. It needs at least 20 rated reference items, otherwise it returns the raw value.
- The map is monotone, so it never changes the ranking. It only restores the output distribution.
- Measured effect (**vintage 2026-07, not re-measured on this corpus**): games MAE 0.652 → 0.598 (ρ=0.71);
  books near-neutral, buying 5★ coverage (ρ=0.70); movies trade about 4 points of ±0.5★ accuracy for 45
  genuine 5★ predictions (MAE 0.474 → 0.509, ρ=0.74); TV skipped (ρ=0.59 is still too weak, and calibration
  costs ~10 accuracy points). The current-corpus ρ values are movies 0.751, TV 0.581, games 0.738, books
  0.872 (§6.3), so the TV decision is unchanged and the games and books cases have both got stronger. The
  MAE deltas themselves need re-measuring.
- Two conditions must hold, not one: the model must **rank** well *and* have **compressed** its range. Movies
  ranks well but was already spread, so calibration there is a deliberate accuracy-for-coverage trade.

### 7.3 Uncertainty quantification
- Every domain has an 80% **split-conformal** interval width: the 80th percentile of the absolute OOF
  residuals. The Oracle shows it directly, for example "3.5 ± 1.0".
- Widths recomputed 2026-08-28 as the 80th percentile of |OOF residual|, taking movies and TV from
  `reports/production_oof.csv` and games and books from `reports/standalone_oof.csv` — in each case the OOF
  of the model in the §6.2 benchmark table:

  | Domain | Width | In-sample fraction covered |
  | :--- | ---: | ---: |
  | Movies | ±0.90 | 80.9% |
  | TV shows | ±1.01 | 80.0% |
  | Games | ±0.85 | 80.3% |
  | Books | ±0.50 | 83.7% |

- The coverage column is close to 80% **by construction** — the width is a quantile of the same residuals it
  is scored on. It is a consistency check, not evidence of coverage. Books over-covers because at N=104 on a
  0.5★ grid the residuals take few distinct values and the quantile lands between them.
- The books width halved (±0.71 → ±0.50) as the books MAE fell from 0.472 to 0.298. The games width fell
  from ±1.00 to ±0.85.
- The width is constant inside a domain. That is a property of vanilla split conformal, which guarantees
  marginal — not conditional — coverage.
- **Caveat:** this is an in-sample OOF approximation and is optimistically biased. Certifying coverage needs a
  held-out calibration set that does not exist at this N. Treat the widths as approximate.
- The games and books trainers also save `conformal_width_80`, `skill_score`, `mae`, `r2` and `train_mean` in
  `model_meta.joblib`, computed on their own 5-fold partition rather than the frozen registry. As of
  2026-08-28 those read books ±0.66 (skill 0.536, MAE 0.405) and games ±0.84 (skill 0.115, MAE 0.821).
  They are the reference point for the next retraining cycle, and they differ from the registry-OOF widths
  above — substantially so for games, where the two partitions disagree by 0.29 MAE. That gap is a small-N
  artifact, not a bug, and it is the reason this project reports registry-fold numbers everywhere else.

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
- **Run the data gate first.** `python -m scripts.data_sanity_check` exits 1 on a failure and is the cheapest
  way to catch a parse regression before a four-hour run consumes it.
- **Not auto-updated.** Hand-written prose in the README and the technical report. Review it if the numbers
  moved materially. Note that `render_docs` prose contains hardcoded numbers and verdicts *inside* the
  auto-generated marker blocks (for example "0.476 local vs 0.475 unified", "it was tested and dropped")
  which do **not** recompute. Those are stale until edited by hand, and they sit next to numbers that are not.

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

## 8. Limitations and open defects

### 8.1 Defects that make a reported number wrong
Items 1–3 are known and **unfixed**, and each one changes how a number in this document must be read.
Items 4–6 were fixed in the 2026-08-28 run; they stay here with what the fix measured, because in two of the
three cases the measurement is the interesting part.

**1. Entity encodings leak across the fold boundary. Every reported unified MAE and R² is optimistic by an
unmeasured amount.**
`src/common/taste_features.py::oof_entity_encode` builds `creator_encoded` (and every other `*_encoded`
column) with its **own internal 5-fold `KFold`, seed 42**, which has no relationship whatsoever to the frozen
fold registry the models are then evaluated on. Two consequences compound:
- **48% of train rows** have a `creator_encoded` value computed from the target of a row that the evaluation
  fold holds out as test. The inner OOF prevents a row from seeing *its own* rating. It does nothing about
  the rows that are its evaluation-fold neighbours.
- The column is computed **once, at feature-build time, and written to the CSV**. All 50 evaluation folds
  then reuse the same values. There is no per-fold recomputation anywhere.

This is not confined to the unified model. Movies (8 crew encodings + `dir_genre_encoded` + 2 neighbour
columns), TV (creator, actors, writer), games (4) and books (5) are all built the same way, and the neighbour
features (`nb_similar_*`, `nb_recommendations_*`) share the defect through `oof_neighbor_encode`.

The size of the optimism has **not been measured**, and it should not be guessed at. The fix is to move the
encoding inside the evaluation fold, which means the feature build can no longer be a one-shot CSV write.
Until then: treat every MAE and R² in §6 as an upper bound on true out-of-sample performance, and treat the
`creator_encoded` SHAP weight in §6.8 the same way. The *relative* comparisons between models are less
affected than the absolute numbers, because every model shares the defect.

**2. FIXED 2026-10-11 — `movies/predict_ratings.py` zero-filled 23 of the 107 movie features.** It now builds every row with `transform_single_movie`; the history below is kept for the record.
This is the **scoring path only** — training and the §6 benchmarks are unaffected. The scorer builds
`median_values`, the language one-hots, the genre multi-hot, three target encodings, the MPAA block and the
PCA block. It does not build: `director_n`, `actors_n`, the six crew encodings and their `_n` counts
(`writer`, `composer`, `cinematographer`, `editor`, `producer`, `production`), the four neighbour columns,
`budget_log`, `revenue_log`, `vote_count_log`, `rel_month`, `age_at_rating`. All 23 are zero-filled.

Measured cost of zeroing those columns: **Stacking MAE 0.375 → 0.529; SVR R² 0.641 → −0.058.** The served
movie predictions in `results/movies/` are made under that penalty. `predict_unified_ratings.py` was fixed for
exactly this class of bug and now warns by name on any unbuilt column (§7.1); `movies/predict_ratings.py`
has not been given the same treatment. It also still carries a local `categorize_rating` with the naive
`'R' in r` test that the canonical implementation exists to replace.

**3. FIXED 2026-10-11 — distillation was switched off in code, not only by verdict.** The dead `pred_Stacking` lookup has been deleted from both trainers; the history below is kept for the record.
§4.4 records the paired test as null, so the feature is correctly dropped on the evidence. But the code could
not have used it either way: the games and books trainers look up a column named `pred_Stacking`, and
`predict_unified_ratings.py` writes `pred_MeanEnsemble`. The trainers print a warning and train without the
prior. Renaming the key is **not** sufficient — `unified_prior` would then arrive as a column absent from
`training_columns`, and the fitted scaler would reject it. If distillation is ever revisited, that is the real
work item.

**4. FIXED 2026-08-28 — the unified language block and the TV country block were each split in two.**
`src/common/parsing.py::norm_language` folds ISO codes onto English names. `unified_feature_engineering.py`
never called it, so the unified matrix held **`lang_en` (983 rows) and `lang_English` (145 rows) as two
orthogonal one-hot columns for one language**, with `lang_hi` (123) as the third and Japanese never reaching
the top-3 cut; the shows block held `ctry_US` (5 shows, stored as `"['US']"`) against `ctry_United States`
(116). Both are normalized now — one country column at **121** shows, and South Korea enters the top-8.
The consequence is recorded rather than buried: merging the language columns made the unified model
**slightly worse, +0.0046 MAE** (0.4979 → 0.5025 on identical items). `lang_en` held movies and TV while
`lang_English` held mostly books, so the split was a partial **domain indicator**, and a cross-domain model
was reading domain identity off a feature whose whole purpose is to bridge domains. A confound removed, not
a regression — and not to be quietly reverted to buy the 0.0046 back.

**5. FIXED 2026-08-28 — the movie embedding header consumed the encoder's whole budget, and capping it is
the one change this cycle with a clean, attributable win.**
Measured on the live corpus before the fix: `_movie_header` emitted the full billed cast — a **median of 41
names**, max 331 — so the header alone was a **median of 230 tokens** against a 256-token encoder limit, and
**42.7% of headers exceeded the limit on their own**. `encode_pooled` windows the *body* to 180 tokens and
prefixes the header to each window, so **69.0% of movie header+window strings were truncated**, and what got
cut was the plot. TV (0%), games (0%) and books (1.5%) were unaffected; their headers run 32–42 tokens.

The movie vibe block was therefore, in practice, an embedding of the title, genre, crew and cast list with
little of the plot surviving — the same length confound §3.6 was written to remove, falling on the largest
domain instead of the smallest. Capping the people lists to 6 actors / 3 directors / 3 writers / 5 genres
takes those two rates to **0.0%** and **0.1%**, and moves the movie benchmark **MAE 0.4925 → 0.4863 and
R² 0.5133 → 0.5226 on the identical 1,062 films**. It was never a train/serve skew — the header is built the
same way in training, batch scoring and the Oracle — so this is a representation fix, and it is the
measurement the earlier draft of this section asked for: chunk-pooling had been a no-op for movies until it
landed.

**6. FIXED 2026-08-28 — the Books Oracle predictor was stale and the pipeline did not rebuild it.**
`models/books/text_knn.joblib` was trained only by running `python -m src.books.text_similarity` by hand, so
it served the Books Oracle from a PCA space fitted on the previous 100-book corpus with the previous text
recipe while `models/books/preprocessor_state.pkl` was current. It is a step in `run_pipeline.py` now and is
rebuilt on every run: **skill 0.507, MAE 0.327** on the 2026-08-28 corpus. The file on disk is dated
2026-08-30, so it was not rebuilt in the 2026-10-01 run and this figure was not re-measured on 110 books.

### 8.2 Standing limits of the design
- **Single user.** Nothing here generalizes across users. Multi-user generalization is out of scope.
- **Aleatoric ceiling.** Self-reported 0.5★ ratings cap the achievable R² and MAE. The same item can receive
  3.5★ or 4.0★ on a different day, and the decomposition in §6.5 counts that noise inside bias².
- **The features mostly measure crowd quality, not personal taste.** IMDb scores and vote counts capture "is
  this generally good". The user's deviation from the crowd is the residual the features cannot see. The TMDB
  neighbour columns are the only collaborative signal in the project.
- **Small-N domains.** Games (66) and books (110) are pilots. Lead with skill score and conformal intervals,
  not R².
- **Music is unrated by design.** It is excluded from every rating metric and every confusion matrix, and from
  the unified frame.
- **Music transfer is gated off.** Music-to-other transfer still measures ≤0.
- **Wikipedia plot text exists but is deliberately not wired into features.** `src/common/wikipedia_plots.py`
  fetches real beat-by-beat plots through the MediaWiki API and reaches 93.5% coverage on movies (median 644
  words, 32 sentences), 65.3% on TV, 54.2% on games, 49.0% on books. It is not used, because partial coverage
  *within* a domain is a register split — half the items get a synopsis and half get a logline — and the
  coverage asymmetry *across* domains is precisely the confound chunk-pooling was introduced to remove.
  Wiring it in needs a plan for the uncovered half first.
- **Several dependencies are imported lazily and are invisible to an import audit:** `tabulate` (pandas
  `.to_markdown()`), `statsmodels` (plotly `trendline=`, six charts), `ftfy` (mojibake repair). An audit that
  greps for imports has already deleted them once, and the missing `tabulate` killed a four-hour run at
  step 8.
