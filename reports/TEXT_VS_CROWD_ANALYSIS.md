# Where Does Personal Taste Live — Text/Plot vs Crowd-Quality?

> **The question.** Crowd-quality features (IMDb rating, votes, Metascore) measure "is this
> *generally* good," not "will *I* like it." The personal signal should live in the plot /
> title / crew **text**. So are we under-weighting text? Measured, not assumed.
> Reproduce: `python -m src.experiments.text_vs_crowd` and `… .text_lift`.

## 1. Skill by feature group (repeated 5×5 CV, skill = 1 − MAE/MAE_mean)

| Domain | TEXT only (plot/title/crew) | CROWD only (IMDb/critic/votes) | FULL | kNN on text (plot-similarity) | Verdict |
|---|---|---|---|---|---|
| **Movies** | 0.063 | **0.329** | 0.373 | 0.069 | crowd-led (you rate ≈ consensus) |
| **TV** | 0.048 | **0.260** | 0.150 | 0.017 | crowd-led; text is near-noise |
| **Games** | 0.105 | 0.124 | **0.204** | 0.106 | mixed — combine both |
| **Books** | **0.387** | 0.170 | 0.244 | **0.426** 🏆 | **text-led — your intuition is right** |
| **Unified** | 0.047 | 0.259 | 0.314 | 0.050 | crowd-led |

**Read:** For **Books**, plot/author text *dominates* — and the purest "books whose descriptions
resemble ones I rated highly" predictor (kNN-in-embedding, **0.426**) beats the full deployed model
(0.244). The Books model is *hurting itself* by mixing weak crowd features into a tree that splits
on them first. For **Movies/TV**, crowd scores win 5–6× over text: you rate those close to critical
consensus, so the personal-beyond-crowd signal the text would carry is small.

## 2. Can we lift the text signal for Movies / TV? (`text_lift.py`)

The movie embedding originally encoded only *Title + Director + Plot* — **omitting cast, writer,
genre, tagline, overview**. We re-embedded with richer text and scored text-only skill under more
components and a **target-supervised** projection (PLS) — the acid test for "is there taste signal
PCA is discarding?"

| Domain | text | PCA15 | PCA50 | RAW-384 | **PLS-15 (supervised)** | kNN | crowd-only |
|---|---|---|---|---|---|---|---|
| Movie | current | 0.060 | 0.071 | 0.076 | −0.146 | 0.064 | 0.309 |
| Movie | **richer** | 0.072 | 0.083 | **0.099** | −0.073 | 0.087 | 0.309 |
| TV | current | 0.055 | 0.028 | 0.046 | −0.205 | −0.005 | 0.202 |
| TV | richer | 0.066 | 0.022 | 0.045 | −0.217 | 0.033 | 0.202 |

**Read:**
- **Movies — a real, modest lever.** Richer text + more components roughly **doubles** the text-only
  signal (0.060 → 0.099). We were genuinely under-using text (the recipe dropped cast/writer/genre).
- **TV — no lever.** Richer text barely moves it, and the **supervised PLS projection goes negative
  (−0.2)** — it overfits and finds *no* generalizable taste direction. That is the decisive evidence
  that TV taste ≈ consensus + irreducible noise; text cannot rescue it.
- **PLS negative in every case** ⇒ there is no hidden linear "taste axis" that PCA discarded. The only
  real lever is *richer input text*, and it only pays off for Movies.

## 3. What we changed (Movies) and the measured result

Enriched the movie embedding text to *Title + Genre + Director + Writer + Cast + Tagline + Plot +
Overview* and raised the vibe PCA from 25 → 40 components (`src/movies/feature_engineering.py`, with
matching parity in `transform_single_movie` and `predict_ratings.py`).

| Metric (Movies) | Before | After |
|---|---|---|
| Text-only skill | 0.063 | **0.071** |
| Full-model skill | 0.373 | **0.379** |
| bias² | 0.397 | **0.385** |
| bootstrap test MAE | 0.485 | **0.479** |
<!-- MOVIE_TEXT_TRAIN:BEGIN -->

**Deployed models retrained on the richer text** (base + Optuna edge-penalty stacking ensemble):
the ensemble's best member lands at **test MAE 0.468 / R² 0.562** vs the prior deployed **0.476 /
0.558** — on par to a hair better, i.e. the richer text is neutral-to-slightly-positive even after
the loud crowd features and target encodings have taken their share. (Unified matrix + frozen-fold
benchmark tables were *not* regenerated for this marginal gain — known drift, re-sync via
`run_pipeline.py`.)
<!-- MOVIE_TEXT_TRAIN:END -->

A small, honest, downside-free gain — consistent with the diagnosis that movie taste is *mostly*
critic-consensus, so richer text helps a little but cannot rival the crowd scores.

## 4. Books text-similarity model — IMPLEMENTED & deployed

Design comparison on the Books matrix (repeated 5×5 CV, skill vs predict-the-mean):

| Books model | skill |
|---|---|
| full SVR (crowd + content) — *old deployed* | 0.244 |
| content SVR (pca + author + category, no crowd) | 0.237 |
| blend 0.6·kNN-text + 0.4·full | 0.390 |
| **embedding-kNN on plot text alone** | **0.429** 🏆 |

Mixing in *any* crowd/author features **hurts** — the honest best model is pure plot-similarity.
Shipped as `src/books/text_similarity.py` (`BooksTextOracle`): a cosine, distance-weighted
KNeighborsRegressor over the book vibe-PCA embedding, saved to `models/books/text_knn.joblib`.
**Deployed CV skill 0.429 / MAE 0.32** (vs 0.24 / ~0.48 for the old SVR). It predicts with a real
*spread* (not the mean-hugging 3.5) and comes with a **"because you rated…" explanation**:

> *The Martian → predicted 4.3 — nearest: Project Hail Mary (Andy Weir, 5.0★), I Have No Mouth… (4.5★)*

The Books active-learning queue (`books_active_learning_queue.py`) now uses this predictor.

## 5. Recommendations

- **Books:** ✅ done — deployed the embedding-kNN text-similarity model (skill 0.24 → **0.43**).
- **Movies:** ✅ done — richer embedding text (added cast/writer/genre) — marginal but free.
- **TV:** don't chase text; it's consensus + noise. (Confirmed by the negative supervised projection.)
- **Games:** keep combining text + local features (Metacritic/tags).
