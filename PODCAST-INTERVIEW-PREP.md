# Personal Media Intelligence Hub — The One-Hour Interview Podcast

**Format:** a mock interview. `HOST` is the interviewer. `YOU` is you.
**Assumed knowledge:** none. Every technical word is defined the first time it appears, in ordinary English, in the sentence where it appears.
**How to use it:** read it out loud once, or listen to it. Then read the Cheat Sheet at the very end (Segment 8) twice.

**One rule before you start:** everything in this script has been checked against the actual code and the actual output files in this repository. Where a resume claim is weaker than it sounds — and three of them are — the script says exactly how, and gives you the sentence to say. Never oversell in a real interview. Interviewers dig, and the digging is where people fall apart.

**A warning specific to this project.** This repository contains a `README.md`, a `docs/ML_SYSTEM_OVERVIEW.md`, a `reports/FINAL_ML_TECHNICAL_REPORT.md` and a LaTeX paper at `paper/PMIH_technical_report.tex`. **They contradict each other**, and they contradict the numbers sitting in `reports/`. That is not a disaster — it has an explanation, and the explanation is one of the strongest things you can say in the room. But do not learn your numbers from the prose. Learn them from this script, which took them from the output files and from re-running the statistics.

---

## SEGMENT 0 — What you actually built (00:00 – 05:00)

**HOST:** Start at zero. Forget the machine learning. In plain English, what is the Personal Media Intelligence Hub?

**YOU:** It is a system that predicts how much *one specific person* — me — will enjoy a book, a film, a TV show, an album or a video game, **before** they consume it. You hand it a title it has never seen. It gives back a star rating on the same half-star scale I rate things on, plus a confidence range, plus a chart showing which properties of that title pushed the prediction up or down.

**HOST:** Netflix already does that. Why build it?

**YOU:** Netflix does something different, and the difference is the whole project. Netflix uses **collaborative filtering** — it finds thousands of people whose viewing looks like mine, and recommends what they liked. That needs thousands of people, and it only works inside one catalogue. Netflix cannot tell me whether I will like a *book*, because it has never seen a book, and it cannot use my film taste to say anything about a book even if it had.

I have exactly one user. Me. 1,514 ratings, spread across four media types. So collaborative filtering is unavailable by construction, and the interesting question becomes: **can you predict taste from the properties of the item itself, and does that carry across media?**

**HOST:** Define "properties of the item."

**YOU:** Everything that is knowable about a title without asking a human whether they liked it. Its genres. Its year. Its runtime or page count. Who made it — director, author, developer, studio. What critics scored it. And, most importantly, what its plot description *means*, converted into numbers by a language model.

**HOST:** So why did this need engineering rather than a spreadsheet?

**YOU:** Four reasons, and these are the four I would actually talk about.

- **The data does not exist in one place.** My film ratings live on Letterboxd. My TV ratings on Serializd. My books in a Hardcover library. My games in a personal CSV export. My music in Spotify, which has no ratings at all. Every one of those needs a different API — an **API** being a web address a program can ask for data instead of a human clicking around a website — and every one of them describes a title differently. Getting them into one table with the same columns is most of the work.
- **The dataset is tiny.** 1,514 rated items sounds like a lot until you split it: 1,163 films, 175 TV shows, 110 books and **66 games**. Sixty-six. At that size, almost anything you measure can be noise, and the standard machine-learning workflow — split the data once, train, report the number — will lie to you.
- **The interesting question is a research question, not a product question.** "Does my film taste predict my game taste?" is a claim about the world. Claims about the world need controls, significance tests, and a decision rule fixed *before* you look at the answer. That is a different discipline from building a model that scores well.
- **It had to be explainable.** A number with no reason attached is useless for the actual use case, which is me standing in front of a shelf deciding what to read next.

**HOST:** And what did you find?

**YOU:** The honest headline is: **taste transfers into games, and only into games.** A model trained on nothing but my film ratings, that has never seen a single game rating, ranks games better than a model trained on my actual game ratings does. In the other direction — using games to predict films — it is actively harmful. I will spend most of Segment 5 on that, including the parts of it I do not fully trust.

> **Interview note — set the size honestly in the first minute.** "1,514 ratings, one user, 66 games in the smallest domain" is a *stronger* opening than any vague claim, because everything you say afterwards is calibrated to it. An interviewer who hears "66 games" and then hears you defend a significance test on 66 games is impressed. An interviewer who discovers the 66 in minute forty is not.

---

## SEGMENT 1 — The vocabulary you need (05:00 – 14:00)

**HOST:** Your resume line reads *"Python, XGBoost, CatBoost, Optuna, Scikit-learn."* Take them one at a time. Plain definition, then where you used it, then why.

**YOU:** Right. And I will add five words that are not on the line but that the whole project is made of, because an interviewer will use them.

### Python

- **What it is:** a programming language. Unlike a compiled language, Python code is read and run line by line by an interpreter, which makes it slow for raw number-crunching but very fast to write. Everything that matters for speed here is done by libraries whose insides are written in C.
- **Where I used it:** all of it. Roughly 100 source files under `src/`, plus a 17-page web app under `app/`.
- **Version pinned in `requirements.lock.txt`:** the whole stack is version-locked, which matters more than it sounds — a model saved by one version of a library often cannot be loaded by the next.

### Scikit-learn

- **What it is:** the standard Python library for classical machine learning. "Classical" meaning everything that is not a deep neural network — linear models, support vector machines, decision trees, and, critically for me, the *plumbing*: splitting data, scaling numbers, chaining steps together, scoring predictions.
- **The one idea worth naming:** scikit-learn defines a **common interface** — every model has `.fit(X, y)` to learn and `.predict(X)` to guess. That is why I can swap an XGBoost model for a support vector machine in one line and the surrounding code does not change.
- **Where I used it:** the fold registry (`RepeatedStratifiedKFold`), the scaling pipelines, the **SVR** models that are the deployed models for Games and Books, and every metric.
- **Version:** scikit-learn 1.8.0.

### What a "model" is here, before the brand names

Before XGBoost and CatBoost mean anything, one definition. A **model** is a function fitted to data. I give it a table where each row is a title and each column is a property — genre, year, critic score, and 384 numbers describing the plot — and one extra column holding my star rating. The model's job is to learn the rule that turns the properties into the rating, well enough that when I show it a title it has never seen, it guesses close.

Two flavours appear in this project:

- **Regression** — predict a number. My rating on a 0.5-to-5.0 star scale. This is the main one.
- **Classification** — predict a bucket. "Skip it / Watchable / Masterpiece," or which of the ten half-star bins it lands in. I use this second one for the confidence display.

### XGBoost — and what a gradient boosted tree actually is

This is the one you must be able to explain slowly, because it is the workhorse.

- **A decision tree** is a flowchart of yes/no questions. "Is the critic score above 3.8? If yes, is the genre Horror? If yes, predict 4.0 stars." One tree on its own is weak and memorises noise.
- **Boosting** is the trick. You build one small, deliberately weak tree. You look at where it was wrong — the **residuals**, meaning actual rating minus predicted rating. Then you build a *second* tree whose job is to predict those errors. Add it to the first. Look at the remaining errors. Build a third. Each tree corrects the mistakes of everything before it.
- **Gradient** boosting is the mathematically general version of "correct the errors": instead of literally fitting the leftover error, each tree follows the direction that reduces a chosen loss function fastest. The **loss function** is the definition of "wrong" you told it to care about.
- **XGBoost** is a fast, well-engineered implementation of that idea, with regularisation — penalties on tree complexity that stop it from memorising.
- **Where I used it:** the Movies model, the base member of the unified ensemble, and **the model inside the transfer grid** (`src/experiments/transfer_study.py`), where it runs at 300 trees, learning rate 0.03, max depth 6.
- **One detail worth volunteering:** I set `objective="reg:absoluteerror"` — I told it to minimise absolute error, not squared error. Squared error punishes a two-star miss four times as hard as a one-star miss, which drags every prediction toward the safe middle. On a rating distribution where most of my ratings sit in three adjacent half-star bins, that produces a model that predicts 3.5 for everything. Absolute error does not have that pull.

### CatBoost

- **What it is:** a different implementation of the same gradient boosting idea, from Yandex. Two things make it different: it builds **symmetric** trees, meaning every branch at the same depth asks the same question, which makes it fast and resistant to overfitting; and it has native, careful handling of **categorical features** — columns whose values are names rather than numbers, like a director's name.
- **Where I used it:** as the second member of the **Mean Ensemble** in `src/unified_model/comprehensive_evaluator.py`, at 500 iterations with an MAE loss, and in the deployed Movies and TV stacks.
- **Why both XGBoost and CatBoost rather than one:** they make different mistakes. Averaging two models that are each about equally good but wrong in different places gives you a better model than either. That is the entire theory of ensembling.
- **The honest footnote I would volunteer:** in my own ablation, adding CatBoost to XGBoost improved mean absolute error by **0.0076 stars**, with a Cohen's *d* of 0.068 (2026-10-01 run). That is statistically significant — *p* below 0.0001 — and practically meaningless, because 0.0066 stars is far below the 0.5-star grid I round to. I kept it and I labelled it "significant-but-trivial" in `reports/latest_metrics.json` rather than quoting the *p*-value and calling it a win.

### Optuna

- **What it is:** a library for **hyperparameter tuning**. A **hyperparameter** is a setting you choose *before* training that controls how the model learns — how many trees, how deep, how fast it steps. The model learns its own internal numbers from data; it cannot learn these. Somebody has to pick them.
- **How Optuna picks them:** rather than trying every combination (a grid search, which explodes) or random ones, it uses a **TPE sampler** — Tree-structured Parzen Estimator. In plain words: it builds a rough picture of which regions of the settings space have produced good results and which have produced bad ones, and it spends its next trial where good is more likely than bad. It is guided guessing that gets better as it goes.
- **Where I used it:** three places. `src/unified_model/advanced_unified_model_trainer.py` runs 10 trials tuning a temporal-decay rate and two loss-function parameters. `src/reporting/production_benchmarks.py` runs 25 trials tuning the Movies and TV loss parameters. `src/shows/model_trainer.py` tunes the TV model.
- **The thing I would raise before I am asked:** the first version of that tuning had a bug I found and fixed, and it is the single best story in the project. The tuning objective and the final evaluation used `train_test_split(..., test_size=0.2, random_state=42)` on the same frame — the *same* split, twice. So Optuna was choosing the settings that minimised error **on the very rows the script then reported its score against**. The score was tuned to its own test set. The fix, which is now in the code with a comment explaining it, is to hold the test fifth out *first*, then tune inside what is left. That is a five-line change that turns a meaningless number into a real one.

### Sentence embeddings

- **The problem:** a model cannot read "A young farm boy discovers he is heir to a desert planet." It needs numbers.
- **An embedding** is a list of numbers — here, 384 of them — that represents the *meaning* of a piece of text, arranged so that texts meaning similar things end up close together in that 384-dimensional space. Not similar words. Similar meaning. "Bank robbery thriller" and "heist film" land near each other with no shared vocabulary.
- **How they are produced:** a pre-trained neural network called a **sentence transformer**. Mine is `all-MiniLM-L6-v2`, a small, fast, widely-used model. I did not train it; it was trained on a large body of text and I use it as a translator from words to numbers.
- **Why this is the load-bearing idea of the whole project:** it is the **only** feature that means the same thing in all four media. A film's genre list and a book's subject list are different vocabularies. A Metacritic score and a Goodreads average are different scales. But "the meaning of this plot description" is one space, and a film, a novel and a game can all be points in it. **Without embeddings there is no cross-media transfer to study.**
- **Where the code is:** `src/common/text_embedding.py`.

### The embedding detail that is worth a whole answer

**HOST:** You said `src/common/text_embedding.py` is worth a whole answer. Why?

**YOU:** Because the naive version quietly destroyed the experiment, and finding that out is the best piece of debugging in the project.

The sentence transformer reads at most 256 tokens — roughly 180 words. The original pipeline handled that by **truncating every description to its first 40 words**. That sounds harmless. It is not, for two reasons:

- It is a *positional* cut, so it threw away 77% of the average game description and 62% of the average book description, while losing nothing from a film plot, which averages 28 words.
- The part it kept — the opening — is exactly where each source's boilerplate lives. So it was preferentially keeping the marketing copy and throwing away the content.

The result was that the four domains were not embedded comparably at all, and a cross-domain study on non-comparable vectors measures nothing.

The fix, which is what runs now, is **chunk pooling**: split the description on sentence boundaries into windows that fit inside the model, embed every window, then average the window vectors and re-normalise to unit length. The whole text is read. And because every window vector is unit length, the average of one window and the average of eight are both unit vectors in the same space, so a 28-word film plot and a 1,074-character game blurb are comparable **by construction** rather than by cutting both to a common length. Each window also carries the title-and-genre header, so a long blurb cannot drift off-topic.

I measured it over five random seeds, in `reports/text_norm_variants.csv`. The book text signal — measured as how much better than a constant predictor the text alone gets you — went from **0.045 to 0.098**, and both cross-domain transfers improved. I also tried removing stopwords (worse) and reducing each description to distinctive keyword terms (much worse — it destroys the word order the transformer encodes, and film-to-game transfer went *negative*).

### Cross-validation, folds, and why fifty

- **The problem:** if I train on 80% of my data and test on the other 20%, my score depends on which 20% I happened to pick. With 66 games, that is a coin flip.
- **Cross-validation** fixes it. Split the data into *k* equal parts, called **folds**. Train on nine, test on the tenth. Repeat ten times so every item gets tested exactly once, on a model that never saw it. Average the ten scores.
- **Repeated** cross-validation does the whole thing again with a different random shuffle. Mine is **10 folds × 5 repeats = 50 folds**, defined in `src/unified_model/create_frozen_folds.py` with `RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)`.
- **Stratified** means each fold is built to contain the same mix of media types and the same mix of low/medium/high ratings as the full dataset, so no fold is accidentally all films or all five-star ratings.
- **"Frozen"** is my word for the thing I am proudest of. The fold assignment is computed once and written to a JSON file, `models/unified/fold_registry.json`, keyed by a per-item ID. Every experiment in the project reads that file. So when I compare a transfer model to a baseline model, they are being tested **on literally the same items in the same folds**, which is what makes a paired statistical test legitimate.

### Mean absolute error, R², skill, and *p*-values

- **MAE** — mean absolute error. Average of |predicted − actual|, in stars. MAE 0.49 means the typical prediction is about half a star out. Lower is better. I always report it because it is in the units of the thing, but it is **not** my primary metric: my ratings cluster in 3–4 stars, so a constant 3.5 already scores MAE 0.749 on my films and lands within half a star 62% of the time. A model that learned nothing still looks respectable on MAE.
- **R²** — pronounced "r squared." The fraction of the variation in my ratings the model explains. 1.0 is perfect, 0.0 is no better than always guessing the average, and it can go negative if the model is worse than the average.
- **Skill** — my own composite: `1 − (model MAE ÷ MAE of always guessing the training mean)`. Skill 0.30 means the model cut the error of a constant guess by 30%. **It is my primary metric.** It is 0 for guessing the mean and about 0 for any other constant, so the clustered rating curve cannot inflate it, and it normalises across domains that have different rating spreads. I pair it with Spearman rank correlation, which shows whether the model puts items in the right order.
- **A *p*-value** — the number that trips people up. It answers: *if there were genuinely no effect, how often would random chance alone produce a result at least this extreme?* A *p* of 0.001 means one time in a thousand. It is **not** the probability the finding is true, and I would say so if asked, because that misreading is the most common one.
- **The paired Wilcoxon signed-rank test** — the specific test I use. It takes the 50 fold-by-fold errors of model A and the 50 of model B, computes the 50 differences, and asks whether they lean systematically to one side. "Paired" is the important word: it compares A and B *on the same fold*, which cancels out the fact that some folds are just harder than others. It is a rank-based test, so it does not assume the errors follow a bell curve — which mine do not.

### SHAP

- **What it is:** SHapley Additive exPlanations. A method for answering "why did the model say 4.5 for *this* title?" It comes from cooperative game theory: it treats each feature as a player in a game whose payoff is the prediction, and fairly allocates credit for the difference between this prediction and the average prediction.
- **Where I used it:** `app/pages/2_Movies_Oracle.py`, `4_TV_Shows_Oracle.py`, `8_Games_Oracle.py` and `10_Books_Oracle.py`, each drawing a **waterfall plot** — a chart that starts at the average prediction and shows each feature pushing it up or down to the final number.
- **The honest limit, and I will come back to it in Segment 5:** that is a chart with feature names on it. It is not a written sentence.

### Streamlit

- **What it is:** a Python library that turns a plain Python script into a web app. You write `st.title("...")` and `st.bar_chart(df)` and it serves a page. No HTML, no JavaScript, no server code.
- **Why it, and not a real web stack:** the audience is one person and the content is charts over pandas dataframes. A React front end with a REST API would have been days of work for zero additional capability. This is a case where the boring choice is the correct one and I would defend it that way.
- **Where:** `app/main.py` plus 16 files in `app/pages/`.

---

## SEGMENT 2 — Resume bullet one: the taste engine (14:00 – 24:00)

**HOST:** First bullet: *"Built a taste engine that predicts how much a person will enjoy a book, film, show, album or game before they ever start it, and explains every prediction in plain language."* Unpack it.

**YOU:** There are two halves — the prediction and the explanation — and I want to be precise about both, including where the wording is generous.

### What actually predicts

There are **two separate model families**, and the distinction matters because an interviewer will conflate them.

- **Five local models**, one per medium, each trained only on its own domain's data and its own domain's features. These are what the app actually serves.
- **One unified model**, trained on all rated domains at once in a shared feature space. This is the **research harness** — the thing the transfer study is built on. It is not the production scorer.

The local models are not all the same architecture, and that is deliberate, not laziness:

| Domain | Rated items | Deployed model | Skill | Spearman ρ | MAE | R² | Within half a star |
|---|---|---|---|---|---|---|---|
| Movies | 1,163 | Tuned ensemble: edge-penalty XGBoost + CatBoost + SVR, fused | 0.355 | 0.751 | 0.483 | 0.530 | 77.1% |
| TV | 175 | Simplex-weighted stack of XGB + CatBoost + SVR | 0.208 | 0.565 | 0.694 | 0.271 | 60.6% |
| Games | 66 | Local **SVR** | 0.376 | 0.719 | 0.515 | 0.497 | 74.2% |
| Books | 110 | Local **SVR** | 0.566 | 0.859 | 0.286 | 0.715 | 88.2% |
| Unified (all 4) | 1,514 | Mean Ensemble (XGB + CatBoost) | 0.287 | 0.674 | 0.542 | 0.445 | 73.1% |

Those numbers are from the 2026-10-01 run: `reports/latest_metrics.json` and `reports/baseline_sanity.json`, which is written by exactly one script and carries the git commit hash of the code that produced it.

**HOST:** Stop. Why is a support vector machine the deployed model for the two smallest domains, when you have gradient boosting available?

**YOU:** Because of the sample size, and this is a real modelling decision rather than a preference.

An **SVR** — support vector regression — fits the smoothest function that keeps almost every training point inside a narrow tube around it. It is one global, smooth surface with a couple of tuning knobs. A gradient boosted forest is hundreds of trees, each carving the feature space into boxes.

With 66 games and roughly 400 features, boosting has enough freedom to draw a box around every single game and memorise the training set perfectly, which teaches it nothing. The SVR physically cannot do that — its capacity is bounded by its regularisation and its kernel, and it degrades toward a smooth average rather than toward memorisation. **The right amount of model complexity is a function of how much data you have,** and 66 rows is not enough to feed a forest.

And it is not a hunch: the benchmark table is generated by running both on the same 50 frozen folds, in `src/reporting/standalone_benchmarks.py`. SVR won on Games and Books. XGBoost won on Movies, where there are 1,163 rows to justify it.

**HOST:** Books has the *best* score of any domain, with the second-smallest dataset. Explain that.

**YOU:** Two reasons, and the second one is a lesson I would offer unprompted.

The first is data quality. Books come from a single **Hardcover** library record via their GraphQL API, so the title, my rating, the authors, the genres, the page count and the description all arrive from one source in one shape. There is no fuzzy title matching, no ISBN join, nothing to mismatch. Movies, by contrast, are a Letterboxd export joined to TMDB joined to OMDb, and every join is a chance to attach the wrong film.

The second is that I made Books worse before I made it better. The old book feature matrix emitted 10 author indicator columns and 155 category indicator columns — **165 near-empty binary columns for about 100 books**, most of them with one or two positive entries. That is the worst feature-to-sample ratio in the project and it is pure noise. I replaced all 165 with **five target encodings**.

**HOST:** Define target encoding.

**YOU:** Instead of a column per author, you have one column holding "my average rating for the books by this author that I have already rated," smoothed toward my global average so that an author with a single book does not get an extreme value. The smoothing formula is `(n × entity_mean + m × global_mean) ÷ (n + m)`, and `m` is the smoothing strength.

Two things about it I would raise myself:

- **`m` matters enormously and the original value was wrong.** The old default was `m = 10`, on a library where the median author has one or two entries. At `m = 10` the formula sits essentially on top of the global mean — the observed spread of the column was 0.05 to 0.13 stars, meaning the feature was a constant. I swept it: `m = 0` gives MAE 0.4769, `m = 1` gives 0.4732, **`m = 2` gives 0.4694**, `m = 10` gives 0.4802. It is at `m = 2` now, and `m = 10` was indistinguishable from having no feature at all.
- **It leaks unless you are careful,** because the average includes the row's own rating. So `oof_entity_encode` in `src/common/taste_features.py` builds the column with an inner five-fold split, so no item's own rating ever reaches its own feature. A control that removed that inner split scored MAE 0.6243 against 0.5075 for simply deleting the columns — leaky features are worse than no features. **I will come back to a place where this protection is not enough, in Segment 4, and it is the biggest single caveat in the project.**

### The explanation half — and where the resume is generous

**HOST:** "Explains every prediction in plain language." Show me.

**YOU:** Here is exactly what the app shows, and then here is the part of that phrase I would soften if I were writing the line again.

For a title you type into one of the four Oracle pages, you get:

- **A verdict row** — the stacking ensemble's star rating, the ordinal model's expected value, and a coarse tier: "Skip it / Watchable / Masterpiece."
- **A confidence engine** — a bar chart of the model's probability across all ten half-star buckets. That is genuinely informative: a model that puts 30% on 3.5 and 28% on 4.0 is telling you something different from one that puts 85% on 3.5.
- **An 80% prediction interval** from **split-conformal prediction**. That is a method with a real guarantee behind it: take the model's absolute errors on held-out data, find the 80th percentile of them, and use that as the half-width. Because it is calibrated on errors the model has not seen, roughly 80% of future predictions genuinely land inside it. Measured coverage per domain is within a few points of the 80% target.
- **A SHAP waterfall chart** under the heading "Why this prediction?" — the top ten features, each with a bar showing how far it moved the prediction from the average.
- **A semantic-neighbours panel** — the three titles I have already rated whose plot embeddings are closest, each with a written caption from `explain_similarity()` in `src/movies/feature_engineering.py` that reads like *"Shared: Franchise (The Dark Knight Collection), Director (Christopher Nolan), Genres (Action, Crime), Keywords (vigilante, gotham)."*

**HOST:** So is "plain language" fair?

**YOU:** **Partly, and I would say so before being asked.** The similar-titles captions are genuine plain English generated from the data. The SHAP output is a labelled chart, not a sentence — the labels are feature names like `actors_encoded` and `critic_avg_5`, which are plain to me and not to a stranger. And "every prediction" is not literally true: the four Oracle pages for Movies, TV, Games and Books have SHAP; the **Music Oracle does not**, because music has no ratings and therefore no supervised model to explain.

The sentence I would actually say is: *"Every rated-domain prediction ships with a feature-attribution chart, a conformal confidence interval, and a written 'you already rated these three similar things' panel. The natural-language part is the similarity explanation; the attribution part is a SHAP chart. Turning the SHAP output into generated sentences is one afternoon of work and it is on my list."* That is honest and it is still a strong answer.

> **Interview note.** The trade here is worth internalising. If you say "plain language explanations" and the interviewer opens the app and sees a chart, you have lost credibility on everything else you said. If you volunteer the distinction, you have demonstrated that you know what your own system does at a level of precision most candidates never reach.

### "Before they ever start it"

**HOST:** And "before they ever start it" — is that real?

**YOU:** Yes, and it is the part that is easiest to demonstrate. The Oracle pages take a title the model has never been trained on, look it up live in the relevant catalogue, build the feature row on the spot, and score it. There is also an **active-learning queue** at `src/evaluation/active_learning_ranker.py` that scores my *watchlists* — the films on my Letterboxd watchlist, the books on my Hardcover want-to-read shelf — and ranks them by how informative rating them would be. It prioritises titles where the models disagree with each other or where the conformal interval is widest, because those are the items whose true rating would teach the model the most. That is the "before you start it" claim doing actual work rather than being a demo.

---

## SEGMENT 3 — Resume bullet two, part one: the data (24:00 – 34:00)

**HOST:** Second bullet: *"Pulled 1,407 personal ratings from 9 public catalogues into one pipeline, engineered 231 shared cross-media features with sentence embeddings, and tuned ensembles over 50 folds."* I want each of those four numbers traced to a file.

**YOU:** Good, because two of the four are exact and two need qualifying, and I would rather qualify them myself.

### 1,407 — verified for the run it came from, 1,514 now

The bullet's 1,407 was exact for the 2026-08-28 run. After the 2026-10-01 retraining, `reports/latest_metrics.json` has a row `{"Domain": "unified_rated", "N": 1514}`. It decomposes as 1,163 movies + 175 TV shows + 66 games + 110 books. The on-disk feature matrix `data/processed/unified/training_features.csv` has 1,520 rows; six of those are games with a status of "Incomplete" that carry no real rating, and they are excluded from the fold registry. The number is generated, not typed — `create_frozen_folds.py` filters to `target_reg > 0` and the evaluator asserts that its slice counts equal the unique item counts.

**Two things I would state plainly.** First, **music contributes zero to that count**, because Spotify has no star ratings and the project deliberately stopped synthesising fake ones. Music is in the system — it has its own ingestion, its own features, its own dashboard and its own unsupervised recommender — but it is not in any accuracy claim. Second, these are one person's ratings, so every number in this project is a statement about *me*, not about people in general.

**HOST:** Why is that first point important?

**YOU:** Because it is where a lot of student projects quietly cheat. If you have no labels, the tempting move is to invent them — "played this song a lot, call it a 4." Then you report an accuracy on your own invention. An earlier version of this project did exactly that with music pseudo-labels, and the code now carries a comment saying music is "de-rated" and excluded from the frozen-fold metrics. Removing a domain from your headline number is not a nice thing to do to your headline number, which is precisely why it is worth mentioning.

### 9 catalogues — **I could not verify this number in the repo**

I want to be exact here. **The number nine does not appear anywhere in the repository as a count of data sources.** I searched the README, the docs, the paper and the code.

What *is* in the repo: `README.md` lists eleven APIs — TMDB, OMDb, TVMaze, RAWG, Hardcover, Open Library, YouTube Data API v3, Spotify, ReccoBeats, MusicBrainz, Genius. And a scan of every URL in the source code finds more than that, including Wikidata, TheAudioDB, Discogs, Deezer, Last.fm, TheTVDB and LRCLIB.

So the honest framing is a **defensible enumeration rather than a memorised count**, and it is stronger than a number anyway:

| Domain | Where my ratings come from | What enriches them |
|---|---|---|
| Movies | Letterboxd CSV export | TMDB, OMDb |
| TV | Serializd export | TMDB, TVMaze, OMDb |
| Games | Personal CSV export | RAWG |
| Books | Hardcover library (GraphQL) | Hardcover; Open Library for unrated lookups |
| Music (no ratings) | Spotify library | ReccoBeats (audio features), MusicBrainz, Genius (lyrics) |
| Cross-domain links | — | Wikidata |

**The sentence I would say:** *"Ratings come out of four personal exports plus a Spotify library; those are enriched against nine or more public catalogues depending on how you count — the README lists eleven APIs and the code touches a few more. If the exact count matters, TMDB, OMDb, TVMaze, RAWG, Hardcover, Open Library, Spotify, ReccoBeats and MusicBrainz are the nine that feed the rated pipeline; Wikidata, Genius and YouTube feed side experiments."*

**HOST:** Why so many for one library?

**YOU:** Because no single one has everything. TMDB has good plots, genres and full credits but no Metacritic or Rotten Tomatoes. OMDb has the critic scores and box office but weak crew data. TVMaze fills in shows TMDB matches badly. Every extra source is one more join, one more cache, one more failure mode — so each one has to earn its place, and every response is cached to JSON under `data/cache/` so a re-run does not re-hit the network.

### 231 features — **the number is real but it is stale, and I would correct it myself**

Here is the exact situation, and it is a good illustration of a real problem in the project rather than an embarrassment.

- **231** is the unified feature count from the version of `README.md` in the last git commit: *"236 columns = 231 features + 5 bookkeeping."*
- The **uncommitted working copy** of that same README now says **239** features, 244 columns.
- The **Streamlit app's own help text**, in all four Oracle pages, says **397**.
- The **actual matrix on disk**, `data/processed/unified/training_features.csv`, has **411 columns**, which is **406 features + 5 bookkeeping** (`target_reg`, `target_ordinal`, `rating_date`, `source_id`, `media_type`).
- The **transfer study's shared space**, which is the subset that matters for the headline finding, is **402 features** — I re-ran `select_features()` to confirm.

**Why they all differ:** the shared text block changed representation. It used to be a **PCA** compression — Principal Component Analysis, a method that takes 384 correlated numbers and finds the 10 directions carrying the most variance, then keeps only those. The project moved from that 10-component compression to the **full 384-dimensional embedding**. That single change adds 374 columns, which is essentially the whole gap between 231 and 406.

**Why the move was made,** because this is the interesting part: PCA keeps *global* variance and degrades the *local* neighbourhood structure, and local neighbourhood structure is exactly what cross-domain similarity is. There is a concrete example in the code comment: in the 10-component PCA space, the nearest neighbours of the *Scott Pilgrim* film are *Ted* and *Ferris Bueller's Day Off* — the "quirky comedy" direction. In the full 384-dimensional space, the nearest neighbours are the *Scott Pilgrim* TV series and all six volumes of the graphic novel. PCA had thrown away precisely the signal the experiment is about.

**The sentence I would say in the room:** *"231 is what my README said at the last commit. The matrix is 406 features today, because the shared text block moved from a 10-component PCA to the full 384-dimensional embedding — and that change is the reason the cross-domain result works, so it is a change I would want to talk about rather than paper over. The correct current number is 406 features, 402 of which the transfer grid uses."*

### What the 406 actually are

Whatever the count, the composition is the interesting thing, and it is built in `src/unified_model/unified_feature_engineering.py`:

| Block | Count | What it is |
|---|---|---|
| `emb_0` … `emb_383` | 384 | The chunk-pooled sentence embedding of the title, genre, tags and full description |
| `gen_*` | 17 | Cross-domain genre indicators — Action, Comedy, Drama, Horror… only genres that appear in **at least two** media types |
| `creator_encoded`, `creator_n` | 2 | Smoothed "my average rating for this director/author/developer," plus how many I have rated |
| `studio_encoded`, `studio_n` | 2 | The same for studio / publisher / production company |
| `critic_avg_5` | 1 | Critic consensus, normalised onto the 5-star scale by averaging IMDb, Metascore, Rotten Tomatoes and TMDB vote average |

**HOST:** Why only 17 genres when your README talks about 183 genre columns?

**YOU:** Another stale-doc case, and the reason is a real design rule. The genre block is filtered by `genres_in_multiple_domains()` — a genre only survives if it appears in at least two media types. `gen_Juvenile fiction` exists only for books, so as a feature in a cross-domain model it is a **domain label wearing a costume**: a model can split on it and immediately know "this is a book," which lets it cheat without transferring anything. Seventeen genres span more than one medium. Those are the ones that stay.

That is the same principle as the biggest bug I found in the transfer study, which is Segment 5.

**HOST:** What about year and runtime? A film and a novel both have a length.

**YOU:** They are in the per-domain matrices and deliberately **not** in the unified one. Runtime in minutes and page count are on incompatible scales; "year" behaves completely differently when your film library spans 1930 to 2026 and your game library spans 1998 to 2024. Every column in the shared space had to survive the question *"does this mean the same thing for a novel and a game?"* Most did not. What survived is: the meaning of the description, a genre if it exists in more than one medium, who made it, and what critics thought.

---

## SEGMENT 4 — Resume bullet two, part two: fifty folds and the ensembles (34:00 – 43:00)

**HOST:** "Tuned ensembles over 50 folds." Fifty is a strange number. Why not five, or ten?

**YOU:** Because ten is not enough when your smallest domain has 66 items, and because of what I want the fifty *for*.

`src/unified_model/create_frozen_folds.py` builds `RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)`. Ten folds, five repeats, fifty train/test cycles. Every item lands in exactly five test folds, once per repeat.

Two reasons for the repeats:

- **Stability of the estimate.** With 66 games, a single 10-fold run tests about 6 or 7 games per fold. One unusual game moves the number visibly. Five repeats with different shuffles average that out.
- **Fifty paired observations for the statistics.** The transfer study's entire method is: run model A and model B on the same fold, record both errors, do that fifty times, and ask whether the fifty differences lean one way. With ten observations a Wilcoxon test has almost no power. With fifty it has some.

**HOST:** So you have fifty independent measurements.

**YOU:** **No, and this is the single most important caveat in the project, so I would correct that immediately rather than let it stand.**

They are fifty *folds*, not fifty independent experiments. There are still only 66 unique games underneath. Each game appears in the test set five times, once per repeat. So the fifty differences the Wilcoxon test sees are correlated with each other — the same items keep reappearing. The test assumes independence. It does not have it.

**The consequence, stated exactly:** my *p*-values are **optimistic**. A reported *p* of 0.0002 is not a true one-in-five-thousand. I do not know the true value, and I would not pretend to.

**What I trust instead, and why it is a better answer than a *p*-value:** the *consistency* of the effect across conditions. My headline transfer cell is positive at 0%, 25%, 50% and 100% target data, and positive in every one of six different source combinations, and it survives a label-shuffled placebo. Four consistent estimates of the same effect is stronger evidence than one small *p*, precisely because the small *p* is the thing I know is inflated.

> **Interview note — this is the moment that decides the interview.** A candidate who says "p equals 0.0003 so it is significant" is a candidate who has read about statistics. A candidate who says "the p-value is optimistic because my fifty folds share 66 items, so I lean on consistency across four data fractions and six source sets instead" is a candidate who has *done* statistics. Practise saying it until it is automatic.

### Deduplication — the bug that inverted a conclusion

**HOST:** You mentioned a measurement bug that changed a scientific conclusion. Tell me.

**YOU:** Two of them, and this is the story I would lead with if asked "what are you proudest of."

**Bug one: 5× N inflation.** Every item is tested five times, once per repeat. The old metrics writer counted every one of those out-of-fold predictions as a separate data point. So a table that should read "N = 1,407 items" read "N = 7,035," and the standard errors that came out of it were about 2.2 times too small — √5 too small. Every confidence interval in the project was too narrow.

**Bug two, which is worse: the benchmark table was comparing a model to itself.** The table had a column for "standalone per-domain models" and a column for "unified model, sliced by domain." The writer was publishing the *unified model's* per-domain slice in **both** columns. So the headline finding — "the unified model is worse than the local models everywhere" — was produced by comparing a model against a differently-labelled copy of itself.

**The fix is the deliverable, not the bug.** I rebuilt the metrics path as a single source of truth:

- `src/reporting/standalone_benchmarks.py` genuinely refits each domain's own architecture on the same 50 registry folds.
- `dedup_per_item()` in `comprehensive_evaluator.py` averages the five raw predictions per item first, rounds once to the nearest half star, *then* scores. So N is always a unique item count.
- Two **assertions** run every time, and they crash the pipeline rather than publish a wrong table:
  - `assert not (benchmarks == slices)` — if a standalone number is bit-identical to the unified slice, the separation has broken again.
  - `assert slice_N == unique_rated_items` — computed from the data, not hardcoded, so it survives the library growing.
- Everything is written by `src/reporting/metrics_writer.py` into one JSON with a timestamp and the git commit hash, atomically, with a round-trip parse check.

**What that changed:** the corrected comparison put standalone and unified on the same estimator, on the same folds, on the same items — and it reversed the direction of the result. It is the clearest possible demonstration that the measurement layer is part of the science, not plumbing underneath it.

**HOST:** So who wins now, unified or local?

**YOU:** **On the current `reports/latest_metrics.json` (2026-10-01 run), the local model wins in all four domains.** By skill, local against the unified slice: 0.355 against 0.315 for movies, 0.208 against 0.186 for TV, 0.376 against 0.202 for games, 0.566 against 0.241 for books. By MAE: 0.483 against 0.513, 0.694 against 0.714, 0.515 against 0.659, 0.286 against 0.500.

And I have to flag something here rather than let you find it. **`reports/RESUME_BULLETS.md` used to say "unified is competitive-or-better in 3 of 4 domains."** That was true of an earlier run of the pipeline, and the file now says so. It is not true of the numbers currently in the metrics file. The correct current statement is the one I just gave you: **local wins everywhere, and the two margins that matter — games 0.14 and books 0.21 stars — are large.** The TV margin, 0.02, is inside TV's fold-reshuffle noise, so I would not lean on it.

**HOST:** That is a big margin. Why is pooling losing?

**YOU:** Because pooling and borrowing are different operations and only one of them helps.

The unified model **includes** the thin domain in training, which means it inherits that domain's noise and has to fit one function that works for films and for novels simultaneously. It also loses every domain-specific feature — Metacritic score, platform, developer for games — because those cannot exist in a shared space.

A zero-shot transfer model **excludes** the thin domain entirely and applies a function fitted on 1,062 films. It never sees the 66 noisy game ratings at all.

So the two results are not in tension even though they sound like it: **pooling a noisy domain in hurts it; borrowing a clean domain's structure instead helps it.** That is why the deployed system keeps local models for scoring and uses the unified model only as a research harness.

### The leak I have not fixed

**HOST:** Is there anything wrong with those unified numbers?

**YOU:** Yes, and I would volunteer it, because volunteering a known flaw is worth more than being caught by it.

**Every unified metric in this project is optimistic by an unmeasured amount**, because of a leak across the fold boundary.

Here is the mechanism, precisely. `oof_entity_encode` protects against leakage using its **own internal five-fold split**, computed once at feature-build time. But the evaluation uses the **frozen 50-fold registry**, which is a completely different partition. The two do not know about each other. The result, which I measured: **48% of training rows carry a `creator_encoded` value computed using a test row's rating** — and the value is computed once and reused across all fifty folds.

So the unified model is being told, faintly, about the answers on its own test set. It is a soft leak, not a catastrophic one — it is one column out of 406, heavily smoothed. But it is real and it is not fixed. It is documented in a comment in the feature builder, in the README, and in the paper's open-gaps appendix.

**Three things I would add immediately after saying that:**

- **The fix is known and is not hard:** re-compute the entity encodings *inside* each registry fold. It costs compute, not cleverness. It is item one on my future-work list.
- **No claim I make rests on a margin smaller than the leak.** The two domains where the local model wins by a lot — 0.14 and 0.21 stars — it wins against an opponent that is being flattered. That direction of the argument is safe.
- **Critically, the transfer study is not affected.** The transfer grid's `shared_space_columns()` keeps only `emb_*`, `gen_*` and `critic_avg_5`. It does **not** keep `creator_encoded` or `studio_encoded`. So the headline transfer finding runs on a feature set that contains no target encodings at all, and this leak cannot touch it. I checked that specifically, because if it did touch it, the finding would be dead.

---

## SEGMENT 5 — Resume bullet three: the transfer study, and every weak claim in it (43:00 – 52:00)

**HOST:** Third bullet: *"Proved taste transfers into games, lifting prediction skill +0.121 (p=0.0003) against four controls including a label-shuffled placebo, and shipped a 16-page dashboard."* This is the claim I would push hardest on. Start with the design.

**YOU:** Good, and I am going to end this segment by taking apart the numbers in that sentence myself, because two of them do not survive contact with the files.

### The design

The question is: **does taste in one medium predict taste in another?** The experiment is a grid, in `src/experiments/transfer_study.py`.

For each of the four rated domains as **target**, and each of the seven non-empty combinations of the other three as **source**, it runs three protocols on all 50 frozen folds:

- **Protocol A — baseline.** Train on the target domain only. This is what you have to beat.
- **Protocol B — zero-shot.** Train on the source domains only. The model has never seen a single rating from the target. Evaluate on the target's test folds.
- **Protocol C — augmented.** Train on the source domains plus 0%, 25%, 50% or 100% of the target's training data. The subsampling is seeded, so it is reproducible.

Everything runs in the shared space only — 402 features — with a fixed XGBoost regressor at 300 trees. Same model everywhere, so the only thing varying between cells is which data went in.

**HOST:** Why fix the model rather than tune each cell?

**YOU:** Because tuning each of roughly 120 cells would mean 120 chances to accidentally tune toward a positive result, and it would make the comparison between cells meaningless — you would no longer know whether a cell won because the data transferred or because it got a luckier hyperparameter draw. A fixed model makes every cell an apples-to-apples comparison. The cost is that no individual cell is at its best. That is the right trade for a study whose output is a *comparison*, not a deployed model.

**HOST:** And the decision rule?

**YOU:** **Pre-registered, before any result was looked at**, written at the top of `src/experiments/transfer_analysis.py`:

> A cell is a **positive transfer finding** if and only if the augmented lift is greater than zero **and** the paired Wilcoxon *p* is below 0.05, at **two or more** of the four target fractions. The project verdict is "taste transfers" if at least one target has a positive finding; otherwise the verdict is the null.

The reason to fix that in advance is that with about 120 cells and no correction, you can always find *something* that clears 0.05. Deciding the bar before you see the numbers is what stops the analysis from drifting toward whatever it found.

### What it found

Reproduced by re-running the analysis myself against `reports/transfer_grid_results.csv`:

| Source → game | 0% target data | 25% | 50% | 100% |
|---|---|---|---|---|
| `movie` | **+0.1024** (p=0.0002) | **+0.1010** (p=0.0005) | **+0.1243** (p<0.0001) | **+0.1452** (p<0.0001) |
| `movie+book` | +0.0883 (p=0.0019) | +0.0967 (p=0.0006) | +0.1090 (p<0.0001) | +0.1362 (p<0.0001) |
| `movie+tv` | +0.0860 (p=0.0041) | +0.0762 (p=0.0098) | +0.0988 (p<0.0001) | +0.1060 (p<0.0001) |
| `movie+tv+book` | +0.0562 (p=0.046) | +0.0798 (p=0.0098) | +0.0757 (p=0.0009) | +0.1090 (p=0.0001) |
| `tv` | −0.0183 (n.s.) | +0.0031 (n.s.) | +0.0481 (p=0.031) | +0.0952 (p=0.0001) |

Six source configurations clear the pre-registered bar, and **every one of them targets Games**. Movies, TV and Books show nothing — into those targets, adding other domains is neutral at best and significantly *harmful* with no target data at all.

**HOST:** What does "+0.1452" mean physically? Be exact.

**YOU:** It is a **reduction in mean absolute error of 0.145 stars**. The games-only baseline predicts my rating with an average error of 0.751 stars. Add my film ratings to its training data and the average error drops to about 0.606. Roughly a fifth of the error, gone, on a domain the extra data is not even about.

**And this is where I have to correct my own resume line.** The word "skill" in *"lifting prediction skill +0.121"* is wrong. In `transfer_analysis.py` the quantity is computed as `lift = a_mae − cell_mae` — it is a **difference in mean absolute error, measured in stars**. "Skill" in this project means something else: `1 − MAE ÷ baseline MAE`, and it is a different number. I would say "reduced mean absolute error by 0.12 stars," not "lifted skill by 0.121."

### The zero-shot result, which is the one I would actually lead with

**HOST:** Which single number is the most striking?

**YOU:** Not the lift. This one, from `reports/transfer_grid_summary.json`:

> A model trained on **nothing but my film ratings**, which has never seen a single game rating, ranks my games at a **Spearman rank correlation of +0.599**. The model trained on my actual game ratings achieves **+0.307**. The film-only model is at **195% of the games model's own performance.**

**Spearman rank correlation** measures whether two orderings agree — 1.0 means identical ranking, 0 means unrelated. It ignores calibration entirely; it only asks whether the model puts my games in the right order.

That number is stark and it is defensible. It says my film taste is a *better* description of my game taste than 66 game ratings are.

**HOST:** Why measure ranking rather than error?

**YOU:** Because of the shape of my ratings. **62% of my films sit in three adjacent half-star bins.** A model that just predicts a constant 3.5 for everything already scores MAE 0.759. The whole achievable range is about a quarter of a star. So error-based metrics barely separate a good model from a constant, and worse, they conflate *ordering* with *calibration* — a model can rank perfectly and still score badly because it is offset by half a star. Games as a source into movies scored −0.069 on error-skill, reading as harmful, while its rank correlation was +0.09, meaning mildly right but badly scaled. Rank is what a recommender needs. Calibration is a one-parameter fix you apply afterwards.

I would also volunteer that reporting Spearman *after* seeing that MAE was unflattering is exactly the kind of metric-shopping a reviewer should be suspicious of. My defence is that the reason is a property of the data — the concentrated rating distribution — that is measurable and was measured, not a property of the result.

### The four controls

**HOST:** "Against four controls including a label-shuffled placebo." Walk them.

**YOU:** The obvious sceptical read is: *Games has 66 items and high variance, so it benefits from **any** added structure, regardless of whether film taste has anything to do with game taste.* Four controls attack that.

**Control 1 — the domain-identity leak, found and fixed.** This is the most important one, because it caught a real false positive. The shared feature set once contained `has_movie_feats`, `has_game_feats` and similar mask columns. A model can split on those and know instantly which medium a row came from — which means it can learn "films are rated like this, games like that" and *look* like transfer while transferring nothing. A diagnostic check flagged it as FAIL. Removing them made the features domain-blind. **The Games finding survived. A different result — an entity-bridge finding that had been reported as significant at ΔMAE +0.032, p=0.004 — collapsed to non-significant.** So the control demonstrably works: it killed one result and left another standing.

**Control 2 — prior against transfer.** Maybe the source model is just a better *constant* than the games average. No: the transfer model scores MAE 0.99 against 1.05 for the game mean, 1.17 for the source mean and 1.05 for the global mean, and it rank-tracks true game ratings at Spearman +0.47. A prior is a near-constant with a rank correlation near zero. This is content.

**Control 3 — text-length normalisation.** Game descriptions run about five times longer than film plots — median 1,074 characters against 214. If length is a hidden domain signal, the transfer could be an artifact of text shape. Re-embedding every domain to a matched median length of about 232 characters and re-running: the effect not only survived, it got *stronger*. On the raw-length space the transfer measured only +0.05 and was not significant at any fraction; normalised, every Games-source configuration cleared all four fractions.

**Control 4 — the label-permutation placebo.** The decisive one. Re-run `movie → game` with the **source ratings shuffled** and the features left intact, on the same 50 folds. If the lift is variance reduction from having more rows, shuffling changes nothing. If the lift is film-taste content, shuffling destroys it. Results, from `reports/transfer_placebo_control.json`:

| Target fraction | Real lift | Placebo lift | p, real vs placebo |
|---|---|---|---|
| 0% | +0.1316 | −0.0036 | <0.0001 |
| 25% | +0.1007 | −0.0245 | 0.0001 |
| 50% | +0.1218 | −0.0152 | <0.0001 |
| 100% | +0.1343 | +0.0529 | 0.011 |

The placebo is **zero at exactly the fractions where transfer matters** — when the target has little or no data of its own. At full target data the placebo recovers +0.053, which is pure variance reduction from the extra rows, and the real effect is still more than twice that.

That is the cleanest evidence in the project, and it is the one I would put in front of a sceptic first.

---

### The honesty section: every weak claim in that bullet, in order

**HOST:** Alright. Anything about that bullet that you would not defend?

**YOU:** **Yes. Four things, and I would rather say them than be caught.**

#### 1. "+0.121 (p=0.0003)" pairs a number from one cell with a *p*-value from a different cell.

I traced both. **+0.121** is `movie+tv+book → game`, lift at 100% target data, from the copy of `reports/transfer_grid_summary.json` in the last git commit: `"lift@100": 0.1211`. Its *p* at that fraction is **0.0014**, not 0.0003. The **0.0003** comes from `docs/ML_SYSTEM_OVERVIEW.md` line 806, which lists that cell's four fractions as `+0.085 p=0.031 · +0.081 p=0.030 · +0.130 p=0.0003 · +0.121 p=0.0014`. So **p=0.0003 belongs to the 50% fraction, where the lift is +0.130.** The bullet takes the lift from one column and the *p* from another.

The correct statement, from re-running the analysis on the current data: **`movie → game` reduces mean absolute error by 0.124 stars at 50% target data with p below 0.0001, and by 0.102 to 0.145 stars across all four fractions, all four significant, over 50 folds.**

**Say this:** *"That figure splices two numbers. The +0.121 is the 100%-fraction lift for one source combination; the p=0.0003 is that combination's 50%-fraction p-value. The honest single-cell version is stronger anyway: movie-to-game, a 0.12-star reduction in mean absolute error at half the target data, p under 0.001, significant at every one of four fractions."*

#### 2. The number is from an older run and no longer reproduces exactly.

The pipeline has been re-run since. On the current `reports/transfer_grid_results.csv`, that same `movie+tv+book → game` cell gives **+0.109** at 100%, not +0.121. **The direction, the significance and the conclusion are unchanged. The exact digits are not.** If asked for a number, quote the strongest current cell — `movie → game`, +0.145 at 100% — and say it is from the current run.

#### 3. This result has flipped its sign three times, and the repository's own documents disagree about it right now.

This is the uncomfortable one, and it is also the most interesting thing in the project.

- A small 6-fold pilot returned **null**.
- The full 50-fold grid on the 2026-07-25 corpus returned **positive**, with five Games cells.
- The grid on an intermediate corpus returned **null** again.
- The grid on the current corpus returns **positive** again, with six Games cells.

Because of that churn, the documents in this repo are not consistent with each other. `paper/PMIH_technical_report.tex` contains a table showing the Games result as significant at 4 out of 4 fractions — and, three pages later, a Conclusion section stating *"An earlier corpus returned this same grid positive into Games at +0.121 skill. That claim is withdrawn."* Both paragraphs are in the same file. `reports/FINAL_ML_TECHNICAL_REPORT.md` says the finding holds. `reports/RESUME_BULLETS.md` says the verdict was null.

**Why that happened, mechanically:** `run_pipeline.py` regenerates numbers into marked blocks in the docs automatically, and it prints an explicit warning at the end listing what it does *not* update — *"prose paragraphs outside the METRICS markers,"* *"prose in the LaTeX paper,"* *"README narrative outside the BENCHMARKS markers."* So the tables refresh and the prose does not. The narrative drifted away from its own tables.

**What changed between runs, from the paper's own list:** six unrated game rows carrying an impossible `target_reg = 0.0` were dropped from the registry; a split `lang_en` / `lang_English` one-hot — a partial domain indicator inside a supposedly domain-blind grid — was merged; the movie embedding header was capped after 69% of movie texts were found truncated inside their own header; and the shared space moved from 10-component PCA to the full 384-dimensional embedding. **None of those four is separated experimentally.** So the reversals have a description and not a cause, and I would say exactly that.

**Say this:** *"The finding is real on the current data and I can reproduce it in front of you. It has also flipped sign across corpus revisions, which is what you should expect from a domain with 66 items — significance is fragile in both directions at that N. Four things changed between runs and I have not ablated them individually, so I can describe the reversal but I cannot yet explain it. That is item seven on my future-work list, and it is one run of the grid away from being answered."*

#### 4. The alignment step in the published grid was silently doing nothing.

The grid was designed to apply **CORAL** — Correlation Alignment, a domain-adaptation method that whitens the source domain's feature covariance and recolours it to match the target's, so the two domains' feature distributions overlap before you train across them.

`_fit_predict()` selected the columns to align by name. It looked for columns starting with `pca_`. When the shared space moved to the full embedding, those columns became `emb_*`. **The selector matched nothing, the alignment silently became a no-op, and no error was raised.** The grid published on 2026-08-30 was run that way.

Two things save it, and one does not:

- **Every arm of the grid was affected identically**, so the comparisons between cells — which is what the finding rests on — remain valid.
- The numbers are therefore **unaligned transfer**, not aligned transfer, and the repository says so in the function's own docstring.
- What it does *not* save: I do not know whether the effect is bigger or smaller with alignment restored. The fix is in the code — one shared `text_feature_cols()` helper that accepts both names, so four separate files can no longer drift — but the grid has not been re-run with it.

**Say this:** *"The CORAL alignment was a silent no-op in the published grid because the column selector still looked for the old PCA column names. Every cell was affected identically, so the comparisons hold, but the absolute numbers are unaligned transfer and I label them that way. The class of bug — a string-prefix selector silently matching nothing — is why there is now exactly one function that decides what the text block is called."*

#### 5. Three of the four controls were run on older corpora.

`reports/transfer_placebo_control.json` is dated 2026-06-27. The text-normalisation control script last ran in June. Only the domain-identity fix is live in the current feature builder. The placebo's real lifts (+0.132 / +0.101 / +0.122 / +0.134) happen to track the current run's movie→game lifts (+0.102 / +0.101 / +0.124 / +0.145) closely, so it is very likely still valid — but *likely* is not *measured*, and I would say "re-running the placebo on the current corpus is one command and I should have run it."

> **Interview note.** Five volunteered flaws in one segment sounds like a lot. It is not. What an interviewer takes away is: *this person knows their own result at the level of individual output files, can distinguish "the conclusion holds" from "the digits reproduce," and does not need to be caught.* The candidate who says "I proved taste transfers, p equals 0.0003" and cannot say which cell it came from loses on the follow-up.

### The 16-page dashboard — verified

`app/pages/` contains exactly 16 files, plus a home page at `app/main.py`:

| # | Page | What it does |
|---|---|---|
| 1–2 | Movies Dashboard / Oracle | Library analytics; score any film |
| 3–4 | TV Shows Dashboard / Oracle | Same for TV, with season-level watch data |
| 5–6 | Music Dashboard / Oracle | Listening analytics; unsupervised recommender (no ratings) |
| 7–8 | Games Dashboard / Oracle | Same for games |
| 9–10 | Books Dashboard / Oracle | Same for books |
| 11 | YouTube Dashboard | Watch history from Google Takeout — ingested, deliberately not modelled |
| 12 | Latent Space Explorer | Interactive map of the 384-dimensional embedding space, dimension-reduced |
| 13 | Model Calibration | Predicted-against-actual, conformal coverage |
| 14 | Taste Drift | How my ratings have moved over time |
| 15 | Transfer Atlas | The transfer grid, rendered |
| 16 | Collaboration Network | The Wikidata entity-link graph — shared creators, adaptations, franchises |

**HOST:** Why is YouTube ingested but not modelled?

**YOU:** Because watch time is a weak and confounded label. I watch things I dislike, I leave things playing, I rewatch things for background noise. Turning "minutes watched" into a rating would be inventing a target, which is exactly the thing I removed for music. It is scoped as a separate implicit-feedback study, and until it is done it stays a dashboard.

### The other real finding — entity bridges

**HOST:** Anything else that came out of this?

**YOU:** One, and it points in the opposite direction from the pooling result, which is why I like it.

`src/linking/build_entity_links.py` resolves every library item to a **Wikidata** entity — Wikidata being a structured, machine-readable database behind Wikipedia. It then follows specific relation properties: *based-on* and *derivative work* for adaptations, *author* / *director* / *composer* for shared creators, and *series* for franchises. It currently produces **2,739 cross-domain links**.

The question is item-level rather than feature-level: *if I rated the book, does that predict my rating for the film?* The evaluation adds fold-safe bridge features — the linked partner's rating, a franchise mean, a has-link flag, all counted only when the partner sits in the training fold — and measures the effect.

Result, from `reports/entity_bridge_results.json`: over **664 linked rated items**, MAE improves from 0.5301 to 0.4940, a **ΔMAE of +0.0361**, 95% confidence interval [+0.0143, +0.0572], Wilcoxon *p* = 0.0014. The adaptation-only slice — 57 items — is directionally positive at +0.0175 but not significant, *p* = 0.317.

Taken with the pooling result, the direction of the project is one sentence: **link items, do not pool features.**

**And the same honesty applies.** This result has also reversed twice, and the entity graph and the corpus changed in the same pipeline run — the no-bridge baseline itself moved from 0.5180 to 0.5301 — so the effect cannot be attributed to the improved graph alone. Scoring the corrected graph against the old corpus would settle it in one run and has not been done.

**HOST:** What improved in the graph?

**YOU:** A resolver bug with a memorable failure mode. `resolve_qid` used to fall back to the first Wikidata search hit whenever nothing matched its type hint. So the film *Batman* resolved to **Batman Province, Turkey**. *Ghostbusters* resolved to the 1984 song. *Toy Story* resolved to the franchise entity rather than the film. Rewriting it to refuse rather than guess raised adaptation links from 21 to 32, and — the one that matters — **film-to-book adaptations went from 2 to 18**, which is the highest-signal link type and the one the test depends on. The total linked-item count barely moved, 665 to 664, so this was not a coverage gain: same items, edges now pointing at the right entities.

---

## SEGMENT 6 — "Why this and not that" (52:00 – 58:00)

**HOST:** Rapid fire. Short answers.

**Why XGBoost and CatBoost, not a neural network?**

**YOU:** 1,407 rows. A neural network needs orders of magnitude more data to beat gradient boosting on tabular features, and gradient boosting is still the default winner on tabular problems at any size. I *do* use a neural network — the sentence transformer — but I use it as a pre-trained feature extractor, which is exactly where a small dataset can afford one: the language understanding was learned on somebody else's corpus and I only consume it.

**Why SVR for games and books rather than the same boosting?**

**YOU:** Capacity control. With 66 items and 400 features, boosting can memorise every row. SVR fits one smooth global function and cannot. I did not guess — both were run on the same 50 folds and SVR won on those two domains and lost on movies.

**Why absolute error rather than squared error?**

**YOU:** Squared error punishes a two-star miss four times as hard as a one-star miss, which drags predictions toward the safe middle. On a rating distribution where most items sit in three adjacent half-star bins, that produces a model that predicts 3.5 for everything and scores well while being useless.

**Why 50 folds rather than one held-out test set?**

**YOU:** Because with 66 games a single split is a coin flip, and because the paired statistical test needs many matched comparisons on identical items. A single split gives me one number and no way to know whether a difference is real.

**Why "frozen" folds written to a file rather than re-splitting each run?**

**YOU:** So that every experiment tests on the same items. A paired test comparing two models that were evaluated on different splits is not a paired test. Freezing the registry is what makes the whole comparison structure legitimate, and it costs one JSON file.

**Why stratify the folds?**

**YOU:** Without it, a random 10-way split of 66 games can easily produce a fold with no five-star games in it. Stratifying on media type crossed with rating bin keeps every fold's domain mix and rating curve close to the whole dataset. There is a small subtlety in the code: small domains cannot support 10-fold on a rare bin, so within each domain the rarest bin is iteratively merged into the largest until every kept bin has at least ten members.

**Why Wilcoxon rather than a t-test?**

**YOU:** A t-test assumes the differences are roughly bell-shaped. My per-fold error differences are not — they are bounded, skewed, and computed on tiny test sets. Wilcoxon works on ranks and makes no distributional assumption. It has slightly less power when a t-test's assumptions do hold, and that is a price worth paying.

**Why report Spearman for the zero-shot result and MAE for everything else?**

**YOU:** Because 62% of my films sit in three adjacent half-star bins, so a constant predictor already scores MAE 0.76 and the achievable range is a quarter of a star — MAE cannot separate ordering from calibration. Rank is what a recommender actually needs. I would also concede that switching metrics after seeing the first one is exactly what a reviewer should be suspicious of, and my defence is that the reason is a measurable property of the data, not of the result.

**Why Streamlit and not React?**

**YOU:** One user, and the content is charts over pandas dataframes. A React front end plus an API layer would be days of work for zero capability I do not already have. When the interesting problem is the modelling, spending the effort on the front end is a way of avoiding the interesting problem.

**Why Hardcover for books and not Goodreads?**

**YOU:** Goodreads shut its public API to new developers. Hardcover exposes a GraphQL API where a single query returns the library record — title, my rating, authors, genres, page count, description, plus mood and tag fields — so there is no join and nothing to mismatch. That single decision is a large part of why Books has the best score of any domain.

**Why exclude language, country and MPAA rating from the shared feature space?**

**YOU:** Because they are near-pure domain indicators. An MPAA rating exists for film and TV and not for books, so a model that splits on it has learned the domain label, not taste. This is not hypothetical: the split `lang_en` / `lang_English` one-hot was exactly that bug — films and TV carried ISO codes from TMDB, books carried spelled-out names from Hardcover, so one language became two orthogonal columns that together encoded "is this a book." Merging them **worsened** unified MAE from 0.4979 to 0.5025 on identical items, and I report that as a **confound removed, not a regression**, because the previous number was partly the model reading a domain label.

**Why cache every API response to disk?**

**YOU:** Because a full re-run of the pipeline would otherwise make tens of thousands of network calls, hit rate limits, and produce a different dataset each time depending on which calls failed. `data/cache/` holds the JSON responses so a re-run is deterministic and offline.

**What surprised you most?**

**YOU:** That eight real data defects were found in an audit and **seven of them moved no offline metric at all.** Wrong genre parsing, a rating-category function that misread "Not Rated" as adult content and mislabelled 12.4% of the movie library, truncated text, double-encoded characters — all present, none visible in MAE. The reason is structural: cross-validation protects against optimism *in the split*. It gives zero protection against a defect present identically in the training rows and the test rows. That gap now has its own gating checker, `scripts/data_sanity_check.py`, which fails the build on row-count disagreement between frames, on values that still coerce to missing after the cleaner has run, on double-encoded UTF-8, and on ratings off the half-star grid. **That is the methodological result I expect to reuse in every future project.**

**What is your biggest single regret?**

**YOU:** Letting the prose drift from the artifacts. The numbers are regenerated by one script into marked blocks; the narrative around them is hand-written and was not. That is how a paper ended up with a table showing a significant result and a conclusion three pages later calling the same result withdrawn. The fix is not more discipline, it is fewer hand-written numbers — every figure in the prose should be a template slot filled from the same JSON the tables read.

---

## SEGMENT 7 — Ten questions to have answers ready for (58:00 – 62:00)

**HOST:** Last stretch. Short, complete answers.

**1. Walk me through what happens end to end.**

Five personal exports and library APIs are pulled — Letterboxd for films, Serializd for TV, a games CSV, the Hardcover GraphQL library for books, Spotify for music — and each is enriched against public catalogues: TMDB and OMDb for films, TMDB plus TVMaze for TV, RAWG for games, ReccoBeats and MusicBrainz and Genius for music. Every response is cached to `data/cache/`. Each domain then builds its own feature matrix. A separate builder, `unified_feature_engineering.py`, projects all four rated domains into one shared 406-column space: 384 chunk-pooled sentence-embedding dimensions, 17 cross-domain genre flags, four smoothed creator and studio target encodings, and one normalised critic average. `create_frozen_folds.py` writes a 10-fold × 5-repeat stratified registry to JSON. `comprehensive_evaluator.py` runs the unified ensemble and every local model on those identical 50 folds, deduplicates the five repeats per item, asserts that N equals unique items and that standalone is not the unified slice, and writes one JSON with a git hash. `transfer_study.py` runs the 120-cell grid on the same folds; `transfer_analysis.py` applies the pre-registered decision rule. `render_docs.py` renders the numbers into marked blocks in the README and reports. A 17-page Streamlit app serves predictions with SHAP attributions, conformal intervals and nearest-rated-neighbour explanations.

**2. What breaks first at scale?**

Nothing about compute — the whole pipeline is a few thousand rows. What breaks is **statistical**, and it breaks in the other direction: the project is already at the edge of what 66 games can support. If I added a second user, the entire design assumption collapses, because every target encoding, every taste feature and the whole framing is per-person. The honest scaling path is not more rows in the same table; it is a per-user model with a shared feature extractor, which is a different system.

The nearest genuine compute limit is the transfer grid: 120 cells × 50 folds × an XGBoost fit each is an overnight run, and it grows as the product of domains and subsets. Adding a fifth rated domain roughly doubles it.

**3. Where is your biggest technical debt?**

Three, in order. **The entity-encoding leak** — `oof_entity_encode` runs its own inner five-fold that is unrelated to the frozen registry, so 48% of training rows carry an encoding computed from a test row's target, and every unified metric is optimistic by an unmeasured amount. **The doc-artifact drift** — the paper's conclusion contradicts the paper's own table because prose is not regenerated. **The movie scoring path** — `movies/predict_ratings.py` zero-fills 23 of its 107 features, so a single-title score from the batch path is worse than the benchmark table implies; the measured cost is stacking MAE 0.375 to 0.529 on that path. Training is unaffected, and it is documented rather than hidden.

**4. What would you do differently?**

Fix the encoding leak first, because it is the only thing that makes a published number wrong rather than incomplete — re-encode inside each registry fold and re-measure everything unified. Then separate the transfer reversal: re-run the grid with each of the four corpus changes reverted in turn, so the reversal has a cause and not just a description. Then re-run the placebo and text-normalisation controls on the current corpus. Then make every number in the prose a template slot fed from the same JSON the tables read.

**5. Convince me this is not just overfitting to 66 games.**

Four independent arguments, and none of them is a *p*-value. The effect is consistent across four target fractions, not one. It is consistent across six different source combinations, not one. A label-permutation placebo — same features, source ratings shuffled — collapses the lift to essentially zero at exactly the fractions where transfer matters. And the zero-shot model, which never touches a single game rating, ranks games at Spearman +0.599 against the games-own model's +0.307, so the effect cannot be memorisation of game data that the model has not seen.

**6. Your p-values assume independence. Do you have it?**

No. Fifty folds over 66 unique items means each item appears in five test folds, so the paired differences are correlated and the *p*-values are optimistic by an amount I have not quantified. That is why I lead with consistency across fractions and sources rather than with a *p*. I would also note that the multiplicity argument cuts both ways: about 120 uncorrected cells at α = 0.05 would throw roughly six false positives by chance, and every single positive cell in this grid lands in the same target domain, which is not what scattered false positives look like.

**7. What is the hardest thing in the codebase?**

Not an algorithm — a discipline. Making the standalone models and the unified model comparable required them to be refit on the same frozen folds, deduplicated the same way, rounded at the same point, and scored by the same function, with assertions that crash the run if any of that separation breaks. Getting that wrong is what produced a table that compared a model to itself and published the resulting conclusion. The Wordle-style "clever algorithm" answer would be the chunk-pooled embedding; the honest answer is the measurement layer.

**8. If a company gave you this project, what would you ship first?**

Nothing about transfer. I would ship the **local models plus the conformal intervals plus the active-learning queue**, because that is the part that is already correct, already explainable, and already useful — it tells you what to read next and how confident it is. The transfer study is research; it belongs in a paper, not in a product, until the reversals are explained.

**9. What did the negative results teach you?**

More than the positive one. Ridge stacking and residual-head correction both destroyed signal at small N and were retired by paired ablation. The distillation prior — using the unified model's prediction as an extra feature for a sparse domain — was null in both domains it was tested on, *p* = 0.72 for games and *p* = 0.25 for books, so it was dropped and the simpler equal-weight mean ensemble kept. The custom asymmetric edge-penalty loss I wrote to make the model braver at the ends of the scale is mis-specified — it weights by the *label* rather than by error magnitude and direction — and is a small net harm. Every one of those is in the repo with its *p*-value and a DROP verdict, and a project where nothing is ever dropped is a project where nothing is ever really tested.

**10. What did you learn?**

That in a small-data project, the measurement layer *is* the project. Two bugs in how numbers were computed each reversed a scientific conclusion, and neither was in a model. That controls are worth more than significance — the label-shuffled placebo told me more in one run than any *p*-value. And that a result which flips sign across corpus revisions is telling you something true about your sample size, which is information rather than failure, as long as you report it as information rather than quietly re-running until it comes back the way you liked.

---

## SEGMENT 8 — Cheat sheet (read this twice)

### Numbers to have on the tip of your tongue

| Thing | Value | Where it comes from |
|---|---|---|
| Rated items total | **1,514** (2026-10-01 run) | `reports/latest_metrics.json`, `unified_rated` N |
| Per domain | 1,163 movies / 175 TV / 110 books / **66 games** | same file, `slices` |
| Rows in the feature matrix | 1,520 (6 unrated games excluded) | `data/processed/unified/training_features.csv` |
| Music rated items | **0** — music has no ratings, ever | by design |
| Unified features, actual | **406** (411 columns − 5 bookkeeping) | measured from the CSV |
| Unified features, per last commit's README | 231 | stale; the resume's number |
| Transfer-grid shared space | **402** features | re-ran `select_features()` |
| Embedding | 384 dims, `all-MiniLM-L6-v2`, chunk-pooled | `src/common/text_embedding.py` |
| Genres in shared space | 17 (must appear in ≥2 media) | `genres_in_multiple_domains()` |
| Folds | **50** = 10 splits × 5 repeats, seed 42 | `create_frozen_folds.py` |
| Local model skill (primary) | movie 0.355 / TV 0.208 / game 0.376 / book 0.566; Spearman 0.751 / 0.565 / 0.719 / 0.859 | `baseline_sanity.json` |
| Local model MAE | movie 0.483 / TV 0.694 / game 0.515 / book 0.286 | `latest_metrics.json` |
| Unified slice MAE | movie 0.513 / TV 0.714 / game 0.659 / book 0.500 | same — **local wins all four** |
| Why skill first | a constant 3.5 already scores MAE 0.749 on films; skill is 0 for guessing the mean | `production_oof.csv` |
| Best transfer cell, current run | `movie → game`, ΔMAE **+0.145** at 100%, +0.124 at 50%, all four fractions p < 0.001 | recomputed from `transfer_grid_results.csv` |
| Resume's +0.121 | `movie+tv+book → game` at 100%, **last commit's** summary; its p there is 0.0014 | `git show HEAD:reports/transfer_grid_summary.json` |
| Resume's p=0.0003 | the **50%** fraction of that same cell, where the lift is +0.130 | `docs/ML_SYSTEM_OVERVIEW.md:806` |
| Zero-shot headline | film-only model ranks games at Spearman **+0.599**; games-own model **+0.307** (195%) | `transfer_grid_summary.json` |
| Placebo control | real +0.132/+0.101/+0.122/+0.134 vs shuffled −0.004/−0.025/−0.015/+0.053 | `transfer_placebo_control.json` (dated 2026-06-27) |
| Entity bridges | 2,739 links; 664 linked rated items; ΔMAE +0.0361, CI [+0.0143, +0.0572], p = 0.0014 | `entity_bridge_results.json` |
| Adaptation-only bridges | 57 items, +0.0175, p = 0.317 — null pilot | `entity_bridge_results_adaptation.json` |
| Entity-encoding leak | **48%** of unified train rows use a test row's target | documented in `unified_feature_engineering.py` |
| Conformal intervals | vanilla split-conformal, **80%** target, coverage within a few points | `active_learning_ranker.py` |
| Dashboard | **16** pages in `app/pages/` + home page | verified by listing |
| CatBoost's contribution | ΔMAE +0.0076, d = 0.068 — "significant-but-trivial" | `latest_metrics.json` ablation |
| Distillation prior | game p = 0.32, book p = 0.47 → **DROP** both | `latest_metrics.json` |
| Optuna trials | 10 (unified trainer) / 25 (production benchmarks) | the trainer scripts |
| Target-encoding smoothing | **m = 2** (swept: 0→0.477, 1→0.473, 2→0.469, 10→0.480) | `taste_features.py` |
| Stack versions | XGBoost 3.2.0, CatBoost 1.2.10, Optuna 4.9.0, scikit-learn 1.8.0, sentence-transformers 5.2.3, SHAP 0.52.0, Streamlit 1.54.0 | `requirements.lock.txt` |

### File map — the fifteen paths worth naming out loud

| Path | What lives there |
|---|---|
| `src/config.py` | Every path and a couple of global constants |
| `src/common/text_embedding.py` | Chunk-pooled sentence embeddings — the cross-domain bridge |
| `src/common/taste_features.py` | Leakage-guarded target encodings; the `m = 2` sweep |
| `src/unified_model/unified_feature_engineering.py` | Builds the 406-column shared matrix; carries the leak caveat |
| `src/unified_model/create_frozen_folds.py` | The 10 × 5 stratified frozen fold registry |
| `src/unified_model/comprehensive_evaluator.py` | Master evaluation, dedup, the two acceptance assertions |
| `src/unified_model/unified_utils.py` | `DomainAligner` (CORAL), temporal weights, `text_feature_cols` |
| `src/unified_model/advanced_unified_model_trainer.py` | Optuna tuning + XGB/CatBoost simplex ensemble; the fixed tuning-split bug |
| `src/reporting/standalone_benchmarks.py` | Genuine local models on the same folds — the fix for the benchmark-inversion bug |
| `src/reporting/metrics_writer.py` | The single source of truth: atomic write, git hash, round-trip check |
| `src/experiments/transfer_study.py` | The 120-cell grid; protocols A/B/C; the CORAL no-op docstring |
| `src/experiments/transfer_analysis.py` | The pre-registered decision rule and the paired Wilcoxon |
| `src/experiments/transfer_placebo_control.py` | The label-shuffled placebo |
| `src/linking/build_entity_links.py` | Wikidata resolution; the Batman-Province-Turkey bug |
| `app/pages/2_Movies_Oracle.py` | SHAP waterfall, conformal interval, similarity explanations |

### The five sentences that carry the whole interview

1. *"One user, 1,514 ratings, and only 66 games — so the entire design is about not fooling yourself at small N: frozen folds shared by every experiment, a decision rule fixed before any result, and controls rather than p-values."*
2. *"The finding is that taste transfers **into games and only into games**: a model trained on nothing but my film ratings ranks my games at Spearman +0.599 against the games-own model's +0.307, and adding film data cuts games' mean absolute error by about 0.12 stars at every level of target data."*
3. *"The sentence embedding is the only feature that means the same thing in a novel and a video game, and chunk-pooling it instead of truncating to 40 words is what made the domains comparable — book text signal went from 0.045 to 0.098 across five seeds."*
4. *"Two measurement bugs each reversed a scientific conclusion — a 5× N inflation and a benchmark table publishing the unified model's slice as the standalone models — so I rebuilt the metrics path as one script writing one JSON with assertions that crash the run rather than publish a wrong table."*
5. *"Every unified metric is optimistic by an unmeasured amount because the entity target encodings use their own inner split rather than the frozen registry — 48% of train rows see a test row's target. I know the size of the exposure, I know the fix, and no claim I make rests on a margin smaller than it."*

### Three things to volunteer before you are caught

- **The "+0.121 (p=0.0003)" in the bullet splices two cells.** +0.121 is the 100%-fraction lift for `movie+tv+book → game` from the last commit's summary; its *p* there is 0.0014. The 0.0003 is that cell's 50%-fraction *p*, where the lift is +0.130. And "skill" is the wrong word — the quantity is a **reduction in mean absolute error, in stars**. Say the current, stronger, single-cell version instead: *movie → game, +0.124 stars at 50% target data, p < 0.0001, significant at all four fractions over 50 folds.*
- **The transfer result has flipped sign three times across corpus revisions, and the repo's own paper contradicts itself about it right now** — a table showing 4-of-4 significance and a conclusion three pages later calling the same result withdrawn. The cause is mechanical: `run_pipeline.py` regenerates the tables and prints a warning that it does *not* regenerate the prose. Four corpus changes fall between runs and none has been ablated, so the reversal has a description and not a cause.
- **The CORAL alignment in the published grid was a silent no-op**, because the column selector still looked for `pca_*` names after the shared space moved to `emb_*`. Every cell was affected identically so the comparisons hold, but the numbers are *unaligned* transfer and the code says so in its own docstring.

Two smaller ones to have ready if pressed: **"231 features" is stale** — the matrix is 406 today, and the reason is the PCA-to-full-embedding change that made the finding work. **"Explains every prediction in plain language"** is really a SHAP attribution chart plus a written similar-titles caption, and the Music Oracle has no explanation at all because music has no ratings.

### If you do not know an answer

Say this, and then actually do it: *"I do not know that off the top of my head. Here is how I would find out —"* and describe where in the repo you would look. In this project that is unusually credible, because the answer to almost every factual question is a specific output file with a git hash in it, and saying *"that would be in `reports/latest_metrics.json`, which is written by one script and carries the commit that produced it"* is a better answer than a remembered number.

The one thing never to do is quote a *p*-value you cannot locate. Everything else in this project can be defended for twenty minutes. A number you cannot trace collapses on the second question.

---

*End of script. Total run time at a conversational pace: roughly one hour.*
