# YouTube: Data Inventory & Modeling Options

*Scope: what YouTube data we actually hold, what signal it carries, and the realistic ways to model it — with the music PU-affinity pipeline as the closest precedent. Written as a decision aid, not a commitment. Numbers below are from the live local data (watch-history export of **2026-06-17**, span **2025-10-20 → 2026-06-17**, 240 days).*

---

## 1. What we have

YouTube currently lives in the project as **ingestion-only** (the dashboard at `app/pages/11_YouTube_Dashboard.py`). Nothing feeds a model yet.

### 1.1 Raw (`data/raw/youtube/`, git-ignored)
| File | Source | Contents |
| :--- | :--- | :--- |
| `watch-history.json` | Google Takeout | 12,200 activity events → **9,542 genuine video watches** after dropping 1,746 community-`/post/` entries and rows with no channel. Fields: `title, titleUrl, subtitles (channel name+url), time, products`. |
| `subscriptions.csv` | Google Takeout | **328 channel subscriptions** (`Channel ID, Channel URL, Channel title`). |

### 1.2 Processed / API-enriched (`data/processed/youtube/`, git-ignored)
Produced by `src/youtube/enrich_yt.py` (YouTube Data API v3), merge-safe and incremental.

| File | Rows | Columns |
| :--- | :--- | :--- |
| `youtube_video_details.csv` | 10,934 | `video_id, channel_id, category_id, published_at, duration_iso, tags, description (≤500 chars), public_views, public_likes, comment_count` |
| `youtube_channel_details.csv` | 4,408 | `Channel ID, channel_title, subscriber_count, video_count` |

**100% of watched videos** are enriched with category/duration/engagement metadata.

### 1.3 The data at a glance
- **8,544 unique videos** across **3,473 unique channels**.
- **Rewatches:** 521 videos watched >1×, 169 watched ≥3×, 44 watched ≥5×, **max 29×** — a graded intensity signal.
- **Format:** ~31% Shorts (<61s).
- **Category mix (watches):** Entertainment 2,479 · People & Blogs 1,490 · Film & Animation 1,151 · Music 1,148 · Gaming 1,007 · Education 766 · Comedy 708 · Science & Tech 390 …
- **Media-adjacent** (Film/Gaming/Music) ≈ **35%** of watches — the slice that could plausibly bridge to the rated domains.
- **Long tail:** 73% of channels were watched exactly once; the top 50 channels are only 28% of watches → taste is broad, not concentrated.
- **Intent vs. algorithm:** only **33%** of watches come from subscribed channels; the other ~67% is search/recommendation/autoplay — i.e. *much of this is not deliberate taste.*

---

## 2. The fundamental challenge (and why Music is the right analogy)

Like Music, YouTube has **no explicit ratings** — only **implicit, positive-only** signal. You see what was watched, never what was deliberately rejected. This is exactly the **Positive-Unlabeled (PU)** setting the music pipeline already solves (`src/music/affinity.py`, `background_pool.py`, quantile-calibrated pseudo-ratings).

But YouTube differs from Music in three ways that shape every option below:

1. **No bounded candidate universe.** Music's "unlabeled" pool is *your Spotify library + playlists* — a finite, taste-relevant set. YouTube's universe is *all of YouTube*. PU needs a deliberately constructed negative/unlabeled pool (see §3.1).
2. **Heavy algorithmic confound.** ~67% of watches are recommendation/autoplay-driven, not chosen. A watch is a much weaker "positive" than a Spotify save. Rewatch count and subscription are the *cleaner* positives.
3. **Global vs. personal signal.** `public_views/likes/comment_count/subscriber_count` are the *same for everyone* — they're popularity, not your taste. Only **presence, frequency (rewatch), recency, and subscription** are personal.

---

## 3. Modeling options

Ordered from "closest to existing infra" to "most speculative." Each notes what it buys, what it costs, and the honest odds.

### 3.1 Option A — YouTube PU-Affinity model *(the direct Music analog; recommended first)*
Mirror the music pipeline. **Positives** = watched videos (weighted by rewatch count + recency). **Unlabeled pool** = a constructed candidate set, e.g. uploads from subscribed-but-unwatched channels + a random/trending sample. Train a PU classifier, calibrate to a 1–5 **pseudo-affinity** via quantiles, expose a "YouTube Oracle" + recommendations.
- **Buys:** a self-consistent YouTube taste model reusing `affinity.py` patterns; a 6th domain for the hub; channel/video recommendations.
- **Granularity matters:** model at the **channel/topic level**, not raw video level — per-video PU is dominated by autoplay noise; channel affinity (watch count, recency, subscribed flag) is far cleaner.
- **Cost / risk:** must define the unlabeled pool (no natural one exists); the pseudo-label inherits the same caveat already documented for music — *you'd partly be predicting your own heuristic.* Evaluate with **ranking metrics** (does it rank held-out watched channels above sampled-unwatched?) and skill-vs-popularity-baseline, never R².

### 3.2 Option B — Graded implicit feedback (confidence-weighted)
Instead of binary PU, treat **rewatch count + recency** as a *confidence* on the positive (the Hu et al. 2008 implicit-ALS framing). The 29×-rewatched video is a strong positive; a single autoplay is weak. Produces a continuous engagement score per channel/topic.
- **Buys:** uses the richest personal signal we have (rewatch up to 29×) and aligns with the project's existing **temporal decay** weighting (`w_i = max(exp(-λΔt), w_min)`).
- **Cost:** still no negatives; best combined with A (confidence weighting *inside* the PU fit).

### 3.3 Option C — YouTube as features for the **unified** taste model *(cross-domain)*
Use watch history to build **taste priors** that feed the rated domains (Movies/TV/Games/Books). Two sub-forms:
- **Channel/topic→genre affinity prior:** aggregate watched topics/categories into genre weights, join to each library item's genres → "you watch a lot of this genre on YouTube." Broad coverage (35% of watches are media-adjacent).
- **Per-item engagement via entity bridge:** link videos to library items (trailers, reviews, OSTs) using the existing `src/linking/` Wikidata bridge → "watched N trailers/reviews of this title before rating it."
- **Buys:** could lift the semantic-only domains (TV/Books) where pooling already earns its keep.
- **Cost / honest odds:** **low.** A global watch-centroid is non-discriminative across items (one constant). The per-item bridge is sparse and carries a **reverse-causation leak** (you watch the trailer *because* you'll watch the film). The project's own track record argues caution: the **entity bridge came back non-significant** (ΔMAE +0.017, p=0.18) and the **distillation prior was dropped**. A YouTube engagement feature is cut from the same cloth. If tried, the **genre-affinity prior** (coverage) beats the per-item bridge (sparsity).

### 3.4 Option D — Standalone "YouTube Oracle" (recommendation, not rating)
Skip ratings entirely; build a **channel/video recommender** (content-based: embed `title + tags + description` with MiniLM, score similarity to a watch-history taste centroid; or collaborative-style via PU from A). Surfaces "channels you'd likely watch" and "ghost-sub cleanup" (already partly in the dashboard).
- **Buys:** genuinely useful UX, no pseudo-rating fiction needed; honest about being a *retrieval* task, not a *prediction* task.
- **Cost:** doesn't integrate with the cross-domain "taste" thesis; it's a parallel utility.

### 3.5 Option E — Behavioral / diagnostic modeling (no taste claim)
Model *habits* rather than *taste*: session/binge prediction, shorts-vs-longform drift, discovery-rate trends, watch-time forecasting. Pure descriptive/temporal analytics (much already in the dashboard).
- **Buys:** robust, no label problem, honest.
- **Cost:** not "taste intelligence" — it's usage analytics.

---

## 4. Signal inventory

| Signal | Type | Personal? | Strength | Caveat |
| :--- | :--- | :--- | :--- | :--- |
| Watched (presence) | implicit positive | ✅ | medium | ~67% algorithm-driven, not chosen |
| Rewatch count (≤29×) | graded positive | ✅ | **high** | rewatch ≠ love (loops, background music) |
| Recency (timestamps) | temporal weight | ✅ | medium | aligns with project's decay model |
| Subscription | explicit positive | ✅ | **high** | 109 of 328 are "ghost" (never watched) |
| Session co-occurrence | contextual | ✅ | low–med | autoplay chains inflate this |
| Category / tags / description | content | ⬜ (item-level) | medium | great for embeddings/genre |
| Duration | content | ⬜ | low | mainly Shorts detection |
| `public_views/likes/comments` | global popularity | ❌ | — | identical for all users; redundant w/ TMDB popularity |
| `subscriber_count` | global popularity | ❌ | — | "creator size," not taste |

---

## 5. Threats to validity (state these up front, per project convention)
- **No negatives, no ratings.** Everything is PU; any "rating" is a calibrated pseudo-label (the same footnoted caveat as the *Unified (full pool)* music row).
- **Algorithmic confound.** 67% non-subscription watches mean the positive class is polluted by what YouTube *pushed*, not what you *chose*. Down-weight or filter to subscription + rewatch positives.
- **Global ≠ personal.** Drop or quarantine popularity columns from any *taste* feature; they measure the video, not you.
- **Rewatch ambiguity.** High replays can mean "favorite" or "background loop" — disambiguate with duration/category (Music-category loops vs. a re-watched essay).
- **Single-user.** Same ceiling as the rest of the hub; nothing generalizes across users.
- **Reverse causation (cross-domain).** Watching a trailer predicts "will rate," not "rates highly."

---

## 6. Recommendation

1. **Start with Option A+B at channel/topic granularity** — it reuses the music PU + temporal-decay machinery, has the cleanest positives (subscription + rewatch), and yields an honest, self-contained YouTube affinity model. Evaluate by **held-out ranking** (watched-channel vs. sampled-unwatched) and **skill over a popularity baseline**, never R².
2. **Treat Option C (cross-domain features) as a pre-registered pilot, not a default.** Test the **genre-affinity prior** first (coverage), gate it through the existing **frozen-fold paired-Wilcoxon ablation**, and be ready to drop it — the entity-bridge and distillation-prior history says the odds are low.
3. **Ship Option D's recommender as the user-facing payoff** regardless — it needs no pseudo-rating and is genuinely useful.
4. **Avoid** leaning on global popularity columns as taste features, and avoid per-video PU (autoplay noise).

---

## 7. Concrete next steps (if we proceed)
- [ ] Define the **unlabeled pool**: subscribed-channel uploads not watched + random/trending sample → `data/processed/youtube/yt_pool.csv`.
- [ ] Build `src/youtube/feature_engineering.py`: channel-level positives (watch count, recency-decayed weight, subscribed flag) + MiniLM embedding of aggregated `title+tags+description`.
- [ ] Adapt `src/music/affinity.py` PU + quantile calibration → `src/youtube/affinity.py`.
- [ ] Evaluation harness: ranking AUC on held-out watched channels + skill vs. "recommend by global popularity."
- [ ] (Pilot, separate) genre-affinity prior into `unified_feature_engineering.py`, judged by the frozen-fold ablation; report the verdict either way.

*Related: [`src/music/readme.md`](../src/music/readme.md) (the PU precedent), the cross-domain transfer study, and the entity-bridge result in `reports/`.*
