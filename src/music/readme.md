# Music — Intelligence Pipeline

Extends the Personal Media Intelligence Hub to your **music** library, following
the same Ingestion → Feature Engineering → Model pattern as Books/Movies/Games.

## The Spotify deprecation (why this isn't pure Spotify)

On **27 Nov 2024** Spotify removed several Web API endpoints for any app created
after that date: **Audio Features, Audio Analysis, Recommendations, Related
Artists**, plus featured/editorial playlists and 30-second previews in multi-get
responses. New apps now get a **403** on `/v1/audio-features`, and in May 2025
Spotify set a **250,000 monthly-active-user** floor for extended access — so a
personal project can't get those back.

This pipeline routes around it with free sources:

| Need | Old (dead) | This pipeline uses |
|------|------------|--------------------|
| Library + metadata | Spotify | **Spotify** (still fine) |
| Audio features (danceability, energy, valence, tempo…) | Spotify `/audio-features` | **ReccoBeats** (free, same metrics, keyed by Spotify ID) |
| Genre / "related" signal | Spotify Related Artists | **MusicBrainz** genres + folksonomy tags |
| Lyrics | — | **Genius** |
| Recommendations | Spotify `/recommendations` | **ReccoBeats** `/track/recommendation` |

ReccoBeats can occasionally miss a track; every stage caches to
`data/cached/music/` and degrades gracefully (a missing feature becomes an
imputed value plus a `*_missing` flag, never a crash).

## Install

```bash
pip install spotipy requests python-dotenv pandas numpy tqdm \
            lyricsgenius vaderSentiment sentence-transformers \
            scikit-learn xgboost catboost joblib
```

Add keys to `.env` (see `.env.example`):
```
SPOTIPY_CLIENT_ID=...
SPOTIPY_CLIENT_SECRET=...
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
GENIUS_ACCESS_TOKEN=...
MUSICBRAINZ_CONTACT=you@example.com   # required by MusicBrainz' fair-use policy
```

## Run order

```bash
python src/music/spotify_ingestion.py        # 1. library + ReccoBeats audio features -> music_library.csv
python src/music/musicbrainz_enrichment.py   # 2. MBIDs + genre tags                  -> music_musicbrainz.csv
python src/music/genius_lyrics.py            # 3. lyrics + VADER sentiment            -> music_lyrics.csv
python src/music/feature_engineering.py      # 4. merge + features                    -> music_processed.csv, music_features.npz
python src/music/model_trainer.py            # 5. train ensemble + save model
```

The Oracle, once trained:

```bash
python src/music/model_trainer.py --predict <spotify_track_id>
python src/music/model_trainer.py --similar <spotify_track_id> --n 10
python src/music/model_trainer.py --discover <id1>,<id2> --n 15
```

## Ratings

Spotify has no star ratings, so `spotify_ingestion.py` synthesises a transparent
1–5 affinity score from: saved status, playlist appearances, top-track rank, and
popularity (see `implicit_rating()`). If you keep real ratings, pass them in and
they override the synthetic ones:

```bash
python src/music/feature_engineering.py --ratings data/raw/my_music_ratings.csv
# CSV needs columns: track_id, rating
```

## Snippets for the main README

**Setup → Run Ingestion**, add: `python src/music/spotify_ingestion.py`

**Train Machine Learning Models**, add:
```bash
# Music
python src/music/spotify_ingestion.py
python src/music/musicbrainz_enrichment.py
python src/music/genius_lyrics.py
python src/music/feature_engineering.py
python src/music/model_trainer.py
```

For the **Unified Model**, the schema aligns the same way the Books domain does:
`artists → director`, `artist_genres/mb_tags → genres/tags`,
`duration_min → runtime`, `popularity → critic/score`, lyric embeddings →
plot/description embeddings. Drop `music_processed.csv` into
`unified_feature_engineering.py` to fold Music into the quad-domain ensemble
(making it a five-domain hub).

## Lyrics & copyright

Genius lyrics are copyrighted. They're cached **locally** for your own analysis
and never written into the model-ready CSV — only derived numeric features and a
short embedding snippet. Use `--sentiment-only` to avoid persisting raw lyrics at
all.


Uncovering free alternative APIs for deprecated Spotify featuresUncovering free alternative APIs for deprecated Spotify featuresI'll dig into the current state of these APIs first, since you flagged the Spotify deprecation. That's the part most likely to trip up the code, so I want to get it right rather than write scripts against dead endpoints.Identified deprecated Spotify features and investigated alternative APIsIdentified deprecated Spotify features and investigated alternative APIsConfirmed the deprecation. One reply mentioned ReccoBeats as a free replacement for audio features. Let me verify that and check the AcousticBrainz status, since both are candidate free sources.Evaluated free audio feature APIs and verified documentation sourcesEvaluated free audio feature APIs and verified documentation sourcesReccoBeats is the standout free drop-in. Let me pull its actual API docs so the endpoints and field names in the script are accurate, then confirm Genius and MusicBrainz/AcousticBrainz.Clarifying ReccoBeats API specifications and request parametersClarifying ReccoBeats API specifications and request parametersThe docs are JS-rendered so the schema didn't load fully. Let me confirm the exact ReccoBeats request flow (Spotify ID → audio features) and field names, which is the crux of the whole pipeline.Architected modular music pipeline integrating deprecated Spotify with ReccoBeats, MusicBrainz, GeniusArchitected modular music pipeline integrating deprecated Spotify with ReccoBeats, MusicBrainz, GeniusThe picture is clear now, and your instinct was right. Here's what's actually true as of today, which shapes the whole design:
Spotify deprecated Audio Features, Audio Analysis, Recommendations, and Related Artists for any app created after November 27, 2024. New apps registered on or after that date can no longer access those endpoints, and requests to /v1/audio-features return 403 Forbidden even with a valid token and correct scopes. No replacements were offered, and in May 2025 Spotify raised the bar for extended access to 250,000 monthly active users, so there's no realistic path back for a personal project. spotify + 2
The best FREE replacement is ReccoBeats, which returns the same metrics (acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence) keyed off Spotify track IDs, no API key. The one catch worth knowing: it isn't always perfectly reliable, so my scripts cache results and degrade gracefully when a track has no features. For genres I lean on MusicBrainz tags (the deprecated Spotify "Related Artists" gap is also best filled by MusicBrainz + Last.fm relationships, per the community consensus), and Genius for lyrics. Spotify Community
So the architecture is: Spotify for library + all track/album/artist metadata, ReccoBeats for audio features, MusicBrainz for canonical metadata + genre tags, Genius for lyrics. Let me build the pipeline matching your existing per-domain format.Crafting documentation and scaffolding project structureCrafting documentation and scaffolding project structureLet me syntax-check all five scripts to make sure they're clean.Orchestrated documentation structure and configuration files for music pipelineOrchestrated documentation structure and configuration files for music pipelineNow a README and .env template so the pipeline drops cleanly into your existing project.Created 2 files, read a fileCreated 2 files, read a fileDone. Eight files under src/music/, matching your existing per-domain structure.
What each does, in run order:

spotify_ingestion.py — pulls your whole library (liked songs, top tracks across all 3 windows, playlist tracks) plus every field Spotify still gives per track/album/artist, attaches audio features from ReccoBeats, derives a 1–5 implicit rating, and writes music_library.csv. That's your "take all data available from Spotify and create CSV" step.
musicbrainz_enrichment.py — adds stable MBIDs and genre/folksonomy tags (the rate limiter and contact User-Agent are built in, since MusicBrainz will throttle you otherwise).
genius_lyrics.py — lyrics plus VADER sentiment and lexical stats, same VADER approach as your Movies dashboard.
feature_engineering.py — merges all three, builds numeric + multi-hot (genres, artists) + SentenceTransformer/PCA embeddings, exactly like your Books pipeline.
model_trainer.py — a stacking ensemble (XGBoost + SVR, plus CatBoost if installed) that predicts your rating, and an Oracle with --predict, --similar, and --discover.

A few design decisions worth flagging:
The whole audio-features layer runs on ReccoBeats, not Spotify, for the deprecation reasons above. It's free and returns the same 9 metrics, but it's the one piece that can be flaky, so every stage caches and never hard-fails on a missing track. If you ever get extended Spotify access, swapping fetch_audio_features() back to the native endpoint is a one-function change.
The rating is synthetic by necessity (Spotify exposes no stars). I made the formula transparent in implicit_rating() so you can tune the weights, and feature_engineering accepts a --ratings CSV to override it with real ratings if you ever log them. Worth knowing: a model trained on a heuristic target partly learns the heuristic, so the headline MAE will look optimistic until you feed it genuine ratings.
For the recommender, --discover uses ReccoBeats' recommendation endpoint to reach outside your library (Spotify's own /recommendations is dead). I built it defensively since I couldn't fully verify its parameter schema against live docs, so if the field names differ slightly it'll report unavailable rather than crash, and you can adjust the params in discover().
The README has copy-paste snippets for folding Music into your main README and the unified five-domain ensemble.