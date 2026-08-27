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
| Lyrics | — | **Genius**, with **LRCLIB** (free, keyless) as fallback |
| Recommendations | Spotify `/recommendations` | **ReccoBeats** `/track/recommendation` |
| Artist origin country | — | **MusicBrainz** artist `country`, cross-filled by **Wikidata** (P495) + **TheAudioDB** |
| Lyrics language | — | **MusicBrainz** work `languages` |
| Tempo (BPM) / loudness gain | Spotify `/audio-features` | **Deezer** `/track/{id}` (free, keyless) |
| Release styles / label / pressing country | — | **Discogs** (free token) |

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
DISCOGS_TOKEN=...                     # optional; https://www.discogs.com/settings/developers
AUDIODB_KEY=2                         # optional; "2" = free/test key (throttled), Patreon key recommended
```

LRCLIB, Deezer and Wikidata need **no key**. Discogs and TheAudioDB are optional —
if their keys are absent, `extra_enrichment.py` simply skips those two and runs the rest.

## Run order

```bash
python src/music/spotify_ingestion.py        # 1. library + ReccoBeats audio features -> music_library.csv
python src/music/musicbrainz_enrichment.py   # 2. MBIDs, genre tags, artist country, lyrics language -> music_musicbrainz.csv
python src/music/extra_enrichment.py         # 2b. Deezer/Discogs/AudioDB/Wikidata    -> music_extra.csv   (optional, additive)
python src/music/genius_lyrics.py            # 3. lyrics (Genius + LRCLIB) + VADER     -> music_lyrics.csv
python src/music/feature_engineering.py      # 4. merge + features (unsupervised)     -> music_processed.csv, music_features.npz
python src/music/profile_builder.py          # 5. build taste profile: KMeans personas + PU classifier + similarity index
```

The Taste/Similarity Oracle (no rating model):

```bash
python -m src.music.model_training --similar <spotify_track_id> --n 10   # closest songs in your library
python -m src.music.model_training --cluster <spotify_track_id>          # which taste persona it belongs to
python -m src.music.model_training --discover <id1>,<id2> --n 15         # new tracks outside your library
```

## No ratings — clustering & similarity

Spotify has no star ratings, and **we deliberately do not synthesise one** — a
fabricated "affinity rating" was misleading, so it was removed. Music is instead
an **unsupervised** taste engine:

- **Taste personas** — the library is clustered (KMeans, silhouette-selected k)
  into listening modes (e.g. *"Chill Acoustic jazz"*, *"hip hop"*), each with its
  signature genres and anchor tracks (`profile_builder.py`).
- **Song-to-song similarity** — cosine similarity in the full feature space finds
  the tracks most like any seed (`model_training.py --similar`).
- **PU affinity** — a Positive-Unlabeled classifier (your library vs a mined
  background pool, with hard-negative weighting) scores how "you" a new track
  sounds — a *fit* score, **not** a star rating.

The behavioral signals a rating would have encoded (saved status, playlist
membership, top-track rank, popularity) are kept as **raw feature columns**, not
collapsed into one synthetic star value.

## Snippets for the main README

**Setup → Run Ingestion**, add: `python src/music/spotify_ingestion.py`

**Train Machine Learning Models**, add:
```bash
# Music
python src/music/spotify_ingestion.py
python src/music/musicbrainz_enrichment.py
python src/music/genius_lyrics.py
python src/music/feature_engineering.py
python src/music/profile_builder.py
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
