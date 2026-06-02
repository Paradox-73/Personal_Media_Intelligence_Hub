"""
config.py — shared paths, constants and environment loading for the Music pipeline.

Keeps every music script (Spotify, MusicBrainz, Genius, feature engineering,
training) pointed at the same folders and credentials. Mirrors the layout used
by the Movies / Games / Books pipelines.

.env keys expected:
    SPOTIPY_CLIENT_ID        (from https://developer.spotify.com/dashboard)
    SPOTIPY_CLIENT_SECRET
    SPOTIPY_REDIRECT_URI     (e.g. http://localhost:8888/callback)
    GENIUS_ACCESS_TOKEN      (from https://genius.com/api-clients)
    MUSICBRAINZ_CONTACT      (an email or app URL; MusicBrainz requires a real
                              contact in the User-Agent. Falls back to a default.)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Project roots (resolve relative to this file so scripts run from anywhere)
SRC_DIR = Path(__file__).resolve().parent          # .../src/music
PROJECT_ROOT = SRC_DIR.parent.parent               # repo root

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "music"
PROCESSED_DIR = DATA_DIR / "processed" / "music"
CACHE_DIR = DATA_DIR / "cache" / "music"
MODELS_DIR = PROJECT_ROOT / "models" / "music"

for _d in (RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Canonical file names used across the pipeline
LIBRARY_CSV = PROCESSED_DIR / "music_library.csv"           # Spotify + ReccoBeats
MUSICBRAINZ_CSV = PROCESSED_DIR / "music_musicbrainz.csv"   # MBIDs + genre tags
LYRICS_CSV = PROCESSED_DIR / "music_lyrics.csv"             # Genius lyrics + sentiment
MASTER_CSV = PROCESSED_DIR / "music_processed.csv"          # merged, model-ready
FEATURES_NPZ = PROCESSED_DIR / "music_features.npz"         # engineered matrix

# --- Credentials
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8888/callback")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
MUSICBRAINZ_CONTACT = os.getenv("MUSICBRAINZ_CONTACT", "personal-media-hub (set MUSICBRAINZ_CONTACT)")

# MusicBrainz requires a descriptive, contactable User-Agent or it may rate-limit/ban.
USER_AGENT = f"PersonalMediaIntelligenceHub/1.0 ( {MUSICBRAINZ_CONTACT} )"

# --- External API base URLs
SPOTIFY_SCOPES = (
    "user-library-read "
    "user-top-read "
    "user-read-recently-played "
    "playlist-read-private "
    "playlist-read-collaborative"
)
RECCOBEATS_BASE = "https://api.reccobeats.com/v1"
MUSICBRAINZ_BASE = "https://musicbrainz.org/ws/2"

# The 9 audio-feature columns ReccoBeats returns (Spotify's deprecated set, minus
# key / mode / time_signature which ReccoBeats does not provide).
AUDIO_FEATURE_COLS = [
    "acousticness", "danceability", "energy", "instrumentalness",
    "liveness", "loudness", "speechiness", "tempo", "valence",
]


def require(value, name: str):
    """Fail fast with a clear message when a credential is missing."""
    if not value:
        raise RuntimeError(
            f"Missing required credential: {name}. Add it to your .env file. "
            f"See src/music/config.py for the full list."
        )
    return value
