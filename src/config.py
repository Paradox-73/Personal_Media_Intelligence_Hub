import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, CACHE_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --- MOVIES ---
# Movie specific directories
MOVIES_RAW_DIR = RAW_DIR / "movies"
MOVIES_PROCESSED_DIR = PROCESSED_DIR / "movies"
MOVIES_MODEL_DIR = MODEL_DIR / "movies"

# Movie Files
RATINGS_PATH = MOVIES_RAW_DIR / "ratings.csv"
LIKED_PATH = MOVIES_RAW_DIR / "liked.csv"

MOVIES_ENRICHED_DATA_PATH = MOVIES_PROCESSED_DIR / "enriched_data.csv"
TRAINING_DATA_PATH = MOVIES_PROCESSED_DIR / "training_features.csv"
FULL_VIEW_PATH = MOVIES_PROCESSED_DIR / "dashboard_view.csv" # For EDA

# Movie Model Artifacts
MODEL_REGRESSOR = MOVIES_MODEL_DIR / "xgb_regressor.pkl"
MODEL_CLASSIFIER = MOVIES_MODEL_DIR / "xgb_classifier.pkl"
PREPROCESSOR_STATE = MOVIES_MODEL_DIR / "preprocessor_state.pkl" # Saves PCA, Threshold lists


# --- TV SHOWS ---
TV_SHOWS_RAW_DIR = RAW_DIR / "shows"
TV_SHOWS_PROCESSED_DIR = PROCESSED_DIR / "shows"
TV_SHOWS_MODEL_DIR = MODEL_DIR / "shows"

TV_SHOWS_ENRICHED_DATA_PATH = TV_SHOWS_PROCESSED_DIR / "enriched_data.csv"
TV_SHOWS_TRAINING_DATA_PATH = TV_SHOWS_PROCESSED_DIR / "training_features.csv"
TV_SHOWS_FULL_VIEW_PATH = TV_SHOWS_PROCESSED_DIR / "dashboard_view.csv"

TV_SHOWS_MODEL_REGRESSOR = TV_SHOWS_MODEL_DIR / "xgb_regressor.pkl"
TV_SHOWS_MODEL_CLASSIFIER = TV_SHOWS_MODEL_DIR / "xgb_classifier.pkl"
TV_SHOWS_PREPROCESSOR_STATE = TV_SHOWS_MODEL_DIR / "preprocessor_state.pkl"


# --- MUSIC ---
MUSIC_RAW_DIR = RAW_DIR / "music"
MUSIC_PROCESSED_DIR = PROCESSED_DIR / "music"
MUSIC_MODEL_DIR = MODEL_DIR / "music"

MUSIC_ENRICHED_DATA_PATH = MUSIC_PROCESSED_DIR / "enriched_data.csv"
MUSIC_TRAINING_DATA_PATH = MUSIC_PROCESSED_DIR / "training_features.csv"
MUSIC_FULL_VIEW_PATH = MUSIC_PROCESSED_DIR / "dashboard_view.csv"

MUSIC_MODEL_REGRESSOR = MUSIC_MODEL_DIR / "xgb_regressor.pkl"
MUSIC_MODEL_CLASSIFIER = MUSIC_MODEL_DIR / "xgb_classifier.pkl"
MUSIC_PREPROCESSOR_STATE = MUSIC_MODEL_DIR / "preprocessor_state.pkl"


# --- GAMES ---
GAMES_RAW_DIR = RAW_DIR / "games"
GAMES_PROCESSED_DIR = PROCESSED_DIR / "games"
GAMES_MODEL_DIR = MODEL_DIR / "games"

GAMES_ENRICHED_DATA_PATH = GAMES_PROCESSED_DIR / "enriched_data.csv"
GAMES_TRAINING_DATA_PATH = GAMES_PROCESSED_DIR / "training_features.csv"
GAMES_FULL_VIEW_PATH = GAMES_PROCESSED_DIR / "dashboard_view.csv"

GAMES_MODEL_REGRESSOR = GAMES_MODEL_DIR / "xgb_regressor.pkl"
GAMES_MODEL_CLASSIFIER = GAMES_MODEL_DIR / "xgb_classifier.pkl"
GAMES_MODEL_PREPROCESSOR_STATE = GAMES_MODEL_DIR / "preprocessor_state.pkl"


# --- BOOKS ---
BOOKS_RAW_DIR = RAW_DIR / "books"
BOOKS_PROCESSED_DIR = PROCESSED_DIR / "books"
BOOKS_MODEL_DIR = MODEL_DIR / "books"

BOOKS_ENRICHED_DATA_PATH = BOOKS_PROCESSED_DIR / "enriched_data.csv"
BOOKS_TRAINING_DATA_PATH = BOOKS_PROCESSED_DIR / "training_features.csv"
BOOKS_FULL_VIEW_PATH = BOOKS_PROCESSED_DIR / "dashboard_view.csv"

BOOKS_MODEL_REGRESSOR = BOOKS_MODEL_DIR / "xgb_regressor.pkl"
BOOKS_MODEL_CLASSIFIER = BOOKS_MODEL_DIR / "xgb_classifier.pkl"
BOOKS_PREPROCESSOR_STATE = BOOKS_MODEL_DIR / "preprocessor_state.pkl"


# Cache Files (These are general and can remain in CACHE_DIR)
TMDB_CACHE_PATH = CACHE_DIR / "tmdb_cache.json"
OMDB_CACHE_PATH = CACHE_DIR / "omdb_cache.json"

# Hyperparameters
MIN_FREQUENCY_THRESHOLD = 3
PCA_COMPONENTS = 10