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

# Files
RATINGS_PATH = RAW_DIR / "ratings.csv"
LIKED_PATH = RAW_DIR / "liked.csv"
TMDB_CACHE_PATH = CACHE_DIR / "tmdb_cache.json"
OMDB_CACHE_PATH = CACHE_DIR / "omdb_cache.json"

ENRICHED_DATA_PATH = PROCESSED_DIR / "enriched_data.csv"
TRAINING_DATA_PATH = PROCESSED_DIR / "training_features.csv"
FULL_VIEW_PATH = PROCESSED_DIR / "dashboard_view.csv" # For EDA

# Model Artifacts
MODEL_REGRESSOR = MODEL_DIR / "xgb_regressor.pkl"
MODEL_CLASSIFIER = MODEL_DIR / "xgb_classifier.pkl"
PREPROCESSOR_STATE = MODEL_DIR / "preprocessor_state.pkl" # Saves PCA, Threshold lists

# Hyperparameters
MIN_FREQUENCY_THRESHOLD = 3
PCA_COMPONENTS = 10