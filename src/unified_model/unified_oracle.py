"""Single-item Unified Model prediction for the Streamlit Oracles.

Exposes `predict_unified(raw_data, media_type)` -> a star rating in the SAME
397-feature cross-domain space the Unified Model was trained on. Each domain's
Oracle metadata dict is mapped onto the unified backbone, then projected with
`transform_single_media` (which applies the fitted CORAL aligner in PCA space),
and scored with the deployed XGB + CatBoost base learners fused by the Mean
Ensemble.

The Mean Ensemble (`SimplexWeightedAverager`) and `MusicProfile` were pickled
from `__main__`; we re-declare and register them so joblib can unpickle them
from any process (e.g. Streamlit). If the ensemble still can't load, we fall back
to the plain mean of the base learners — which is what the batch predictor does too.
"""

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib
import numpy as np

from src import config
from src.unified_model.unified_feature_engineering import transform_single_media


class SimplexWeightedAverager:
    """Simplex-weighted average of base-model predictions (matches the trainer)."""
    def __init__(self):
        self.weights = None
        self.n_models = 0

    def predict(self, X):
        return np.asarray(X) @ self.weights


@dataclass
class MusicProfile:
    centroids: np.ndarray
    cluster_labels: List[str]
    cluster_meta: List[Dict[str, Any]]
    X_lib: np.ndarray
    pu_model: Any
    pool_score_dist: np.ndarray
    top_genres: List[tuple]
    audio_fingerprint: Dict[str, float]
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]


def _register_for_unpickle():
    """Expose the custom classes under __main__ so joblib can resolve them."""
    main = sys.modules.get("__main__")
    if main is None:
        main = types.ModuleType("__main__")
        sys.modules["__main__"] = main
    for cls in (SimplexWeightedAverager, MusicProfile):
        if not hasattr(main, cls.__name__):
            setattr(main, cls.__name__, cls)


_CACHE: Dict[str, Any] = {}


def _load():
    if not _CACHE:
        _register_for_unpickle()
        ed = config.UNIFIED_ENSEMBLE_DIR
        _CACHE["state"] = joblib.load(config.UNIFIED_PREPROCESSOR_STATE)
        _CACHE["xgb"] = joblib.load(ed / "xgb_base_regressor.joblib")
        _CACHE["cat"] = joblib.load(ed / "catboost_base_regressor.joblib")
        try:
            _CACHE["mean"] = joblib.load(ed / "stacking_ensemble_regressor.joblib")
        except Exception:
            _CACHE["mean"] = None
    return _CACHE


def _to_backbone(raw: dict, media_type: str) -> dict:
    """Map a domain Oracle's metadata dict onto the unified backbone keys."""
    m = dict(raw)
    if media_type == "tv":
        m.setdefault("title", raw.get("name"))
        m["director"] = raw.get("created_by") or raw.get("director")
        m["genre"] = raw.get("genres") or raw.get("genre")
        m["rated"] = raw.get("age_rating") or raw.get("rated")
    elif media_type == "game":
        m["title"] = raw.get("name") or raw.get("title")
        m["director"] = raw.get("developers") or raw.get("developer")
        m["genre"] = raw.get("genres") or raw.get("genre")
        m["overview"] = raw.get("description_raw") or raw.get("overview")
        m["metascore"] = raw.get("metacritic") or raw.get("metascore")
        m["imdb_rating"] = raw.get("rating") or raw.get("imdb_rating")  # 0-5 scale
        m["imdb_votes"] = raw.get("ratings_count") or raw.get("imdb_votes")
    elif media_type == "book":
        m["director"] = raw.get("authors") or raw.get("author") or raw.get("director")
        m["genre"] = raw.get("categories") or raw.get("genre")
        m["overview"] = raw.get("description") or raw.get("overview")
        m["imdb_rating"] = raw.get("averageRating") or raw.get("imdb_rating")  # 0-5 scale
        m["imdb_votes"] = raw.get("ratingsCount") or raw.get("imdb_votes")
        m["runtime"] = raw.get("pageCount") or raw.get("runtime")
    elif media_type == "music":
        # Mirror the music mapping used to build the unified training data
        # (name->title, artists->director, artist_genres->genre, popularity->imdb_rating,
        #  overview = mb_tags + genres + lyric snippet, runtime = length in minutes).
        m["title"] = raw.get("name") or raw.get("title")
        m["director"] = raw.get("artists") or raw.get("director")
        m["genre"] = raw.get("artist_genres") or raw.get("mb_genres") or raw.get("genre")
        m["overview"] = " ".join(str(raw.get(k) or "") for k in
                                 ("mb_tags", "artist_genres", "lyric_embed_text")).strip()
        m["imdb_rating"] = raw.get("popularity") or raw.get("imdb_rating")
        m["year"] = raw.get("release_year") or raw.get("year")
        length_ms = raw.get("mb_length_ms") or raw.get("duration_ms") or 0
        try:
            m["runtime"] = float(length_ms) / 60000
        except (TypeError, ValueError):
            m["runtime"] = 0
    # movie: keys already match the backbone
    return m


def predict_unified(raw_data: dict, media_type: str):
    """Return the Unified Model's star rating for one item, or None on failure."""
    try:
        c = _load()
        meta = _to_backbone(raw_data, media_type)
        X = transform_single_media(meta, c["state"], media_type=media_type)
        p_xgb = float(c["xgb"].predict(X)[0])
        p_cat = float(c["cat"].predict(X)[0])
        if c.get("mean") is not None:
            try:
                pred = float(c["mean"].predict(np.column_stack([[p_xgb], [p_cat]]))[0])
            except Exception:
                pred = (p_xgb + p_cat) / 2
        else:
            pred = (p_xgb + p_cat) / 2
        return float(np.clip(pred, 0.5, 5.0))
    except Exception:
        return None
