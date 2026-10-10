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


def _join(*vals):
    """Comma-join the non-empty values, matching how the batch builder concatenates
    a domain's several tag columns into the single shared `tags` slot."""
    parts = []
    for v in vals:
        if v is None:
            continue
        s = ", ".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        s = s.strip().strip(",").strip()
        if s and s.lower() != "nan":
            parts.append(s)
    return ", ".join(parts)


def _first(raw: dict, *keys):
    """First key that is actually PRESENT, rather than first truthy value.

    `raw.get("rating") or raw.get("imdb_rating")` silently discards a legitimate
    0 (an unrated game, a zero metascore) and falls through to a missing key,
    which then gets median-imputed as if the item had no score at all.
    """
    for k in keys:
        v = raw.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        try:
            if v != v:  # NaN
                continue
        except Exception:  # noqa: BLE001
            pass
        return v
    return None


def _to_backbone(raw: dict, media_type: str) -> dict:
    """Map a domain Oracle's metadata dict onto the unified backbone keys.

    The `tags` and `country` slots matter as much as the scores: the batch builder
    fills both for every row, so leaving them empty here changed the embedded text
    and blanked the ctry_* one-hot block relative to training.
    """
    m = dict(raw)
    # Movies arrive on backbone names already; union in the writers so the creator
    # set matches what the batch builder encodes (see UNIFIED_ENCODE).
    if media_type == "movie" and raw.get("writer"):
        m["director"] = _join(raw.get("director"), raw.get("writer"))
    # The studio/publisher/network axis is one shared column, exactly like `director`.
    # Movies have no branch below (they already arrive on backbone names), so seed it
    # here and let each domain override.
    m.setdefault("studio", _first(raw, "production", "production_companies", "studio", "publisher"))
    if media_type == "tv":
        m.setdefault("title", raw.get("name"))
        m["director"] = _join(raw.get("created_by") or raw.get("director"), raw.get("writer"))
        m["studio"] = _first(raw, "network", "networks", "production_companies", "studio")
        m["genre"] = raw.get("genres") or raw.get("genre")
        m["rated"] = raw.get("age_rating") or raw.get("rated")
        m["tags"] = _join(raw.get("keywords"), raw.get("tvdb_tags"))
        # The batch builder appends the season-by-season narrative to a show's
        # overview; match it here by tmdb_id or this path embeds a shorter string
        # than the one the model was fitted on.
        try:
            from src.shows.season_data import season_overview_by_show
            _sn = season_overview_by_show().get(
                int(float(raw.get("tmdb_id") or raw.get("id") or 0)), "")
        except Exception:
            _sn = ""
        if _sn:
            m["overview"] = (str(raw.get("overview") or "").strip() + " " + _sn).strip()
    elif media_type == "game":
        m["title"] = _first(raw, "name", "title")
        m["director"] = _first(raw, "developers", "developer")
        m["studio"] = _first(raw, "publishers", "publisher", "studio")
        m["genre"] = _first(raw, "genres", "genre")
        m["overview"] = _first(raw, "description_raw", "overview")
        m["metascore"] = _first(raw, "metacritic", "metascore")
        m["imdb_rating"] = _first(raw, "rating", "imdb_rating")  # 0-5 scale
        m["imdb_votes"] = _first(raw, "ratings_count", "imdb_votes")
        m["rated"] = _first(raw, "age_rating", "esrb_rating", "rated")
        m["year"] = _first(raw, "year", "released")
        m["tags"] = _join(raw.get("tags"))
    elif media_type == "book":
        m["director"] = _first(raw, "authors", "author", "director")
        m["studio"] = _first(raw, "publisher", "publishers", "studio")
        m["genre"] = _first(raw, "categories", "genre")
        m["overview"] = _first(raw, "description", "overview")
        m["imdb_rating"] = _first(raw, "averageRating", "imdb_rating")  # 0-5 scale
        m["imdb_votes"] = _first(raw, "ratingsCount", "imdb_votes")
        m["runtime"] = _first(raw, "pageCount", "runtime")
        # Books store the year as `publishedDate`; without this the unified path saw
        # no year at all and median-imputed it.
        m["year"] = _first(raw, "year", "publishedDate", "released")
        m["tags"] = _join(raw.get("moods"), raw.get("hc_tags"))
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
        m["tags"] = _join(raw.get("mb_tags"))
    else:
        # movie: keys already match the backbone; `keywords` is its tag channel.
        m["tags"] = _join(raw.get("tags"), raw.get("keywords"))
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
    except Exception as e:  # noqa: BLE001 -- the Oracle page shows "no prediction", so say why
        print(f"predict_unified({media_type}) failed: {e!r}")
        return None
