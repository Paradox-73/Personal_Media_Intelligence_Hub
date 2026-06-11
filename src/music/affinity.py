"""
affinity.py — Pure functions for music taste scoring and calibration.

Implements density-based (centroids, kNN) and discriminative (PU classifier)
scoring methods, plus the calibration to human-friendly percentiles.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def density_score(X_query: np.ndarray, centroids: np.ndarray, weights=None, mode="max") -> np.ndarray:
    """Score by proximity to cluster centroids (multi-prototype density)."""
    # X_query: (n_query, d), centroids: (k, d)
    sims = cosine_similarity(X_query, centroids)  # (n_query, k)
    
    if mode == "max":
        return np.max(sims, axis=1)
    elif mode == "mean":
        if weights is not None:
            return np.average(sims, axis=1, weights=weights)
        return np.mean(sims, axis=1)
    elif mode == "softmax":
        # Softmax-weighted similarity
        exp_sims = np.exp(sims * 10) # Temperature scaling
        weights = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
        return np.sum(sims * weights, axis=1)
    return np.max(sims, axis=1)


def knn_density(X_query: np.ndarray, X_lib: np.ndarray, k=25) -> np.ndarray:
    """Score = mean cosine similarity to the k nearest library tracks."""
    sims = cosine_similarity(X_query, X_lib)  # (n_query, n_lib)
    # Sort similarities and take the top k for each query track
    top_k_sims = np.sort(sims, axis=1)[:, -k:]
    return np.mean(top_k_sims, axis=1)


def pu_proba(X_query: np.ndarray, pu_model) -> np.ndarray:
    """Probability of 'positive' from the PU classifier (XGBoost/SVM)."""
    # If model has predict_proba, use it; else fallback to predict
    if hasattr(pu_model, "predict_proba"):
        return pu_model.predict_proba(X_query)[:, 1]
    return pu_model.predict(X_query)


def blend(scores: dict, weights: dict = None) -> np.ndarray:
    """Weighted combination of multiple score vectors."""
    if weights is None:
        weights = {k: 1.0 / len(scores) for k in scores}
    
    # Normalize weights
    total_w = sum(weights.values())
    norm_weights = {k: v / total_w for k, v in weights.items()}
    
    first_score = next(iter(scores.values()))
    result = np.zeros_like(first_score)
    for k, s in scores.items():
        result += s * norm_weights.get(k, 0.0)
    return result


def calibrate(raw_scores: np.ndarray, pool_scores: np.ndarray) -> np.ndarray:
    """Map raw affinity scores to 0-100 percentiles relative to the pool distribution."""
    # We want to know: what % of the pool has a score LOWER than this one.
    # High percentile = very high affinity compared to general population.
    sorted_pool = np.sort(pool_scores)
    percentiles = np.searchsorted(sorted_pool, raw_scores) / len(sorted_pool)
    return percentiles * 100.0
