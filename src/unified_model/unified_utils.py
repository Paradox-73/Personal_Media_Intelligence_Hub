import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

from sklearn.base import BaseEstimator, TransformerMixin

class DomainAligner(BaseEstimator, TransformerMixin):
    """
    Aligns domain-specific distributions to a common space (e.g., Movies).
    Supports Centroid matching and CORAL (Correlation Alignment).
    """
    def __init__(self, method='centroid', reference_domain='movie'):
        self.method = method
        self.reference_domain = reference_domain
        self.stats = {}
        self.ref_stats = None

    def fit(self, X, media_types):
        unique_domains = np.unique(media_types)
        for domain in unique_domains:
            mask = (media_types == domain)
            x_domain = X[mask]
            
            mean = np.mean(x_domain, axis=0)
            cov = np.cov(x_domain, rowvar=False) + np.eye(X.shape[1]) * 1e-5
            
            self.stats[domain] = {'mean': mean, 'cov': cov}
            
            if domain == self.reference_domain:
                self.ref_stats = self.stats[domain]
        
        # If reference domain not found in data, use the first available or global
        if self.ref_stats is None and self.stats:
            self.ref_stats = next(iter(self.stats.values()))
            
        return self

    def transform(self, X, media_types):
        X_aligned = np.copy(X)
        # Coerce to ndarray so a Python list/Series both yield a boolean mask
        # (a raw list == str returns a scalar bool, breaking single-item inference).
        media_types = np.asarray(media_types)
        for domain in np.unique(media_types):
            mask = (media_types == domain)
            if domain not in self.stats or not mask.any():
                continue
            
            # 1. Centroid alignment (Subtract domain mean, add reference mean)
            X_aligned[mask] -= self.stats[domain]['mean']
            if self.ref_stats is not None:
                X_aligned[mask] += self.ref_stats['mean']
            
            # 2. CORAL (Correlation Alignment)
            if self.method == 'coral' and self.ref_stats is not None:
                # Whiten source
                d, v = np.linalg.eigh(self.stats[domain]['cov'])
                whiten = v @ np.diag(1.0 / np.sqrt(d + 1e-8)) @ v.T
                
                # Recolor with target
                d_ref, v_ref = np.linalg.eigh(self.ref_stats['cov'])
                recolor = v_ref @ np.diag(np.sqrt(d_ref + 1e-8)) @ v_ref.T
                
                # Apply to zero-meaned data
                # Note: mean was already handled above
                X_aligned[mask] = (X_aligned[mask] - self.ref_stats['mean']) @ whiten @ recolor + self.ref_stats['mean']
                
        return X_aligned

def compute_temporal_weights(dates, lambda_decay=0.001, w_min=0.2):
    """
    Floored exponential sample weighting: w_i = max( exp(−λ · Δt_i), w_min )
    Δt in days since rating.
    """
    latest_date = dates.max()
    delta_t = (latest_date - dates).dt.days
    weights = np.exp(-lambda_decay * delta_t)
    return np.maximum(weights, w_min)

def get_music_affinity_features(texts, profile, bundle):
    """
    Compute features describing how an item relates to music taste.
    Uses the music profile's centroids in embedding space.
    """
    if not texts or len(texts) == 0:
        return pd.DataFrame()
        
    # 1. Embed text using music's embedding model
    model = SentenceTransformer(bundle.get("embed_model", 'all-MiniLM-L6-v2'))
    # Handle single string
    if isinstance(texts, str): texts = [texts]
    
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    # 2. Reduce with music's fitted PCA
    reduced = bundle["pca"].transform(embeddings)
    
    # 3. Identify embedding columns in profile centroids
    # The centroids were fit on X_lib which has feature_names
    emb_cols = [i for i, name in enumerate(profile.feature_names) if name.startswith("emb_")]
    if not emb_cols:
        return pd.DataFrame(np.zeros((len(texts), len(profile.cluster_labels) + 2)))
        
    music_centroids_emb = profile.centroids[:, emb_cols]
    
    # 4. Similarity to music centroids (in the PCA-reduced embedding space)
    sims = cosine_similarity(reduced, music_centroids_emb)
    
    feats = pd.DataFrame(sims, columns=[f"music_affinity_c{i}" for i in range(sims.shape[1])])
    feats["music_affinity_max"] = np.max(sims, axis=1)
    feats["music_affinity_mean"] = np.mean(sims, axis=1)
    
    return feats

def get_music_gate_mask(media_types):
    """
    Returns a mask (multiplier) for music features based on the transfer matrix.
    If a domain has negative lift from music, we gate (zero out) the features.
    """
    matrix_path = config.MUSIC_PROCESSED_DIR / "transfer_matrix.csv"
    gate_map = {'movie': 1.0, 'book': 1.0, 'tv': 0.0, 'game': 0.0, 'music': 1.0}
    
    if matrix_path.exists():
        try:
            matrix = pd.read_csv(matrix_path)
            # Lift (Exp A) is the column to check
            for _, row in matrix.iterrows():
                target = row['Target'].lower()
                lift = row.get('Lift (Exp A)', 0)
                gate_map[target] = 1.0 if lift >= 0 else 0.0
        except:
            pass
            
    # Ensure 'music' domain itself has 1.0
    gate_map['music'] = 1.0
    
    if isinstance(media_types, str):
        return gate_map.get(media_types.lower(), 0.0)
        
    return media_types.map(lambda x: gate_map.get(str(x).lower(), 0.0)).fillna(0.0).values
