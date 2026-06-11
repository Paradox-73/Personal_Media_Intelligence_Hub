"""
cross_domain_transfer.py — Study how well music taste predicts other media ratings.

Experiments:
A. Lift: Domain Features vs. Domain + Music Features.
B. Raw Power: Predict rating using ONLY music-derived features.
C. Probe: Correlation of music affinity with ratings.
"""

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, RepeatedKFold
from xgboost import XGBRegressor
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from sklearn.metrics.pairwise import cosine_similarity

# Define MusicProfile class locally to avoid unpickling issues
from dataclasses import dataclass
from typing import List, Dict, Any

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

def get_music_affinity_features(texts, profile, bundle):
    """Compute features describing how an item relates to music taste."""
    print(f"Computing music affinity features for {len(texts)} items...")
    
    # 1. Embed text using music's embedding model
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(bundle["embed_model"])
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    
    # 2. Reduce with music's fitted PCA
    reduced = bundle["pca"].transform(embeddings)
    
    # 3. Identify embedding columns in profile centroids
    # The centroids were fit on X_lib which has feature_names
    emb_cols = [i for i, name in enumerate(profile.feature_names) if name.startswith("emb_")]
    if not emb_cols:
        print("Warning: No 'emb_' features found in music profile centroids.")
        # Fallback to zero features if something is wrong
        return pd.DataFrame(np.zeros((len(texts), len(profile.cluster_labels) + 2)))
        
    music_centroids_emb = profile.centroids[:, emb_cols]
    
    # 4. Similarity to music centroids (in the PCA-reduced embedding space)
    sims = cosine_similarity(reduced, music_centroids_emb)
    
    feats = pd.DataFrame(sims, columns=[f"music_affinity_c{i}" for i in range(sims.shape[1])])
    feats["music_affinity_max"] = np.max(sims, axis=1)
    feats["music_affinity_mean"] = np.mean(sims, axis=1)
    
    return feats

def load_domain_text(domain):
    """Load raw titles and overviews for a domain."""
    if domain == 'movie':
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        df['text'] = "Title: " + df['title'].fillna('') + ". " + df['overview'].fillna('')
    elif domain == 'tv':
        df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        df['text'] = "Title: " + df['name'].fillna('') + ". " + df['overview'].fillna('')
    elif domain == 'game':
        try:
            df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        except:
            df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
        df['text'] = "Title: " + df['name'].fillna('') + ". " + df['description_raw'].fillna('')
    elif domain == 'book':
        try:
            df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        except:
            df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
        df['text'] = "Title: " + df['title'].fillna('') + ". " + df['description'].fillna('')
    else:
        return None
    
    # We need to return something that can be joined back to the unified data.
    # The unified feature engineering uses indices, but we don't have them here.
    # However, for this transfer study, we'll just return the text column.
    # To be safe, we'll return a series indexed by whatever unique ID we can find.
    return df

def run_experiments():
    # 1. Load Music Profile and Preprocessors
    profile_path = config.MUSIC_MODEL_DIR / "profile.joblib"
    bundle_path = config.MUSIC_MODEL_DIR / "preprocessors.joblib"
    
    if not profile_path.exists() or not bundle_path.exists():
        print("Music profile or preprocessors not found. Run music pipeline first.")
        return
        
    profile = joblib.load(profile_path)
    bundle = joblib.load(bundle_path)
    
    # 2. Load Unified Data
    df_all = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # We need to reconstruct the unified dataset with text to ensure row alignment
    # This is similar to unified_feature_engineering.py but simpler
    print("Reconstructing unified text data...")
    all_texts = []
    
    # Movies
    df_m = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_m['text'] = "Title: " + df_m['title'].fillna('') + ". " + df_m['overview'].fillna('')
    df_m['media_type'] = 'movie'
    
    # Shows
    df_s = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_s['text'] = "Title: " + df_s['name'].fillna('') + ". " + df_s['overview'].fillna('')
    df_s['media_type'] = 'tv'
    
    # Games
    try: df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except: df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    df_g = df_g.dropna(subset=['my_rating'])
    df_g['text'] = "Title: " + df_g['name'].fillna('') + ". " + df_g['description_raw'].fillna('')
    df_g['media_type'] = 'game'
    
    # Books
    try: df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except: df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    df_b = df_b.dropna(subset=['my_rating'])
    
    b_desc = df_b.get('description')
    if b_desc is None: b_desc = pd.Series([''] * len(df_b), index=df_b.index)
    df_b['text'] = "Title: " + df_b['title'].fillna('') + ". " + b_desc.fillna('')
    
    df_b['media_type'] = 'book'
    
    df_text_combined = pd.concat([
        df_m[['media_type', 'text']], 
        df_s[['media_type', 'text']], 
        df_g[['media_type', 'text']], 
        df_b[['media_type', 'text']]
    ], ignore_index=True)
    
    # Ensure df_all and df_text_combined are aligned
    # Both are created by concatenating movies, tv, games, books in that order.
    # NOTE: Music is in unified_feature_engineering but we skip it here as target.
    df_all = df_all[df_all['media_type'] != 'music'].copy().reset_index(drop=True)
    if len(df_all) != len(df_text_combined):
        print(f"Warning: Alignment mismatch! Unified: {len(df_all)}, Text: {len(df_text_combined)}")
        # This might happen if unified_feature_engineering dropped more rows later.
        # But both starts from the same enriched files.
    
    domains = ['movie', 'tv', 'game', 'book']
    transfer_results = []
    
    for domain in domains:
        print(f"\n>>> Analyzing Domain: {domain}")
        idx = df_all['media_type'] == domain
        df_domain = df_all[idx].copy()
        df_text_domain = df_text_combined[idx].copy()
        
        if len(df_domain) < 20:
            print(f"Skipping {domain} (too few samples: {len(df_domain)})")
            continue
            
        # Target
        y = df_domain['target_reg'].values
        
        # Domain Features (X_domain)
        exclude = ['target_reg', 'target_ordinal', 'source_id', 'media_type', 'rating_date']
        X_domain = df_domain.drop(columns=[c for c in exclude if c in df_domain.columns])
        
        # Real Music Affinity Features
        X_music = get_music_affinity_features(df_text_domain['text'].tolist(), profile, bundle)
        
        # Experiment B: Raw Music Power
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        model = XGBRegressor(n_estimators=50, max_depth=3, learning_rate=0.1)
        
        scores_b = cross_val_score(model, X_music, y, cv=rkf, scoring='r2')
        r2_b = np.mean(scores_b)
        
        # Experiment A: Lift
        scores_baseline = cross_val_score(model, X_domain, y, cv=rkf, scoring='r2')
        r2_baseline = np.mean(scores_baseline)
        
        X_combined = pd.concat([X_domain.reset_index(drop=True), X_music.reset_index(drop=True)], axis=1)
        scores_combined = cross_val_score(model, X_combined, y, cv=rkf, scoring='r2')
        r2_combined = np.mean(scores_combined)
        
        lift = r2_combined - r2_baseline
        
        transfer_results.append({
            "Source": "Music",
            "Target": domain,
            "Raw R2 (Exp B)": r2_b,
            "Baseline R2": r2_baseline,
            "Combined R2": r2_combined,
            "Lift (Exp A)": lift
        })
        
    df_matrix = pd.DataFrame(transfer_results)
    print("\n--- Cross-Domain Transfer Matrix ---")
    print(df_matrix.to_string(index=False))
    
    matrix_path = config.MUSIC_PROCESSED_DIR / "transfer_matrix.csv"
    df_matrix.to_csv(matrix_path, index=False)
    print(f"\nMatrix saved to {matrix_path}")

if __name__ == "__main__":
    run_experiments()
