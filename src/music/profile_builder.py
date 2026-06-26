"""
profile_builder.py — Builds the Music Taste Profile from library and pool data.

Includes Hard Negative Mining:
- Identifies tracks in the pool that are "polar opposites" in audio fingerprint.
- Identifies pool tracks that are outliers to all library clusters.
- Prioritizes/Weights these as strong negatives for the PU classifier.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from xgboost import XGBClassifier

import sys
from pathlib import Path

# Add current directory to path to handle imports when run from elsewhere
file_path = Path(__file__).resolve()
if str(file_path.parent) not in sys.path:
    sys.path.append(str(file_path.parent))

import config
import feature_engineering
import affinity
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

def build_profile():
    print("Loading library and pool data...")
    # Load library
    df_lib = pd.read_csv(config.MASTER_CSV)
    if "primary_artist" not in df_lib.columns:
        df_lib["primary_artist"] = (df_lib.get("artists", pd.Series([""] * len(df_lib)))
                                    .fillna("").astype(str).str.split(",").str[0].str.strip())
    data_lib = np.load(config.FEATURES_NPZ, allow_pickle=True)
    X_lib = data_lib["X"]
    feature_names = data_lib["feature_names"].tolist()
    
    # Load preprocessors for group map
    bundle = joblib.load(config.MODELS_DIR / "preprocessors.joblib")
    feature_groups = bundle["feature_groups"]
    
    # Load/Build pool
    from background_pool import POOL_CSV
    if not POOL_CSV.exists():
        import background_pool
        background_pool.build_pool(5000)
    df_pool = pd.read_csv(POOL_CSV)
    print(f"Transforming pool tracks ({len(df_pool)})...")
    X_pool = feature_engineering.transform(df_pool, bundle)
    
    # 2. Clustering
    print("Clustering library into taste modes...")
    k_range = range(4, 11)
    best_k = 6
    best_score = -1
    
    if len(X_lib) > 20:
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_lib)
            score = silhouette_score(X_lib, labels)
            if score > best_score:
                best_score = score
                best_k = k
    
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    lib_labels = km.fit_predict(X_lib)
    centroids = km.cluster_centers_
    
    # 3. Auto-labeling
    print("Analyzing clusters...")
    cluster_meta = []
    cluster_label_names = []
    
    audio_cols = feature_groups["audio"]
    # Global audio fingerprint for polar-opposite check. Audio features may be absent
    # (ReccoBeats not run / thin library export); fall back to 0 so the profile still
    # builds from genres + popularity + embeddings.
    global_fingerprint_vals = pd.Series(
        {c: (float(df_lib[c].mean()) if c in df_lib.columns and df_lib[c].notna().any() else 0.0)
         for c in config.AUDIO_FEATURE_COLS}
    )
    
    for i in range(best_k):
        mask = (lib_labels == i)
        c_df = df_lib[mask]
        c_X = X_lib[mask]
        
        # Top genres
        from feature_engineering import split_terms
        combined_genres = c_df["artist_genres"].fillna("") + ";" + c_df["mb_tags"].fillna("")
        c_genres = pd.Series([g for row in split_terms(combined_genres) for g in row]).value_counts().head(5)
        top_g = c_genres.index.tolist()
        
        # Audio fingerprint
        base_audio = config.AUDIO_FEATURE_COLS
        fingerprint = {col: float(c_df[col].mean()) for col in base_audio if col in c_df.columns}
        
        # Exemplars
        dists = np.linalg.norm(c_X - centroids[i], axis=1)
        closest_idx = np.argsort(dists)[:3]
        exemplars = c_df.iloc[closest_idx][["name", "primary_artist"]].to_dict("records")
        exemplar_names = [f"{e['name']} by {e['primary_artist']}" for e in exemplars]
        
        # Name the cluster
        main_genre = top_g[0] if top_g else "Eclectic"
        energy = "High-Energy" if fingerprint.get("energy", 0) > 0.7 else "Chill" if fingerprint.get("energy", 0) < 0.4 else ""
        acoustic = "Acoustic" if fingerprint.get("acousticness", 0) > 0.6 else ""
        label = f"{energy} {acoustic} {main_genre}".strip()
        cluster_label_names.append(label)
        
        cluster_meta.append({
            "id": i,
            "label": label,
            "genres": top_g,
            "fingerprint": fingerprint,
            "exemplars": exemplar_names,
            "size": int(mask.sum())
        })
    
    # 4. Hard Negative Mining
    print("Mining hard negatives from pool...")
    # Calculate similarity to centroids
    pool_sims = affinity.cosine_similarity(X_pool, centroids)
    max_sims = np.max(pool_sims, axis=1)
    
    # Outliers: very low similarity to all centroids
    outlier_mask = max_sims < np.percentile(max_sims, 10) # Bottom 10% similarity
    
    # Polar opposites: distant from global audio fingerprint
    # Find indices for audio columns in feature matrix
    audio_idx = [feature_names.index(c) for c in config.AUDIO_FEATURE_COLS if c in feature_names]
    pool_audio = X_pool[:, audio_idx]
    lib_audio_mean = X_lib[:, audio_idx].mean(axis=0)
    
    audio_dists = np.linalg.norm(pool_audio - lib_audio_mean, axis=1)
    polar_mask = audio_dists > np.percentile(audio_dists, 90) # Top 10% distance
    
    hard_neg_mask = outlier_mask | polar_mask
    print(f"Identified {hard_neg_mask.sum()} hard negatives (outliers/polar opposites).")
    
    # 5. PU Classifier
    print("Training PU classifier (Library vs Pool with hard negative focus)...")
    X_train = np.vstack([X_lib, X_pool])
    y_train = np.array([1] * len(X_lib) + [0] * len(X_pool))
    
    # Assign higher weights to hard negatives to sharpen the boundary
    weights = np.ones(len(y_train))
    weights[len(X_lib):][hard_neg_mask] = 2.0 # Double weight for hard negatives
    
    pu_model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    pu_model.fit(X_train, y_train, sample_weight=weights)
    
    # Calibration distribution
    print("Computing calibration distribution...")
    d_scores = affinity.density_score(X_pool, centroids)
    k_scores = affinity.knn_density(X_pool, X_lib)
    p_scores = pu_model.predict_proba(X_pool)[:, 1]
    
    pool_score_dist = affinity.blend({"density": d_scores, "knn": k_scores, "pu": p_scores})
    
    # Global summary data
    global_fingerprint = {col: float(df_lib[col].mean()) for col in config.AUDIO_FEATURE_COLS if col in df_lib.columns}
    from feature_engineering import build_genre_vocab
    genre_vocab, _ = build_genre_vocab(df_lib)
    
    profile = MusicProfile(
        centroids=centroids,
        cluster_labels=cluster_label_names,
        cluster_meta=cluster_meta,
        X_lib=X_lib,
        pu_model=pu_model,
        pool_score_dist=pool_score_dist,
        top_genres=genre_vocab[:10],
        audio_fingerprint=global_fingerprint,
        feature_names=feature_names,
        feature_groups=feature_groups
    )
    
    # Save
    joblib.dump(profile, config.MODELS_DIR / "profile.joblib")
    
    # Write MD report
    with open(config.PROCESSED_DIR / "music_profile.md", "w") as f:
        f.write("# Music Taste Profile\n\n")
        f.write(f"Based on your library of {len(X_lib)} tracks.\n\n")
        f.write("## Your Taste Personas\n\n")
        for meta in cluster_meta:
            f.write(f"### {meta['label']}\n")
            f.write(f"- **Size:** {meta['size']} tracks\n")
            f.write(f"- **Primary Genres:** {', '.join(meta['genres'])}\n")
            f.write(f"- **Exemplars:** {', '.join(meta['exemplars'])}\n\n")
        f.write("## Global Audio Fingerprint\n")
        for k, v in global_fingerprint.items():
            f.write(f"- **{k.capitalize()}:** {v:.2f}\n")
        
    print(f"Profile saved to {config.MODELS_DIR / 'profile.joblib'}")
    return profile

if __name__ == "__main__":
    build_profile()
