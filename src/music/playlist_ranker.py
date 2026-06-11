"""
playlist_ranker.py — Scores and ranks external Spotify playlists by affinity.

Input: One or more Spotify playlist IDs.
Output: Ranked list of playlists with per-track explanations.
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import config
import feature_engineering
import affinity
from profile_builder import MusicProfile

def get_playlist_tracks_mock(playlist_id):
    """Mock fetching tracks from a playlist."""
    # In reality, this would use spotipy.playlist_items()
    print(f"Mock fetching tracks for playlist {playlist_id}...")
    n = 50
    df = pd.DataFrame({
        "track_id": [f"playlist_{playlist_id}_{i}" for i in range(n)],
        "name": [f"Playlist Track {i}" for i in range(n)],
        "primary_artist": [f"Artist {i%10}" for i in range(n)],
        "artist_genres": ["pop;rock"] * n,
        "mb_tags": ["indie"] * n,
        "release_year": [2022] * n,
        "popularity": [70] * n,
        "duration_ms": [180000] * n,
        "acousticness": [0.2] * n,
        "danceability": [0.6] * n,
        "energy": [0.8] * n,
        "instrumentalness": [0.0] * n,
        "liveness": [0.1] * n,
        "loudness": [-5.0] * n,
        "speechiness": [0.05] * n,
        "tempo": [120] * n,
        "valence": [0.5] * n,
    })
    return df

def rank_playlists(playlist_ids: list[str], profile_path=None):
    if profile_path is None:
        profile_path = config.MODELS_DIR / "profile.joblib"
    
    print(f"Loading profile from {profile_path}...")
    profile = joblib.load(profile_path)
    bundle = joblib.load(config.MODELS_DIR / "preprocessors.joblib")
    
    playlist_results = []
    
    for pid in playlist_ids:
        # 1. Ingest
        df_tracks = get_playlist_tracks_mock(pid)
        
        # 2. Transform
        X_cand = feature_engineering.transform(df_tracks, bundle)
        
        # 3. Score
        d_scores = affinity.density_score(X_cand, profile.centroids)
        k_scores = affinity.knn_density(X_cand, profile.X_lib)
        p_scores = profile.pu_model.predict_proba(X_cand)[:, 1]
        
        raw_scores = affinity.blend({"density": d_scores, "knn": k_scores, "pu": p_scores})
        cal_scores = affinity.calibrate(raw_scores, profile.pool_score_dist)
        
        df_tracks["affinity_score"] = cal_scores
        
        # 4. Cluster matching
        sims = affinity.cosine_similarity(X_cand, profile.centroids)
        best_cluster_idx = np.argmax(sims, axis=1)
        df_tracks["best_cluster"] = [profile.cluster_labels[i] for i in best_cluster_idx]
        
        # 5. Aggregate
        top_k = 10
        mean_top_k = df_tracks["affinity_score"].nlargest(top_k).mean()
        pct_high = (df_tracks["affinity_score"] > 70).mean() * 100
        
        playlist_results.append({
            "playlist_id": pid,
            "mean_top_k": mean_top_k,
            "pct_high_affinity": pct_high,
            "best_tracks": df_tracks.nlargest(3, "affinity_score")[["name", "primary_artist", "affinity_score", "best_cluster"]].to_dict("records")
        })
        
    # Rank
    df_rank = pd.DataFrame(playlist_results).sort_values("mean_top_k", ascending=False)
    
    print("\n--- Playlist Rankings ---")
    for i, row in df_rank.iterrows():
        print(f"\n{i+1}. Playlist: {row['playlist_id']}")
        print(f"   Score (Mean Top 10): {row['mean_top_k']:.1f}%")
        print(f"   % Tracks > 70th pct: {row['pct_high_affinity']:.1f}%")
        print(f"   Top Gems:")
        for track in row['best_tracks']:
            print(f"     - {track['name']} by {track['primary_artist']} ({track['affinity_score']:.1f}%, matches '{track['best_cluster']}')")
            
    return df_rank

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlists", nargs="+", default=["daily_mix_1", "discover_weekly"], help="playlist IDs")
    args = ap.parse_args()
    rank_playlists(args.playlists)
