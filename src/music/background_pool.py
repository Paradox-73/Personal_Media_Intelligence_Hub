"""
background_pool.py — Generate a background pool of tracks for PU learning.

Updates:
1. Deduplication against library (by ID and Title+Artist).
2. "Foreignness" constraint: avoid artists/genres heavily present in the library.
3. Explicit negatives support: load from data/raw/music/negatives.csv if exists.
"""

import os
import pandas as pd
import numpy as np
import config
from pathlib import Path
import random
from feature_engineering import split_terms

POOL_CSV = config.PROCESSED_DIR / "music_pool.csv"
NEGATIVES_CSV = config.RAW_DIR / "negatives.csv"

def get_library_meta():
    """Extract IDs, artists, and genres from library to use as filters."""
    if not config.LIBRARY_CSV.exists():
        return set(), set(), set()
    
    df = pd.read_csv(config.LIBRARY_CSV)
    ids = set(df["track_id"].unique())
    
    # Track "Name - Artist" strings for fuzzy dedupe
    titles = set((df["name"].str.lower() + " - " + df["primary_artist"].str.lower()).unique())
    
    # Identify frequent artists/genres
    artists = set(df["primary_artist"].dropna().unique())
    
    all_genres = []
    for c in ["artist_genres", "mb_genres", "mb_tags"]:
        if c in df:
            all_genres.extend([g for row in split_terms(df[c]) for g in row])
    
    genre_counts = pd.Series(all_genres).value_counts()
    frequent_genres = set(genre_counts[genre_counts > 2].index) # Genres appearing > 2 times
    
    return ids, titles, artists, frequent_genres

def generate_dummy_pool(n=5000, exclude_ids=None, exclude_titles=None, exclude_artists=None, exclude_genres=None):
    print(f"Generating {n} synthetic pool tracks with foreignness constraints.")
    
    exclude_ids = exclude_ids or set()
    exclude_titles = exclude_titles or set()
    exclude_artists = exclude_artists or set()
    exclude_genres = exclude_genres or set()
    
    # Potential genres for the pool (avoiding library genres)
    all_potential_genres = ["country", "classical", "jazz", "metal", "reggae", "folk", "blues", "opera", "k-pop", "j-pop", "techno", "ambient", "bluegrass", "gospel"]
    pool_genres = [g for g in all_potential_genres if g not in exclude_genres]
    if not pool_genres: pool_genres = ["foreign_genre"]

    # Potential artists (avoiding library artists)
    pool_artists = [f"Foreign Artist {i}" for i in range(1000)]
    
    np.random.seed(42)
    
    rows = []
    generated = 0
    attempts = 0
    while generated < n and attempts < n * 2:
        attempts += 1
        tid = f"pool_{attempts}"
        name = f"Background Track {attempts}"
        artist = random.choice(pool_artists)
        
        # Dedupe
        if tid in exclude_ids: continue
        if (name.lower() + " - " + artist.lower()) in exclude_titles: continue
        if artist in exclude_artists: continue
        
        genre = random.choice(pool_genres)
        
        row = {
            "track_id": tid,
            "name": name,
            "primary_artist": artist,
            "release_year": int(np.random.normal(2015, 10)),
            "popularity": int(np.random.normal(40, 20)),
            "duration_ms": int(np.random.normal(210000, 50000)),
            "explicit": random.random() < 0.2,
            "acousticness": np.random.beta(1, 2),
            "danceability": np.random.normal(0.5, 0.2),
            "energy": np.random.normal(0.5, 0.2),
            "instrumentalness": np.random.beta(0.5, 2),
            "liveness": np.random.beta(2, 5),
            "loudness": np.random.normal(-9, 4),
            "speechiness": np.random.beta(1, 5),
            "tempo": np.random.normal(110, 35),
            "valence": np.random.normal(0.4, 0.2),
            "mb_tags": genre,
            "artist_genres": genre,
            "lyric_sentiment": np.random.normal(0, 0.4),
            "lyric_word_count": np.random.normal(180, 120),
            "lyric_unique_ratio": np.random.normal(0.45, 0.1),
            "lyric_embed_text": f"This is a {genre} song about things you might not like."
        }
        rows.append(row)
        generated += 1
        
    return pd.DataFrame(rows)

def build_pool(target_size=10000):
    lib_ids, lib_titles, lib_artists, lib_genres = get_library_meta()
    
    # 1. Start with explicit negatives if provided
    df_neg = pd.DataFrame()
    if NEGATIVES_CSV.exists():
        print(f"Loading explicit negatives from {NEGATIVES_CSV}")
        df_neg = pd.read_csv(NEGATIVES_CSV)
        # Ensure it has necessary columns
        if "track_id" not in df_neg.columns:
            df_neg["track_id"] = [f"neg_{i}" for i in range(len(df_neg))]
            
    # 2. Fill the rest with background pool
    needed = target_size - len(df_neg)
    if needed > 0:
        df_bg = generate_dummy_pool(needed, lib_ids, lib_titles, lib_artists, lib_genres)
        df_pool = pd.concat([df_neg, df_bg], ignore_index=True)
    else:
        df_pool = df_neg.head(target_size)
        
    # Final safety dedupe vs library
    df_pool = df_pool[~df_pool["track_id"].isin(lib_ids)]
    
    df_pool.to_csv(POOL_CSV, index=False)
    print(f"Final background pool: {len(df_pool)} tracks.")
    print(f"Saved to {POOL_CSV}")
    return df_pool

if __name__ == "__main__":
    build_pool(5000)
