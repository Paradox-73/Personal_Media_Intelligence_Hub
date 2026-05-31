import pandas as pd
import json
import os
import argparse
import sys
import requests
import numpy as np
import time
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Load Environment Variables
load_dotenv()

RAWG_KEY = os.getenv("RAWG_API_KEY")

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return None
        # Handle formats like DD-MM-YYYY or YYYY
        s = str(val).strip()
        if '-' in s:
            parts = s.split('-')
            if len(parts[0]) == 4: return int(parts[0])
            if len(parts[-1]) == 4: return int(parts[-1])
        return int(float(s[:4]))
    except:
        return None

def is_effectively_empty(val):
    if val is None: return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    try:
        if pd.isna(val): return True
    except:
        pass
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown']

def fetch_rawg_data(game_name, platform=None):
    if not RAWG_KEY:
        print("❌ RAWG_API_KEY missing.")
        return None
    
    url = f"https://api.rawg.io/api/games?key={RAWG_KEY}&search={game_name}&search_precise=true"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        if data.get('results'):
            # Try to find the best match
            best_match = data['results'][0]
            game_id = best_match['id']
            
            # Fetch full details
            detail_url = f"https://api.rawg.io/api/games/{game_id}?key={RAWG_KEY}"
            detail_res = requests.get(detail_url, timeout=10)
            return detail_res.json()
    except Exception as e:
        print(f"Error fetching from RAWG: {e}")
    return None

def enrich_game_data():
    print("🎮 Starting Game Data Enrichment...")
    
    raw_path = config.GAMES_RAW_DIR / "games_data.csv"
    if not raw_path.exists():
        print(f"❌ Raw data not found at {raw_path}")
        return

    try:
        df = pd.read_csv(raw_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(raw_path, encoding='latin1')
    print(f"📊 Loaded {len(df)} games from raw data.")

    enriched_rows = []
    
    for idx, row in df.iterrows():
        game_name = row['name']
        print(f"[{idx+1}/{len(df)}] Processing: {game_name}")
        
        # Check if we need to fetch more data
        needs_fetch = False
        cols_to_check = ['genres', 'developers', 'publishers', 'metacritic', 'rating', 'description_raw']
        for col in cols_to_check:
            if col not in row or is_effectively_empty(row[col]):
                needs_fetch = True
                break
        
        if needs_fetch:
            rawg_data = fetch_rawg_data(game_name)
            if rawg_data:
                # Fill missing columns
                if is_effectively_empty(row.get('genres')):
                    row['genres'] = ", ".join([g['name'] for g in rawg_data.get('genres', [])])
                if is_effectively_empty(row.get('developers')):
                    row['developers'] = ", ".join([d['name'] for d in rawg_data.get('developers', [])])
                if is_effectively_empty(row.get('publishers')):
                    row['publishers'] = ", ".join([p['name'] for p in rawg_data.get('publishers', [])])
                if is_effectively_empty(row.get('metacritic')):
                    row['metacritic'] = rawg_data.get('metacritic')
                if is_effectively_empty(row.get('rating')):
                    row['rating'] = rawg_data.get('rating')
                if is_effectively_empty(row.get('ratings_count')):
                    row['ratings_count'] = rawg_data.get('ratings_count')
                if is_effectively_empty(row.get('reviews_count')):
                    row['reviews_count'] = rawg_data.get('reviews_count')
                if is_effectively_empty(row.get('cover')):
                    row['cover'] = rawg_data.get('background_image')
                if is_effectively_empty(row.get('tags')):
                    row['tags'] = ", ".join([t['name'] for t in rawg_data.get('tags', [])])
                if is_effectively_empty(row.get('description_raw')):
                    row['description_raw'] = rawg_data.get('description_raw') or rawg_data.get('description')
                
                # Update release year if missing
                if is_effectively_empty(row.get('released')):
                    row['released'] = rawg_data.get('released')
            
            # Throttle API calls
            time.sleep(0.2)

        enriched_rows.append(row.to_dict())

    df_enriched = pd.DataFrame(enriched_rows)
    
    # Save to processed directory
    config.GAMES_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_enriched.to_csv(config.GAMES_ENRICHED_DATA_PATH, index=False)
    print(f"✅ Enrichment complete. Saved to {config.GAMES_ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    enrich_game_data()
