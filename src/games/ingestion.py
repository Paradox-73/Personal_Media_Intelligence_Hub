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

def search_games_by_query(query, max_results=5):
    """Live RAWG search for the Oracle. Returns lightweight result dicts."""
    if not RAWG_KEY or not query:
        return []
    url = f"https://api.rawg.io/api/games?key={RAWG_KEY}&search={query}&page_size={max_results}"
    try:
        results = requests.get(url, timeout=10).json().get('results', [])
    except Exception as e:
        print(f"RAWG search error: {e}")
        return []

    out = []
    for g in results[:max_results]:
        plats = g.get('platforms') or []
        platform = plats[0]['platform']['name'] if plats else 'Unknown'
        out.append({
            'id': g.get('id'),
            'title': g.get('name', 'Unknown'),
            'platform': platform,
            'year': (str(g.get('released')) or '')[:4],
            'poster_path': g.get('background_image'),
        })
    return out


def fetch_game_details_by_id(game_id):
    """Fetch full RAWG detail for a selected game, shaped for the Oracle transform."""
    if not RAWG_KEY or not game_id:
        return None
    try:
        g = requests.get(f"https://api.rawg.io/api/games/{game_id}?key={RAWG_KEY}", timeout=10).json()
    except Exception as e:
        print(f"RAWG details error: {e}")
        return None
    if not g or not g.get('name'):
        return None

    plats = g.get('platforms') or []
    platform = plats[0]['platform']['name'] if plats else 'Unknown'
    genres = [x['name'] for x in g.get('genres', [])]
    developers = [x['name'] for x in g.get('developers', [])]
    tags = [x['name'] for x in (g.get('tags') or [])[:15]]

    return {
        'title': g.get('name'),
        'name': g.get('name'),
        'platform': platform,
        'year': (str(g.get('released')) or '')[:4],
        'released': g.get('released'),
        'genre': genres,
        'genres': ", ".join(genres),
        'developer': developers,
        'developers': ", ".join(developers),
        'tags': ", ".join(tags),
        'description_raw': g.get('description_raw') or '',
        'metacritic': g.get('metacritic'),
        'rating': g.get('rating'),
        'ratings_count': g.get('ratings_count'),
        'reviews_count': g.get('reviews_count'),
        'cover': g.get('background_image'),
        'poster_path': g.get('background_image'),
        'processing_status': 'success',
    }


def enrich_game_data(merge=True):
    """Enrich the games dataset from RAWG.

    merge=True (default): keep every existing (hand-cleaned) enriched row untouched
    and only fetch+append games that are newly in the raw games_data.csv list.
    merge=False: re-enrich the whole raw list from scratch.
    """
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

    # Merge: preserve existing cleaned rows, enrich only games new to the raw list.
    existing_by_name = {}
    if merge and config.GAMES_ENRICHED_DATA_PATH.exists():
        try:
            ex = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        except Exception:
            ex = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
        if not ex.empty and 'name' in ex.columns:
            existing_by_name = {str(r['name']).strip().lower(): r.to_dict()
                                for _, r in ex.iterrows()}
        print(f"   Merge: {len(existing_by_name)} existing (cleaned) games preserved.")

    enriched_rows = list(existing_by_name.values())  # keep all cleaned rows first
    n_new = 0

    for idx, row in df.iterrows():
        game_name = row['name']
        if merge and str(game_name).strip().lower() in existing_by_name:
            continue  # already preserved above
        n_new += 1
        print(f"[new {n_new}] Processing: {game_name}")

        # Check if we need to fetch more data
        needs_fetch = False
        cols_to_check = ['genres', 'developers', 'publishers', 'metacritic', 'rating', 'description_raw', 'playtime']
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
                if is_effectively_empty(row.get('playtime')):
                    # RAWG 'playtime' = average hours players take to beat the game
                    row['playtime'] = rawg_data.get('playtime')

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
