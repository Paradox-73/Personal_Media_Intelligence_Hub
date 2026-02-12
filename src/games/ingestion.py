import pandas as pd
import json
import os
import argparse
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Adjusted path for src/games
from src import config

# Load Environment Variables
load_dotenv()

# --- 1. SETUP & SAFETY ---
# No external APIs for games in this initial version.
# RAWG_KEY = os.getenv("RAWG_API_KEY")
# IGDB_KEY = os.getenv("IGDB_API_KEY")

# --- 2. HELPERS ---

def clean_year(val):
    try:
        s = str(val).replace('.0', '').strip()
        if not s or s.lower() in ['nan', 'none', '']: return None
        return int(s[:4])
    except:
        return None

def clean_float(val):
    try:
        if not val or str(val).lower() in ['nan', 'n/a', 'none', '', 'null']: return None
        return float(val)
    except:
        return None

def clean_int(val):
    try:
        if not val or str(val).lower() in ['nan', 'n/a', 'none', '', 'null']: return None
        return int(str(val).replace(',', '').split('.')[0])
    except:
        return None

def is_effectively_empty(val):
    """Checks if a value is effectively missing (NaN, None, empty string, '0', 'N/A')."""
    if val is None: return True
    
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
        
    try:
        if pd.isna(val): return True
    except:
        pass
        
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown', 'false', '0', '0.0', 'not rated', 'unrated']

# --- 3. CORE LOGIC ---

def get_game_metadata(title, platform, year):
    """
    Placeholder: Fetches available data for games from local sources.
    This would be extended to external APIs like RAWG/IGDB.
    """
    
    # For now, just return basic info from input
    metadata = {
        'title': title,
        'platform': platform,
        'year': year,
        'genre': None, # Placeholder for API-fetched data
        'developer': None, # Placeholder for API-fetched data
        'publisher': None, # Placeholder for API-fetched data
        'release_date': None,
        'rawg_id': None,
        'igdb_id': None,
    }
    return metadata

def fetch_fresh_data(row):
    """Wrapper for initial ingestion of game data."""
    item_title = str(row['Title']) # Assuming 'Title' column in raw CSV
    item_platform = str(row['Platform']) # Assuming 'Platform' column in raw CSV
    item_year = clean_year(row['Year']) # Assuming 'Year' column in raw CSV
    
    metadata = get_game_metadata(item_title, item_platform, item_year)
    
    metadata['user_rating'] = row.get('Rating') # Assuming 'Rating' column
    metadata['is_liked'] = row.get('is_liked', 0)
    
    return metadata

# --- 4. ORACLE / UI FUNCTIONS ---

def search_games_by_query(query):
    """
    Placeholder: Searches local data or an API for games by query string.
    """
    print(f"Searching games for: {query} (placeholder)")
    return []

def fetch_game_details_by_id(item_id):
    """
    Placeholder: Fetches full metadata for a specific ID.
    """
    print(f"Fetching game details for ID: {item_id} (placeholder)")
    return {'processing_status': 'failed'}

# --- 5. MODES ---

def run_repair(limit=None):
    print("🔧 STARTING SMART REPAIR (Game Ingestion Mode)...")
    try:
        # Assuming a simple CSV with Title, Platform, Year, Rating, is_liked
        ratings = pd.read_csv(config.GAMES_RAW_DIR / "game_ratings.csv") # Placeholder file name
        try:
            liked = pd.read_csv(config.GAMES_RAW_DIR / "game_liked.csv") # Placeholder file name
            ratings['is_liked'] = 0
            if not liked.empty and 'Title' in liked.columns:
                ratings.loc[ratings['Title'].isin(liked['Title']), 'is_liked'] = 1
        except FileNotFoundError:
            print("⚠️ No game_liked.csv found, proceeding without liked data.")
            ratings['is_liked'] = 0
    except Exception as e:
        print(f"❌ Error loading source CSVs: {e}")
        return

    existing_data_map = {}
    if os.path.exists(config.GAMES_ENRICHED_DATA_PATH):
        try:
            df_exist = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
            for _, row in df_exist.iterrows():
                key = f"{str(row.get('title'))}_{str(row.get('platform'))}_{clean_year(row.get('year'))}"
                existing_data_map[key] = row.to_dict()
            print(f"📂 Loaded {len(df_exist)} existing records.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")

    final_rows = []
    if limit: ratings = ratings.head(limit)
    
    for idx, row in ratings.iterrows():
        item_title = str(row['Title'])
        item_platform = str(row['Platform'])
        item_year = clean_year(row['Year'])
        key = f"{item_title}_{item_platform}_{item_year}"
        
        if key in existing_data_map:
            final_rows.append(existing_data_map[key])
            continue
        
        print(f"[{idx+1}] 🔄 Processing: {item_title} ({item_platform}, {item_year})")
        new_data = fetch_fresh_data(row)
        final_rows.append(new_data)

    df_final = pd.DataFrame(final_rows)
    # Clean null characters before saving
    df_final = df_final.replace({r'\x00': ''}, regex=True)
    df_final.to_csv(config.GAMES_ENRICHED_DATA_PATH, index=False)
    print(f"✅ DONE. Saved to: {config.GAMES_ENRICHED_DATA_PATH}")

def run_fill_missing():
    print("🩹 STARTING UNIVERSAL GAP FILL (Games)...")
    print("Currently no external APIs integrated, so no missing data to fill.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fill-missing", action="store_true", help="Scan ALL columns and fill ANY missing value from APIs")
    args = parser.parse_args()
    
    if args.fill_missing:
        run_fill_missing()
    else:
        run_repair(limit=args.limit)