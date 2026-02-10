import pandas as pd
import json
import os
import time
import argparse
import sys
import requests
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from tmdbv3api import TMDb, Movie

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

# Load Environment Variables
load_dotenv()

# --- 1. SETUP & SAFETY ---
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")

if not TMDB_KEY or not OMDB_KEY:
    print("❌ ERROR: API Keys missing in .env file.")
    sys.exit(1)

tmdb = TMDb()
tmdb.api_key = TMDB_KEY
tmdb_movie = Movie()

# --- 2. HELPERS ---

def clean_year(val):
    """Robust year extraction."""
    try:
        s = str(val).replace('.0', '').strip()
        if not s or s.lower() == 'nan': return None
        return int(s[:4])
    except:
        return None

def fetch_omdb_direct(imdb_id=None, title=None, year=None):
    """Fallback OMDB Fetcher."""
    try:
        if imdb_id:
            url = f"http://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_KEY}"
            res = requests.get(url, timeout=5).json()
            if res.get('Response') == 'True': return res

        if title and year:
            clean_t = title.replace(':', '').strip()
            url = f"http://www.omdbapi.com/?t={clean_t}&y={year}&apikey={OMDB_KEY}"
            res = requests.get(url, timeout=5).json()
            if res.get('Response') == 'True': return res
    except: pass
    return None

def smart_search_tmdb(title, target_year):
    """Find movie on TMDB matching Name AND Year (+/- 1 year)."""
    if not target_year: return None
    try:
        results = tmdb_movie.search(title)
        for res in results:
            if not hasattr(res, 'release_date') or not res.release_date: continue
            res_year = int(res.release_date.split('-')[0])
            if abs(res_year - target_year) <= 1:
                return res
    except: pass
    return None

def load_cache(path):
    if path.exists():
        try:
            with open(path, 'r') as f: return json.load(f)
        except: return {}
    return {}

def save_cache(data, path):
    try:
        with open(path, 'w') as f: json.dump(data, f)
    except: pass

# --- 3. ROW PROCESSING ---

def fetch_fresh_data(row, tmdb_cache, omdb_cache):
    """Fetches data from scratch for a single row."""
    lb_name = str(row['Name'])
    lb_year = clean_year(row['Year'])
    
    # 1. TMDB
    target = smart_search_tmdb(lb_name, lb_year)
    tmdb_data = {}
    
    if target:
        details = tmdb_movie.details(target.id)
        credits = tmdb_movie.credits(target.id)
        
        def get_names(obj_list): return [x['name'] for x in obj_list] if obj_list else []

        tmdb_data = {
            'tmdb_id': target.id,
            'imdb_id': getattr(details, 'imdb_id', None),
            'title': getattr(details, 'title', lb_name),
            'overview': getattr(details, 'overview', ''),
            'tagline': getattr(details, 'tagline', ''),
            'runtime': getattr(details, 'runtime', 0),
            'release_date': getattr(details, 'release_date', ''),
            'genres_list': get_names(getattr(details, 'genres', [])),
            'production_companies': get_names(getattr(details, 'production_companies', [])),
            'cast_list': get_names(getattr(credits, 'cast', [])),
            'directors_list': [c['name'] for c in getattr(credits, 'crew', []) if c['job'] == 'Director'],
            'writers_list': [c['name'] for c in getattr(credits, 'crew', []) if c['department'] == 'Writing'],
            'poster_path': getattr(details, 'poster_path', ''),
            'backdrop_path': getattr(details, 'backdrop_path', ''),
            'vote_average': getattr(details, 'vote_average', 0),
            'vote_count': getattr(details, 'vote_count', 0),
            'popularity': getattr(details, 'popularity', 0),
            'original_language': getattr(details, 'original_language', ''),
            'homepage': getattr(details, 'homepage', '')
        }
    
    # 2. OMDB
    imdb_id = tmdb_data.get('imdb_id') or row.get('imdb_id')
    omdb_res = fetch_omdb_direct(imdb_id=imdb_id, title=lb_name, year=lb_year)
    omdb_data = {}

    if omdb_res:
        rt_score = next((r['Value'] for r in omdb_res.get('Ratings', []) if r['Source'] == 'Rotten Tomatoes'), None)
        omdb_data = {
            'rated': omdb_res.get('Rated'),
            'awards': omdb_res.get('Awards'),
            'country': omdb_res.get('Country'),
            'imdb_rating': omdb_res.get('imdbRating'),
            'imdb_votes': omdb_res.get('imdbVotes'),
            'metascore': omdb_res.get('Metascore'),
            'rotten_tomatoes_rating': rt_score,
            'box_office': omdb_res.get('BoxOffice'),
            'type': omdb_res.get('Type'),
            'dvd': omdb_res.get('DVD'),
            'omdb_genre': omdb_res.get('Genre'),
            'omdb_director': omdb_res.get('Director'),
            'omdb_actors': omdb_res.get('Actors')
        }

    # 3. MERGE
    def pick(*args):
        for val in args:
            if val is not None and str(val) != '' and str(val).lower() != 'nan' and str(val) != 'N/A':
                return val
        return None

    return {
        'letterboxd_name': lb_name,
        'year': lb_year,
        'tmdb_id': tmdb_data.get('tmdb_id'),
        'imdb_id': pick(tmdb_data.get('imdb_id'), imdb_id),
        'letterboxd_uri': row.get('Letterboxd URI'),
        'user_rating': row.get('Rating'),
        'is_liked': row.get('is_liked', 0),
        
        'title': pick(tmdb_data.get('title'), omdb_res.get('Title') if omdb_res else None, lb_name),
        'tagline': tmdb_data.get('tagline'),
        'overview': tmdb_data.get('overview'), # Plot removed as requested
        'genre': pick(tmdb_data.get('genres_list'), omdb_data.get('omdb_genre')),
        
        'director': pick(tmdb_data.get('directors_list'), omdb_data.get('omdb_director')),
        'writer': tmdb_data.get('writers_list'), 
        'actors': pick(tmdb_data.get('cast_list'), omdb_data.get('omdb_actors')),
        
        'imdb_rating': omdb_data.get('imdb_rating'),
        'imdb_votes': omdb_data.get('imdb_votes'),
        'metascore': omdb_data.get('metascore'),
        'rotten_tomatoes_rating': omdb_data.get('rotten_tomatoes_rating'),
        'vote_average': tmdb_data.get('vote_average'),
        
        'runtime': pick(tmdb_data.get('runtime'), omdb_res.get('Runtime') if omdb_res else None),
        'released': pick(tmdb_data.get('release_date'), omdb_res.get('Released') if omdb_res else None),
        'rated': omdb_data.get('rated'),
        'language': tmdb_data.get('original_language'),
        'country': omdb_data.get('country'),
        'awards': omdb_data.get('awards'),
        
        'poster': pick(tmdb_data.get('poster_path'), omdb_res.get('Poster') if omdb_res else None),
        'backdrop_path': tmdb_data.get('backdrop_path'),
        
        'production': tmdb_data.get('production_companies'),
        'website': tmdb_data.get('homepage'),
        'processing_status': 'success' if tmdb_data.get('tmdb_id') else 'failed'
    }

def run_repair(limit=None):
    print("🔧 STARTING SMART REPAIR...")
    
    # 1. Load Master List
    try:
        ratings = pd.read_csv(config.RATINGS_PATH)
        liked = pd.read_csv(config.LIKED_PATH)
        ratings['is_liked'] = 0
        if not liked.empty and 'Letterboxd URI' in liked.columns:
            ratings.loc[ratings['Letterboxd URI'].isin(liked['Letterboxd URI']), 'is_liked'] = 1
    except Exception as e:
        print(f"❌ Error loading source CSVs: {e}")
        return

    # 2. Load Existing Data (To preserve it)
    existing_data_map = {}
    if os.path.exists(config.ENRICHED_DATA_PATH):
        try:
            df_exist = pd.read_csv(config.ENRICHED_DATA_PATH)
            # Create a lookup key based on Name+Year to find existing entries easily
            for _, row in df_exist.iterrows():
                key = f"{str(row.get('letterboxd_name'))}_{clean_year(row.get('year'))}"
                existing_data_map[key] = row.to_dict()
            print(f"📂 Loaded {len(df_exist)} existing records.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")

    # 3. Load Caches
    tmdb_cache = load_cache(config.TMDB_CACHE_PATH)
    omdb_cache = load_cache(config.OMDB_CACHE_PATH)
    
    if limit: ratings = ratings.head(limit)

    final_rows = []
    
    # 4. Iterate Master List
    for idx, row in ratings.iterrows():
        lb_name = str(row['Name'])
        lb_year = clean_year(row['Year'])
        key = f"{lb_name}_{lb_year}"
        
        # --- DECISION LOGIC ---
        use_existing = False
        
        if key in existing_data_map:
            existing_row = existing_data_map[key]
            
            # CHECK: Is the existing data actually valid?
            has_tmdb_id = existing_row.get('tmdb_id') and str(existing_row.get('tmdb_id')) != 'nan'
            
            # STRICT YEAR CHECK: Does the data we fetched actually match the CSV year?
            fetched_year = clean_year(str(existing_row.get('released', ''))[:4])
            year_match = True
            if fetched_year and lb_year:
                if abs(fetched_year - lb_year) > 1:
                    year_match = False
            
            if has_tmdb_id and year_match:
                use_existing = True
        
        if use_existing:
            # print(f"[{idx+1}] ✅ Keeping: {lb_name}")
            final_rows.append(existing_data_map[key])
        else:
            print(f"[{idx+1}] 🔄 Refetching: {lb_name} ({lb_year})")
            # Fetch fresh
            try:
                new_data = fetch_fresh_data(row, tmdb_cache, omdb_cache)
                final_rows.append(new_data)
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                final_rows.append(row.to_dict()) # Fallback

        if idx % 20 == 0:
            save_cache(tmdb_cache, config.TMDB_CACHE_PATH)

    # 5. Save
    save_cache(tmdb_cache, config.TMDB_CACHE_PATH)
    
    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(config.ENRICHED_DATA_PATH, index=False)
    
    print("-" * 30)
    print(f"✅ DONE. Total Rows: {len(df_final)}")
    print(f"📂 Saved to: {config.ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_repair(limit=args.limit)