import pandas as pd
import json
import os
import argparse
import sys
import requests
import re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from tmdbv3api import TMDb, TV

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Load Environment Variables
load_dotenv()

# --- 1. SETUP & SAFETY ---
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")

if not TMDB_KEY:
    print("❌ ERROR: TMDB_API_KEY missing in .env file.")
    sys.exit(1)

tmdb = TMDb()
tmdb.api_key = TMDB_KEY
tmdb.language = 'en'
tmdb.debug = True 
tmdb_tv = TV()

# --- 2. HELPERS ---

def clean_title(title):
    """Removes common noise from titles for better search results."""
    if not title: return ""
    # Remove year in parenthesis: "Show Name (2015)" -> "Show Name"
    t = re.sub(r'\(\d{4}\)', '', str(title))
    # Remove country codes: "The Office (US)" -> "The Office"
    t = re.sub(r'\((US|UK|AU)\)', '', t)
    return t.strip()

def clean_year(val):
    try:
        s = str(val).replace('.0', '').strip()
        if not s or s.lower() in ['nan', 'none', '']: return None
        if '-' in s: return int(s.split('-')[0])
        if '/' in s: return int(s.split('/')[-1])
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
    if val is None: return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    try:
        if pd.isna(val): return True
    except:
        pass
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown', 'false', '0', '0.0', 'not rated', 'unrated', '[]']

# --- 3. API FETCHERS (ROBUST) ---

def fetch_tvmaze_robust(tvmaze_id=None, imdb_id=None, title=None):
    try:
        if tvmaze_id and not is_effectively_empty(tvmaze_id):
            res = requests.get(f"https://api.tvmaze.com/shows/{int(tvmaze_id)}", timeout=5)
            if res.status_code == 200: return res.json()
        
        if imdb_id and not is_effectively_empty(imdb_id):
            res = requests.get(f"https://api.tvmaze.com/lookup/shows?imdb={imdb_id}", timeout=5)
            if res.status_code == 200: return res.json()
        
        if title:
            clean_t = str(title).replace(' ', '%20')
            res = requests.get(f"https://api.tvmaze.com/singlesearch/shows?q={clean_t}", timeout=5)
            if res.status_code == 200: return res.json()
    except:
        pass
    return {}

def fetch_omdb_robust(title=None, imdb_id=None, year=None):
    if not OMDB_KEY: return {}
    try:
        params = {'apikey': OMDB_KEY, 'type': 'series'}
        
        # 1. Try IMDb ID (Best)
        if imdb_id and not is_effectively_empty(imdb_id):
            params['i'] = imdb_id
            res = requests.get("http://www.omdbapi.com/", params=params, timeout=5).json()
            if res.get('Response') == 'True': return res
        
        # 2. Try Title + Year (Exact)
        if title:
            clean_t = clean_title(title)
            params['t'] = clean_t
            if year: 
                params['y'] = year
                res = requests.get("http://www.omdbapi.com/", params=params, timeout=5).json()
                if res.get('Response') == 'True': return res
            
            # 3. Try Title Only (Fallback for mismatched years)
            if 'y' in params: del params['y']
            res = requests.get("http://www.omdbapi.com/", params=params, timeout=5).json()
            if res.get('Response') == 'True': return res

            # 4. FUZZY SEARCH (s=...)
            # If t=... failed, search for the list and take the first result
            params = {'apikey': OMDB_KEY, 'type': 'series', 's': clean_t}
            if year: params['y'] = year
            
            search_res = requests.get("http://www.omdbapi.com/", params=params, timeout=5).json()
            
            # If search found results, grab the first ID and recurse
            if search_res.get('Response') == 'True' and search_res.get('Search'):
                best_match_id = search_res['Search'][0]['imdbID']
                # Call myself again with the ID we just found
                return fetch_omdb_robust(imdb_id=best_match_id)

    except Exception as e:
        print(f"OMDb Error: {e}")
        pass
    return {}

def search_tmdb_robust(title, target_year):
    if not title: return None
    clean_t = clean_title(title)
    
    # 1. Search with Clean Title
    try:
        results = tmdb_tv.search(clean_t)
        
        # A. Try Exact Year Match
        if target_year:
            for res in results:
                if not hasattr(res, 'first_air_date') or not res.first_air_date: continue
                res_year = int(res.first_air_date.split('-')[0])
                if abs(res_year - target_year) <= 1:
                    return res
        
        # B. Popularity Match (if year fails or not provided)
        if results:
            return results[0] 
    except: pass
    
    return None

# --- 4. CORE LOGIC ---

def get_tv_show_metadata_ultimate(title, year, tmdb_id=None, tvmaze_id=None, imdb_id=None):
    tmdb_data = {}
    tvmaze_data = {}
    omdb_data = {}
    
    # --- PHASE 1: TMDB ANCHORING ---
    target_tmdb_obj = None
    
    # A. Try by TMDB ID
    if tmdb_id and not is_effectively_empty(tmdb_id):
        try: target_tmdb_obj = tmdb_tv.details(int(float(tmdb_id)))
        except: pass
    
    # B. Try Search
    if not target_tmdb_obj:
        target_tmdb_obj = search_tmdb_robust(title, year)
        if target_tmdb_obj:
            try: target_tmdb_obj = tmdb_tv.details(target_tmdb_obj.id)
            except: pass

    # --- PHASE 2: TMDB DEEP DIVE (Credits & Content Ratings) ---
    found_imdb_id = imdb_id
    tmdb_actors = []
    tmdb_writers = []
    tmdb_age_rating = None
    
    if target_tmdb_obj:
        # 1. IDs
        try:
            ex_ids = tmdb_tv.external_ids(target_tmdb_obj.id)
            if not found_imdb_id: found_imdb_id = ex_ids.get('imdb_id')
        except: pass

        # 2. Credits (Actors & Writers)
        try:
            credits = tmdb_tv.credits(target_tmdb_obj.id)
            if hasattr(credits, 'cast'):
                tmdb_actors = [c['name'] for c in credits['cast'][:6]]
            if hasattr(credits, 'crew'):
                tmdb_writers = [c['name'] for c in credits['crew'] if c['department'] == 'Writing'][:3]
        except: pass

        # 3. Content Ratings
        try:
            ratings = tmdb_tv.content_ratings(target_tmdb_obj.id)
            if hasattr(ratings, 'results'):
                us_rating = next((r['rating'] for r in ratings['results'] if r['iso_3166_1'] == 'US'), None)
                if not us_rating:
                    us_rating = ratings['results'][0]['rating'] if ratings['results'] else None
                tmdb_age_rating = us_rating
        except: pass
        
        # Populate TMDB Dict
        def get_names(obj_list): return [x['name'] for x in obj_list] if obj_list else []
        runtimes = getattr(target_tmdb_obj, 'episode_run_time', [])
        tmdb_runtime = int(sum(runtimes)/len(runtimes)) if runtimes else None

        tmdb_data = {
            'tmdb_id': target_tmdb_obj.id,
            'imdb_id': found_imdb_id,
            'name': getattr(target_tmdb_obj, 'name', title),
            'overview': getattr(target_tmdb_obj, 'overview', ''),
            'tagline': getattr(target_tmdb_obj, 'tagline', ''),
            'status': getattr(target_tmdb_obj, 'status', ''),
            'type': getattr(target_tmdb_obj, 'type', ''),
            'number_of_episodes': getattr(target_tmdb_obj, 'number_of_episodes', 0),
            'number_of_seasons': getattr(target_tmdb_obj, 'number_of_seasons', 0),
            'first_air_date': getattr(target_tmdb_obj, 'first_air_date', ''),
            'last_air_date': getattr(target_tmdb_obj, 'last_air_date', ''),
            'genres': get_names(getattr(target_tmdb_obj, 'genres', [])),
            'production_companies': get_names(getattr(target_tmdb_obj, 'production_companies', [])),
            'created_by': get_names(getattr(target_tmdb_obj, 'created_by', [])),
            'vote_average': getattr(target_tmdb_obj, 'vote_average', 0),
            'vote_count': getattr(target_tmdb_obj, 'vote_count', 0),
            'original_language': getattr(target_tmdb_obj, 'original_language', ''),
            'poster_path': getattr(target_tmdb_obj, 'poster_path', ''),
            'backdrop_path': getattr(target_tmdb_obj, 'backdrop_path', ''),
            'homepage': getattr(target_tmdb_obj, 'homepage', ''),
            'tmdb_runtime': tmdb_runtime
        }

    # --- PHASE 3: TVMAZE & OMDB ---
    
    # Fetch TVMaze
    tvmaze_data = fetch_tvmaze_robust(tvmaze_id=tvmaze_id, imdb_id=found_imdb_id, title=title)
    
    # If TMDB didn't have IMDb ID, maybe TVMaze does?
    if not found_imdb_id and tvmaze_data.get('externals', {}).get('imdb'):
        found_imdb_id = tvmaze_data['externals']['imdb']
        
    # Fetch OMDb (Now with Fuzzy Search logic)
    omdb_data = fetch_omdb_robust(title=title, imdb_id=found_imdb_id, year=year)

    # --- PHASE 4: AGGREGATION ---
    def pick(*args):
        for val in args:
            if not is_effectively_empty(val): return val
        return None
    
    network_val = None
    if isinstance(tvmaze_data.get('network'), dict): network_val = tvmaze_data['network'].get('name')
    elif isinstance(tvmaze_data.get('webChannel'), dict): network_val = tvmaze_data['webChannel'].get('name')

    omdb_runtime = None
    if omdb_data.get('Runtime'):
        try: omdb_runtime = int(omdb_data['Runtime'].split(' ')[0])
        except: pass

    return {
        # IDs
        'tmdb_id': tmdb_data.get('tmdb_id'),
        'tvmaze_id': tvmaze_data.get('id') or tvmaze_id,
        'imdb_id': pick(found_imdb_id, omdb_data.get('imdbID'), tmdb_data.get('imdb_id')),
        
        # Details
        'name': pick(tmdb_data.get('name'), tvmaze_data.get('name'), omdb_data.get('Title'), title),
        'year': clean_year(pick(tmdb_data.get('first_air_date'), tvmaze_data.get('premiered'), omdb_data.get('Released'), year)),
        'overview': pick(tmdb_data.get('overview'), tvmaze_data.get('summary'), omdb_data.get('Plot')),
        'tagline': tmdb_data.get('tagline'),
        'status': pick(tmdb_data.get('status'), tvmaze_data.get('status')),
        'type': pick(tvmaze_data.get('type'), tmdb_data.get('type')),
        
        # People
        'created_by': pick(tmdb_data.get('created_by'), None),
        'actors': pick(tmdb_actors, omdb_data.get('Actors'), None), 
        'writer': pick(tmdb_writers, omdb_data.get('Writer'), None), 
        
        # Meta
        'genres': pick(tmdb_data.get('genres'), tvmaze_data.get('genres'), omdb_data.get('Genre')),
        'language': pick(tmdb_data.get('original_language'), tvmaze_data.get('language'), omdb_data.get('Language')),
        'country': pick(omdb_data.get('Country'), tmdb_data.get('production_countries')),
        'network': network_val,
        'production_companies': tmdb_data.get('production_companies'),
        'homepage': pick(tmdb_data.get('homepage'), tvmaze_data.get('officialSite')),
        
        # Tech
        'number_of_episodes': pick(tmdb_data.get('number_of_episodes'), omdb_data.get('totalSeasons')),
        'number_of_seasons': pick(tmdb_data.get('number_of_seasons'), omdb_data.get('totalSeasons')),
        'runtime': pick(tvmaze_data.get('runtime'), tmdb_data.get('tmdb_runtime'), omdb_runtime),
        'first_air_date': pick(tmdb_data.get('first_air_date'), tvmaze_data.get('premiered'), omdb_data.get('Released')),
        'last_air_date': tmdb_data.get('last_air_date'),
        
        # Stats
        'vote_average': clean_float(tmdb_data.get('vote_average')),
        'vote_count': clean_int(tmdb_data.get('vote_count')),
        'imdb_rating': clean_float(omdb_data.get('imdbRating')),
        'imdb_votes': clean_int(omdb_data.get('imdbVotes')),
        'age_rating': pick(tmdb_age_rating, omdb_data.get('Rated')),
        'awards': omdb_data.get('Awards'),
        
        # Visuals
        'poster_path': pick(tmdb_data.get('poster_path'), omdb_data.get('Poster')),
        'backdrop_path': tmdb_data.get('backdrop_path'),
        
        'processing_status': 'success' if (tmdb_data or tvmaze_data or omdb_data) else 'failed'
    }

# --- 5. EXECUTION ---

def run_ingestion(limit=None):
    print("📺 STARTING TV SHOW INGESTION (Corrected Ultimate Mode)...")
    
    # Paths
    raw_path = config.TV_SHOWS_RAW_DIR / "ratings.csv"
    enriched_path = config.TV_SHOWS_ENRICHED_DATA_PATH
    
    try:
        df_raw = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"❌ Error: Raw file not found at {raw_path}")
        return

    # Normalize Raw Columns
    df_raw.rename(columns={
        'title': 'name', 
        'my_rating': 'user_rating', 
        'premiered': 'year_raw',
        'tvmaze_id': 'tvmaze_id_input'
    }, inplace=True)
    
    # Load Existing
    if os.path.exists(enriched_path):
        try: df_enriched = pd.read_csv(enriched_path)
        except: df_enriched = pd.DataFrame()
    else:
        df_enriched = pd.DataFrame()

    # Map Existing
    existing_map = {}
    if not df_enriched.empty:
        for idx, row in df_enriched.iterrows():
            if not is_effectively_empty(row.get('tvmaze_id')):
                existing_map[f"tvmaze_{int(float(row['tvmaze_id']))}"] = row.to_dict()
            name_key = str(row.get('name')).lower().strip()
            year_key = clean_year(row.get('year'))
            if name_key:
                existing_map[f"{name_key}_{year_key}"] = row.to_dict()

    # --- CRITICAL UPDATE: ADDED MISSING COLUMNS TO CHECK LIST ---
    ALL_COLUMNS = [
        'tmdb_id', 'tvmaze_id', 'imdb_id', 'name', 'year', 'overview',
        'created_by', 'actors', 'writer', 'genres', 'age_rating',
        'number_of_episodes', 'vote_average', 'poster_path',
        'imdb_rating', 'imdb_votes', 'awards' 
    ]

    final_rows = []
    
    if limit: df_raw = df_raw.head(limit)

    for idx, raw_row in df_raw.iterrows():
        name = str(raw_row['name']).strip()
        year = clean_year(raw_row.get('year_raw')) or clean_year(raw_row.get('year_watched'))
        tvmaze_id_input = clean_int(raw_row.get('tvmaze_id_input'))
        
        # Identify Existing Data
        existing_data = None
        if tvmaze_id_input:
            existing_data = existing_map.get(f"tvmaze_{tvmaze_id_input}")
        if not existing_data:
            key = f"{name.lower()}_{year}"
            existing_data = existing_map.get(key)

        # Check for Missing Data
        needs_fetch = False
        if not existing_data:
            needs_fetch = True
        else:
            for col in ALL_COLUMNS:
                if is_effectively_empty(existing_data.get(col)):
                    needs_fetch = True
                    break
        
        final_row = {}
        
        if needs_fetch:
            print(f"[{idx+1}] 🔄 Fetching: {name}")
            
            known_tmdb = existing_data.get('tmdb_id') if existing_data else None
            known_imdb = existing_data.get('imdb_id') if existing_data else None
            
            fetched_data = get_tv_show_metadata_ultimate(
                name, year, 
                tmdb_id=known_tmdb, 
                tvmaze_id=tvmaze_id_input, 
                imdb_id=known_imdb
            )
            
            if existing_data:
                final_row = existing_data.copy()
                for k, v in fetched_data.items():
                    if not is_effectively_empty(v):
                        final_row[k] = v
            else:
                final_row = fetched_data
        else:
            final_row = existing_data.copy()

        # Update User Stats
        final_row['user_rating'] = raw_row.get('user_rating')
        final_row['watch_count'] = raw_row.get('watch_count')
        final_row['year_watched'] = raw_row.get('year_watched')
        
        if is_effectively_empty(final_row.get('name')): final_row['name'] = name
        if is_effectively_empty(final_row.get('year')): final_row['year'] = year
        
        final_rows.append(final_row)

    # Save
    df_final = pd.DataFrame(final_rows)
    df_final = df_final.replace({r'\x00': ''}, regex=True)
    df_final.to_csv(config.TV_SHOWS_ENRICHED_DATA_PATH, index=False)
    print(f"✅ DONE. Saved to: {config.TV_SHOWS_ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    run_ingestion(limit=args.limit)