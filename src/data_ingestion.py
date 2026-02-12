import pandas as pd
import json
import os
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
    
    # CRASH FIX: Check for lists/arrays BEFORE calling pd.isna
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
        
    try:
        if pd.isna(val): return True
    except:
        pass
        
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown', 'false', '0', '0.0', 'not rated', 'unrated']

def fetch_omdb_direct(imdb_id=None, title=None, year=None):
    try:
        if imdb_id and str(imdb_id).lower() != 'nan':
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

# --- 3. CORE LOGIC ---

def get_movie_metadata(title, year, tmdb_id=None, imdb_id=None):
    """
    Fetches available data from TMDB and OMDB.
    """
    
    tmdb_data = {}
    omdb_data = {}
    
    # --- A. FETCH TMDB ---
    target = None
    if tmdb_id and not is_effectively_empty(tmdb_id):
        try:
            target = tmdb_movie.details(int(float(tmdb_id)))
            target.credits = tmdb_movie.credits(target.id) 
        except: pass
    
    if not target:
        target = smart_search_tmdb(title, year)
        if target:
            try:
                detailed = tmdb_movie.details(target.id)
                detailed.credits = tmdb_movie.credits(target.id)
                target = detailed
            except: pass

    if target:
        def get_names(obj_list): return [x['name'] for x in obj_list] if obj_list else []
        
        tmdb_data = {
            'tmdb_id': target.id,
            'imdb_id': getattr(target, 'imdb_id', None),
            'title': getattr(target, 'title', title),
            'overview': getattr(target, 'overview', ''),
            'tagline': getattr(target, 'tagline', ''),
            'runtime': getattr(target, 'runtime', 0),
            'release_date': getattr(target, 'release_date', ''),
            'genres': get_names(getattr(target, 'genres', [])),
            'production': get_names(getattr(target, 'production_companies', [])),
            'cast': get_names(getattr(target.credits, 'cast', [])),
            'director': [c['name'] for c in getattr(target.credits, 'crew', []) if c['job'] == 'Director'],
            'writer': [c['name'] for c in getattr(target.credits, 'crew', []) if c['department'] == 'Writing'],
            'vote_average': getattr(target, 'vote_average', 0),
            'vote_count': getattr(target, 'vote_count', 0),
            'original_language': getattr(target, 'original_language', ''),
            # Removed: poster_path, backdrop_path, homepage (website)
        }

    # --- B. FETCH OMDB ---
    oid = tmdb_data.get('imdb_id') or imdb_id
    omdb_res = fetch_omdb_direct(imdb_id=oid, title=title, year=year)
    
    if omdb_res:
        rt_score = next((r['Value'] for r in omdb_res.get('Ratings', []) if r['Source'] == 'Rotten Tomatoes'), None)
        omdb_data = {
            'title': omdb_res.get('Title'),
            'rated': omdb_res.get('Rated'),
            'released': omdb_res.get('Released'),
            'runtime': omdb_res.get('Runtime'),
            'genre': omdb_res.get('Genre'),
            'director': omdb_res.get('Director'),
            'writer': omdb_res.get('Writer'),
            'actors': omdb_res.get('Actors'),
            'plot': omdb_res.get('Plot'),
            'language': omdb_res.get('Language'),
            'country': omdb_res.get('Country'),
            'awards': omdb_res.get('Awards'),
            'metascore': omdb_res.get('Metascore'),
            'imdb_rating': omdb_res.get('imdbRating'),
            'imdb_votes': omdb_res.get('imdbVotes'),
            'box_office': omdb_res.get('BoxOffice'),
            'rotten_tomatoes_rating': rt_score,
            'dvd': omdb_res.get('DVD'),
            # Removed: Poster, Type
        }

    # --- C. MERGE ---
    def pick(*args):
        for val in args:
            if not is_effectively_empty(val): return val
        return None

    return {
        'tmdb_id': tmdb_data.get('tmdb_id'),
        'imdb_id': pick(tmdb_data.get('imdb_id'), imdb_id),
        'title': pick(tmdb_data.get('title'), omdb_data.get('title'), title),
        
        # DISTINCT FIELDS FOR OVERVIEW AND PLOT
        'overview': tmdb_data.get('overview'),
        'plot': omdb_data.get('plot'),
        
        'tagline': tmdb_data.get('tagline'),
        'director': pick(tmdb_data.get('director'), omdb_data.get('director')),
        'writer': pick(tmdb_data.get('writer'), omdb_data.get('writer')),
        'actors': pick(tmdb_data.get('cast'), omdb_data.get('actors')),
        'genre': pick(tmdb_data.get('genres'), omdb_data.get('genre')),
        'imdb_rating': clean_float(omdb_data.get('imdb_rating')),
        'imdb_votes': clean_int(omdb_data.get('imdb_votes')),
        'metascore': clean_int(omdb_data.get('metascore')),
        'rotten_tomatoes_rating': omdb_data.get('rotten_tomatoes_rating'),
        'vote_average': clean_float(tmdb_data.get('vote_average')),
        'vote_count': clean_int(tmdb_data.get('vote_count')),
        'rated': omdb_data.get('rated'),
        'year': year,
        'released': pick(tmdb_data.get('release_date'), omdb_data.get('released')),
        'runtime': pick(tmdb_data.get('runtime'), omdb_data.get('runtime')),
        'language': pick(tmdb_data.get('original_language'), omdb_data.get('language')),
        'country': omdb_data.get('country'),
        'awards': omdb_data.get('awards'),
        'box_office': omdb_data.get('box_office'),
        'production': pick(tmdb_data.get('production'), None),
        # Removed: poster, backdrop_path, website
        'poster': getattr(target, 'poster_path', None) if target else None # Retaining poster purely for UI visualization if needed
    }

def fetch_fresh_data(row, tmdb_cache, omdb_cache):
    """Wrapper for initial ingestion."""
    lb_name = str(row['Name'])
    lb_year = clean_year(row['Year'])
    
    metadata = get_movie_metadata(lb_name, lb_year, imdb_id=row.get('imdb_id'))
    
    metadata['letterboxd_name'] = lb_name
    metadata['letterboxd_uri'] = row.get('Letterboxd URI')
    metadata['user_rating'] = row.get('Rating')
    metadata['is_liked'] = row.get('is_liked', 0)
    # Removed: processing_status
    
    return metadata

# --- 4. ORACLE / UI FUNCTIONS ---

def search_movies_by_query(query):
    """
    Searches TMDB for a movie by query string.
    Returns a list of dicts with basic info for the UI.
    """
    results = []
    try:
        search_res = tmdb_movie.search(query)
        for res in search_res:
            results.append({
                'id': res.id,
                'title': getattr(res, 'title', 'Unknown'),
                'year': clean_year(getattr(res, 'release_date', '')),
                'poster_path': getattr(res, 'poster_path', ''),
                'director': 'Unknown' # TMDB search result doesn't give director easily without details fetch
            })
    except Exception as e:
        print(f"Search error: {e}")
    return results

def fetch_movie_details_by_tmdb_id(tmdb_id):
    """
    Fetches full metadata for a specific TMDB ID (used by Oracle).
    """
    try:
        # 1. Fetch basic details first to get Title/Year/IMDb ID
        target = tmdb_movie.details(int(tmdb_id))
        title = getattr(target, 'title', '')
        year = clean_year(getattr(target, 'release_date', ''))
        imdb_id = getattr(target, 'imdb_id', None)
        
        # 2. Use the main metadata fetcher to get the full enriched dict (TMDB + OMDB)
        metadata = get_movie_metadata(title, year, tmdb_id=tmdb_id, imdb_id=imdb_id)
        metadata['processing_status'] = 'success'
        return metadata
    except Exception as e:
        print(f"Error fetching details by ID: {e}")
        return {'processing_status': 'failed'}

# --- 5. MODES ---

def run_repair(limit=None):
    print("🔧 STARTING SMART REPAIR (Ingestion Mode)...")
    try:
        ratings = pd.read_csv(config.RATINGS_PATH)
        liked = pd.read_csv(config.LIKED_PATH)
        ratings['is_liked'] = 0
        if not liked.empty and 'Letterboxd URI' in liked.columns:
            ratings.loc[ratings['Letterboxd URI'].isin(liked['Letterboxd URI']), 'is_liked'] = 1
    except Exception as e:
        print(f"❌ Error loading source CSVs: {e}")
        return

    existing_data_map = {}
    if os.path.exists(config.ENRICHED_DATA_PATH):
        try:
            df_exist = pd.read_csv(config.ENRICHED_DATA_PATH)
            for _, row in df_exist.iterrows():
                key = f"{str(row.get('letterboxd_name'))}_{clean_year(row.get('year'))}"
                existing_data_map[key] = row.to_dict()
            print(f"📂 Loaded {len(df_exist)} existing records.")
        except Exception as e:
            print(f"⚠️ Could not read existing file: {e}")

    final_rows = []
    if limit: ratings = ratings.head(limit)
    
    for idx, row in ratings.iterrows():
        lb_name = str(row['Name'])
        lb_year = clean_year(row['Year'])
        key = f"{lb_name}_{lb_year}"
        
        if key in existing_data_map:
            if not is_effectively_empty(existing_data_map[key].get('tmdb_id')):
                final_rows.append(existing_data_map[key])
                continue
        
        print(f"[{idx+1}] 🔄 Refetching: {lb_name} ({lb_year})")
        new_data = fetch_fresh_data(row, {}, {})
        final_rows.append(new_data)

    df_final = pd.DataFrame(final_rows)
    df_final.to_csv(config.ENRICHED_DATA_PATH, index=False)
    print(f"✅ DONE. Saved to: {config.ENRICHED_DATA_PATH}")

def run_fill_missing():
    print("🩹 STARTING UNIVERSAL GAP FILL...")
    
    if not os.path.exists(config.ENRICHED_DATA_PATH):
        print("❌ Enriched data file not found.")
        return

    try:
        # Load as object to avoid type inference issues
        df = pd.read_csv(config.ENRICHED_DATA_PATH, dtype=object)
    except Exception as e:
        print(f"❌ Error loading enriched CSV: {e}")
        return

    modified_count = 0
    
    for index, row in df.iterrows():
        missing_cols = []
        for col in df.columns:
            if is_effectively_empty(row[col]):
                missing_cols.append(col)
        
        if not missing_cols:
            continue

        print(f"[{index+1}] {row.get('title')} - Missing: {len(missing_cols)} fields")
        
        metadata = get_movie_metadata(
            title=row.get('title') or row.get('letterboxd_name'),
            year=clean_year(row.get('year')),
            tmdb_id=row.get('tmdb_id'),
            imdb_id=row.get('imdb_id')
        )

        updates = []
        for col in missing_cols:
            # Check if we have data for this missing column in our fresh metadata
            if col in metadata and not is_effectively_empty(metadata[col]):
                df.at[index, col] = metadata[col]
                updates.append(col)
        
        if updates:
            print(f"   ✨ Filled {len(updates)} cols: {', '.join(updates[:5])}...")
            modified_count += 1
        else:
            print("   ⚠️ No new data available from APIs.")

        if index > 0 and index % 20 == 0:
            df.to_csv(config.ENRICHED_DATA_PATH, index=False)

    df.to_csv(config.ENRICHED_DATA_PATH, index=False)
    print("-" * 30)
    print(f"✅ DONE. Updated {modified_count} rows.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fill-missing", action="store_true", help="Scan ALL columns and fill ANY missing value from APIs")
    args = parser.parse_args()
    
    if args.fill_missing:
        run_fill_missing()
    else:
        run_repair(limit=args.limit)