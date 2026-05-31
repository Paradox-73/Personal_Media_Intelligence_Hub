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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Load Environment Variables
load_dotenv()

# --- 1. SETUP & SAFETY ---
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")

if not TMDB_KEY or not OMDB_KEY:
    print("❌ ERROR: API Keys missing in .env file.")
    # Don't exit, just print warning so app doesn't crash on import
    # sys.exit(1) 

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
    if val is None: return True
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) == 0
    try:
        if pd.isna(val): return True
    except:
        pass
    s = str(val).strip().lower()
    return s in ['', 'nan', 'n/a', 'none', 'null', 'unknown', 'false', '0', '0.0', 'not rated', 'unrated']

def fetch_omdb_direct(imdb_id=None, title=None, year=None):
    """Internal helper to fetch from OMDB."""
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

# --- 3. EXPORTED FUNCTIONS (FOR APPS) ---

def get_omdb_data(imdb_id):
    """
    Wrapper for Oracle App to fetch OMDB data by ID.
    Returns specific fields needed for the model (Metascore, RT, etc.)
    """
    data = fetch_omdb_direct(imdb_id=imdb_id)
    if not data: return None
    
    # Extract specific fields needed for the Oracle
    rt_score = next((r['Value'] for r in data.get('Ratings', []) if r['Source'] == 'Rotten Tomatoes'), None)
    return {
        'metascore': data.get('Metascore'),
        'rotten_tomatoes_rating': rt_score,
        'awards': data.get('Awards'),
        'box_office': data.get('BoxOffice'),
        'rated': data.get('Rated'),
        'poster_omdb': data.get('Poster'),
        'plot': data.get('Plot')
    }

def search_movies_by_query(query):
    """Searches TMDB for movies matching the query. Used by Oracle UI."""
    results = []
    if not query: return []
    try:
        search_res = tmdb_movie.search(query)
        for res in search_res:
            results.append({
                'id': res.id,
                'title': getattr(res, 'title', 'Unknown'),
                'year': clean_year(getattr(res, 'release_date', '')),
                'poster_path': getattr(res, 'poster_path', ''),
                'director': 'Unknown' 
            })
    except Exception as e:
        print(f"Search error: {e}")
    return results

def fetch_movie_details_by_tmdb_id(tmdb_id):
    """Fetches details from TMDB. Used by Oracle UI."""
    try:
        # Fetch details
        m = tmdb_movie.details(int(tmdb_id))
        
        # Fetch credits (Directors/Cast)
        credits = tmdb_movie.credits(tmdb_id)
        
        director = [c.name for c in credits.crew if c.job == 'Director']
        actors = [c.name for c in credits.cast][:5]
        genres = [g.name for g in m.genres] if hasattr(m, 'genres') else []
        prod = [p.name for p in getattr(m, 'production_companies', [])]

        return {
            'tmdb_id': m.id,
            'imdb_id': getattr(m, 'imdb_id', None),
            'title': m.title,
            'year': clean_year(getattr(m, 'release_date', '')),
            'runtime': getattr(m, 'runtime', 0),
            'overview': getattr(m, 'overview', ''),
            'tagline': getattr(m, 'tagline', ''),
            'director': director,
            'actors': actors,
            'genre': genres,
            'production': prod,
            'poster': m.poster_path,
            'vote_average': getattr(m, 'vote_average', 0),
            'vote_count': getattr(m, 'vote_count', 0),
            'popularity': getattr(m, 'popularity', 0),
            'processing_status': 'success'
        }
    except Exception as e:
        return {'processing_status': 'error', 'error': str(e)}

# --- 4. CORE BATCH LOGIC ---

def get_movie_metadata(title, year, tmdb_id=None, imdb_id=None):
    """
    Fetches available data from TMDB and OMDB for batch processing.
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
        'poster': getattr(target, 'poster_path', None) if target else None 
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
    
    return metadata

def run_repair(limit=None):
    # (Same code as before, omitted for brevity but presumed to exist)
    pass

def sync_letterboxd_data():
    """
    Syncs the raw ratings.csv and liked.csv with enriched_data.csv and new_ratings.csv.
    - Updates user_rating and is_liked for existing movies in enriched_data.csv.
    - Updates Rating and is_liked for existing movies in new_ratings.csv.
    - Adds truly new movies (not in either) to new_ratings.csv.
    """
    print("🔄 Starting Letterboxd Sync...")

    raw_path = config.RATINGS_PATH
    liked_path = config.LIKED_PATH
    enriched_path = config.MOVIES_ENRICHED_DATA_PATH
    new_ratings_path = config.DATA_DIR / "raw" / "movies" / "new_ratings.csv"

    if not raw_path.exists():
        print(f"❌ Error: Raw ratings file not found at {raw_path}")
        return

    # 1. Load Data
    try:
        df_raw = pd.read_csv(raw_path)
    except Exception as e:
        print(f"❌ Error reading {raw_path}: {e}")
        return

    liked_uris = set()
    if liked_path.exists():
        try:
            df_liked = pd.read_csv(liked_path)
            liked_uris = set(df_liked['Letterboxd URI'].dropna())
            print(f"❤️ Loaded {len(liked_uris)} liked movies.")
        except Exception as e:
            print(f"⚠️ Warning: Could not read {liked_path}: {e}")

    # Standardize column names for Letterboxd export
    rename_map = {'Name': 'Name', 'Year': 'Year', 'Rating': 'Rating', 'Letterboxd URI': 'Letterboxd URI'}
    df_raw.rename(columns={c: rename_map.get(c, c) for c in df_raw.columns}, inplace=True)

    df_enriched = pd.read_csv(enriched_path) if enriched_path.exists() else pd.DataFrame()
    df_new = pd.read_csv(new_ratings_path) if new_ratings_path.exists() else pd.DataFrame()

    print(f"📊 Raw: {len(df_raw)} | Enriched: {len(df_enriched)} | New: {len(df_new)}")

    updates_enriched = 0
    updates_new = 0
    new_entries = []

    # Identify existing movies
    # Use Letterboxd URI as primary key
    enriched_uris = set(df_enriched['letterboxd_uri'].dropna()) if not df_enriched.empty and 'letterboxd_uri' in df_enriched.columns else set()
    new_uris = set(df_new['Letterboxd URI'].dropna()) if not df_new.empty and 'Letterboxd URI' in df_new.columns else set()

    for _, row in df_raw.iterrows():
        uri = row.get('Letterboxd URI')
        rating = row.get('Rating')
        is_liked = 1 if uri in liked_uris else 0

        # Check Enriched Data
        if uri in enriched_uris:
            mask = df_enriched['letterboxd_uri'] == uri
            changed = False
            if df_enriched.loc[mask, 'user_rating'].iloc[0] != rating:
                df_enriched.loc[mask, 'user_rating'] = rating
                changed = True
            
            if 'is_liked' in df_enriched.columns:
                if df_enriched.loc[mask, 'is_liked'].iloc[0] != is_liked:
                    df_enriched.loc[mask, 'is_liked'] = is_liked
                    changed = True
            else:
                df_enriched.loc[mask, 'is_liked'] = is_liked
                changed = True
            
            if changed:
                updates_enriched += 1

        # Check New Ratings
        elif uri in new_uris:
            mask = df_new['Letterboxd URI'] == uri
            changed = False
            if df_new.loc[mask, 'Rating'].iloc[0] != rating:
                df_new.loc[mask, 'Rating'] = rating
                changed = True
            
            if 'is_liked' in df_new.columns:
                if df_new.loc[mask, 'is_liked'].iloc[0] != is_liked:
                    df_new.loc[mask, 'is_liked'] = is_liked
                    changed = True
            else:
                df_new.loc[mask, 'is_liked'] = is_liked
                changed = True
                
            if changed:
                updates_new += 1

        # Truly New
        else:
            row_dict = row.to_dict()
            row_dict['is_liked'] = is_liked
            new_entries.append(row_dict)

    # 2. Save Updates
    if updates_enriched > 0:
        df_enriched.to_csv(enriched_path, index=False)
        print(f"✅ Updated {updates_enriched} movies in enriched_data.csv")

    if updates_new > 0:
        df_new.to_csv(new_ratings_path, index=False)
        print(f"✅ Updated {updates_new} movies in new_ratings.csv")

    if new_entries:
        df_new_to_add = pd.DataFrame(new_entries)
        if not df_new.empty:
            # Avoid re-adding if Name/Year match but URI was missing for some reason
            # (Safety check)
            existing_names_years = set(zip(df_new['Name'], df_new['Year']))
            df_new_to_add = df_new_to_add[~df_new_to_add.apply(lambda r: (r['Name'], r['Year']) in existing_names_years, axis=1)]

            if not df_new_to_add.empty:
                df_new_combined = pd.concat([df_new, df_new_to_add], ignore_index=True)
                df_new_combined.to_csv(new_ratings_path, index=False)
                print(f"✨ Added {len(df_new_to_add)} truly new movies to new_ratings.csv")
        else:
            df_new_to_add.to_csv(new_ratings_path, index=False)
            print(f"✨ Created new_ratings.csv with {len(df_new_to_add)} movies")
    else:
        print("🙌 No new movies found.")

    print("🏁 Sync complete.")

if __name__ == "__main__":
    sync_letterboxd_data()