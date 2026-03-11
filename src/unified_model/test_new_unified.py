import pandas as pd
import numpy as np
import joblib
import sys
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Import both metadata fetchers
from src.movies.ingestion import get_movie_metadata
# Assumes you have a parallel function for TV shows. Adjust import if named differently.
#from src.shows.ingestion import get_show_metadata 

def round_to_nearest_half(rating):
    return np.clip(np.round(rating * 2) / 2, 0.5, 5.0)

def transform_single_media(media_data, state, is_tv_show):
    """Transforms a single raw metadata dictionary into unified features."""
    import re, ast
    row = {col: state['median_values'].get(col, 0) for col in state['median_values']}
    row['is_tv_show'] = 1 if is_tv_show else 0

    def safe_num(val, default_val):
        if pd.isna(val) or str(val).lower() in ['nan', 'n/a', 'none', '']: return default_val
        try: return float(str(val).replace(',', '').replace('%', '').strip())
        except: return default_val

    # Standard feature mappings
    row['year'] = safe_num(media_data.get('year'), row.get('year'))
    row['runtime'] = safe_num(media_data.get('runtime'), row.get('runtime'))
    row['imdb_rating'] = safe_num(media_data.get('imdb_rating'), row.get('imdb_rating'))
    row['metascore'] = safe_num(media_data.get('metascore'), row.get('metascore'))
    row['rotten_tomatoes_rating'] = safe_num(media_data.get('rotten_tomatoes_rating'), row.get('rotten_tomatoes_rating'))
    row['vote_average'] = safe_num(media_data.get('vote_average'), row.get('vote_average'))
    row['popularity'] = safe_num(media_data.get('popularity'), row.get('popularity'))
    row['imdb_votes'] = safe_num(media_data.get('imdb_votes', media_data.get('vote_count')), row.get('imdb_votes'))
    
    # TV specifics
    row['number_of_seasons'] = safe_num(media_data.get('number_of_seasons'), 1) if is_tv_show else 1
    row['number_of_episodes'] = safe_num(media_data.get('number_of_episodes'), 1) if is_tv_show else 1

    bo_str = str(media_data.get('box_office', '0')).replace('$', '').replace(',', '').strip()
    try: bo = float(bo_str)
    except: bo = 0
    row['box_office_log'] = np.log1p(bo)

    awards_str = str(media_data.get('awards', ''))
    wins_match = re.search(r'(\d+)\s+win', awards_str, re.IGNORECASE)
    row['total_wins'] = float(wins_match.group(1)) if wins_match else 0
    noms_match = re.search(r'(\d+)\s+nomination', awards_str, re.IGNORECASE)
    row['total_nominations'] = float(noms_match.group(1)) if noms_match else 0
    
    imdb_100 = safe_num(media_data.get('imdb_rating'), 0) * 10
    meta = safe_num(media_data.get('metascore'), 0)
    rt = safe_num(media_data.get('rotten_tomatoes_rating'), 0)
    va = safe_num(media_data.get('vote_average'), 0) * 10
    critic_scores = [s for s in [imdb_100, meta, rt, va] if s > 0]
    row['critic_avg_5'] = ((np.mean(critic_scores) if critic_scores else 0) / 100) * 5
    
    final_df = pd.DataFrame(0, index=[0], columns=state['training_columns'])
    for col in row:
        if col in final_df.columns: final_df[col] = row[col]

    # Languages
    langs = [l.strip() for l in str(media_data.get('language', '')).split(',')]
    lang = next((l for l in langs if l in state['top_languages']), 'Other')
    lang_col = re.sub(r"[\[\]<']", "", f'lang_{lang}')
    if lang_col in final_df.columns: final_df[lang_col] = 1.0

    # Genres
    raw_genres = str(media_data.get('genre', media_data.get('genres', '')))
    genres = [g.strip() for g in raw_genres.split(',')] if ',' in raw_genres else [raw_genres]
    genre_mapping = {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}
    mapped_genres = []
    for g in genres: mapped_genres.extend(genre_mapping.get(g, [g]))
    
    gen_vec = state['mlb_genre'].transform([list(set(mapped_genres))])
    for i, gen_name in enumerate(state['mlb_genre'].classes_):
        col_name = re.sub(r"[\[\]<']", "", f"gen_{gen_name}")
        if col_name in final_df.columns: final_df[col_name] = float(gen_vec[0, i])

    # MPAA
    r = str(media_data.get('rated', media_data.get('age_rating', ''))).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: mpaa = 'Adult'
    elif 'PG' in r or 'TV-14' in r: mpaa = 'Teen'
    else: mpaa = 'General'
    mpaa_col = f"rated_{mpaa}"
    if mpaa_col in final_df.columns: final_df[mpaa_col] = 1.0

    # Text
    title = str(media_data.get('title', media_data.get('name', 'Unknown')))
    director = str(media_data.get('director', media_data.get('created_by', 'Unknown')))
    overview = str(media_data.get('overview', media_data.get('plot', '')))
    text_content = f"Title: {title}. Lead Creative: {director}. {overview}"
    
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    pca_vec = state['pca'].transform(transformer_model.encode([text_content]))
    for i in range(pca_vec.shape[1]):
        col_name = f'pca_{i}'
        if col_name in final_df.columns: final_df[col_name] = float(pca_vec[0, i])

    return final_df.astype(float)


def run_test_on_new_unified(csv_path):
    print(f"🚀 Loading new unified media from: {csv_path}")
    
    try:
        df_new = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Automatically add 'Type' if it doesn't exist (assumes legacy movie files)
    if 'Type' not in df_new.columns and 'type' not in df_new.columns:
        df_new['Type'] = 'movie'

    # Standardize column names from Letterboxd format
    required_cols = {'Name': 'Name', 'Year': 'Year', 'Rating': 'Rating', 'Type': 'Type'}
    
    # Capitalize column names dynamically so 'name' becomes 'Name'
    col_mapping = {c: c.title() for c in df_new.columns}
    df_new.rename(columns=col_mapping, inplace=True)
    
    if not all(col in df_new.columns for col in required_cols.values()):
        print(f"❌ Missing required columns. Found: {list(df_new.columns)}. Ensure CSV has: Name, Year, Rating")
        return
        
    df_new.dropna(subset=['Name', 'Year', 'Rating', 'Type'], inplace=True)
    df_new['Year'] = df_new['Year'].astype(int)


    # Load Models & State
    ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
    models = {}
    for name in ["XGB", "SVR", "CatBoost", "Stacking"]:
        path = ensemble_dir / f"{name.lower() if name != 'Stacking' else 'stacking_ensemble'}{'_base' if name != 'Stacking' else ''}.joblib"
        if path.exists(): models[name] = joblib.load(path)
        
    state_path = config.UNIFIED_PREPROCESSOR_STATE
    if not state_path.exists() or not models:
        print("❌ ERROR: Unified Models or Preprocessor State missing.")
        return
        
    state = joblib.load(state_path)
    print(f"✅ Loaded {len(models)} unified models.")

    results = []
    
    # --- 1. Load the Cache ---
    METADATA_CACHE_FILE = config.CACHE_DIR / "movies_test_metadata_cache.csv"
    if METADATA_CACHE_FILE.exists():
        cached_metadata_df = pd.read_csv(METADATA_CACHE_FILE)
    else:
        cached_metadata_df = pd.DataFrame() 
        
    print(f"🔍 Enriching and predicting for {len(df_new)} items...")
    
    for _, row in df_new.iterrows():
        name, year, m_type, actual = str(row['Name']), int(row['Year']), str(row['Type']).lower(), float(row['Rating'])
        is_tv_show = m_type in ['show', 'tv', 'series']
        
        try:
            # --- 2. Check the Cache ---
            # Try matching by 'title' (movies) or 'name' (shows)
            title_match = (cached_metadata_df.get('title', pd.Series(dtype=str)).str.lower() == name.lower())
            name_match = (cached_metadata_df.get('name', pd.Series(dtype=str)).str.lower() == name.lower())
            year_match = (cached_metadata_df.get('year', pd.Series(dtype=int)) == year)
            
            cached_entry_mask = (title_match | name_match) & year_match
            cached_entry = cached_metadata_df[cached_entry_mask]

            if not cached_entry.empty:
                print(f"   [CACHE HIT] Using cached data for '{name} ({year})'")
                metadata = cached_entry.iloc[0].to_dict()
                
                # Un-flatten JSON lists/dicts
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try: metadata[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError): pass
            else:
                # --- 3. Cache Miss: Fetch from API ---
                print(f"   [CACHE MISS] Fetching data for {m_type.upper()}: '{name} ({year})'")
                if is_tv_show:
                    # metadata = get_show_metadata(name, year) # Uncomment when you have the show ingestor
                    metadata = {} 
                else:
                    metadata = get_movie_metadata(name, year)
                    
                if metadata and (metadata.get('title') or metadata.get('name')):
                    # Save back to cache
                    flattened_metadata = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in metadata.items()}
                    new_cache_entry = pd.DataFrame([flattened_metadata])
                    cached_metadata_df = pd.concat([cached_metadata_df, new_cache_entry], ignore_index=True)
                    cached_metadata_df.to_csv(METADATA_CACHE_FILE, index=False)
            
            if not metadata or not (metadata.get('title') or metadata.get('name')):
                print(f"⚠️ Skipping '{name}': Could not fetch metadata.")
                continue
                
            # --- 4. Transform and Predict ---
            features = transform_single_media(metadata, state, is_tv_show)
            
            result_row = {'Name': name, 'Type': m_type, 'Year': year, 'Actual': actual}
            for model_name, model in models.items():
                result_row[f'Pred_{model_name}'] = round_to_nearest_half(model.predict(features)[0])
            results.append(result_row)
            
        except Exception as e:
            print(f"⚠️ Error processing '{name}': {e}")
            continue
                
        # features = transform_single_media(metadata, state, is_tv_show)
            
        # result_row = {'Name': name, 'Type': m_type, 'Year': year, 'Actual': actual}
        # for model_name, model in models.items():
        #     result_row[f'Pred_{model_name}'] = round_to_nearest_half(model.predict(features)[0])
        # results.append(result_row)
            
            
    if not results: return
    df_results = pd.DataFrame(results)
    
    # Evaluate & Report
    y_true = df_results['Actual']
    print("\n" + "="*55 + "\n📊 UNIFIED NEW MEDIA TEST REPORT\n" + "="*55)
    for name in models.keys():
        y_pred = df_results[f'Pred_{name}']
        rmse, mae, r2 = np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred), r2_score(y_true, y_pred)
        print(f"   Model: {name}\n   📉 RMSE: {rmse:.4f} | 📉 MAE: {mae:.4f} | 📈 R²: {r2:.4f}\n" + "-"*55)

    pred_dir = config.UNIFIED_PREDICTIONS_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / "new_unified_test_results.csv"
    df_results.to_csv(out_path, index=False)
    print(f"✅ Unified test results saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test unified model on new media.")
    parser.add_argument("csv_path", help="Path to CSV (Name, Year, Type, Rating)")
    args = parser.parse_args()
    run_test_on_new_unified(args.csv_path)