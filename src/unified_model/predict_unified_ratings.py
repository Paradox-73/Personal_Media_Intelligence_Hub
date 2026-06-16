import pandas as pd
import joblib
import numpy as np
import sys
import ast
import re
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

from src.unified_model.unified_utils import get_music_affinity_features, get_music_gate_mask

# Define MusicProfile in __main__ to satisfy joblib
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MusicProfile:
    centroids: np.ndarray
    cluster_labels: List[str]
    cluster_meta: List[Dict[str, Any]]
    X_lib: np.ndarray
    pu_model: Any
    pool_score_dist: np.ndarray
    top_genres: List[tuple]
    audio_fingerprint: Dict[str, float]
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]

class SimplexWeightedAverager:
    def __init__(self):
        self.weights = None
        self.n_models = 0
    def predict(self, X):
        return X @ self.weights

def clean_money(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return 0

def parse_list(x):
    if isinstance(x, (list, np.ndarray)): return x if isinstance(x, list) else x.tolist()
    if pd.isna(x) or x == "": return []
    try:
        if "[" in str(x): return ast.literal_eval(str(x))
        return [item.strip() for item in str(x).split(',')]
    except: return [str(x)]

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return None
        s = str(val).strip()
        match = re.search(r'(\d{4})', s)
        if match: return int(match.group(1))
        return int(float(s[:4]))
    except:
        return None

def batch_predict_unified_ratings():
    print("🚀 Starting Batch Prediction (Unified Ensemble Mode)...")

    # --- 1. Load Models & State ---
    ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
    model_paths = {
        "MeanEnsemble": ensemble_dir / "stacking_ensemble_regressor.joblib",
        "XGB_Base": ensemble_dir / "xgb_base_regressor.joblib",
        "CatBoost_Base": ensemble_dir / "catboost_base_regressor.joblib",
        "Ordinal_EV": ensemble_dir / "ordinal_classifier.joblib"
    }
    
    models = {}
    for name, path in model_paths.items():
        if path.exists():
            models[name] = joblib.load(path)
        else:
            print(f"⚠️ Warning: Model file not found for '{name}' at {path}.")
            
    if not models:
        print("❌ ERROR: No models loaded.")
        return
    
    state_path = config.UNIFIED_PREPROCESSOR_STATE
    if not state_path.exists():
        print(f"❌ ERROR: Unified Preprocessor State not found at {state_path}.")
        return
    state = joblib.load(state_path)
    print(f"✅ Loaded {len(models)} models and preprocessor state.")
    
    # --- 2. Load and Combine Data ---
    try:
        df_movies = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        df_shows = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        df_games = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        df_books = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        
        df_movies['media_type'] = 'movie'
        df_movies['is_tv_show'] = 0; df_movies['is_game'] = 0; df_movies['is_book'] = 0
        
        df_shows['media_type'] = 'tv'
        df_shows['is_tv_show'] = 1; df_shows['is_game'] = 0; df_shows['is_book'] = 0
        df_shows = df_shows.rename(columns={'name': 'title', 'created_by': 'director', 'genres': 'genre', 'age_rating': 'rated'})

        df_games['media_type'] = 'game'
        df_games = df_games.rename(columns={'name': 'title', 'my_rating': 'user_rating', 'genres': 'genre', 'developers': 'director', 'metacritic': 'metascore', 'rating': 'imdb_rating', 'ratings_count': 'imdb_votes', 'description_raw': 'overview', 'age_rating': 'rated'})
        df_games['is_tv_show'] = 0; df_games['is_game'] = 1; df_games['is_book'] = 0
        df_games['year'] = df_games['released'].apply(clean_year)
        
        df_books['media_type'] = 'book'
        df_books = df_books.rename(columns={'my_rating': 'user_rating', 'authors': 'director', 'categories': 'genre', 'averageRating': 'imdb_rating', 'ratingsCount': 'imdb_votes', 'description': 'overview', 'publishedDate': 'released', 'pageCount': 'runtime'})
        df_books['is_tv_show'] = 0; df_books['is_game'] = 0; df_books['is_book'] = 1
        df_books['year'] = df_books['released'].apply(clean_year)
        
        dfs = [df_movies, df_shows, df_games, df_books]
        # Check if music is available and intended
        if config.MUSIC_ENRICHED_DATA_PATH.exists():
            df_music = pd.read_csv(config.MUSIC_ENRICHED_DATA_PATH)
            df_music = df_music.rename(columns={'name': 'title', 'artists': 'director', 'artist_genres': 'genre', 'rating': 'user_rating', 'release_year': 'year', 'popularity': 'imdb_rating'})
            df_music['media_type'] = 'music'
            df_music['is_tv_show'] = 0; df_music['is_game'] = 0; df_music['is_book'] = 0; df_music['is_music'] = 1
            dfs.append(df_music)
            
        df = pd.concat(dfs, ignore_index=True)
        print(f"   Total: {len(df)} records across {df['media_type'].nunique()} domains.")
    except Exception as e:
        print(f"❌ ERROR loading enriched data: {e}")
        return

    # --- 3. Feature Engineering ---
    print("   Processing unified features...")
    df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'].astype(str).str.replace('%', '', regex=False), errors='coerce')
    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    
    mask_short_scale = (df['is_game'] == 1) | (df['is_book'] == 1)
    df['ir_100'] = pd.to_numeric(df['imdb_rating'], errors='coerce') * 10
    df.loc[mask_short_scale, 'ir_100'] = pd.to_numeric(df.loc[mask_short_scale, 'imdb_rating'], errors='coerce') * 20
    df['va_100'] = pd.to_numeric(df['vote_average'], errors='coerce') * 10
    df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
    df['critic_avg_5'] = (df[['ir_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']].mean(axis=1) / 100 * 5)
    
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med_val)
        else:
            df[col] = med_val
    
    df['language_cleaned'] = df['language'].apply(lambda x: next((l.strip() for l in str(x).split(',') if l.strip() in state['top_languages']), 'Other'))
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')
    
    df['text_content'] = "Title: " + df['title'].fillna('Unknown') + ". Lead Creative: " + df['director'].fillna('Unknown') + ". " + df['overview'].fillna(df.get('plot', '')).fillna('')
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    pca_vec = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(pca_vec.shape[1])], index=df.index)

    df['genre_list'] = df['genre'].apply(parse_list)
    genre_encoded = state['mlb_genre'].transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{sanitize_col(c)}" for c in state['mlb_genre'].classes_], index=df.index)

    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # Music Affinity (Gated)
    profile_path = config.MUSIC_MODEL_DIR / "profile.joblib"
    bundle_path = config.MUSIC_MODEL_DIR / "preprocessors.joblib"
    if profile_path.exists() and bundle_path.exists():
        profile = joblib.load(profile_path)
        bundle = joblib.load(bundle_path)
        X_music = get_music_affinity_features(df['text_content'].tolist(), profile, bundle).multiply(get_music_gate_mask(df['media_type']), axis=0)
        X_music.index = df.index
    else:
        X_music = pd.DataFrame()

    # Missingness Masks
    for domain in ['movie', 'tv', 'game', 'book', 'music']:
        df[f'has_{domain}_feats'] = (df['media_type'] == domain).astype(int)

    mask_cols = [f'has_{d}_feats' for d in ['movie', 'tv', 'game', 'book', 'music']]
    X_temp = pd.concat([df[list(state['median_values'].keys())], df[mask_cols], X_lang, X_genre, X_mpaa, X_text, X_music], axis=1)
    X_temp.columns = [sanitize_col(col) for col in X_temp.columns]
    X_temp = X_temp.loc[:, ~X_temp.columns.duplicated()]

    X_final = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    common_cols = [c for c in state['training_columns'] if c in X_temp.columns]
    X_final[common_cols] = X_temp[common_cols]

    # --- 4. Predict ---
    print("   Predicting ratings with unified models...")
    df['display_name'] = df['title'].fillna(df.get('letterboxd_name', 'Unknown'))
    
    def get_model_features(model):
        if hasattr(model, 'feature_names_in_'): return list(model.feature_names_in_)
        if hasattr(model, 'feature_names'): return list(model.feature_names)
        if hasattr(model, 'get_booster'): return model.get_booster().feature_names
        return None

    for name, model in models.items():
        feat_names = get_model_features(model)
        if feat_names and name != "MeanEnsemble":
            X_model = pd.DataFrame(0, index=df.index, columns=feat_names)
            common = [c for c in feat_names if c in X_final.columns]
            X_model[common] = X_final[common]
        else:
            X_model = X_final

        if name == "Ordinal_EV":
            try:
                probs = model.predict_proba(X_model)
                present_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])[joblib.load(config.UNIFIED_ENSEMBLE_DIR / "ordinal_classes.joblib")]
                raw_preds = np.sum(probs * present_vals, axis=1)
            except: continue
        elif name == "MeanEnsemble":
            # For MeanEnsemble, we need base model predictions
            try:
                # Use base models with their own aligned features if needed
                p_xgb = models['XGB_Base'].predict(pd.DataFrame(X_final, columns=get_model_features(models['XGB_Base'])))
                p_cat = models['CatBoost_Base'].predict(pd.DataFrame(X_final, columns=get_model_features(models['CatBoost_Base'])))
                raw_preds = model.predict(np.column_stack([p_xgb, p_cat]))
            except:
                raw_preds = (models['XGB_Base'].predict(X_final) + models['CatBoost_Base'].predict(X_final)) / 2
        else:
            raw_preds = model.predict(X_model)
            
        df[f'raw_pred_{name}'] = raw_preds
        df[f'pred_{name}'] = np.round(np.clip(raw_preds, 0.5, 5.0) * 2) / 2

    # Include PCA features for the ranker
    pca_cols = [c for c in X_final.columns if c.startswith('pca_')]
    df = pd.concat([df, X_final[pca_cols]], axis=1)

    # --- 5. Report ---
    if 'user_rating' in df.columns:
        df['user_rating_numeric'] = pd.to_numeric(df['user_rating'], errors='coerce')
        eval_df = df.dropna(subset=['user_rating_numeric']).copy()
        if not eval_df.empty:
            y_true = eval_df['user_rating_numeric']
            print("\n" + "="*55 + "\n📊 UNIFIED ML ENSEMBLE PERFORMANCE REPORT\n" + "="*55)
            for name in models.keys():
                pred_col = f'pred_{name}'
                if pred_col not in eval_df.columns: continue
                y_pred = eval_df[pred_col]
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"   Model: {name:<15} | MAE: {mae:.4f} | R²: {r2:.4f} | ±0.5 Acc: {((np.abs(y_true-y_pred)<=0.5).sum()/len(y_true))*100:.1f}%")
            print("="*55 + "\n")

    # Save
    out_dir = config.UNIFIED_PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = ['display_name', 'year', 'media_type', 'user_rating'] + [f'pred_{name}' for name in models.keys() if f'pred_{name}' in df.columns] + [f'raw_pred_{name}' for name in models.keys() if f'raw_pred_{name}' in df.columns] + pca_cols
    df[out_cols].to_csv(out_dir / "unified_predictions_ensemble.csv", index=False)
    print(f"✅ Saved unified predictions to {out_dir / 'unified_predictions_ensemble.csv'}")

if __name__ == "__main__":
    batch_predict_unified_ratings()
