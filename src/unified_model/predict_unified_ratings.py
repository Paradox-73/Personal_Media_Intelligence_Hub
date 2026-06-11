import pandas as pd
import joblib
import numpy as np
import sys
import ast
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.advanced_unified_model_trainer import DomainResidualCorrector

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
        "Stacking": ensemble_dir / "stacking_ensemble_regressor.joblib",
        "XGB_Base": ensemble_dir / "xgb_base_regressor.joblib",
        "SVR_Base": ensemble_dir / "svr_base_regressor.joblib",
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
    
    # Use config.UNIFIED_PREPROCESSOR_STATE
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
        
        try:
            df_games = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        except:
            df_games = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
            
        try:
            df_books = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        except:
            df_books = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
        
        df_movies['media_type'] = 'movie'
        df_movies['is_tv_show'] = 0
        df_movies['is_game'] = 0
        df_movies['is_book'] = 0
        
        df_shows['media_type'] = 'tv'
        df_shows['is_tv_show'] = 1
        df_shows['is_game'] = 0
        df_shows['is_book'] = 0
        # Align shows schema
        df_shows = df_shows.rename(columns={
            'name': 'title',
            'created_by': 'director',
            'genres': 'genre',
            'age_rating': 'rated'
        })

        # Align games schema
        df_games['media_type'] = 'game'
        df_games = df_games.rename(columns={
            'name': 'title',
            'my_rating': 'user_rating',
            'genres': 'genre',
            'developers': 'director',
            'metacritic': 'metascore',
            'rating': 'imdb_rating',
            'ratings_count': 'imdb_votes',
            'description_raw': 'overview',
            'age_rating': 'rated'
        })
        df_games['is_tv_show'] = 0
        df_games['is_game'] = 1
        df_games['is_book'] = 0
        df_games['year'] = df_games['released'].apply(clean_year)
        
        # Align books schema
        df_books['media_type'] = 'book'
        df_books = df_books.rename(columns={
            'my_rating': 'user_rating',
            'authors': 'director',
            'categories': 'genre',
            'averageRating': 'imdb_rating',
            'ratingsCount': 'imdb_votes',
            'description': 'overview',
            'publishedDate': 'released',
            'pageCount': 'runtime'
        })
        if 'released' not in df_books.columns:
            df_books['released'] = None
        df_books['is_tv_show'] = 0
        df_books['is_game'] = 0
        df_books['is_book'] = 1
        df_books['year'] = df_books['released'].apply(clean_year)
        
        df = pd.concat([df_movies, df_shows, df_games, df_books], ignore_index=True)
        print(f"   Counts -> Movies: {len(df_movies)}, Shows: {len(df_shows)}, Games: {len(df_games)}, Books: {len(df_books)}")
        print(f"   Total: {len(df)} records.")
    except Exception as e:
        print(f"❌ ERROR loading enriched data: {e}")
        return

    print("   Processing unified features...")
    
    # --- 3. Feature Engineering ---
    df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].astype(str).str.replace('%', '', regex=False)
    df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'], errors='coerce')
    
    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    
    # Pre-clean numeric cols
    num_cols = ['is_tv_show', 'is_game', 'is_book', 'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'imdb_votes', 'box_office_log', 'popularity', 'total_wins', 'total_nominations', 'critic_avg_5']
    
    # Normalize short scales
    mask_short_scale = (df['is_game'] == 1) | (df['is_book'] == 1)
    df['ir_100'] = pd.to_numeric(df['imdb_rating'], errors='coerce') * 10
    df.loc[mask_short_scale, 'ir_100'] = pd.to_numeric(df.loc[mask_short_scale, 'imdb_rating'], errors='coerce') * 20
    
    df['va_100'] = pd.to_numeric(df['vote_average'], errors='coerce') * 10
    df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
    
    critic_scores = df[['ir_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']]
    df['critic_avg_5'] = (critic_scores.mean(axis=1) / 100 * 5)
    
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med_val)
        else:
            df[col] = med_val
    
    def map_language(lang_str):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in str(lang_str).split(',')]
        for lang in langs:
            if lang in state['top_languages']: return lang
        return 'Other'
        
    df['language_cleaned'] = df['language'].apply(map_language)
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')
    
    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('Unknown') + \
                         ". Lead Creative: " + df['director'].fillna('Unknown') + \
                         ". " + df['overview'].fillna(df.get('plot', '')).fillna('')
                         
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    pca_vec = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(pca_vec.shape[1])], index=df.index)

    df['genre_list'] = df['genre'].apply(parse_list)
    genre_mapping = {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}
    def unify_genres(g_list):
        mapped = []
        for g in g_list: mapped.extend(genre_mapping.get(g, [g]))
        return list(set(mapped))
    df['genre_list'] = df['genre_list'].apply(unify_genres)
    
    genre_encoded = state['mlb_genre'].transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{sanitize_col(c)}" for c in state['mlb_genre'].classes_], index=df.index)

    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # Music Affinity Features (Gated)
    print("   Applying Gated Music Affinity features...")
    profile_path = config.MUSIC_MODEL_DIR / "profile.joblib"
    bundle_path = config.MUSIC_MODEL_DIR / "preprocessors.joblib"
    
    if profile_path.exists() and bundle_path.exists():
        profile = joblib.load(profile_path)
        bundle = joblib.load(bundle_path)
        X_music_raw = get_music_affinity_features(df['text_content'].tolist(), profile, bundle)
        gate_mask = get_music_gate_mask(df['media_type'])
        X_music = X_music_raw.multiply(gate_mask, axis=0)
        X_music.index = df.index
    else:
        X_music = pd.DataFrame()

    X_temp = pd.concat([df[list(state['median_values'].keys())], X_lang, X_genre, X_mpaa, X_text, X_music], axis=1)
    X_temp.columns = [sanitize_col(col) for col in X_temp.columns]
    X_temp = X_temp.loc[:, ~X_temp.columns.duplicated()]

    X_final = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    common_cols = list(set(X_temp.columns) & set(state['training_columns']))
    X_final[common_cols] = X_temp[common_cols]

    # --- 4. Predict with all models ---
    print("   Predicting ratings with unified models...")
    df['display_name'] = df['title'].fillna(df.get('letterboxd_name', 'Unknown'))
    
    for name, model in models.items():
        # Try to get expected features
        expected_cols = None
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
        elif hasattr(model, 'feature_names_'):
            expected_cols = list(model.feature_names_)
        elif hasattr(model, 'get_booster'):
            expected_cols = model.get_booster().feature_names
        elif name == "Stacking" and hasattr(model, 'base_model') and hasattr(model.base_model, 'estimators_'):
            expected_cols = model.base_model.estimators_[0].get_booster().feature_names
            
        if expected_cols is not None:
            # Align features exactly in the order expected
            X_model = pd.DataFrame(0, index=df.index, columns=expected_cols)
            common = [c for c in expected_cols if c in X_final.columns]
            X_model[common] = X_final[common]
        else:
            X_model = X_final

        if name == "Ordinal_EV":
            try:
                probs = model.predict_proba(X_model)
                classes = joblib.load(config.UNIFIED_ENSEMBLE_DIR / "ordinal_classes.joblib")
                bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
                present_vals = np.array([bucket_map[c] for c in classes])
                raw_preds = np.sum(probs * present_vals, axis=1)
                df[f'raw_pred_{name}'] = raw_preds
                df[f'pred_{name}'] = np.round(np.clip(raw_preds, 0.5, 5.0) * 2) / 2
            except:
                print(f"⚠️ Warning: Could not run Ordinal_EV prediction.")
        elif name == "Stacking":
            try:
                preds = model.predict(X_model, df['media_type'])
                df[f'raw_pred_{name}'] = preds
                df[f'pred_{name}'] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2
            except TypeError:
                preds = model.predict(X_model)
                df[f'raw_pred_{name}'] = preds
                df[f'pred_{name}'] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2
        else:
            preds = model.predict(X_model)
            df[f'raw_pred_{name}'] = preds
            df[f'pred_{name}'] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2

    # --- 5. Performance Report ---
    if 'user_rating' in df.columns:
        df['user_rating_numeric'] = pd.to_numeric(df['user_rating'], errors='coerce')
        if not df['user_rating_numeric'].isna().all():
            eval_df = df.dropna(subset=['user_rating_numeric']).copy()
            y_true = eval_df['user_rating_numeric']
            
            print("\n" + "="*55)
            print("📊 UNIFIED ML ENSEMBLE PERFORMANCE REPORT")
            print("="*55)
            print(f"Total Evaluated: {len(eval_df)}")
        
        for name in models.keys():
            pred_col = f'pred_{name}'
            if pred_col not in eval_df.columns: continue
            y_pred = eval_df[pred_col]
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            diffs = np.abs(y_true - y_pred)
            total = len(diffs)
            
            print("-" * 55)
            print(f"   Model: {name}")
            print(f"   📉 MSE: {mse:.4f} | 📉 RMSE: {rmse:.4f} | 📉 MAE: {mae:.4f} | 📈 R²: {r2:.4f}")
            print(f"   Distribution -> Exact: {((diffs == 0.0).sum()/total)*100:.1f}% | ±0.5: {((diffs <= 0.5).sum()/total)*100:.1f}% | >1.0: {((diffs > 1.0).sum()/total)*100:.1f}%")
        print("="*55 + "\n")
        
        if 'pred_Stacking' in df.columns:
            df['abs_diff_stacking'] = np.abs(df['user_rating_numeric'] - df['pred_Stacking'])

        # --- 6. Visualizations ---
        pred_dir = config.UNIFIED_PREDICTIONS_DIR
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 6a. KDE Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.kdeplot(data=eval_df, x='user_rating_numeric', ax=ax, label='Actual Ratings', color='black', linewidth=3, fill=True, alpha=0.1)
        for name in models.keys():
            pred_col = f'pred_{name}'
            if pred_col in eval_df.columns:
                sns.kdeplot(data=eval_df, x=pred_col, ax=ax, label=f'{name} Preds', linestyle='--', linewidth=2)
            
        ax.set_title('Comparison of Predicted Rating Distributions (Unified Dataset - KDE)', fontsize=18, pad=20)
        ax.set_xlabel('Rating (0.5 - 5.0)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xticks(np.arange(0.5, 5.5, 0.5))
        ax.legend(title='Model', fontsize='medium')
        plt.tight_layout()
        kde_path = pred_dir / "unified_ratings_kde.png"
        plt.savefig(kde_path, dpi=150)
        print(f"📈 KDE distribution plot saved to {kde_path}")
        plt.close(fig) 

        # 6b. Histogram Grid
        data_cols = ['user_rating_numeric'] + [f'pred_{name}' for name in models.keys() if f'pred_{name}' in eval_df.columns]
        titles = ['Actual Ratings'] + [f'{name} Predictions' for name in models.keys() if f'pred_{name}' in eval_df.columns]
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
        axes = axes.flatten()
        
        for i, col in enumerate(data_cols):
            sns.histplot(eval_df[col], bins=np.arange(0.25, 5.75, 0.5), ax=axes[i], kde=False)
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Rating (0.5 - 5.0)')
            axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        for i in range(len(data_cols), len(axes)):
            axes[i].set_visible(False)

        fig.suptitle('Histogram of Rating Distributions (Unified Dataset)', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        hist_path = pred_dir / "unified_ratings_histograms.png"
        plt.savefig(hist_path, dpi=150)
        print(f"📈 Histogram grid plot saved to {hist_path}")
        plt.close(fig)
        
        # Save CSV Results
        out_cols = ['display_name', 'year', 'media_type', 'user_rating'] + [f'pred_{name}' for name in models.keys() if f'pred_{name}' in df.columns]
        if 'abs_diff_stacking' in df.columns: out_cols.append('abs_diff_stacking')
        
        final_df = df[[c for c in out_cols if c in df.columns]]
        if 'pred_Stacking' in df.columns:
            final_df = final_df.sort_values(by='pred_Stacking', ascending=False)
            
        out_path = pred_dir / "unified_predictions_ensemble.csv"
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved unified model predictions to {out_path}")

if __name__ == "__main__":
    batch_predict_unified_ratings()
