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
        if '-' in s:
            parts = s.split('-')
            if len(parts[0]) == 4: return int(parts[0])
            if len(parts[-1]) == 4: return int(parts[-1])
        return int(float(s[:4]))
    except:
        return None

def batch_predict_unified_ratings():
    print("🚀 Starting Batch Prediction (Unified Ensemble Mode)...")

    # --- 1. Load Models & State ---
    ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
    model_paths = {
        "XGB": ensemble_dir / "xgb_base_regressor.joblib",
        "SVR": ensemble_dir / "svr_base_regressor.joblib",
        "CatBoost": ensemble_dir / "catboost_base_regressor.joblib",
        "Stacking": ensemble_dir / "stacking_ensemble_regressor.joblib",
        "Ordinal_EV": ensemble_dir / "ordinal_classifier.joblib"
    }
    
    models = {}
    for name, path in model_paths.items():
        if not path.exists():
            print(f"❌ ERROR: Model file not found for '{name}' at {path}.")
            return
        models[name] = joblib.load(path)
    
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
        
        df_movies['is_tv_show'] = 0
        df_movies['is_game'] = 0
        
        df_shows['is_tv_show'] = 1
        df_shows['is_game'] = 0
        # Align shows schema
        df_shows = df_shows.rename(columns={
            'name': 'title',
            'created_by': 'director',
            'production_companies': 'production',
            'genres': 'genre',
            'age_rating': 'rated'
        })

        # Align games schema
        df_games = df_games.rename(columns={
            'name': 'title',
            'my_rating': 'user_rating',
            'genres': 'genre',
            'developers': 'director',
            'metacritic': 'metascore',
            'rating': 'imdb_rating',
            'ratings_count': 'imdb_votes',
            'description_raw': 'overview',
            'age_rating': 'rated',
            'cover': 'poster'
        })
        df_games['is_tv_show'] = 0
        df_games['is_game'] = 1
        df_games['year'] = df_games['released'].apply(clean_year)
        
        df = pd.concat([df_movies, df_shows, df_games], ignore_index=True)
        print(f"   Loaded {len(df_movies)} movies, {len(df_shows)} shows, and {len(df_games)} games. Total: {len(df)} records.")
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
    
    df['number_of_seasons'] = pd.to_numeric(df.get('number_of_seasons', 1), errors='coerce').fillna(1)
    df['number_of_episodes'] = pd.to_numeric(df.get('number_of_episodes', 1), errors='coerce').fillna(1)

    df['imdb_rating_100'] = pd.to_numeric(df['imdb_rating'], errors='coerce') * 10
    df['vote_average_100'] = pd.to_numeric(df['vote_average'], errors='coerce') * 10
    df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
    
    critic_scores = df[['imdb_rating_100', 'metascore', 'rotten_tomatoes_rating', 'vote_average_100']]
    df['critic_avg_100'] = critic_scores.mean(axis=1)
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5
    
    if 'imdb_votes' in df.columns and 'vote_count' in df.columns:
        df['imdb_votes'] = df['imdb_votes'].fillna(df['vote_count'])
    elif 'vote_count' in df.columns:
        df.rename(columns={'vote_count': 'imdb_votes'}, inplace=True)
        
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            # Pre-clean strings by removing commas
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
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in state['mlb_genre'].classes_], index=df.index)

    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    X_temp = pd.concat([df[list(state['median_values'].keys())], X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_temp.columns = [sanitize_col(col) for col in X_temp.columns]
    X_temp = X_temp.loc[:, ~X_temp.columns.duplicated()]

    X_final = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    common_cols = list(set(X_temp.columns) & set(state['training_columns']))
    X_final[common_cols] = X_temp[common_cols]

    # --- 4. Predict with all models ---
    print("   Predicting ratings with unified models...")
    df['display_name'] = df['title'].fillna(df.get('letterboxd_name', 'Unknown'))
    
    for name, model in models.items():
        preds = model.predict(X_final)
        df[f'pred_{name}'] = np.round(np.clip(preds, 0, 5) * 2) / 2

    # --- 5. Performance Report ---
    if 'user_rating' in df.columns and not df['user_rating'].isna().all():
        eval_df = df.dropna(subset=['user_rating']).copy()
        y_true = eval_df['user_rating']
        
        print("\n" + "="*55)
        print("📊 UNIFIED ML ENSEMBLE PERFORMANCE REPORT")
        print("="*55)
        print(f"Total Evaluated: {len(eval_df)}")
        
        for name in models.keys():
            pred_col = f'pred_{name}'
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
        
        df['abs_diff_stacking'] = np.abs(df['user_rating'] - df['pred_Stacking'])

        # --- 6. Visualizations ---
        pred_dir = config.UNIFIED_PREDICTIONS_DIR
        pred_dir.mkdir(parents=True, exist_ok=True)

        # 6a. KDE Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.kdeplot(data=eval_df, x='user_rating', ax=ax, label='Actual Ratings', color='black', linewidth=3, fill=True, alpha=0.1)
        for name in models.keys():
            sns.kdeplot(data=eval_df, x=f'pred_{name}', ax=ax, label=f'{name} Preds', linestyle='--', linewidth=2)
            
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
        data_cols = ['user_rating'] + [f'pred_{name}' for name in models.keys()]
        titles = ['Actual Ratings'] + [f'{name} Predictions' for name in models.keys()]
        
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
        out_cols = ['display_name', 'year', 'is_tv_show', 'user_rating'] + [f'pred_{name}' for name in models.keys()] + ['abs_diff_stacking']
        final_df = df[[c for c in out_cols if c in df.columns]].sort_values(by='pred_Stacking', ascending=False)
        out_path = pred_dir / "unified_predictions_ensemble.csv"
        final_df.to_csv(out_path, index=False)
        print(f"✅ Saved unified model predictions to {out_path}")

if __name__ == "__main__":
    batch_predict_unified_ratings()