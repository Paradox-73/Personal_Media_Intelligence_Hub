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
    if pd.isna(x) or x == "": return []
    try:
        if isinstance(x, list): return x
        if "[" in str(x): return ast.literal_eval(str(x))
        return [str(x)]
    except: return []

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

def batch_predict_ratings():
    print("🚀 Starting Batch Prediction (Scale 0-5 Fixed)...")

    if not config.MODEL_REGRESSOR.exists() or not config.PREPROCESSOR_STATE.exists():
        print("❌ Model or Preprocessor State not found.")
        return
        
    model = joblib.load(config.MODEL_REGRESSOR)
    state = joblib.load(config.PREPROCESSOR_STATE)
    
    try:
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        print(f"   Loaded {len(df)} movies.")
    except FileNotFoundError:
        print("❌ Data not found.")
        return

    print("   Processing features...")
    
    # --- Feature Engineering ---
    # 1. Numerics
    if df['rotten_tomatoes_rating'].dtype == 'object':
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].str.replace('%', '', regex=False).astype(float)

    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    
    df['total_wins'] = df['awards'].str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)

    # Re-calculate the critic average feature
    df['imdb_rating_100'] = df['imdb_rating'] * 10
    df['vote_average_100'] = df['vote_average'] * 10
    critic_scores = df[['imdb_rating_100', 'metascore', 'rotten_tomatoes_rating', 'vote_average_100']]
    df['critic_avg_100'] = critic_scores.mean(axis=1)
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5

    if 'imdb_votes' in df.columns:
        df.rename(columns={'imdb_votes': 'vote_count'}, inplace=True)
    
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med_val)
        else:
            df[col] = med_val
    
    # 2. Languages
    def map_language(lang_str):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in lang_str.split(',')]
        for lang in langs:
            if lang in state['top_languages']:
                return lang
        return 'Other'
    df['language_cleaned'] = df['language'].apply(map_language)
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')

    # 3. Text Embeddings
    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('Unknown') + \
                         ". Directed by: " + df['director'].fillna('Unknown') + \
                         ". Starring: " + df['actors'].fillna('Unknown') + \
                         ". Written by: " + df['writer'].fillna('Unknown') + \
                         ". Produced by: " + df['production'].fillna('Unknown') + \
                         ". " + df['tagline'].fillna('') + \
                         " " + df['overview'].fillna('') + \
                         " " + df['plot'].fillna('')
    
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    pca_vec = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(pca_vec.shape[1])], index=df.index)

    # 4. Genres
    df['genre_list'] = df['genre'].apply(parse_list)
    genre_encoded = state['mlb_genre'].transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in state['mlb_genre'].classes_], index=df.index)

    # 5. MPAA
    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # 6. Assemble
    X_temp = pd.concat([df[list(state['median_values'].keys())], X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_temp.columns = [sanitize_col(col) for col in X_temp.columns]
    X_temp = X_temp.loc[:, ~X_temp.columns.duplicated()]

    # 7. Alignment with training columns (Implicitly drops popularity if removed from training)
    X_final = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    common_cols = list(set(X_temp.columns) & set(state['training_columns']))
    X_final[common_cols] = X_temp[common_cols]

    print("   Predicting ratings...")
    
    # --- Predict ---
    preds = model.predict(X_final)
    
    df['predicted_rating'] = np.round(np.clip(preds, 0, 5) * 2) / 2
    df['name'] = df['title'].fillna(df.get('letterboxd_name', 'Unknown'))

    # --- Report ---
    if 'user_rating' in df.columns and not df['user_rating'].isna().all():
        eval_df = df.dropna(subset=['user_rating']).copy()
        y_true, y_pred = eval_df['user_rating'], eval_df['predicted_rating']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        diffs = np.abs(y_true - y_pred)
        total = len(diffs)
        
        print("\n" + "="*40)
        print("📊 ML ENGINEER PERFORMANCE REPORT")
        print("="*40)
        print(f"Total Evaluated: {len(eval_df)}")
        print("-" * 40)
        print(f"📉 MAE:  {mae:.4f}")
        print(f"📉 RMSE: {rmse:.4f}")
        print(f"📈 R²:   {r2:.4f}")
        print("-" * 40)
        print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
        print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
        print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
        print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
        print("="*40 + "\n")
        
        df.loc[eval_df.index, 'abs_diff'] = diffs

        # Side-by-Side Histograms
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        
        sns.histplot(y_true, bins=np.arange(0.25, 5.75, 0.5), 
                     ax=axes[0], color='royalblue', kde=True)
        axes[0].set_title('My Actual Ratings Distribution')
        axes[0].set_xlabel('Rating (0.5 - 5.0)')
        axes[0].set_ylabel('Count')
        axes[0].set_xticks(np.arange(0.5, 5.5, 0.5))
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        sns.histplot(y_pred, bins=np.arange(0.25, 5.75, 0.5), 
                     ax=axes[1], color='darkorange', kde=True)
        axes[1].set_title("ML's Predicted Ratings Distribution")
        axes[1].set_xlabel('Rating (0.5 - 5.0)')
        axes[1].set_ylabel('')
        axes[1].set_xticks(np.arange(0.5, 5.5, 0.5))
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        fig.suptitle('Comparison of Rating Distributions (Full Dataset)', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = config.PREDICTIONS_DIR / "ratings_distribution_histogram.png"
        plt.savefig(plot_path)
        print(f"📈 Histogram saved to {plot_path}")

    # Save
    out_cols = ['name', 'year', 'user_rating', 'predicted_rating', 'abs_diff', 'director', 'production']
    final_df = df[[c for c in out_cols if c in df.columns]].sort_values(by='predicted_rating', ascending=False)
    
    config.MOVIES_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.MOVIES_PREDICTIONS_DIR / "predicted_ratings.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    batch_predict_ratings()