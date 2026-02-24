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

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def clean_money(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return 0

def clean_text_value(x):
    if pd.isna(x): return 'Unknown'
    s = str(x).strip()
    s = s.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
    return s.strip()

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

def batch_predict_ratings():
    print("🚀 Starting Batch Prediction (Scale 0-5 Fixed)...")

    if not config.MODEL_REGRESSOR.exists():
        print("❌ Model not found.")
        return
        
    model = joblib.load(config.MODEL_REGRESSOR)
    state = joblib.load(config.PREPROCESSOR_STATE)
    
    try:
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        print(f"   Loaded {len(df)} movies.")
    except FileNotFoundError:
        print("❌ Data not found.")
        return

    # --- Feature Engineering ---
    # A. Numerics
    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med_val)
        else:
            df[col] = med_val

    # B. Categoricals
    get_primary = lambda x: parse_list(x)[0] if parse_list(x) else 'Unknown'
    
    def create_dummy_df(raw_col, prefix, valid_list, clean_func=None):
        if clean_func:
            clean_vals = df[raw_col].apply(clean_func)
            clean_vals = clean_vals.apply(clean_text_value)
        else:
            clean_vals = df[raw_col].apply(clean_text_value)
            
        clean_vals = clean_vals.apply(lambda x: x if x in valid_list else 'Other')
        all_categories = sorted(list(set(valid_list) | {'Other'}))
        cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
        return pd.get_dummies(clean_vals.astype(cat_type), prefix=prefix)

    X_dir = create_dummy_df('director', 'dir', state['valid_directors'], get_primary)
    X_act = create_dummy_df('actors', 'act', state['valid_actors'], get_primary)
    X_wri = create_dummy_df('writer', 'wri', state['valid_writers'], get_primary)
    
    df['primary_prod'] = df['production'].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown')
    X_prod = create_dummy_df('primary_prod', 'studio', state['valid_studios'], None)

    # C. MPAA & Genres
    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.DataFrame()
    for cat in ['Adult', 'Teen', 'General']:
        X_mpaa[f'rated_{cat}'] = (df['mpaa_cat'] == cat).astype(int)

    df['genre_list'] = df['genre'].apply(parse_list)
    genre_encoded = state['mlb_genre'].transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in state['mlb_genre'].classes_], index=df.index)

    # D. Text
    df['text_content'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    tfidf_mat = state['tfidf'].transform(df['text_content'])
    pca_mat = state['pca'].transform(tfidf_mat.toarray())
    X_text = pd.DataFrame(pca_mat, columns=[f'pca_{i}' for i in range(pca_mat.shape[1])], index=df.index)

    # E. Assembly
    X_temp = pd.concat([df[list(state['median_values'].keys())], X_dir, X_act, X_wri, X_prod, X_mpaa, X_genre, X_text], axis=1)
    
    # Sanitize
    new_cols = [re.sub(r"[\[\]<']", "", str(col)) for col in X_temp.columns]
    X_temp.columns = new_cols

    # F. Alignment
    X_final = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    common_cols = list(set(X_temp.columns) & set(state['training_columns']))
    X_final[common_cols] = X_temp[common_cols]

    # --- Predict ---
    preds = model.predict(X_final)
    
    df['predicted_rating'] = np.round(np.clip(preds, 0, 5) * 2) / 2
    df['name'] = df['title'].fillna(df['letterboxd_name'])

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

        # Add histogram plot
        plt.figure(figsize=(10, 6))
        sns.histplot(y_true, bins=np.arange(0, 5.75, 0.5), 
                     color='blue', alpha=0.5, label='Actual Ratings', kde=False)
        sns.histplot(y_pred, bins=np.arange(0, 5.75, 0.5), 
                     color='orange', alpha=0.5, label='Predicted Ratings', kde=False)
        plt.title('Distribution of Actual vs Predicted Ratings (Full Dataset)')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.xticks(np.arange(0.5, 5.5, 0.5))
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plot_path = config.PREDICTIONS_DIR / "ratings_distribution_histogram.png"
        plt.savefig(plot_path)
        print(f"📈 Histogram of rating distributions saved to {plot_path}")

    # Save
    out_cols = ['name', 'year', 'user_rating', 'predicted_rating', 'abs_diff', 'director', 'production']
    final_df = df[[c for c in out_cols if c in df.columns]].sort_values(by='predicted_rating', ascending=False)
    
    out_path = config.MOVIES_PREDICTIONS_DIR / "predicted_ratings.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✅ Saved to {out_path}")

if __name__ == "__main__":
    batch_predict_ratings()