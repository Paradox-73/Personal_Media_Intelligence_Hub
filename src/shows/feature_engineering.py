import pandas as pd
import numpy as np
import ast
import joblib
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def safe_eval(val):
    if pd.isna(val) or val == "": return []
    try:
        if isinstance(val, list): return val
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(val).split(',') if x.strip()]

def run_feature_engineering():
    print("⚙️ STARTING TV SHOW FEATURE ENGINEERING...")

    if not config.TV_SHOWS_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.TV_SHOWS_ENRICHED_DATA_PATH}")
        return

    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    
    # Filter for Training: Rows with user_rating
    df_train = df[df['user_rating'].notna()].copy()
    print(f"   Training on {len(df_train)} rated shows.")

    # --- 1. Numerical Features ---
    num_cols = [
        'year', 'vote_average', 'vote_count', 
        'imdb_rating', 'imdb_votes', 
        'number_of_episodes', 'number_of_seasons', 'runtime'
    ]
    
    median_values = {}
    for col in num_cols:
        df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)
        median_values[col] = median_val

    # --- 2. Genres (Multi-Label) ---
    df_train['genres'] = df_train['genres'].apply(safe_eval)
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(df_train['genres'])
    genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c}" for c in mlb_genres.classes_], index=df_train.index)

    # --- 3. Categoricals ---
    cat_cols = ['status', 'age_rating', 'type', 'language']
    df_encoded_cats = pd.get_dummies(df_train[cat_cols], dummy_na=True)

    # --- 4. Combine ---
    X = pd.concat([df_train[num_cols], genres_df, df_encoded_cats], axis=1)
    
    # --- 5. Create Targets (ROBUST) ---
    y_reg = df_train['user_rating']
    
    # Calculate dynamic threshold to ensure we have both 0s and 1s
    user_median = df_train['user_rating'].median()
    print(f"   User Median Rating: {user_median}")
    
    # If median is high (e.g. 8), we split there. If median is low, we split there.
    # We use '>' strictly to try and force a split if many values equal the median
    y_class = (df_train['user_rating'] > user_median).astype(int)
    
    # FALLBACK: If strictly greater results in only one class (e.g. all ratings are identical), 
    # try >=. If that still fails, the model trainer will handle it.
    if len(y_class.unique()) < 2:
         y_class = (df_train['user_rating'] >= user_median).astype(int)

    class_counts = y_class.value_counts()
    print(f"   Class Balance: {class_counts.to_dict()} (0=Lower, 1=Higher)")

    # --- 6. Save ---
    train_output = X.copy()
    train_output['target_reg'] = y_reg
    train_output['target_class'] = y_class
    
    train_output.to_csv(config.TV_SHOWS_TRAINING_DATA_PATH, index=False)
    
    artifacts = {
        'features_columns': X.columns.tolist(),
        'mlb_genres': mlb_genres,
        'median_values': median_values,
        'cat_cols': cat_cols
    }
    
    config.TV_SHOWS_PREPROCESSOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.TV_SHOWS_PREPROCESSOR_STATE)
    
    print(f"✅ Features processed. Shape: {X.shape}")
    print(f"✅ Data saved to: {config.TV_SHOWS_TRAINING_DATA_PATH}")

if __name__ == "__main__":
    run_feature_engineering()