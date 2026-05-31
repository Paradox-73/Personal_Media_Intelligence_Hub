import pandas as pd
import numpy as np
import ast
import joblib
import sys
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
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

def get_primary_network(val):
    if pd.isna(val) or val == "": return "Other"
    try:
        if "[" in str(val):
            val_list = ast.literal_eval(str(val))
            if val_list and len(val_list) > 0:
                return val_list[0].strip()
        return str(val).split(',')[0].strip()
    except:
        return "Other"

def run_feature_engineering():
    print("⚙️ STARTING TV SHOW FEATURE ENGINEERING (Robust Mode)...")

    if not config.TV_SHOWS_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.TV_SHOWS_ENRICHED_DATA_PATH}")
        return

    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    
    # Filter for Training (Must have a user rating)
    df = df[df['user_rating'].notna()].copy()
    print(f"   Input Data: {len(df)} rated shows.")

    # --- 1. Numericals ---
    df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
    
    # Calculate medians BEFORE filling NaNs (Critical for Model State)
    median_vote = df['vote_average'].median()
    if pd.isna(median_vote): median_vote = 7.0 # Fallback
    
    median_year = df['year'].median()
    if pd.isna(median_year): median_year = 2010 # Fallback

    # Fill NaNs
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(median_vote)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(median_year)
    df['season_count'] = pd.to_numeric(df['number_of_seasons'], errors='coerce').fillna(1)
    
    # --- 2. Vibe Features (Network) ---
    df['primary_network'] = df['network'].apply(get_primary_network)
    
    # Determine top networks
    top_networks = df['primary_network'].value_counts().nlargest(10).index.tolist()
    
    df['network_clean'] = df['primary_network'].apply(lambda x: x if x in top_networks else "Other")
    network_dummies = pd.get_dummies(df['network_clean'], prefix='net')

    # Adult Content
    df['is_adult'] = df['age_rating'].apply(lambda x: 1 if str(x) in ['TV-MA', 'R', '18+'] else 0)

    # --- 3. Genres ---
    df['genres_list'] = df['genres'].apply(safe_eval)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres_list'])
    
    # Create temp dataframe for frequency calculation
    temp_genre_df = pd.DataFrame(genres_encoded, columns=[f"g_{c}" for c in mlb.classes_], index=df.index)
    
    # Remove rare genres to reduce noise
    min_freq = 0.05 * len(df)
    kept_genres = temp_genre_df.columns[temp_genre_df.sum() >= min_freq].tolist()
    genre_df = temp_genre_df[kept_genres]
    
    print(f"   Selected {len(kept_genres)} common genres.")

    # --- 4. NLP ---
    print("   Processing NLP features...")
    df['overview'] = df['overview'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', max_features=30, min_df=2)
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    tfidf_cols = [f"txt_{word}" for word in tfidf.get_feature_names_out()]
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_cols, index=df.index)

    # --- 5. Combine ---
    # Ensure numerical columns are clearly selected
    feature_cols = ['year', 'vote_average', 'vote_count_log', 'season_count', 'is_adult']
    
    X = pd.concat([df[feature_cols], genre_df, network_dummies, tfidf_df], axis=1)
    
    # --- 6. Targets & Save Data ---
    y_reg = df['user_rating']
    
    # Classification Target (for potential future use)
    user_median = df['user_rating'].median()
    y_class = (df['user_rating'] > user_median).astype(int)
    if y_class.value_counts().min() < 5: 
        y_class = (df['user_rating'] >= user_median).astype(int)

    # NEW: Ordinal target mapping (10 buckets: 0 to 9)
    mapping = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}
    y_ord = df['user_rating'].map(mapping).fillna(5).astype(int)

    output = X.copy()
    output['target_reg'] = y_reg
    output['target_class'] = y_class
    output['target_ordinal'] = y_ord
    
    output.to_csv(config.TV_SHOWS_TRAINING_DATA_PATH, index=False)
    
    # --- 7. Save Artifacts (CRITICAL FIX) ---
    artifacts = {
        # Feature Columns (Required for prediction alignment)
        'features_columns': X.columns.tolist(),
        
        # --- DUAL COMPATIBILITY MODE ---
        # 1. Flat keys (for predict_ratings.py)
        'median_vote_avg': median_vote,
        'median_year': median_year,
        
        # 2. Nested keys (for Streamlit Oracle App)
        'median_values': {
            'year': median_year,
            'vote_average': median_vote,
        },
        
        # Encoders & Lists
        'top_networks': top_networks,
        'mlb_genres': mlb,              # For Oracle App input transformation
        'kept_genres': kept_genres,     # For predict_ratings.py reconstruction
        'tfidf_model': tfidf,
        'tfidf_cols': tfidf_cols
    }
    
    config.TV_SHOWS_PREPROCESSOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.TV_SHOWS_PREPROCESSOR_STATE)
    
    print(f"✅ Features processed & Artifacts saved.")
    print(f"   Shape: {X.shape}")
    print(f"   Artifacts path: {config.TV_SHOWS_PREPROCESSOR_STATE}")

if __name__ == "__main__":
    run_feature_engineering()