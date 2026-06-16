import pandas as pd
import numpy as np
import ast
import joblib
import sys
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

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
    print("⚙️ STARTING TV SHOW FEATURE ENGINEERING (MiniLM Upgrade)...")

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

    # --- 4. Upgraded Text Representation (Sentence Transformers) ---
    print("   Generating MiniLM embeddings for TV shows...")
    df['text_content'] = "Title: " + df['name'].fillna('Unknown') + \
                         ". Network: " + df['primary_network'].fillna('Unknown') + \
                         ". " + df['overview'].fillna('')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)

    print("   Applying PCA (15 components) to embeddings...")
    pca = PCA(n_components=15)
    X_text_pca = pca.fit_transform(text_embeddings)
    pca_cols = [f'pca_{i}' for i in range(15)]
    text_df = pd.DataFrame(X_text_pca, columns=pca_cols, index=df.index)

    # --- 5. Combine ---
    feature_cols = ['year', 'vote_average', 'vote_count_log', 'season_count', 'is_adult']
    X = pd.concat([df[feature_cols], genre_df, network_dummies, text_df], axis=1)
    
    # --- 6. Targets & Save Data ---
    y_reg = df['user_rating']
    
    # Classification Target (for legacy compatibility)
    user_median = df['user_rating'].median()
    y_class = (df['user_rating'] >= user_median).astype(int)

    # NEW: Ordinal target mapping (10 buckets: 0 to 9)
    mapping = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}
    y_ord = df['user_rating'].map(mapping).fillna(5).astype(int)

    output = X.copy()
    output['target_reg'] = y_reg
    output['target_class'] = y_class
    output['target_ordinal'] = y_ord
    
    output.to_csv(config.TV_SHOWS_TRAINING_DATA_PATH, index=False)
    
    # --- 7. Save Artifacts ---
    artifacts = {
        'features_columns': X.columns.tolist(),
        'median_values': {
            'year': median_year,
            'vote_average': median_vote,
            'vote_count_log': df['vote_count_log'].median(),
            'season_count': df['season_count'].median()
        },
        'top_networks': top_networks,
        'mlb_genres': mlb,
        'kept_genres': kept_genres,
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'pca': pca,
        'pca_cols': pca_cols
    }
    
    config.TV_SHOWS_PREPROCESSOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.TV_SHOWS_PREPROCESSOR_STATE)
    
    print(f"✅ Features processed & Artifacts saved.")
    print(f"   Shape: {X.shape}")
    print(f"   Artifacts path: {config.TV_SHOWS_PREPROCESSOR_STATE}")

if __name__ == "__main__":
    run_feature_engineering()

if __name__ == "__main__":
    run_feature_engineering()