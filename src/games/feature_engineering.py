import pandas as pd
import numpy as np
import joblib
import ast
import sys
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# --- CONFIG ---
PCA_COMPONENTS = 15 # Reduced for small dataset

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return 0
        s = str(val).strip()
        if '-' in s:
            parts = s.split('-')
            if len(parts[0]) == 4: return int(parts[0])
            if len(parts[-1]) == 4: return int(parts[-1])
        return int(float(s[:4]))
    except:
        return 0

def parse_list(x):
    if pd.isna(x) or x == "": return []
    try:
        # Check if it looks like a list
        if isinstance(x, list): return x
        s = str(x).strip()
        if s.startswith('[') and s.endswith(']'):
            return ast.literal_eval(s)
        # Otherwise split by comma
        return [item.strip() for item in s.split(',')]
    except:
        return [str(x)]

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

def process_features():
    print("🛠️ Starting Game Feature Engineering...")

    if not config.GAMES_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.GAMES_ENRICHED_DATA_PATH}")
        return

    # Use latin1 for robustness if needed
    try:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    
    # 1. Clean my_rating
    # Map 'I' to NaN (Incomplete)
    df['my_rating_clean'] = pd.to_numeric(df['my_rating'], errors='coerce')
    
    # Track status
    df['status'] = 'Rated'
    df.loc[df['my_rating'] == 'I', 'status'] = 'Incomplete'
    df.loc[df['my_rating'].isna(), 'status'] = 'Unrated'
    
    print(f"   Ratings Breakdown -> Rated: {(df['status']=='Rated').sum()}, Incomplete: {(df['status']=='Incomplete').sum()}, Unrated: {(df['status']=='Unrated').sum()}")
    
    # 2. Extract Year
    df['year'] = df['released'].apply(clean_year)
    
    # 3. Numeric Features
    num_cols = ['year', 'metacritic', 'rating', 'ratings_count', 'reviews_count']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 4. Platform One-Hot
    X_platform = pd.get_dummies(df['platform_from_text'], prefix='plat')
    
    # 5. Genres Multi-Hot
    df['genre_list'] = df['genres'].apply(parse_list)
    mlb_genre = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb_genre.fit_transform(df['genre_list']), 
                           columns=[f"gen_{sanitize_col(c)}" for c in mlb_genre.classes_], 
                           index=df.index)
    
    # 6. Developers Multi-Hot (Top ones to avoid explosion)
    df['dev_list'] = df['developers'].apply(parse_list)
    mlb_dev = MultiLabelBinarizer()
    dev_encoded = mlb_dev.fit_transform(df['dev_list'])
    dev_counts = dev_encoded.sum(axis=0)
    valid_dev_indices = np.where(dev_counts >= 2)[0]
    X_dev = pd.DataFrame(dev_encoded[:, valid_dev_indices], 
                         columns=[f"dev_{sanitize_col(mlb_dev.classes_[i])}" for i in valid_dev_indices], 
                         index=df.index)

    # 7. Text Embeddings
    print("   Generating text embeddings...")
    df['text_content'] = "Name: " + df['name'].fillna('') + \
                         ". Genres: " + df['genres'].fillna('') + \
                         ". Tags: " + df['tags'].fillna('') + \
                         ". Description: " + df['description_raw'].fillna('')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    
    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=min(PCA_COMPONENTS, len(df)-1))
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # 8. Combine
    X_final = pd.concat([df[num_cols], X_platform, X_genre, X_dev, X_text], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    # Targets
    X_final['target_reg'] = df['my_rating_clean'].values
    # Also categorical for dashboard if needed (Low: 1-2, Med: 3-4, High: 5)
    X_final['target_class'] = pd.cut(df['my_rating_clean'], bins=[-1, 2.5, 4.5, 5.1], labels=[0, 1, 2])

    # Save
    X_final.to_csv(config.GAMES_TRAINING_DATA_PATH, index=False)
    
    state = {
        'mlb_genre': mlb_genre,
        'mlb_dev': mlb_dev,
        'valid_dev_indices': valid_dev_indices,
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': X_final.drop(columns=['target_reg', 'target_class']).columns.tolist(),
        'platforms': X_platform.columns.tolist(),
        'genres': X_genre.columns.tolist()
    }
    joblib.dump(state, config.GAMES_MODEL_PREPROCESSOR_STATE)
    print(f"✅ Features processed. Shape: {X_final.shape}")

if __name__ == "__main__":
    process_features()
