import pandas as pd
import numpy as np
import joblib
import ast
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

# --- UPDATED CONFIG (Stricter Thresholds) ---
MIN_DIRECTOR_COUNT = 4  # Increased to reduce columns
MIN_ACTOR_COUNT = 8     # Increased to reduce columns
PCA_COMPONENTS = 8      # Reduced dimensionality

def clean_percentage(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('%', '').strip()
    try: return float(x)
    except: return np.nan

def parse_list(x):
    if pd.isna(x): return []
    try:
        if '[' not in str(x): return [str(x)]
        return ast.literal_eval(str(x))
    except: return []

def process_features():
    print("🛠️ Starting Feature Engineering...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(config.ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv not found.")
        return

    # 2. Clean Target
    df = df.dropna(subset=['user_rating'])
    y_reg = df['user_rating']
    
    def classify(r):
        if r <= 2.5: return 0
        elif r < 4.0: return 1
        else: return 2
    y_class = df['user_rating'].apply(classify)

    # 3. Numeric Features
    if df['rotten_tomatoes_rating'].dtype == object:
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_percentage)
    
    num_cols = ['year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'popularity']
    for col in num_cols:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    X_num = df[num_cols].reset_index(drop=True)

    # 4. Categorical: Director
    df['director_list'] = df['director'].apply(parse_list)
    df['primary_director'] = df['director_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    dir_counts = df['primary_director'].value_counts()
    valid_dirs = dir_counts[dir_counts >= MIN_DIRECTOR_COUNT].index
    df['primary_director'] = df['primary_director'].apply(lambda x: x if x in valid_dirs else 'Other_Director')
    
    X_dir = pd.get_dummies(df['primary_director'], prefix='dir')

    # 5. Categorical: Actors
    df['actors_list'] = df['actors'].apply(parse_list)
    df['primary_actor'] = df['actors_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    act_counts = df['primary_actor'].value_counts()
    valid_actors = act_counts[act_counts >= MIN_ACTOR_COUNT].index
    df['primary_actor'] = df['primary_actor'].apply(lambda x: x if x in valid_actors else 'Other_Actor')
    
    X_act = pd.get_dummies(df['primary_actor'], prefix='act')

    # 6. Categorical: Genres
    df['genre_list'] = df['genre'].apply(parse_list)
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=mlb.classes_, index=df.index)
    X_genre.columns = [f"gen_{c}" for c in X_genre.columns]

    # 7. Text Features
    df['text_content'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_content'])
    
    pca = PCA(n_components=PCA_COMPONENTS)
    pca_features = pca.fit_transform(tfidf_matrix.toarray())
    X_text = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 8. Combine
    X_final = pd.concat([X_num, X_dir.reset_index(drop=True), X_act.reset_index(drop=True), X_genre.reset_index(drop=True), X_text.reset_index(drop=True)], axis=1)
    
    print(f"📊 Training Data Shape: {X_final.shape}")
    
    # 9. Save
    state = {
        'valid_directors': valid_dirs,
        'valid_actors': valid_actors,
        'mlb_genre': mlb,
        'tfidf': tfidf,
        'pca': pca,
        'training_columns': X_final.columns.tolist(),
        'median_values': df[num_cols].median().to_dict()
    }
    joblib.dump(state, config.PREPROCESSOR_STATE)
    
    X_final['target_reg'] = y_reg.reset_index(drop=True)
    X_final['target_class'] = y_class.reset_index(drop=True)
    
    X_final.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"✅ Features saved to {config.TRAINING_DATA_PATH}")

if __name__ == "__main__":
    process_features()