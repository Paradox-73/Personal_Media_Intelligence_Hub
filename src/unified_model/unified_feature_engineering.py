import pandas as pd
import numpy as np
import joblib
import ast
import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config # Ensure your config has UNIVERSAL paths defined

# --- CONFIG ---
PCA_COMPONENTS = 25

# --- HELPERS ---
def clean_money(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return 0

def parse_list(x):
    if isinstance(x, (list, np.ndarray)): return x if isinstance(x, list) else x.tolist()
    if pd.isna(x): return []
    s = str(x).strip()
    if not s: return []
    try:
        if s.startswith('[') and s.endswith(']'): return ast.literal_eval(s)
        return [item.strip() for item in s.split(',')]
    except: return [s]

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

# --- MAIN UNIFICATION FUNCTION ---
def build_universal_dataset():
    print("🌍 Building Universal Media Dataset (Movies + TV Shows)...")

    # 1. Load Data
    df_movies = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_shows = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])

    # 2. Align Schemas & Add the Golden Flag
    df_movies['is_tv_show'] = 0
    df_shows['is_tv_show'] = 1

    # Map Show columns to Movie equivalents where naming differs
    df_shows = df_shows.rename(columns={
        'name': 'title',
        'created_by': 'director', # Mapping Creator to Director as Lead Creative
        'production_companies': 'production',
        'genres': 'genre',
        'age_rating': 'rated'
    })

    # Combine them
    df = pd.concat([df_movies, df_shows], ignore_index=True)
    print(f"   Combined Dataset: {len(df_movies)} Movies + {len(df_shows)} Shows = {len(df)} total records.")

    # 3. Clean and Engineer Numerical Features
    df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].astype(str).str.replace('%', '', regex=False)
    df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'], errors='coerce')
    
    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])

    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)

    # TV Specific logic (fill missing for movies)
    df['number_of_seasons'] = pd.to_numeric(df.get('number_of_seasons', 1), errors='coerce').fillna(1)
    df['number_of_episodes'] = pd.to_numeric(df.get('number_of_episodes', 1), errors='coerce').fillna(1)

    # Critic averages
    df['imdb_rating_100'] = pd.to_numeric(df['imdb_rating'], errors='coerce') * 10
    df['vote_average_100'] = pd.to_numeric(df['vote_average'], errors='coerce') * 10
    df['metascore'] = pd.to_numeric(df['metascore'], errors='coerce')
    critic_scores = df[['imdb_rating_100', 'metascore', 'rotten_tomatoes_rating', 'vote_average_100']]
    df['critic_avg_100'] = critic_scores.mean(axis=1)
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5

    # Define Universal Numerical Columns
    num_cols = [
        'is_tv_show', 'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating',
        'vote_average', 'imdb_votes', 'box_office_log', 'popularity',
        'total_wins', 'total_nominations', 'critic_avg_5', 'number_of_seasons', 'number_of_episodes'
    ]

    # Fill NaNs with medians calculated across the UNIVERSAL dataset
    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        df[col] = numeric_series.fillna(numeric_series.median())

    X_num = df[num_cols].reset_index(drop=True)

    # 4. Language One-Hot Encoding
    top_3_languages = df['language'].astype(str).str.split(', ').explode().value_counts().nlargest(3).index.tolist()
    def map_language(lang_str):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in str(lang_str).split(',')]
        for lang in langs:
            if lang in top_3_languages: return lang
        return 'Other'
    
    X_lang = pd.get_dummies(df['language'].apply(map_language), prefix='lang')

    # 5. Text Embeddings
    print("   Generating Universal text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('Unknown') + \
                         ". Lead Creative: " + df['director'].fillna('Unknown') + \
                         ". Starring: " + df['actors'].fillna('Unknown') + \
                         ". Written by: " + df['writer'].fillna('Unknown') + \
                         ". Produced by: " + df['production'].fillna('Unknown') + \
                         ". " + df['tagline'].fillna('') + \
                         " " + df['overview'].fillna('')
                         
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer.encode(df['text_content'].tolist(), show_progress_bar=True)

    pca = PCA(n_components=PCA_COMPONENTS)
    X_text = pd.DataFrame(pca.fit_transform(text_embeddings), columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 6. Genres (Unified mapping)
    df['genre_list'] = df['genre'].apply(parse_list)
    
    # Optional: Map TV genres to Movie ones if desired, or let the MLB handle all unique strings
    genre_mapping = {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}
    def unify_genres(g_list):
        mapped = []
        for g in g_list:
            mapped.extend(genre_mapping.get(g, [g]))
        return list(set(mapped))
        
    df['genre_list'] = df['genre_list'].apply(unify_genres)
    
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=[f"gen_{c}" for c in mlb.classes_], index=df.index)

    # 7. MPAA / TV Rating
    X_mpaa = pd.get_dummies(df['rated'].apply(categorize_rating), prefix='rated')

    # 8. Combine & Save
    X_final = pd.concat([X_num, X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    X_final['target_reg'] = df['user_rating'].reset_index(drop=True)
    X_final['source_id'] = df['tmdb_id'].reset_index(drop=True) # Keep ID for debugging

    # Save to a new Unified directory
    # IMPORTANT: Ensure UNIFIED_TRAINING_DATA_PATH and UNIFIED_PREPROCESSOR_STATE are in your config.py
    X_final.to_csv(config.UNIFIED_TRAINING_DATA_PATH, index=False)
    
    state = {
        'top_languages': top_3_languages,
        'mlb_genre': mlb,
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': [c for c in X_final.columns if c not in ['target_reg', 'source_id']]
    }
    joblib.dump(state, config.UNIFIED_PREPROCESSOR_STATE)
    print(f"✅ Unified Feature Engineering Complete. Shape: {X_final.shape}")

if __name__ == "__main__":
    build_universal_dataset()