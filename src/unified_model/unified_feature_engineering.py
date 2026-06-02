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
from src import config

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
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r or 'MATURE' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r or 'TEEN' in r: return 'Teen'
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

def build_universal_dataset():
    print("🌍 Building Universal Media Dataset (Movies + TV Shows + Games + Books + Music)...")
    
    # 1. Load Movies
    df_m = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_m['media_type'] = 'movie'
    
    # 2. Load TV Shows
    df_s = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_s['media_type'] = 'tv'
    df_s = df_s.rename(columns={
        'name': 'title', 
        'created_by': 'director', 
        'genres': 'genre', 
        'age_rating': 'rated'
    })

    # 3. Load Games
    try:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    
    df_g = df_g.rename(columns={
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
    df_g['media_type'] = 'game'
    df_g['year'] = df_g['released'].apply(clean_year)
    df_g = df_g.dropna(subset=['user_rating'])

    # 4. Load Books
    try:
        df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except:
        df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    
    df_b = df_b.rename(columns={
        'my_rating': 'user_rating',
        'authors': 'director',
        'categories': 'genre',
        'averageRating': 'imdb_rating',
        'ratingsCount': 'imdb_votes',
        'description': 'overview',
        'publishedDate': 'released',
        'pageCount': 'runtime' # Map pages to runtime as a volume proxy
    })
    df_b['media_type'] = 'book'
    df_b['year'] = df_b['released'].apply(clean_year)
    df_b = df_b.dropna(subset=['user_rating'])

    # 5. Load Music
    try:
        df_mu = pd.read_csv(config.MUSIC_ENRICHED_DATA_PATH)
        df_mu = df_mu.rename(columns={
            'name': 'title',
            'primary_artist': 'director',
            'artist_genres': 'genre',
            'rating': 'user_rating',
            'release_year': 'year',
            'popularity': 'imdb_rating', # Use popularity as a proxy for crowd score
        })
        df_mu['media_type'] = 'music'
        df_mu['runtime'] = df_mu['duration_ms'].fillna(0) / 60000 # ms to min
        # For overview, combine MB tags if available
        df_mu['overview'] = df_mu['mb_tags'].fillna('') + " " + df_mu['artist_genres'].fillna('')
        df_mu = df_mu.dropna(subset=['user_rating'])
    except Exception as e:
        print(f"   ⚠️ Could not load Music data: {e}")
        df_mu = pd.DataFrame()

    # Combine all
    dfs = [df_m, df_s, df_g, df_b]
    if not df_mu.empty:
        dfs.append(df_mu)
        
    df = pd.concat(dfs, ignore_index=True)
    print(f"   Counts -> Movies: {len(df_m)}, Shows: {len(df_s)}, Games: {len(df_g)}, Books: {len(df_b)}, Music: {len(df_mu)}")
    print(f"   Total: {len(df)} records.")
    
    # Feature Engineering
    df['is_tv_show'] = (df['media_type'] == 'tv').astype(int)
    df['is_game'] = (df['media_type'] == 'game').astype(int)
    df['is_book'] = (df['media_type'] == 'book').astype(int)
    df['is_music'] = (df['media_type'] == 'music').astype(int)

    df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'].astype(str).str.replace('%', ''), errors='coerce')
    df['box_office_log'] = np.log1p(df['box_office'].apply(clean_money))
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', re.I)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', re.I)[0].astype(float).fillna(0)
    
    # Numeric Normalization
    critic = df[['imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average']].copy()
    
    # IMDb (0-10), Metascore (0-100), RT (0-100)
    # Games/Books imdb_rating (0-5)
    mask_short_scale = (df['is_game'] == 1) | (df['is_book'] == 1)
    critic['ir_100'] = critic['imdb_rating'] * 10
    critic.loc[mask_short_scale, 'ir_100'] = critic.loc[mask_short_scale, 'imdb_rating'] * 20
    
    critic['va_100'] = critic['vote_average'] * 10
    df['critic_avg_5'] = (critic[['ir_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']].mean(axis=1) / 100 * 5)

    num_cols = ['is_tv_show', 'is_game', 'is_book', 'is_music', 'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'imdb_votes', 'box_office_log', 'popularity', 'total_wins', 'total_nominations', 'critic_avg_5']
    
    for c in num_cols:
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str).str.replace(',', '', regex=False)
        numeric_series = pd.to_numeric(df[c], errors='coerce')
        median_val = numeric_series.median() if not numeric_series.isna().all() else 0
        df[c] = numeric_series.fillna(median_val)

    # NLP
    top_3_langs = df['language'].astype(str).str.split(', ').explode().value_counts().nlargest(3).index.tolist()
    X_lang = pd.get_dummies(df['language'].apply(lambda x: next((l.strip() for l in str(x).split(',') if l.strip() in top_3_langs), 'Other')), prefix='lang')

    print("   Generating text embeddings...")
    df['txt'] = "Title: " + df['title'].fillna('') + ". Lead: " + df['director'].fillna('') + ". " + df['overview'].fillna('')
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    pca = PCA(n_components=10)
    X_text = pd.DataFrame(pca.fit_transform(transformer.encode(df['txt'].tolist())), columns=[f'pca_{i}' for i in range(10)])

    # Genres
    df['gen_list'] = df['genre'].apply(parse_list).apply(lambda gl: list(set([item for g in gl for item in {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}.get(g, [g])])))
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['gen_list']), columns=[f"gen_{sanitize_col(c)}" for c in mlb.classes_], index=df.index)
    X_mpaa = pd.get_dummies(df['rated'].apply(categorize_rating), prefix='rated')

    # Final
    X_final = pd.concat([df[num_cols], X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_final.columns = [sanitize_col(c) for c in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce').fillna(0)
    X_final['target_reg'] = df['user_rating']
    X_final['target_ordinal'] = df['user_rating'].map({0.5:0, 1.0:1, 1.5:2, 2.0:3, 2.5:4, 3.0:5, 3.5:6, 4.0:7, 4.5:8, 5.0:9}).fillna(5).astype(int)
    X_final['source_id'] = df.index
    X_final['media_type'] = df['media_type']

    X_final.to_csv(config.UNIFIED_TRAINING_DATA_PATH, index=False)
    state = {
        'top_languages': top_3_langs, 
        'mlb_genre': mlb, 
        'sentence_transformer': 'all-MiniLM-L6-v2', 
        'pca': pca, 
        'median_values': df[num_cols].median().to_dict(), 
        'training_columns': [c for c in X_final.columns if c not in ['target_reg', 'target_ordinal', 'source_id', 'media_type']]
    }
    joblib.dump(state, config.UNIFIED_PREPROCESSOR_STATE)
    print(f"✅ Unified Feature Engineering Complete (Movies+Shows+Games+Books). Shape: {X_final.shape}")

if __name__ == "__main__":
    build_universal_dataset()
