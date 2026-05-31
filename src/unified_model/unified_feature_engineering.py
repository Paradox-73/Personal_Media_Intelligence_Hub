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
        if '-' in s:
            parts = s.split('-')
            if len(parts[0]) == 4: return int(parts[0])
            if len(parts[-1]) == 4: return int(parts[-1])
        return int(float(s[:4]))
    except:
        return None

def transform_single_media(m, state, media_type='movie'):
    """
    Transforms raw metadata for a single item into the unified feature space.
    media_type: 'movie', 'tv', or 'game'
    """
    is_tv_show = 1 if media_type == 'tv' else 0
    is_game = 1 if media_type == 'game' else 0

    # 1. Numerical Features
    data = {
        'is_tv_show': is_tv_show,
        'is_game': is_game,
        'year': clean_year(m.get('year', m.get('released'))),
        'runtime': pd.to_numeric(m.get('runtime'), errors='coerce'),
        'imdb_rating': pd.to_numeric(m.get('imdb_rating', m.get('rating')), errors='coerce'),
        'metascore': pd.to_numeric(m.get('metascore', m.get('metacritic')), errors='coerce'),
        'rotten_tomatoes_rating': float(str(m.get('rotten_tomatoes_rating', '0')).replace('%', '') or 0),
        'vote_average': pd.to_numeric(m.get('vote_average'), errors='coerce'),
        'imdb_votes': pd.to_numeric(m.get('imdb_votes', m.get('vote_count', m.get('ratings_count'))), errors='coerce'),
        'box_office_log': np.log1p(clean_money(m.get('box_office'))),
        'popularity': pd.to_numeric(m.get('popularity'), errors='coerce'),
        'number_of_seasons': pd.to_numeric(m.get('number_of_seasons', 1), errors='coerce') if is_tv_show else 1,
        'number_of_episodes': pd.to_numeric(m.get('number_of_episodes', 1), errors='coerce') if is_tv_show else 1
    }
    
    # Awards
    awd = str(m.get('awards', ''))
    w = re.search(r'(\d+)\s+win', awd, re.I)
    n = re.search(r'(\d+)\s+nomination', awd, re.I)
    data['total_wins'] = float(w.group(1)) if w else 0.0
    data['total_nominations'] = float(n.group(1)) if n else 0.0
    
    # Critic Avg
    s = [data['rotten_tomatoes_rating'], data['metascore'], data['imdb_rating']*10, data['vote_average']*10]
    s = [v for v in s if v > 0]
    data['critic_avg_5'] = (sum(s)/len(s)/100*5) if s else 0.0

    # Fill NaNs with state medians
    for col, med in state['median_values'].items():
        if pd.isna(data.get(col)): data[col] = med

    df_num = pd.DataFrame([data])

    # 2. Language
    langs = [l.strip() for l in str(m.get('language', 'English')).split(',')]
    found = next((l for l in langs if l in state['top_languages']), 'Other')
    lang_data = {f'lang_{l}': (1.0 if l == found else 0.0) for l in state['top_languages'] + ['Other']}
    df_lang = pd.DataFrame([lang_data])

    # 3. Text
    title = m.get('title', m.get('name', 'Unknown'))
    lead = m.get('director', m.get('developers', 'Unknown'))
    plot = m.get('overview', m.get('plot', m.get('description_raw', '')))
    text = f"Title: {title}. Lead: {lead}. Plot: {plot}"
    
    transformer = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    pca_vec = state['pca'].transform(transformer.encode([text]))
    df_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(pca_vec.shape[1])])

    # 4. Genres
    genres = parse_list(m.get('genre', m.get('genres')))
    g_map = {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}
    mapped = []
    for g in genres: mapped.extend(g_map.get(g, [g]))
    gen_encoded = state['mlb_genre'].transform([list(set(mapped))])
    df_genre = pd.DataFrame(gen_encoded, columns=[f"gen_{c}" for c in state['mlb_genre'].classes_])

    # 5. MPAA
    cat = categorize_rating(m.get('rated', m.get('age_rating')))
    mpaa_data = {f'rated_{c}': (1.0 if c == cat else 0.0) for c in ['Adult', 'Teen', 'General']}
    df_mpaa = pd.DataFrame([mpaa_data])

    X = pd.concat([df_num, df_lang, df_genre, df_mpaa, df_text], axis=1)
    X.columns = [sanitize_col(c) for c in X.columns]
    X_final = pd.DataFrame(0.0, index=[0], columns=state['training_columns'])
    common = list(set(X.columns) & set(state['training_columns']))
    X_final[common] = X[common]
    return X_final

def build_universal_dataset():
    print("🌍 Building Universal Media Dataset (Movies + TV Shows + Games)...")
    
    # Load Movies
    df_m = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_m['is_tv_show'] = 0
    df_m['is_game'] = 0
    
    # Load TV Shows
    df_s = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_s['is_tv_show'] = 1
    df_s['is_game'] = 0
    df_s = df_s.rename(columns={
        'name': 'title', 
        'created_by': 'director', 
        'production_companies': 'production', 
        'genres': 'genre', 
        'age_rating': 'rated'
    })

    # Load Games
    try:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    
    # Align Game Schema
    df_g = df_g.rename(columns={
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
    df_g['is_tv_show'] = 0
    df_g['is_game'] = 1
    # Extract year from released
    df_g['year'] = df_g['released'].apply(clean_year)
    
    # Drop items with no rating
    df_g = df_g.dropna(subset=['user_rating'])

    # Combine all
    df = pd.concat([df_m, df_s, df_g], ignore_index=True)
    print(f"   Loaded {len(df_m)} movies, {len(df_s)} shows, and {len(df_g)} games. Total: {len(df)} records.")
    
    # Feature Engineering Logic (Numerical)
    df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'].astype(str).str.replace('%', ''), errors='coerce')
    df['box_office_log'] = np.log1p(df['box_office'].apply(clean_money))
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', re.I)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', re.I)[0].astype(float).fillna(0)
    df['number_of_seasons'] = pd.to_numeric(df.get('number_of_seasons', 1), errors='coerce').fillna(1)
    df['number_of_episodes'] = pd.to_numeric(df.get('number_of_episodes', 1), errors='coerce').fillna(1)
    
    critic = df[['imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average']].copy()
    critic['ir_100'] = critic['imdb_rating'] * 10
    # RAWG rating (imdb_rating here) is often 0-5, so maybe check if it's already on 0-5 scale
    # Movie/Show imdb_rating is 0-10. Game RAWG rating is 0-5.
    # Let's normalize game rating to 10 if it's <= 5? 
    # Actually RAWG rating is already 0-5, so multiplying by 20 makes it 0-100.
    # But let's be careful.
    mask_game = df['is_game'] == 1
    critic.loc[mask_game, 'ir_100'] = critic.loc[mask_game, 'imdb_rating'] * 20
    critic.loc[~mask_game, 'ir_100'] = critic.loc[~mask_game, 'imdb_rating'] * 10
    
    critic['va_100'] = critic['vote_average'] * 10
    df['critic_avg_5'] = (critic[['ir_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']].mean(axis=1) / 100 * 5)

    num_cols = ['is_tv_show', 'is_game', 'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'imdb_votes', 'box_office_log', 'popularity', 'total_wins', 'total_nominations', 'critic_avg_5', 'number_of_seasons', 'number_of_episodes']
    for c in num_cols:
        # Pre-clean strings by removing commas
        if df[c].dtype == 'object':
            df[c] = df[c].astype(str).str.replace(',', '', regex=False)
            
        numeric_series = pd.to_numeric(df[c], errors='coerce')
        median_val = numeric_series.median() if not numeric_series.isna().all() else 0
        df[c] = numeric_series.fillna(median_val)

    # NLP & Categorical
    top_3_langs = df['language'].astype(str).str.split(', ').explode().value_counts().nlargest(3).index.tolist()
    X_lang = pd.get_dummies(df['language'].apply(lambda x: next((l.strip() for l in str(x).split(',') if l.strip() in top_3_langs), 'Other')), prefix='lang')

    print("   Generating text embeddings...")
    df['txt'] = "Title: " + df['title'].fillna('') + ". Lead: " + df['director'].fillna('') + ". " + df['overview'].fillna('')
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    pca = PCA(n_components=10)
    X_text = pd.DataFrame(pca.fit_transform(transformer.encode(df['txt'].tolist())), columns=[f'pca_{i}' for i in range(10)])

    df['gen_list'] = df['genre'].apply(parse_list).apply(lambda gl: list(set([item for g in gl for item in {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}.get(g, [g])])))
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['gen_list']), columns=[f"gen_{c}" for c in mlb.classes_], index=df.index)
    X_mpaa = pd.get_dummies(df['rated'].apply(categorize_rating), prefix='rated')

    # Final Assembly
    X_final = pd.concat([df[num_cols], X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_final.columns = [sanitize_col(c) for c in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    # Target cleaning - ensure all are numeric
    df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce').fillna(0)
    
    X_final['target_reg'] = df['user_rating']
    X_final['target_ordinal'] = df['user_rating'].map({0.5:0, 1.0:1, 1.5:2, 2.0:3, 2.5:4, 3.0:5, 3.5:6, 4.0:7, 4.5:8, 5.0:9}).fillna(5).astype(int)
    X_final['source_id'] = df.get('tmdb_id', df.index)
    X_final['is_tv_flag'] = df['is_tv_show']
    X_final['is_game_flag'] = df['is_game']

    X_final.to_csv(config.UNIFIED_TRAINING_DATA_PATH, index=False)
    state = {
        'top_languages': top_3_langs, 
        'mlb_genre': mlb, 
        'sentence_transformer': 'all-MiniLM-L6-v2', 
        'pca': pca, 
        'median_values': df[num_cols].median().to_dict(), 
        'training_columns': [c for c in X_final.columns if c not in ['target_reg', 'target_ordinal', 'source_id', 'is_tv_flag', 'is_game_flag']]
    }
    joblib.dump(state, config.UNIFIED_PREPROCESSOR_STATE)
    print(f"✅ Unified Feature Engineering Complete (Including Games). Shape: {X_final.shape}")

if __name__ == "__main__":
    build_universal_dataset()
