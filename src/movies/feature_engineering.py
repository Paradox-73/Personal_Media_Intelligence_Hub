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
from sklearn.metrics.pairwise import cosine_similarity 

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

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
    if isinstance(x, (list, np.ndarray)):
        return x if isinstance(x, list) else x.tolist()
    if pd.isna(x): return []
    s = str(x).strip()
    if not s: return []
    try:
        if s.startswith('[') and s.endswith(']'):
            return ast.literal_eval(s)
        return [item.strip() for item in s.split(',')]
    except (ValueError, SyntaxError):
        return [s]

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

# --- MAIN TRAINING FUNCTION ---
def process_features():
    print("🛠️ Starting Movie Feature Engineering (v2)...")

    if not config.MOVIES_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.MOVIES_ENRICHED_DATA_PATH}")
        return

    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    df = df.dropna(subset=['user_rating']).copy()
    print(f"   Input Data: {len(df)} rated movies.")

    # 1. Drop unused columns
    df = df.drop(columns=['type', 'website'], errors='ignore')

    # 2. Clean and Engineer Numerical Features
    if df['rotten_tomatoes_rating'].dtype == 'object':
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].str.replace('%', '', regex=False).astype(float)

    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])

    df['total_wins'] = df['awards'].str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)

    # Engineer Critic vs. User Rating Feature
    df['imdb_rating_100'] = df['imdb_rating'] * 10
    df['vote_average_100'] = df['vote_average'] * 10
    critic_scores = df[['imdb_rating_100', 'metascore', 'rotten_tomatoes_rating', 'vote_average_100']]
    df['critic_avg_100'] = critic_scores.mean(axis=1)
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5

    if 'vote_count' in df.columns and 'imdb_votes' not in df.columns:
        df.rename(columns={'vote_count': 'imdb_votes'}, inplace=True)

    num_cols = [
        'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating',
        'vote_average', 'imdb_votes', 'box_office_log', 'popularity',
        'total_wins', 'total_nominations', 'critic_avg_5'
    ]

    for col in num_cols:
        if col not in df.columns:
            df[col] = 0
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        median_val = numeric_series.median()
        df[col] = numeric_series.fillna(median_val)

    X_num = df[num_cols].reset_index(drop=True)

    # 3. Language One-Hot Encoding
    top_3_languages = df['language'].str.split(', ').explode().value_counts().nlargest(3).index.tolist()
    def map_language(lang_str):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in lang_str.split(',')]
        for lang in langs:
            if lang in top_3_languages:
                return lang
        return 'Other'
    df['language_cleaned'] = df['language'].apply(map_language)
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')

    # 4. Upgraded Text Representation (Sentence Transformers)
    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('Unknown') + \
                         ". Directed by: " + df['director'].fillna('Unknown') + \
                         ". " + df['plot'].fillna('')
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)

    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 5. Genres
    df['genre_list'] = df['genre'].apply(parse_list)
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=[f"gen_{c}" for c in mlb.classes_], index=df.index)

    # 5.1 Target Encoding for Directors/Creators (Leakage-Safe)
    print("   Applying Leakage-Safe Target Encoding...")
    from sklearn.model_selection import KFold
    
    def target_encode(train_df, col, target='user_rating', m=10):
        # Bayesian smoothing: (n * mean + m * global_mean) / (n + m)
        global_mean = train_df[target].mean()
        
        # We must compute this strictly out-of-fold to avoid leakage
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        encoded = pd.Series(index=train_df.index, dtype=float)
        
        # Flatten creators if it's a list (for directors/actors)
        temp_df = train_df[[col, target]].copy()
        if isinstance(temp_df[col].iloc[0], (list, str)) and '[' in str(temp_df[col].iloc[0]):
             temp_df[col] = temp_df[col].apply(parse_list)
        elif isinstance(temp_df[col].iloc[0], str):
             temp_df[col] = temp_df[col].str.split(', ')

        # Function to get mean for a set of items (e.g. multiple directors)
        def get_item_mean(items, mapping, g_mean):
            if not items: return g_mean
            vals = [mapping.get(i, g_mean) for i in items]
            return np.mean(vals)

        for train_idx, val_idx in kf.split(temp_df):
            # Compute mapping on training folds
            fold_train = temp_df.iloc[train_idx].explode(col)
            stats = fold_train.groupby(col)[target].agg(['count', 'mean'])
            mapping = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
            mapping = mapping.to_dict()
            
            # Apply to validation fold
            encoded.iloc[val_idx] = temp_df.iloc[val_idx][col].apply(lambda x: get_item_mean(x, mapping, global_mean))
        
        return encoded.fillna(global_mean)

    df['director_encoded'] = target_encode(df, 'director')
    df['actors_encoded'] = target_encode(df, 'actors')
    
    # 5.2 Director x Genre Interaction
    # Simplification: Top genre x Director encoding
    df['primary_genre'] = df['genre_list'].apply(lambda x: x[0] if x else 'Unknown')
    df['dir_genre_key'] = df['director'].astype(str) + "_" + df['primary_genre']
    df['dir_genre_encoded'] = target_encode(df, 'dir_genre_key')

    X_te = df[['director_encoded', 'actors_encoded', 'dir_genre_encoded']].reset_index(drop=True)

    # 6. MPAA Rating
    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # 7. Combine Final Features
    print("   Combining all features...")
    X_final = pd.concat([X_num, X_lang, X_genre, X_te, X_mpaa, X_text], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    # 8. Define Targets
    X_final['target_reg'] = df['user_rating'].reset_index(drop=True)
    X_final['target_class'] = pd.cut(
        df['user_rating'],
        bins=[0, 2.5, 3.5, 5],
        labels=[0, 1, 2],
        right=True
    ).astype(int).reset_index(drop=True)
    
    # NEW: Ordinal target mapping (10 buckets: 0 to 9)
    mapping = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}
    X_final['target_ordinal'] = df['user_rating'].map(mapping).fillna(5).astype(int).reset_index(drop=True)

    # 9. Save Data and Preprocessor State
    # Compute global mappings for inference
    def get_global_mapping(train_df, col, target='user_rating', m=10):
        global_mean = train_df[target].mean()
        temp_df = train_df[[col, target]].copy().explode(col)
        stats = temp_df.groupby(col)[target].agg(['count', 'mean'])
        mapping = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        return mapping.to_dict(), global_mean

    dir_map, dir_mean = get_global_mapping(df, 'director')
    act_map, act_mean = get_global_mapping(df, 'actors')
    dg_map, dg_mean = get_global_mapping(df, 'dir_genre_key')

    X_final.to_csv(config.TRAINING_DATA_PATH, index=False)
    state = {
        'top_languages': top_3_languages,
        'mlb_genre': mlb,
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': [c for c in X_final.columns if 'target' not in c],
        'target_encodings': {
            'director': {'map': dir_map, 'mean': dir_mean},
            'actors': {'map': act_map, 'mean': act_mean},
            'dir_genre': {'map': dg_map, 'mean': dg_mean}
        }
    }
    joblib.dump(state, config.PREPROCESSOR_STATE)
    print(f"✅ Features processed. Shape: {X_final.shape}")

# --- INFERENCE FUNCTIONS (FOR APP) ---
def transform_single_movie(movie_data, state):
    # Initialize with medians
    row = {col: state['median_values'].get(col, 0) for col in state['median_values']}

    def safe_num(val, default_val):
        if pd.isna(val) or str(val).lower() in ['nan', 'n/a', 'none', '']: return default_val
        try: return float(str(val).replace(',', '').replace('%', '').strip())
        except: return default_val

    # Mapping movie_data keys to feature columns
    row['year'] = safe_num(movie_data.get('year'), row.get('year'))
    row['runtime'] = safe_num(movie_data.get('runtime'), row.get('runtime'))
    row['imdb_rating'] = safe_num(movie_data.get('imdb_rating'), row.get('imdb_rating'))
    row['metascore'] = safe_num(movie_data.get('metascore'), row.get('metascore'))
    row['rotten_tomatoes_rating'] = safe_num(movie_data.get('rotten_tomatoes_rating'), row.get('rotten_tomatoes_rating'))
    row['vote_average'] = safe_num(movie_data.get('vote_average'), row.get('vote_average'))
    row['popularity'] = safe_num(movie_data.get('popularity'), row.get('popularity'))
    
    # Handle the vote_count -> imdb_votes mapping
    votes = safe_num(movie_data.get('imdb_votes'), None)
    if votes is None:
        votes = safe_num(movie_data.get('vote_count'), row.get('imdb_votes'))
    row['imdb_votes'] = votes

    bo = clean_money(movie_data.get('box_office', 0))
    row['box_office_log'] = np.log1p(bo)

    awards_str = str(movie_data.get('awards', ''))
    wins_match = re.search(r'(\d+)\s+win', awards_str, re.IGNORECASE)
    row['total_wins'] = float(wins_match.group(1)) if wins_match else 0
    noms_match = re.search(r'(\d+)\s+nomination', awards_str, re.IGNORECASE)
    row['total_nominations'] = float(noms_match.group(1)) if noms_match else 0
    
    # Engineer Critic vs. User Rating Feature (Matches predict_ratings.py)
    imdb_val = safe_num(movie_data.get('imdb_rating'), np.nan)
    imdb_100 = imdb_val * 10 if not pd.isna(imdb_val) else np.nan
    
    meta = safe_num(movie_data.get('metascore'), np.nan)
    rt = safe_num(movie_data.get('rotten_tomatoes_rating'), np.nan)
    
    va_val = safe_num(movie_data.get('vote_average'), np.nan)
    va = va_val * 10 if not pd.isna(va_val) else np.nan
    
    critic_scores = pd.Series([imdb_100, meta, rt, va])
    avg_100 = critic_scores.mean() # mean() ignores NaN by default
    row['critic_avg_5'] = (avg_100 / 100) * 5 if not pd.isna(avg_100) else 0.0
    
    # Initialize DF with all training columns
    final_df = pd.DataFrame(0.0, index=[0], columns=state['training_columns'])
    for col in row:
        if col in final_df.columns:
            final_df[col] = float(row[col])

    def map_language_single(lang_str, top_langs):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in str(lang_str).split(',')]
        for lang in langs:
            if lang in top_langs:
                return lang
        return 'Other'

    lang = map_language_single(movie_data.get('language'), state['top_languages'])
    lang_col = sanitize_col(f'lang_{lang}')
    if lang_col in final_df.columns: final_df[lang_col] = 1.0

    genres = parse_list(movie_data.get('genre', []))
    gen_vec = state['mlb_genre'].transform([genres])
    for i, gen_name in enumerate(state['mlb_genre'].classes_):
        col_name = sanitize_col(f"gen_{gen_name}")
        if col_name in final_df.columns: final_df[col_name] = float(gen_vec[0, i])

    mpaa = categorize_rating(movie_data.get('rated', ''))
    mpaa_col = sanitize_col(f"rated_{mpaa}")
    if mpaa_col in final_df.columns: final_df[mpaa_col] = 1.0

    # Apply Target Encodings
    te = state.get('target_encodings', {})
    
    def apply_te(val, te_meta):
        if not te_meta: return 0.0
        mapping, g_mean = te_meta['map'], te_meta['mean']
        items = parse_list(val)
        if not items: return g_mean
        return np.mean([mapping.get(i, g_mean) for i in items])

    final_df['director_encoded'] = apply_te(movie_data.get('director'), te.get('director'))
    final_df['actors_encoded'] = apply_te(movie_data.get('actors'), te.get('actors'))
    
    primary_genre = parse_list(movie_data.get('genre', []))
    primary_genre = primary_genre[0] if primary_genre else 'Unknown'
    dir_genre_key = str(movie_data.get('director')) + "_" + primary_genre
    final_df['dir_genre_encoded'] = apply_te(dir_genre_key, te.get('dir_genre'))

    # Match predict_ratings.py text_content construction exactly
    text_content = "Title: " + str(movie_data.get('title', 'Unknown')) + \
                   ". Directed by: " + str(movie_data.get('director', 'Unknown')) + \
                   ". " + str(movie_data.get('plot', ''))
    
    transformer_model = SentenceTransformer(state['sentence_transformer'])
    embedding = transformer_model.encode([text_content])
    pca_vec = state['pca'].transform(embedding)
    for i in range(pca_vec.shape[1]):
        col_name = f'pca_{i}'
        if col_name in final_df.columns: final_df[col_name] = float(pca_vec[0, i])

    return final_df.astype(float)

# --- NEW SIMILARITY FUNCTIONS FOR ORACLE ---
def find_similar_movies(movie_data, input_df, state, top_n=3):
    try:
        df_train = pd.read_csv(config.TRAINING_DATA_PATH)
        df_enriched = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        df_enriched = df_enriched.dropna(subset=['user_rating']).reset_index(drop=True)
        
        sim_cols = [c for c in state['training_columns'] if c.startswith('pca_')]
        
        X_train_sim = df_train[sim_cols]
        input_sim = input_df[sim_cols]
        
        sim_scores = cosine_similarity(input_sim, X_train_sim)[0]
        top_indices = np.argsort(sim_scores)[::-1][:top_n]
        
        similar_movies = []
        for idx in top_indices:
            row = df_enriched.iloc[idx]
            similar_movies.append({
                'title': row['title'],
                'year': row['year'],
                'score': sim_scores[idx],
                'director': row.get('director', 'Unknown'),
                'genre': row.get('genre', 'Unknown'),
            })
        return similar_movies
    except Exception as e:
        print(f"Error finding similar movies: {e}")
        return []

def explain_similarity(target_movie, similar_movie):
    shared = []
    
    target_dirs = set(parse_list(target_movie.get('director', [])))
    sim_dirs = set(parse_list(similar_movie.get('director', [])))
    common_dirs = target_dirs.intersection(sim_dirs)
    if common_dirs and 'Unknown' not in common_dirs:
        shared.append(f"Director ({', '.join(common_dirs)})")
        
    target_genres = set(parse_list(target_movie.get('genre', [])))
    sim_genres = set(parse_list(similar_movie.get('genre', [])))
    common_genres = target_genres.intersection(sim_genres)
    if common_genres and 'Unknown' not in common_genres:
        shared.append(f"Genres ({', '.join(common_genres)})")
        
    if shared:
        return " & ".join(shared)
    
    return "Similar plot/vibe (Semantic Match)"

if __name__ == "__main__":
    process_features()
