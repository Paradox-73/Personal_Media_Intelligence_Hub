import pandas as pd
import numpy as np
import joblib
import ast
import sys
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# --- CONFIG ---
MIN_DIRECTOR_COUNT = 3  
MIN_ACTOR_COUNT = 5     
MIN_WRITER_COUNT = 3    
MIN_PROD_COUNT = 5      
PCA_COMPONENTS = 10     

# --- HELPERS ---
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
    # FIX: Check for list/array types FIRST to avoid "Ambiguous truth value" error
    if isinstance(x, list): return x
    if isinstance(x, np.ndarray): return x.tolist()
    
    if pd.isna(x) or x == "": return []
    try:
        if "[" in str(x): return ast.literal_eval(str(x))
        return [str(x)]
    except: return []

def get_primary(val_list):
    return val_list[0] if val_list and len(val_list) > 0 else 'Unknown'

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    # Matches XGBoost strict rules
    return re.sub(r"[\[\]<']", "", str(col_name))

# --- MAIN TRAINING FUNCTION ---
def process_features():
    print("🛠️ Starting Movie Feature Engineering (Data Type Fix)...")
    
    if not config.MOVIES_ENRICHED_DATA_PATH.exists(): return

    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    df = df.dropna(subset=['user_rating']).copy()
    print(f"   Input Data: {len(df)} rated movies.")

    # 1. Numerics
    if df['rotten_tomatoes_rating'].dtype == object:
        def clean_pct(x):
            if pd.isna(x): return np.nan
            return float(str(x).replace('%', '').strip())
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_pct)
    
    df['box_office_clean'] = df['box_office'].apply(clean_money)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    
    num_cols = ['year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'vote_count', 'box_office_log']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

    X_num = df[num_cols].reset_index(drop=True)

    # 2. Categoricals
    for role, min_cnt in [('director', MIN_DIRECTOR_COUNT), ('actors', MIN_ACTOR_COUNT), ('writer', MIN_WRITER_COUNT)]:
        df[f'{role}_list'] = df[role].apply(parse_list)
        df[f'primary_{role}'] = df[f'{role}_list'].apply(get_primary)
        df[f'primary_{role}'] = df[f'primary_{role}'].apply(clean_text_value)
        
        counts = df[f'primary_{role}'].value_counts()
        valid = counts[counts >= min_cnt].index.tolist()
        df[f'clean_{role}'] = df[f'primary_{role}'].apply(lambda x: x if x in valid else 'Other')

    X_dir = pd.get_dummies(df['clean_director'], prefix='dir')
    X_act = pd.get_dummies(df['clean_actors'], prefix='act')
    X_wri = pd.get_dummies(df['clean_writer'], prefix='wri')

    # 3. Production
    df['primary_prod'] = df['production'].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown')
    df['primary_prod'] = df['primary_prod'].apply(clean_text_value)
    prod_counts = df['primary_prod'].value_counts()
    valid_prod = prod_counts[prod_counts >= MIN_PROD_COUNT].index.tolist()
    df['clean_prod'] = df['primary_prod'].apply(lambda x: x if x in valid_prod else 'Other')
    X_prod = pd.get_dummies(df['clean_prod'], prefix='studio')

    # 4. MPAA
    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # 5. Genres
    df['genre_list'] = df['genre'].apply(parse_list)
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=[f"gen_{c}" for c in mlb.classes_], index=df.index)

    # 6. Text
    df['text_content'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_content'])
    pca = PCA(n_components=PCA_COMPONENTS)
    pca_features = pca.fit_transform(tfidf_matrix.toarray())
    X_text = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 7. Combine & Sanitize
    X_final = pd.concat([X_num, X_dir, X_act, X_wri, X_prod, X_mpaa, X_genre, X_text], axis=1)
    
    new_cols = [sanitize_col(col) for col in X_final.columns]
    X_final.columns = new_cols
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    # Targets
    X_final['target_reg'] = df['user_rating'].reset_index(drop=True)
    user_median = df['user_rating'].median()
    X_final['target_class'] = (df['user_rating'] > user_median).astype(int).reset_index(drop=True)
    
    X_final.to_csv(config.TRAINING_DATA_PATH, index=False)
    
    state = {
        'valid_directors': df['clean_director'].unique().tolist(),
        'valid_actors': df['clean_actors'].unique().tolist(),
        'valid_writers': df['clean_writer'].unique().tolist(),
        'valid_studios': df['clean_prod'].unique().tolist(),
        'mlb_genre': mlb,
        'tfidf': tfidf,
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': [c for c in X_final.columns if 'target' not in c]
    }
    joblib.dump(state, config.PREPROCESSOR_STATE)
    print(f"✅ Features processed. Shape: {X_final.shape}")

# --- INFERENCE FUNCTIONS (FOR APP) ---

def transform_single_movie(movie_data, state):
    # 1. Numerics
    # Initialize with median defaults
    row = {col: state['median_values'].get(col, 0) for col in state['median_values']}
    
    # --- HELPER: Safe Numeric Cast ---
    def safe_num(val, default_val):
        try:
            if val is None or str(val).lower() in ['nan', 'n/a', '', 'none']:
                return default_val
            # Clean string artifacts like commas
            clean_s = str(val).replace(',', '').strip()
            return float(clean_s)
        except:
            return default_val

    # Apply safe cast to all potential string fields
    row['year'] = safe_num(movie_data.get('year'), row.get('year'))
    row['runtime'] = safe_num(movie_data.get('runtime'), row.get('runtime'))
    row['imdb_rating'] = safe_num(movie_data.get('imdb_rating'), row.get('imdb_rating'))
    row['metascore'] = safe_num(movie_data.get('metascore'), row.get('metascore')) # FIX: Safe Cast
    row['vote_average'] = safe_num(movie_data.get('vote_average'), row.get('vote_average'))
    row['vote_count'] = safe_num(movie_data.get('vote_count'), row.get('vote_count'))
    
    # Handle Box Office & RT special cases
    bo = clean_money(movie_data.get('box_office', 0))
    row['box_office_log'] = np.log1p(bo)
    
    rt = movie_data.get('rotten_tomatoes_rating')
    if isinstance(rt, str) and '%' in rt: rt = float(rt.replace('%', ''))
    row['rotten_tomatoes_rating'] = safe_num(rt, row.get('rotten_tomatoes_rating'))

    df_row = pd.DataFrame([row])

    # 2. Categoricals
    def set_cat(raw_val, prefix, valid_list):
        clean_val = clean_text_value(get_primary(parse_list(raw_val)))
        if clean_val not in valid_list: clean_val = 'Other'
        return sanitize_col(f"{prefix}_{clean_val}")

    dir_col = set_cat(movie_data.get('director'), 'dir', state['valid_directors'])
    act_col = set_cat(movie_data.get('actors'), 'act', state['valid_actors'])
    wri_col = set_cat(movie_data.get('writer'), 'wri', state['valid_writers'])
    
    raw_prod = str(movie_data.get('production', '')).split(',')[0]
    prod_val = clean_text_value(raw_prod)
    if prod_val not in state['valid_studios']: prod_val = 'Other'
    prod_col = sanitize_col(f"studio_{prod_val}")

    mpaa = categorize_rating(movie_data.get('rated', ''))
    mpaa_col = sanitize_col(f"rated_{mpaa}")

    # 3. Text & Genres
    txt = str(movie_data.get('overview', '')) + " " + str(movie_data.get('tagline', ''))
    tfidf_vec = state['tfidf'].transform([txt])
    pca_vec = state['pca'].transform(tfidf_vec.toarray())
    
    genres = parse_list(movie_data.get('genre', []))
    gen_vec = state['mlb_genre'].transform([genres])

    # 4. Construct Final DataFrame
    final_df = pd.DataFrame(0, index=[0], columns=state['training_columns'])
    
    # Ensure all inputs are numeric in the final dataframe
    for col in df_row.columns:
        if col in final_df.columns: final_df[col] = df_row[col].astype(float)
            
    for col in [dir_col, act_col, wri_col, prod_col, mpaa_col]:
        if col in final_df.columns: final_df[col] = 1.0
            
    for i, gen_name in enumerate(state['mlb_genre'].classes_):
        col_name = sanitize_col(f"gen_{gen_name}")
        if col_name in final_df.columns: final_df[col_name] = float(gen_vec[0, i])
            
    for i in range(pca_vec.shape[1]):
        col_name = sanitize_col(f"pca_{i}")
        if col_name in final_df.columns: final_df[col_name] = float(pca_vec[0, i])

    # Explicitly force float type for safety
    final_df = final_df.astype(float)
    return final_df

def find_similar_movies(input_data, input_features, state, top_n=5):
    try:
        db_df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        train_df = pd.read_csv(config.TRAINING_DATA_PATH)
        train_feats = train_df.drop(columns=['target_reg', 'target_class'], errors='ignore')
        common_cols = train_feats.columns.intersection(input_features.columns)
        
        sim_scores = cosine_similarity(input_features[common_cols], train_feats[common_cols])
        top_indices = sim_scores[0].argsort()[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            if idx < len(db_df):
                row = db_df.iloc[idx]
                results.append({
                    'title': row.get('title'),
                    'year': row.get('year'),
                    'score': sim_scores[0][idx],
                    'overview': row.get('overview')
                })
        return results
    except Exception as e:
        print(f"Error in similarity: {e}")
        return []

def explain_similarity(movie_a, movie_b_row):
    reasons = []
    if movie_a.get('director') and str(movie_b_row.get('director')) in str(movie_a.get('director')):
        reasons.append("Shared Director")
    
    genres_a = set(parse_list(movie_a.get('genre')))
    genres_b = set(parse_list(movie_b_row.get('genre')))
    if not genres_b and 'genre' in movie_b_row: genres_b = set(parse_list(movie_b_row['genre']))
         
    common_gen = genres_a.intersection(genres_b)
    if common_gen: reasons.append(f"Shared Genres ({', '.join(list(common_gen)[:2])})")
        
    if not reasons: return "Similar Plot & Vibe"
    return ", ".join(reasons)

if __name__ == "__main__":
    process_features()