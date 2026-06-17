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

# --------------------------------------------------------------------------- #
# Oracle helpers (single-item inference + similarity)
# --------------------------------------------------------------------------- #
_ST_MODEL = None
def _get_transformer():
    """Lazy-load the sentence transformer once (heavy import)."""
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _ST_MODEL


def _game_text(raw):
    """Reproduce the exact training text_content string for one game."""
    genres = raw.get('genres')
    if not isinstance(genres, str):
        genres = ", ".join(raw.get('genre') or [])
    tags = raw.get('tags') if isinstance(raw.get('tags'), str) else ""
    desc = raw.get('description_raw') or raw.get('description') or ""
    name = raw.get('name') or raw.get('title') or ""
    return f"Name: {name}. Genres: {genres}. Tags: {tags}. Description: {desc}"


def transform_single_game(raw_data, state):
    """Turn one raw game dict into a model-ready 1-row frame aligned to training_columns."""
    cols = state['training_columns']
    X = pd.DataFrame(0.0, index=[0], columns=cols)
    medians = state.get('median_values', {})

    # Numerics (fall back to training medians when missing)
    def num(key):
        try:
            v = float(raw_data.get(key))
            return v if not np.isnan(v) else medians.get(key, 0)
        except (TypeError, ValueError):
            return medians.get(key, 0)

    year = clean_year(raw_data.get('year') or raw_data.get('released')) or medians.get('year', 0)
    numeric_vals = {'year': year, 'metacritic': num('metacritic'), 'rating': num('rating'),
                    'ratings_count': num('ratings_count'), 'reviews_count': num('reviews_count')}
    for k, v in numeric_vals.items():
        if k in X.columns:
            X.at[0, k] = v

    # Platform one-hot
    plat = raw_data.get('platform')
    if isinstance(plat, list):
        plat = plat[0] if plat else ''
    pcol = sanitize_col(f"plat_{plat}")
    if pcol in X.columns:
        X.at[0, pcol] = 1

    # Genres multi-hot
    genres = raw_data.get('genre') or raw_data.get('genres') or []
    genres = genres if isinstance(genres, list) else parse_list(genres)
    mlb_genre = state.get('mlb_genre')
    if mlb_genre is not None:
        try:
            gv = mlb_genre.transform([genres])[0]
            for i, c in enumerate(mlb_genre.classes_):
                col = sanitize_col(f"gen_{c}")
                if col in X.columns:
                    X.at[0, col] = gv[i]
        except Exception:
            pass

    # Developers multi-hot (frequency-gated indices)
    devs = raw_data.get('developer') or raw_data.get('developers') or []
    devs = devs if isinstance(devs, list) else parse_list(devs)
    mlb_dev = state.get('mlb_dev')
    valid = state.get('valid_dev_indices', [])
    if mlb_dev is not None:
        try:
            dv = mlb_dev.transform([devs])[0]
            for i in valid:
                col = sanitize_col(f"dev_{mlb_dev.classes_[i]}")
                if col in X.columns:
                    X.at[0, col] = dv[i]
        except Exception:
            pass

    # Text embedding -> PCA
    try:
        emb = _get_transformer().encode([_game_text(raw_data)])
        pcav = state['pca'].transform(emb)[0]
        for i in range(len(pcav)):
            col = f"pca_{i}"
            if col in X.columns:
                X.at[0, col] = pcav[i]
    except Exception:
        pass

    return X


def _load_enriched_games():
    try:
        return pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except Exception:
        try:
            return pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
        except Exception:
            return pd.DataFrame()


def find_similar_games(raw_data, input_df, state, n=3):
    """Cosine similarity in the shared text-embedding space against your rated library."""
    lib = _load_enriched_games()
    if lib.empty:
        return []

    model = _get_transformer()
    target_text = _game_text(raw_data)

    lib = lib.copy()
    lib['_text'] = ("Name: " + lib['name'].fillna('') +
                    ". Genres: " + lib.get('genres', pd.Series('', index=lib.index)).fillna('') +
                    ". Tags: " + lib.get('tags', pd.Series('', index=lib.index)).fillna('') +
                    ". Description: " + lib.get('description_raw', pd.Series('', index=lib.index)).fillna(''))

    embs = model.encode([target_text] + lib['_text'].tolist())
    target_vec, lib_vecs = embs[0], embs[1:]
    sims = lib_vecs @ target_vec / (
        (np.linalg.norm(lib_vecs, axis=1) * np.linalg.norm(target_vec)) + 1e-9)

    target_name = (raw_data.get('name') or raw_data.get('title') or '').lower().strip()
    out = []
    for idx in np.argsort(sims)[::-1]:
        row = lib.iloc[idx]
        if str(row.get('name', '')).lower().strip() == target_name:
            continue  # skip self
        out.append({
            'title': row.get('name', 'Unknown'),
            'platform': row.get('platform_from_text', 'Unknown'),
            'year': clean_year(row.get('released')),
            'similarity': float(sims[idx]),
            'raw_data': row.to_dict(),
        })
        if len(out) >= n:
            break
    return out


def explain_similarity_games(raw_data, other_raw, state):
    """Human-readable reason two games are alike (shared genres / dev / platform)."""
    def to_set(val):
        if isinstance(val, list):
            return {str(v).strip().lower() for v in val}
        return {g.strip().lower() for g in str(val or '').replace('[', '').replace(']', '').split(',') if g.strip()}

    g1 = to_set(raw_data.get('genre') or raw_data.get('genres'))
    g2 = to_set(other_raw.get('genres'))
    shared_g = g1 & g2

    d1 = to_set(raw_data.get('developer') or raw_data.get('developers'))
    d2 = to_set(other_raw.get('developers'))
    shared_d = d1 & d2

    bits = []
    if shared_g:
        bits.append(f"shared genres: {', '.join(sorted(shared_g))}")
    if shared_d:
        bits.append(f"same developer: {', '.join(sorted(shared_d))}")
    p1 = str(raw_data.get('platform', '')).lower()
    p2 = str(other_raw.get('platform_from_text', '')).lower()
    if p1 and p1 == p2:
        bits.append(f"same platform ({raw_data.get('platform')})")
    return "Matched on " + "; ".join(bits) if bits else "Matched on overall semantic vibe."


if __name__ == "__main__":
    process_features()
