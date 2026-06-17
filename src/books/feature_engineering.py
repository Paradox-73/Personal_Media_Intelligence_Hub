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
        # Handle formats like DD-MM-YYYY, YYYY-MM-DD, or just YYYY
        match = re.search(r'(\d{4})', s)
        if match:
            return int(match.group(1))
        return 0
    except:
        return 0

def parse_list(x):
    if pd.isna(x) or x == "": return []
    try:
        if isinstance(x, list): return x
        s = str(x).strip()
        if s.startswith('[') and s.endswith(']'):
            return ast.literal_eval(s)
        return [item.strip() for item in s.split(',')]
    except:
        return [str(x)]

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

def process_features():
    print("🛠️ Starting Book Feature Engineering...")

    if not config.BOOKS_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.BOOKS_ENRICHED_DATA_PATH}")
        return

    try:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    
    # 1. Clean Target
    df = df.dropna(subset=['my_rating']).copy()
    df['my_rating_clean'] = pd.to_numeric(df['my_rating'], errors='coerce').fillna(0)
    
    # 2. Extract Year
    df['year'] = df['publishedDate'].apply(clean_year)
    
    # 3. Numeric Features
    num_cols = ['year', 'pageCount', 'averageRating', 'ratingsCount']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 4. Authors Multi-Hot (Top ones to avoid explosion)
    df['author_list'] = df['authors'].apply(parse_list)
    mlb_author = MultiLabelBinarizer()
    author_encoded = mlb_author.fit_transform(df['author_list'])
    # Only keep authors that appear at least twice
    author_counts = author_encoded.sum(axis=0)
    valid_author_indices = np.where(author_counts >= 2)[0]
    X_author = pd.DataFrame(author_encoded[:, valid_author_indices], 
                            columns=[f"aut_{sanitize_col(mlb_author.classes_[i])}" for i in valid_author_indices], 
                            index=df.index)
    
    # 5. Categories Multi-Hot
    df['cat_list'] = df['categories'].apply(parse_list)
    mlb_cat = MultiLabelBinarizer()
    X_cat = pd.DataFrame(mlb_cat.fit_transform(df['cat_list']), 
                         columns=[f"cat_{sanitize_col(c)}" for c in mlb_cat.classes_], 
                         index=df.index)
    
    # 6. Text Embeddings
    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('') + \
                         ". Authors: " + df['authors'].fillna('') + \
                         ". Categories: " + df['categories'].fillna('') + \
                         ". Description: " + df['description'].fillna('')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    
    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=min(PCA_COMPONENTS, len(df)-1))
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # 7. Combine
    X_final = pd.concat([df[num_cols].reset_index(drop=True), 
                         X_author.reset_index(drop=True), 
                         X_cat.reset_index(drop=True), 
                         X_text.reset_index(drop=True)], axis=1)
    
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    # Targets
    X_final['target_reg'] = df['my_rating_clean'].values
    X_final['target_class'] = pd.cut(df['my_rating_clean'], bins=[-1, 2.5, 4.5, 5.1], labels=[0, 1, 2]).astype(int)

    # Save
    X_final.to_csv(config.BOOKS_TRAINING_DATA_PATH, index=False)
    
    state = {
        'mlb_author': mlb_author,
        'valid_author_indices': valid_author_indices,
        'mlb_cat': mlb_cat,
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': X_final.drop(columns=['target_reg', 'target_class']).columns.tolist(),
        'sentence_transformer': 'all-MiniLM-L6-v2'
    }
    joblib.dump(state, config.BOOKS_PREPROCESSOR_STATE)
    print(f"✅ Features processed. Shape: {X_final.shape}")

# --------------------------------------------------------------------------- #
# Oracle helpers (single-item inference + similarity)
# --------------------------------------------------------------------------- #
_ST_MODEL = None
def _get_transformer():
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _ST_MODEL


def _join(val):
    """Render authors/categories (list or string) as a comma string."""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val or '')


def _book_text(raw):
    """Reproduce the exact training text_content string for one book."""
    title = raw.get('title') or ''
    authors = _join(raw.get('authors') or raw.get('author'))
    cats = _join(raw.get('categories') or raw.get('genre'))
    desc = raw.get('description') or ''
    return f"Title: {title}. Authors: {authors}. Categories: {cats}. Description: {desc}"


def transform_single_book(raw_data, state):
    """Turn one raw book dict into a model-ready 1-row frame aligned to training_columns."""
    cols = state['training_columns']
    X = pd.DataFrame(0.0, index=[0], columns=cols)
    medians = state.get('median_values', {})

    def num(key):
        try:
            v = float(raw_data.get(key))
            return v if not np.isnan(v) else medians.get(key, 0)
        except (TypeError, ValueError):
            return medians.get(key, 0)

    year = clean_year(raw_data.get('year') or raw_data.get('publishedDate')) or medians.get('year', 0)
    numeric_vals = {'year': year, 'pageCount': num('pageCount'),
                    'averageRating': num('averageRating'), 'ratingsCount': num('ratingsCount')}
    for k, v in numeric_vals.items():
        if k in X.columns:
            X.at[0, k] = v

    # Authors multi-hot (frequency-gated indices)
    authors = raw_data.get('author') or raw_data.get('authors') or []
    authors = authors if isinstance(authors, list) else parse_list(authors)
    mlb_author = state.get('mlb_author')
    valid = state.get('valid_author_indices', [])
    if mlb_author is not None:
        try:
            av = mlb_author.transform([authors])[0]
            for i in valid:
                col = sanitize_col(f"aut_{mlb_author.classes_[i]}")
                if col in X.columns:
                    X.at[0, col] = av[i]
        except Exception:
            pass

    # Categories multi-hot
    cats = raw_data.get('genre') or raw_data.get('categories') or []
    cats = cats if isinstance(cats, list) else parse_list(cats)
    mlb_cat = state.get('mlb_cat')
    if mlb_cat is not None:
        try:
            cv = mlb_cat.transform([cats])[0]
            for i, c in enumerate(mlb_cat.classes_):
                col = sanitize_col(f"cat_{c}")
                if col in X.columns:
                    X.at[0, col] = cv[i]
        except Exception:
            pass

    # Text embedding -> PCA
    try:
        emb = _get_transformer().encode([_book_text(raw_data)])
        pcav = state['pca'].transform(emb)[0]
        for i in range(len(pcav)):
            col = f"pca_{i}"
            if col in X.columns:
                X.at[0, col] = pcav[i]
    except Exception:
        pass

    return X


def _load_enriched_books():
    try:
        return pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except Exception:
        try:
            return pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
        except Exception:
            return pd.DataFrame()


def find_similar_books(raw_data, input_df, state, n=3):
    """Cosine similarity in the shared text-embedding space against your rated library."""
    lib = _load_enriched_books()
    if lib.empty:
        return []

    model = _get_transformer()
    target_text = _book_text(raw_data)

    lib = lib.copy()
    lib['_text'] = ("Title: " + lib['title'].fillna('') +
                    ". Authors: " + lib.get('authors', pd.Series('', index=lib.index)).fillna('') +
                    ". Categories: " + lib.get('categories', pd.Series('', index=lib.index)).fillna('') +
                    ". Description: " + lib.get('description', pd.Series('', index=lib.index)).fillna(''))

    embs = model.encode([target_text] + lib['_text'].tolist())
    target_vec, lib_vecs = embs[0], embs[1:]
    sims = lib_vecs @ target_vec / (
        (np.linalg.norm(lib_vecs, axis=1) * np.linalg.norm(target_vec)) + 1e-9)

    target_title = str(raw_data.get('title', '')).lower().strip()
    out = []
    for idx in np.argsort(sims)[::-1]:
        row = lib.iloc[idx]
        if str(row.get('title', '')).lower().strip() == target_title:
            continue
        out.append({
            'title': row.get('title', 'Unknown'),
            'author': row.get('authors', 'Unknown'),
            'year': clean_year(row.get('publishedDate')),
            'similarity': float(sims[idx]),
            'raw_data': row.to_dict(),
        })
        if len(out) >= n:
            break
    return out


def explain_similarity_books(raw_data, other_raw, state):
    """Human-readable reason two books are alike (shared categories / author)."""
    def to_set(val):
        if isinstance(val, list):
            return {str(v).strip().lower() for v in val}
        return {g.strip().lower() for g in str(val or '').replace('[', '').replace(']', '').split(',') if g.strip()}

    c1 = to_set(raw_data.get('genre') or raw_data.get('categories'))
    c2 = to_set(other_raw.get('categories'))
    shared_c = c1 & c2

    a1 = to_set(raw_data.get('author') or raw_data.get('authors'))
    a2 = to_set(other_raw.get('authors'))
    shared_a = a1 & a2

    bits = []
    if shared_a:
        bits.append(f"same author: {', '.join(sorted(shared_a))}")
    if shared_c:
        bits.append(f"shared categories: {', '.join(sorted(shared_c))}")
    return "Matched on " + "; ".join(bits) if bits else "Matched on overall semantic vibe."


if __name__ == "__main__":
    process_features()
