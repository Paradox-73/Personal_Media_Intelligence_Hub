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
from src.common.parsing import (  # canonical shared parsers
    parse_list, people_key, sanitize_col, clean_money)
from src.common import taste_features as tf
from src.common.text_embedding import encode_pooled, encode_pooled_single

# --- CONFIG ---
PCA_COMPONENTS = 15 # Reduced for small dataset

# Smoothing prior for the entity encodings (see src/common/taste_features.py).
TE_PRIOR = 2

# (column, names to keep). Replaces 165 one-hot dummies that were being fitted on
# 93 rows. `moods` and `hc_tags` are Hardcover fields that were never read.
BOOK_ENCODE = [
    ("authors", 2), ("categories", 4), ("publisher", 1),
    ("moods", 4), ("hc_tags", 5),
]

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
    
    # 4-5. Authors / categories / publisher as TARGET ENCODINGS, not multi-hot.
    #
    # This block used to emit 10 author dummies and 155 category dummies. On 93
    # rated books that is 165 near-empty binary columns for 93 samples -- most with
    # one or two positives -- which is the single worst feature-to-sample ratio in
    # the project and pure variance. A smoothed mean rating per author/category
    # carries the same information in one dense column each, and the smoothing
    # prior handles the long tail instead of the model having to.
    #
    # `moods` and `hc_tags` come from Hardcover (65% / 69% coverage) and were never
    # read at all.
    y_books = df['my_rating_clean'].astype(float).values
    df['author_list'] = df['authors'].apply(parse_list)
    df['cat_list'] = df['categories'].apply(parse_list)

    te_frames, te_artifacts = {}, {}
    for col, limit in BOOK_ENCODE:
        if col not in df.columns:
            continue
        enc, sup, art = tf.oof_entity_encode(df[col].tolist(), y_books,
                                             m=TE_PRIOR, limit=limit)
        te_frames[f'{col}_encoded'] = enc
        te_frames[f'{col}_n'] = sup
        te_artifacts[col] = art
    X_entity = pd.DataFrame(te_frames)
    # Kept only so an older preprocessor state / the inference path can still be
    # unpickled; the dummy columns themselves are no longer emitted.
    mlb_author, mlb_cat = MultiLabelBinarizer(), MultiLabelBinarizer()
    mlb_author.fit(df['author_list'])
    mlb_cat.fit(df['cat_list'])
    valid_author_indices = np.array([], dtype=int)


    # 6. Text Embeddings
    print("   Generating text embeddings...")
    # Header + description are embedded SEPARATELY so a long blurb is split into
    # sentence windows and read in full, instead of MiniLM silently dropping every
    # token past 256. Book blurbs average ~104 words and routinely run far longer, so
    # the tail was being cut off. `_book_header` is shared with the inference path
    # and the similarity search, so the three cannot drift apart.
    df['text_header'] = [_book_header(r) for r in df.to_dict('records')]
    _body = "Description: " + df['description'].fillna('').astype(str)
    df['text_content'] = df['text_header'] + " " + _body

    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)
    
    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=min(PCA_COMPONENTS, len(df)-1))
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # 7. Combine
    X_final = pd.concat([df[num_cols].reset_index(drop=True),
                         X_entity.reset_index(drop=True),
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
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'te_artifacts': te_artifacts,
        'book_encode': BOOK_ENCODE,
        'te_prior': TE_PRIOR,
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


def _book_header(raw):
    """The per-item prefix repeated on every window, so a long blurb cannot drift
    off-item. One definition, shared by training, inference and similarity."""
    title = raw.get('title') or ''
    authors = _join(raw.get('authors') or raw.get('author'))
    cats = _join(raw.get('categories') or raw.get('genre'))
    return f"Title: {title}. Authors: {authors}. Categories: {cats}."


def _book_body(raw):
    return "Description: " + str(raw.get('description') or '')


def _book_text(raw):
    """Header + body as one flat string, for display and for callers that want it."""
    return f"{_book_header(raw)} {_book_body(raw)}"


def transform_single_book(raw_data, state):
    """Turn one raw book dict into a model-ready 1-row frame aligned to training_columns."""
    cols = state['training_columns']
    X = pd.DataFrame(0.0, index=[0], columns=cols)
    medians = state.get('median_values', {})

    # `process_features` fills missing numerics with 0 (`.fillna(0)`), so inference
    # must too — falling back to the median served a value the batch never produces.
    def num(key):
        try:
            v = float(raw_data.get(key))
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    year = clean_year(raw_data.get('year') or raw_data.get('publishedDate')) or 0
    numeric_vals = {'year': year, 'pageCount': num('pageCount'),
                    'averageRating': num('averageRating'), 'ratingsCount': num('ratingsCount')}
    for k, v in numeric_vals.items():
        if k in X.columns:
            X.at[0, k] = v

    # Entity target encodings, applied through the same helper that built the
    # training columns. Replaces the old author/category multi-hot blocks (165
    # dummies fitted on 93 rows). `genre` is accepted as an alias for `categories`
    # and `author` for `authors`, because the watchlist/queue callers use those.
    ALIASES = {'authors': ('authors', 'author'), 'categories': ('categories', 'genre'),
               'publisher': ('publisher',), 'moods': ('moods',), 'hc_tags': ('hc_tags',)}
    arts = state.get('te_artifacts') or {}
    for col, limit in state.get('book_encode', BOOK_ENCODE):
        art = arts.get(col)
        if not art:
            continue
        raw_val = next((raw_data.get(k) for k in ALIASES.get(col, (col,))
                        if raw_data.get(k) not in (None, '', [])), None)
        enc, sup = tf.apply_entity_map(raw_val, art['map'], art.get('counts', {}),
                                       art['mean'], art.get('limit', limit))
        if f'{col}_encoded' in X.columns:
            X.at[0, f'{col}_encoded'] = enc
        if f'{col}_n' in X.columns:
            X.at[0, f'{col}_n'] = sup

    # Text embedding -> PCA
    try:
        emb = encode_pooled_single(_book_header(raw_data), _book_body(raw_data),
                                   _get_transformer())
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

    lib = lib.copy()
    # Same header/body split and same pooled encoder as training, so "similar" is
    # measured in the space the model was actually fitted on.
    recs = lib.to_dict('records')
    embs = encode_pooled([_book_header(raw_data)] + [_book_header(r) for r in recs],
                         [_book_body(raw_data)] + [_book_body(r) for r in recs], model)
    target_vec, lib_vecs = embs[0], embs[1:]
    # encode_pooled returns unit vectors, so the dot product IS the cosine.
    sims = lib_vecs @ target_vec

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
