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
from src.common.wikipedia_plots import plot_for
from src.common.parsing import fix_mojibake, clean_year
from src.common.text_embedding import encode_pooled, encode_pooled_single

# --- CONFIG ---
PCA_COMPONENTS = 15 # Reduced for small dataset

# Smoothing prior for the entity encodings (see src/common/taste_features.py).
TE_PRIOR = 2

# (column, names to keep). Replaces the 17 developer dummies that were fitted on
# 66 rated games; `publishers` and `tags` were fetched and never read.
GAME_ENCODE = [("developers", 2), ("publishers", 2), ("tags", 6), ("genres", 3)]

RAWG_PLATFORM_LABELS = {
    "PlayStation 2": "PS2", "PSP": "PSP", "PC": "PC",
    "Android": "Mobile", "iOS": "Mobile", "macOS": "PC", "Linux": "PC",
}

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
    df['year'] = df['released'].apply(lambda v: clean_year(v) or 0)
    
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
    
    # 6. Developers / publishers / tags as TARGET ENCODINGS, not multi-hot.
    # 17 developer dummies on 66 rated games is variance, not signal, and it left
    # `publishers` (49 distinct), `tags` (90% coverage) and `playtime` unread.
    y_games = df['my_rating_clean'].astype(float).values
    df['dev_list'] = df['developers'].apply(parse_list)
    te_frames, te_artifacts = {}, {}
    for col, limit in GAME_ENCODE:
        if col not in df.columns:
            continue
        enc, sup, art = tf.oof_entity_encode(df[col].tolist(), y_games,
                                             m=TE_PRIOR, limit=limit)
        te_frames[f'{col}_encoded'] = enc
        te_frames[f'{col}_n'] = sup
        te_artifacts[col] = art
    te_frames['playtime'] = pd.to_numeric(df.get('playtime'), errors='coerce').fillna(0).values
    X_dev = pd.DataFrame(te_frames, index=df.index)
    # Kept for unpickling older states; the dummies are no longer emitted.
    mlb_dev = MultiLabelBinarizer()
    mlb_dev.fit(df['dev_list'])
    valid_dev_indices = np.array([], dtype=int)

    # 7. Text Embeddings
    print("   Generating text embeddings...")
    # Header + description are embedded SEPARATELY so a long blurb is split into
    # sentence windows and read in full. RAWG descriptions average ~267 tokens and
    # 48.6% of this library runs past MiniLM's 256-token limit -- a plain `.encode()`
    # silently dropped that tail, which is the single largest text defect in the
    # games pipeline. `_game_header` is shared with the inference path and the
    # similarity search, so the three cannot drift apart.
    _records = df.to_dict('records')
    df['text_header'] = [_game_header(r) for r in _records]
    _body = pd.Series([_game_body(r) for r in _records], index=df.index)
    df['text_content'] = df['text_header'] + " " + _body

    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)

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
        'genres': X_genre.columns.tolist(),
        'te_artifacts': te_artifacts,
        'game_encode': GAME_ENCODE,
        'te_prior': TE_PRIOR,
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


def _game_header(raw):
    """The per-item prefix repeated on every window, so a long blurb cannot drift
    off-item. One definition, shared by training, inference and similarity.

    Tags are capped: RAWG ships up to ~40 of them and an uncapped tag list ran to
    224 tokens on the worst item, which plus a 180-token window would overflow the
    model again. 150 chars keeps header+window inside 256 for every row.
    """
    genres = raw.get('genres')
    if not isinstance(genres, str):
        genres = ", ".join(raw.get('genre') or [])
    tags = raw.get('tags') if isinstance(raw.get('tags'), str) else ""
    name = raw.get('name') or raw.get('title') or ""
    return fix_mojibake(f"Name: {name}. Genres: {genres}. Tags: {str(tags)[:150]}.")


def _game_body(raw):
    desc = raw.get('description_raw') or raw.get('description') or ""
    blurb = fix_mojibake(str(desc or ""))
    return "Description: " + plot_for('game', raw.get('name') or raw.get('title'), blurb)


def _game_text(raw):
    """Header + body as one flat string, for display and for callers that want it."""
    return f"{_game_header(raw)} {_game_body(raw)}"


def transform_single_game(raw_data, state):
    """Turn one raw game dict into a model-ready 1-row frame aligned to training_columns."""
    cols = state['training_columns']
    X = pd.DataFrame(0.0, index=[0], columns=cols)
    medians = state.get('median_values', {})

    # Numerics. The batch builder fills missing numerics with 0 (`.fillna(0)`), not
    # with the median — so a game with no Metacritic score trains as 0 and must be
    # served as 0. Falling back to the median here told the model an unreviewed game
    # scored 80.
    def num(key):
        try:
            v = float(raw_data.get(key))
            return 0.0 if np.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    year = clean_year(raw_data.get('year') or raw_data.get('released')) or 0
    numeric_vals = {'year': year, 'metacritic': num('metacritic'), 'rating': num('rating'),
                    'ratings_count': num('ratings_count'), 'reviews_count': num('reviews_count')}
    for k, v in numeric_vals.items():
        if k in X.columns:
            X.at[0, k] = v

    # Platform one-hot. The batch builder one-hots `platform_from_text`, which is the
    # column the enriched CSV actually carries; reading only `platform` (a key that
    # exists in neither the CSV nor the RAWG payload) left the whole plat_* block at
    # zero for every inference, while every training row has exactly one set.
    plat = next((raw_data.get(k) for k in ('platform_from_text', 'platform', 'platforms')
                 if raw_data.get(k) not in (None, '', [])), '')
    if isinstance(plat, (list, tuple)):
        plat = plat[0] if len(plat) else ''
    if isinstance(plat, dict):
        plat = plat.get('name', '')
    # RAWG names a platform differently from the training labels {Mobile, PC, PS2, PSP}.
    plat = RAWG_PLATFORM_LABELS.get(plat, plat)
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

    # Developer / publisher / tag target encodings, applied through the same helper
    # that built the training columns (replaces the old developer multi-hot block).
    ALIASES = {'developers': ('developers', 'developer'),
               'publishers': ('publishers', 'publisher'),
               'tags': ('tags',), 'genres': ('genres', 'genre')}
    arts = state.get('te_artifacts') or {}
    for col, limit in state.get('game_encode', GAME_ENCODE):
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
    if 'playtime' in X.columns:
        try:
            X.at[0, 'playtime'] = float(raw_data.get('playtime') or 0)
        except (TypeError, ValueError):
            X.at[0, 'playtime'] = 0.0

    # Text embedding -> PCA. No try/except: the batch CSV is built through this
    # function, and a swallowed failure would silently score every game with zeros.
    emb = encode_pooled_single(_game_header(raw_data), _game_body(raw_data),
                              _get_transformer())
    pcav = state['pca'].transform(emb)[0]
    for i in range(len(pcav)):
        col = f"pca_{i}"
        if col in X.columns:
            X.at[0, col] = pcav[i]

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
    lib = lib.copy()
    _lib_recs = lib.to_dict('records')
    # Same chunk-pooled recipe as training, so "similar games" is measured in the
    # space the model actually learned in and long blurbs are read in full.
    headers = [_game_header(raw_data)] + [_game_header(r) for r in _lib_recs]
    bodies = [_game_body(raw_data)] + [_game_body(r) for r in _lib_recs]
    lib['_text'] = [f"{h} {b}" for h, b in zip(headers[1:], bodies[1:])]

    embs = encode_pooled(headers, bodies, model)
    target_vec, lib_vecs = embs[0], embs[1:]
    # encode_pooled returns unit vectors, so the dot product IS the cosine.
    sims = lib_vecs @ target_vec

    target_name = (raw_data.get('name') or raw_data.get('title') or '').lower().strip()
    out = []
    for idx in np.argsort(sims)[::-1]:
        row = lib.iloc[idx]
        if str(row.get('name', '')).lower().strip() == target_name:
            continue  # skip self
        out.append({
            'title': row.get('name', 'Unknown'),
            'platform': row.get('platform_from_text', 'Unknown'),
            'year': clean_year(row.get('released')) or 0,
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
