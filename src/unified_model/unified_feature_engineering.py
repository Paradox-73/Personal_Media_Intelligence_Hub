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

from src.unified_model.unified_utils import get_music_affinity_features, get_music_gate_mask, DomainAligner

# Define MusicProfile in __main__ to satisfy joblib
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class MusicProfile:
    centroids: np.ndarray
    cluster_labels: List[str]
    cluster_meta: List[Dict[str, Any]]
    X_lib: np.ndarray
    pu_model: Any
    pool_score_dist: np.ndarray
    top_genres: List[tuple]
    audio_fingerprint: Dict[str, float]
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]

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

def transform_single_media(meta, state, media_type='movie'):
    """
    Transforms a single item's metadata into the unified feature space.
    Used for live inference/evaluation.
    """
    # 1. Basic Metadata
    is_tv = 1 if media_type == 'tv' else 0
    is_game = 1 if media_type == 'game' else 0
    is_book = 1 if media_type == 'book' else 0
    is_music = 1 if media_type == 'music' else 0
    
    runtime = meta.get('runtime', 0)
    if isinstance(runtime, str):
        runtime_match = re.search(r'\d+', runtime)
        runtime = float(runtime_match.group()) if runtime_match else 0
    
    year = clean_year(meta.get('year') or meta.get('released'))
    
    # Missingness Masks
    masks = {f'has_{d}_feats': (1 if media_type == d else 0) for d in ['movie', 'tv', 'game', 'book', 'music']}
    
    # Scores
    imdb = pd.to_numeric(meta.get('imdb_rating', 0), errors='coerce')
    meta_score = pd.to_numeric(meta.get('metascore', 0), errors='coerce')
    rt = pd.to_numeric(str(meta.get('rotten_tomatoes_rating', 0)).replace('%', ''), errors='coerce')
    va = pd.to_numeric(meta.get('vote_average', 0), errors='coerce')
    
    # Normalize Critic Avg
    ir_100 = imdb * 10 if media_type in ['movie', 'tv'] else imdb * 20
    critic_avg = np.nanmean([ir_100, meta_score, rt, va * 10]) / 100 * 5 if any([imdb, meta_score, rt, va]) else 0
    
    box_office = np.log1p(clean_money(meta.get('box_office', 0)))
    
    data = {
        'is_tv_show': is_tv, 'is_game': is_game, 'is_book': is_book, 'is_music': is_music,
        'year': year, 'runtime': runtime, 'imdb_rating': imdb, 'metascore': meta_score,
        'rotten_tomatoes_rating': rt, 'vote_average': va, 'imdb_votes': pd.to_numeric(meta.get('imdb_votes', 0), errors='coerce'),
        'box_office_log': box_office, 'popularity': pd.to_numeric(meta.get('popularity', 0), errors='coerce'),
        'total_wins': 0, 'total_nominations': 0, 'critic_avg_5': critic_avg,
        **masks
    }
    
    # NLP — length-normalized, genre-templated text (matches the batch builder):
    # cap title/genre and truncate the description to its first 40 words so domains
    # share the same text shape, removing the cross-domain length confound.
    _g = re.sub(r"[\[\]']", "", str(meta.get('genre') or meta.get('categories') or ''))[:80]
    _d = " ".join(str(meta.get('overview') or meta.get('description') or '').split()[:40])
    txt = f"Title: {str(meta.get('title') or '')[:120]}. Genre: {_g}. {_d}"
    transformer = SentenceTransformer(state['sentence_transformer'])
    emb = transformer.encode([txt], normalize_embeddings=True)

    # PCA first (384 -> 10), THEN domain alignment: the aligner was fit on the
    # PCA components (pca_cols) in the trainer, so it operates in PCA space.
    pca_vec = state['pca'].transform(emb)
    if 'aligner' in state and state['aligner'] is not None:
        pca_vec = state['aligner'].transform(pca_vec, [media_type])

    X_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(10)])
    
    # Languages & Genres
    lang = next((l.strip() for l in str(meta.get('language', 'English')).split(',') if l.strip() in state['top_languages']), 'Other')
    X_lang = pd.DataFrame(0, index=[0], columns=[f'lang_{l}' for l in state['top_languages']] + ['lang_Other'])
    X_lang[f'lang_{lang}'] = 1
    
    genres = parse_list(meta.get('genre') or meta.get('categories', []))
    X_genre = pd.DataFrame(state['mlb_genre'].transform([genres]), columns=[f"gen_{sanitize_col(c)}" for c in state['mlb_genre'].classes_])
    
    mpaa = categorize_rating(meta.get('rated', 'NR'))
    X_mpaa = pd.DataFrame(0, index=[0], columns=['rated_Adult', 'rated_Teen', 'rated_General'])
    X_mpaa[f'rated_{mpaa}'] = 1
    
    # Music Affinity (gated to zero for non-music items, so a failure here is
    # harmless for movie/tv/game/book — those columns just stay 0 via alignment).
    X_music = pd.DataFrame()
    profile_path = config.MUSIC_MODEL_DIR / "profile.joblib"
    bundle_path = config.MUSIC_MODEL_DIR / "preprocessors.joblib"
    if profile_path.exists() and bundle_path.exists():
        try:
            profile = joblib.load(profile_path)
            bundle = joblib.load(bundle_path)
            X_music = get_music_affinity_features([txt], profile, bundle)
            X_music = X_music * get_music_gate_mask(media_type)
        except Exception:
            X_music = pd.DataFrame()

    # Final concat. Coerce the numeric backbone to floats: missing fields (e.g. a
    # manually-entered book with no page count) arrive as None and would otherwise
    # leave an object-dtype column that XGBoost/CatBoost reject.
    X_meta = pd.DataFrame([data]).apply(pd.to_numeric, errors='coerce')
    X_final = pd.concat([X_meta.reset_index(drop=True), X_lang.reset_index(drop=True),
                         X_genre.reset_index(drop=True), X_mpaa.reset_index(drop=True), 
                         X_text.reset_index(drop=True), X_music.reset_index(drop=True)], axis=1)
    
    # Fill missing columns from training
    for col in state['training_columns']:
        if col not in X_final.columns:
            X_final[col] = 0
            
    return X_final[state['training_columns']]

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
    # Convert 'I' to NaN for numeric handling but keep for the universal dataset
    df_g['user_rating'] = pd.to_numeric(df_g['user_rating'], errors='coerce')

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
        'pageCount': 'runtime'
    })
    
    if 'released' not in df_b.columns:
        df_b['released'] = None

    df_b['media_type'] = 'book'
    df_b['year'] = df_b['released'].apply(clean_year)
    df_b['user_rating'] = pd.to_numeric(df_b['user_rating'], errors='coerce')

    # 5. Load Music
    try:
        df_mu = pd.read_csv(config.MUSIC_ENRICHED_DATA_PATH)
        
        df_mu = df_mu.rename(columns={
            'name': 'title',
            'artists': 'director',
            'artist_genres': 'genre',
            'rating': 'user_rating',
            'release_year': 'year',
            'popularity': 'imdb_rating', 
            'mb_length_ms': 'duration_ms'
        })
        df_mu['media_type'] = 'music'
        df_mu['runtime'] = df_mu['duration_ms'].fillna(0) / 60000

        # CROSS-DOMAIN ALIGNMENT: music now carries the same geography/language axes
        # the other domains use. Lyrics language -> shared `language`; artist origin
        # country -> shared `country` (which movies/TV already populate).
        _blank = pd.Series([''] * len(df_mu))
        df_mu['language'] = df_mu.get('mb_language', _blank)
        df_mu['country'] = df_mu.get('mb_artist_country', _blank)

        # Build overview with the richest aligned signals so the text embedding (and
        # therefore the cross-domain music-affinity features) carry genre + crowd tags +
        # mood + origin, not just MusicBrainz tags.
        df_mu['overview'] = (df_mu['mb_tags'].fillna('') + " "
                             + df_mu['genre'].fillna('') + " "
                             + df_mu.get('lastfm_tags', _blank).fillna('') + " "
                             + df_mu.get('audiodb_mood', _blank).fillna('') + " "
                             + df_mu.get('mb_artist_country', _blank).fillna('') + " "
                             + df_mu.get('mb_language', _blank).fillna('') + " "
                             + df_mu.get('lyric_embed_text', _blank).fillna(''))
        df_mu['user_rating'] = pd.to_numeric(df_mu['user_rating'], errors='coerce')
    except Exception as e:
        print(f"   ⚠️ Could not load Music data: {e}")
        df_mu = pd.DataFrame()

    # Combine all
    dfs = [df_m, df_s, df_g, df_b]
    if not df_mu.empty:
        dfs.append(df_mu)
        
    df = pd.concat(dfs, ignore_index=True)
    
    # Missingness Masks (Multi-Modal Fusion)
    for domain in ['movie', 'tv', 'game', 'book', 'music']:
        df[f'has_{domain}_feats'] = (df['media_type'] == domain).astype(int)

    print(f"   Counts -> Movies: {len(df_m)}, Shows: {len(df_s)}, Games: {len(df_g)}, Books: {len(df_b)}, Music: {len(df_mu)}")
    print(f"   Total: {len(df)} records.")
    
    # Date Handling for Temporal Weighting
    df['rating_date'] = pd.to_datetime(df.get('rating_date'), errors='coerce')
    df['rating_date'] = df['rating_date'].fillna(pd.Timestamp('2000-01-01'))
    
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

    # Shared GEOGRAPHY axis (movies + TV + music now populate `country`). A new
    # cross-domain bridge: top origin countries one-hot, everything else -> Other.
    if 'country' not in df.columns:
        df['country'] = np.nan
    _ctry_counts = (df['country'].astype(str).str.split(',').explode().str.strip().value_counts())
    top_countries = [c for c in _ctry_counts.index
                     if c and c.lower() not in ('nan', 'none', 'unknown', '')][:6]
    X_country = pd.get_dummies(df['country'].apply(
        lambda x: next((c.strip() for c in str(x).split(',') if c.strip() in top_countries), 'Other')),
        prefix='ctry')

    print("   Generating text embeddings (length-normalized, genre-templated)...")
    # Cap title/genre and truncate the description to its first 40 words so a movie
    # and a game share the same text shape. This removes the cross-domain text-length
    # confound (movie plots ~214 chars vs game descriptions ~1074) that masked
    # cross-domain transfer in the shared vibe space; see text_norm_transfer_control.
    _gen = df['genre'].fillna('').astype(str).str.replace(r"[\[\]']", "", regex=True).str.slice(0, 80)
    _desc = df['overview'].fillna('').astype(str).apply(lambda s: " ".join(s.split()[:40]))
    df['txt'] = ("Title: " + df['title'].fillna('').astype(str).str.slice(0, 120)
                 + ". Genre: " + _gen + ". " + _desc)
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer.encode(df['txt'].tolist(), normalize_embeddings=True)
    
    # --- DOMAIN CENTROID ALIGNMENT (REMOVED: Handle in trainer per-fold to avoid leakage) ---
    # Centering here on full dataset leaks test info.
    # We'll use the raw embeddings for PCA and let the trainer apply DomainAligner.
    pca = PCA(n_components=10)
    X_text = pd.DataFrame(pca.fit_transform(text_embeddings), columns=[f'pca_{i}' for i in range(10)])

    # Genres
    df['gen_list'] = df['genre'].apply(parse_list).apply(lambda gl: list(set([item for g in gl for item in {'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'], 'Action & Adventure': ['Action', 'Adventure']}.get(g, [g])])))
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['gen_list']), columns=[f"gen_{sanitize_col(c)}" for c in mlb.classes_], index=df.index)
    X_mpaa = pd.get_dummies(df['rated'].apply(categorize_rating), prefix='rated')

    # Music Affinity Features (Cross-Domain Gating)
    print("   Applying Gated Music Affinity features...")
    profile_path = config.MUSIC_MODEL_DIR / "profile.joblib"
    bundle_path = config.MUSIC_MODEL_DIR / "preprocessors.joblib"
    
    if profile_path.exists() and bundle_path.exists():
        profile = joblib.load(profile_path)
        bundle = joblib.load(bundle_path)
        X_music_raw = get_music_affinity_features(df['txt'].tolist(), profile, bundle)
        
        # Apply gate based on transfer matrix
        gate_mask = get_music_gate_mask(df['media_type'])
        X_music = X_music_raw.multiply(gate_mask, axis=0)
    else:
        print("   ⚠️ Music profile not found. Skipping music affinity features.")
        X_music = pd.DataFrame()

    # Mask indicator columns
    mask_cols = [f'has_{d}_feats' for d in ['movie', 'tv', 'game', 'book', 'music']]

    # Final
    X_final = pd.concat([df[num_cols + mask_cols], X_lang, X_country, X_genre, X_mpaa, X_text, X_music], axis=1)
    X_final.columns = [sanitize_col(c) for c in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]

    df['user_rating'] = pd.to_numeric(df['user_rating'], errors='coerce').fillna(0)
    X_final['target_reg'] = df['user_rating']
    X_final['target_ordinal'] = df['user_rating'].map({0.5:0, 1.0:1, 1.5:2, 2.0:3, 2.5:4, 3.0:5, 3.5:6, 4.0:7, 4.5:8, 5.0:9}).fillna(5).astype(int)
    X_final['rating_date'] = df['rating_date']
    X_final['source_id'] = df.index
    X_final['media_type'] = df['media_type']

    X_final.to_csv(config.UNIFIED_TRAINING_DATA_PATH, index=False)
    state = {
        'top_languages': top_3_langs, 
        'mlb_genre': mlb, 
        'sentence_transformer': 'all-MiniLM-L6-v2', 
        'pca': pca, 
        'median_values': df[num_cols].median().to_dict(), 
        'training_columns': [c for c in X_final.columns if c not in ['target_reg', 'target_ordinal', 'source_id', 'media_type', 'rating_date']]
    }
    joblib.dump(state, config.UNIFIED_PREPROCESSOR_STATE)
    print(f"✅ Unified Feature Engineering Complete (Movies+Shows+Games+Books). Shape: {X_final.shape}")

if __name__ == "__main__":
    build_universal_dataset()
