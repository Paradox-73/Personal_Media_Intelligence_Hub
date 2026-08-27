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
from src.common.parsing import (  # canonical shared parsers
    parse_list, people_key, sanitize_col, clean_money, canon_genres)
# categorize_rating lives in src/common/parsing.py. Three copies of it had drifted;
# two used a bare `'R' in r` test that read 'Not Rated'/'Unrated'/'Approved' as
# Adult, mislabelling 12.4% of the movie library.
from src.common.parsing import categorize_rating, norm_language
from src.common import taste_features as tf
from src.common.text_embedding import encode_pooled, encode_pooled_single

def _movie_header(raw):
    """The per-item prefix repeated on every window. One definition, shared by
    training, single-item inference and the scoring pipeline, so they cannot drift."""
    def g(k):
        v = raw.get(k, '')
        # NaN is not None: `str(float('nan'))` is the literal "nan", and 121 of 1062
        # movies have a NaN tagline, so 11% of the corpus was embedded ending in
        # ". Tagline: nan." while the Oracle -- whose metadata dict returns '' --
        # produced ". Tagline: ." for the same film. _show_header already guards this.
        if v is None or (isinstance(v, float) and v != v):
            return ''
        return str(v)
    return ("Title: " + (g('title') or 'Unknown') +
            ". Genre: " + people_key(raw.get('genre')) +
            ". Directed by: " + (people_key(raw.get('director')) or 'Unknown') +
            ". Written by: " + people_key(raw.get('writer')) +
            ". Starring: " + people_key(raw.get('actors')) +
            ". Tagline: " + g('tagline') + ".")


def _movie_body(raw):
    def g(k):
        v = raw.get(k, '')
        if v is None or (isinstance(v, float) and v != v):
            return ''
        return str(v)
    return (g('plot') + " " + g('overview')).strip()


def _movie_text(raw):
    return f"{_movie_header(raw)} {_movie_body(raw)}"


# --- CONFIG ---
PCA_COMPONENTS = 40  # raised from 25: richer text carries more taste signal in higher comps

# Bayesian smoothing prior for every entity target encoding. Was 10, which on a
# library where the median director has 1-2 credits pinned the encoding to the
# global mean (observed std 0.05-0.13). Measured sweep on movies, 10-fold with
# inner OOF: m=0 0.4769 | m=1 0.4732 | m=2 0.4694 | m=3 0.4765 | m=5 0.4713 |
# m=10 0.4802 | m=20 0.4769.  See reports/crew_te_experiment.json.
TE_PRIOR = 2

# (enriched column, how many names to keep). The cast list is ordered by billing,
# so the top 3 carry the signal; crew columns are short already.
# Leave-one-out rho against user_rating: writer .563, director .460, producer .424,
# cinematographer .418, composer .390, editor .385, production .352.
# `country` is deliberately absent -- it measured rho 0.017, pure noise.
CREW_ENCODE = [
    ("director", 2), ("actors", 3), ("writer", 3), ("composer", 1),
    ("cinematographer", 1), ("editor", 2), ("producer", 3), ("production", 2),
]


def tf_neighbors_cache():
    """TMDB neighbour graph, or {} if it has not been fetched yet.

    Populate with `python -m src.common.tmdb_neighbors`. Absent cache degrades
    gracefully: every neighbour feature falls back to the global mean.
    """
    import json
    p = config.CACHE_DIR / "tmdb_neighbors.json"
    if not p.exists():
        print("   ⚠️ TMDB neighbour cache missing -- run "
              "`python -m src.common.tmdb_neighbors` for the item-item features.")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}

# --- HELPERS ---




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
    # OMDb ships Rotten Tomatoes as a percent STRING ("92%"). Coerce unconditionally.
    # The old guard was `if df['rotten_tomatoes_rating'].dtype == 'object'`, which
    # stopped firing once pandas began inferring an Arrow-backed `str` dtype for text
    # columns. The strings then survived into `critic_scores.mean(axis=1)` below and
    # raised `TypeError: unsupported operand type(s) for +: 'float' and 'str'`, taking
    # the whole movie feature build down. It fails loudly rather than dropping RT
    # quietly, so this is a pandas-upgrade regression, not silent long-running data
    # loss -- under the older dtype the guard fired and RT parsed fine. Coercing
    # without a dtype test means no future dtype change can reintroduce either failure.
    df['rotten_tomatoes_rating'] = pd.to_numeric(
        df['rotten_tomatoes_rating'].astype('string').str.replace('%', '', regex=False),
        errors='coerce')

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
        # Strip thousands separators and stray % before coercing. OMDb returns
        # `imdb_votes` grouped ("7,13,701"), so a bare to_numeric turned 756 of
        # 1,062 movies into NaN and then into the median — 71% of the column was
        # a constant, while transform_single_movie's safe_num() parsed the real
        # value. The model trained on the median and was served the truth.
        # Clean UNCONDITIONALLY -- never gate on dtype. This was `if df[col].dtype
        # == object`, which stopped firing when pandas began inferring an Arrow
        # `str` dtype for text columns, silently RE-introducing the very bug the
        # comment above describes as fixed: 747 of 1,062 imdb_votes values carry
        # thousands separators, bare to_numeric salvaged only 306 of them, and the
        # other 756 were median-filled while transform_single_movie's safe_num()
        # parsed the true value. Same failure mode as the rotten_tomatoes_rating
        # guard above. A no-op on already-numeric columns, so there is no reason
        # to gate it at all.
        df[col] = (df[col].astype('string')
                   .str.replace(',', '', regex=False)
                   .str.replace('%', '', regex=False)
                   # OMDb returns runtime as "87 min" -- 95 of 1062 films (8.9%).
                   # Without stripping the unit those coerce to NaN and collapse onto
                   # the median (116), which is exactly what imdb_votes was doing with
                   # its thousands separators. Strip any trailing non-numeric unit.
                   .str.replace(r'\s*(min|mins|minutes|hrs|h)\.?$', '', regex=True)
                   .str.strip())
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        median_val = numeric_series.median()
        df[col] = numeric_series.fillna(median_val)

    X_num = df[num_cols].reset_index(drop=True)

    # 3. Language One-Hot Encoding
    # Normalise BEFORE picking the top 3. The column mixes TMDB ISO codes with OMDb
    # English names, so 'en' (839) and 'English' (23) became two orthogonal one-hots
    # for one language, while Japanese split 25/6 and never made the top 3 at all --
    # 65 rows fell to lang_Other. The serve path inherits this: an OMDb-only title
    # sets lang_English, a column only 40 training rows ever had.
    df['language'] = df['language'].astype('string').apply(norm_language)
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
    # Richer text: include cast, writer, genre, tagline, overview (previously only
    # Title+Director+Plot). text_lift.py showed this ~doubles the text-only taste signal
    # for movies (crew/cast/theme carry personal preference the crowd scores don't).
    # people_key() so "['Nolan']" and "Nolan" embed as the same sentence.
    # Header + plot/overview are embedded SEPARATELY so any long synopsis is split
    # into sentence windows and read in full, and so the vectors are L2-normalised the
    # same way every other domain's now are. Movie text is short (max 244 tokens today)
    # so nothing was being truncated here -- this is for parity: the unified model and
    # the books/games pipelines pool, and a movie vector has to live in the same space.
    _records = df.to_dict('records')
    df['text_header'] = [_movie_header(r) for r in _records]
    _body = pd.Series([_movie_body(r) for r in _records], index=df.index)
    df['text_content'] = df['text_header'] + " " + _body
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)

    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=PCA_COMPONENTS)
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 5. Genres
    df['genre_list'] = df['genre'].apply(canon_genres)
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
        
        # Flatten creators if it's a list (for directors/actors).
        # ALWAYS go through parse_list. This used to pick a branch by inspecting
        # row 0 only -- if row 0 happened to be comma-form, every bracket-form row
        # ("['Christopher Nolan', 'X']", which is 967 of 1062 director cells) was
        # sent through .str.split(', ') and shredded into keys like "['Christopher".
        # parse_list handles both storage shapes, so the branch was never needed.
        temp_df = train_df[[col, target]].copy()
        temp_df[col] = temp_df[col].apply(parse_list)

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

    # 5.2 Director x Genre Interaction
    # Simplification: Top genre x Director encoding
    df['primary_genre'] = df['genre_list'].apply(lambda x: x[0] if x else 'Unknown')
    # Canonical key (not the raw cell) so a repr-list row and a comma-string row for
    # the same director produce the SAME key here and at inference.
    df['dir_genre_key'] = df['director'].apply(people_key) + "_" + df['primary_genre']

    # 5.3 CREW + neighbour taste block (see src/common/taste_features.py).
    # The old matrix encoded only director and actors, at m=10 -- which on a library
    # where the median entity has 1-2 credits smoothed the encoding onto the global
    # mean and made it worthless. enrich_extended.py had already fetched composer /
    # cinematographer / editor / producer and nothing ever read them; `writer` was
    # never encoded for movies at all despite being the STRONGEST entity signal here
    # (leave-one-out rho 0.563 vs 0.460 for director).
    y_rating = df['user_rating'].astype(float).values
    te_artifacts = {}
    te_frames = {}
    for col, limit in CREW_ENCODE:
        if col not in df.columns:
            continue
        enc, sup, art = tf.oof_entity_encode(df[col].tolist(), y_rating,
                                             m=TE_PRIOR, limit=limit)
        te_frames[f'{col}_encoded'] = enc
        te_frames[f'{col}_n'] = sup
        te_artifacts[col] = art
    # Composite key: pass it pre-wrapped as a one-element list so parse_people does
    # not split "A, B_Action" (a two-director cell) on its comma.
    enc, sup, art = tf.oof_entity_encode([[k] for k in df['dir_genre_key']],
                                         y_rating, m=TE_PRIOR)
    te_frames['dir_genre_encoded'] = enc
    te_artifacts['dir_genre'] = art

    # director_encoded / actors_encoded keep their exact old names -- predict_ratings,
    # the Streamlit app and the unified feature builder all reference them.
    assert 'director_encoded' in te_frames and 'actors_encoded' in te_frames, \
        "CREW_ENCODE must keep director and actors: downstream code depends on these names"

    X_te = pd.DataFrame(te_frames).reset_index(drop=True)

    # 5.4 TMDB item-item neighbours -- the only collaborative signal in the project.
    nb_cache = tf_neighbors_cache()
    tmdb_ids = pd.to_numeric(df.get('tmdb_id'), errors='coerce').tolist()
    nb_frames, nb_artifacts = {}, {}
    for endpoint in ('similar', 'recommendations'):
        lists = tf.load_neighbor_lists(nb_cache, 'movie', tmdb_ids, endpoint)
        enc, sup, art = tf.oof_neighbor_encode(tmdb_ids, lists, y_rating)
        nb_frames[f'nb_{endpoint}_mean'] = enc
        nb_frames[f'nb_{endpoint}_n'] = sup
        nb_artifacts[endpoint] = art
    X_nb = pd.DataFrame(nb_frames).reset_index(drop=True)

    # 5.5 Numerics that were fetched into enriched_data.csv and never read.
    X_extra = tf.movie_numeric_extras(df).reset_index(drop=True)

    # 6. MPAA Rating
    df['mpaa_cat'] = df['rated'].apply(categorize_rating)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # 7. Combine Final Features
    print("   Combining all features...")
    X_final = pd.concat([X_num, X_lang, X_genre, X_te, X_nb, X_extra, X_mpaa, X_text],
                        axis=1)
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
    def get_global_mapping(train_df, col, target='user_rating', m=10, split=True):
        """Saved lookup table for inference. MUST be keyed the same way inference
        looks it up (apply_te -> parse_list -> individual names).

        `split=True` parses the cell into its individual names first. Without that
        parse, .explode() on a column of *strings* is a no-op and the keys end up
        being whole cells ("['Denis Villeneuve']"), which inference never asks for
        -- so every lookup missed and the encoding silently flattened to the mean.
        `split=False` for composite keys (dir_genre) that are looked up whole.
        """
        global_mean = train_df[target].mean()
        temp_df = train_df[[col, target]].copy()
        if split:
            temp_df[col] = temp_df[col].apply(parse_list)
        temp_df = temp_df.explode(col)
        stats = temp_df.groupby(col)[target].agg(['count', 'mean'])
        mapping = (stats['count'] * stats['mean'] + m * global_mean) / (stats['count'] + m)
        return mapping.to_dict(), global_mean

    # Legacy inference maps. `te_artifacts` (built by taste_features with the SAME
    # parser used here) is now the authoritative source; these three are kept so an
    # older joblib state or a caller reading state['target_encodings'] still works.
    dir_map, dir_mean = get_global_mapping(df, 'director', m=TE_PRIOR)
    act_map, act_mean = get_global_mapping(df, 'actors', m=TE_PRIOR)
    dg_map, dg_mean = get_global_mapping(df, 'dir_genre_key', m=TE_PRIOR, split=False)

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
        },
        # Authoritative serve-time artifacts, produced by the same code path that
        # built the training columns. Every entry is {map, counts, mean, m, limit}.
        'te_artifacts': te_artifacts,
        'crew_encode': CREW_ENCODE,
        'te_prior': TE_PRIOR,
        # {endpoint: {"index": {tmdb_id: rating}, "mean": float}}
        'neighbor_artifacts': nb_artifacts,
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

    genres = canon_genres(movie_data.get('genre', []))
    # Only labels the fitted binarizer knows: an unseen one makes transform raise.
    genres = [g for g in genres if g in set(state['mlb_genre'].classes_)]
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

    primary_genre = canon_genres(movie_data.get('genre', []))
    primary_genre = primary_genre[0] if primary_genre else 'Unknown'
    # Composite key: canonicalised like training, and looked up WHOLE (splitting it
    # on commas would ask for "B_Action" instead of "A, B_Action").
    dir_genre_key = people_key(movie_data.get('director')) + "_" + primary_genre
    dg = te.get('dir_genre')
    final_df['dir_genre_encoded'] = (
        dg['map'].get(dir_genre_key, dg['mean']) if dg else 0.0)

    # --- crew / neighbour / numeric block, applied through the SAME helper that
    # built the training columns. Doing it by hand here is exactly how the
    # creator/actor encodings drifted out of sync on the shows side twice.
    arts = state.get('te_artifacts') or {}
    for col, limit in state.get('crew_encode', CREW_ENCODE):
        art = arts.get(col)
        if not art:
            continue
        enc, sup = tf.apply_entity_map(movie_data.get(col), art['map'],
                                       art.get('counts', {}), art['mean'],
                                       art.get('limit', limit))
        if f'{col}_encoded' in final_df.columns:
            final_df[f'{col}_encoded'] = enc
        if f'{col}_n' in final_df.columns:
            final_df[f'{col}_n'] = sup
    dg_art = arts.get('dir_genre')
    if dg_art and 'dir_genre_encoded' in final_df.columns:
        enc, _ = tf.apply_entity_map([dir_genre_key], dg_art['map'],
                                     dg_art.get('counts', {}), dg_art['mean'])
        final_df['dir_genre_encoded'] = enc

    nb_arts = state.get('neighbor_artifacts') or {}
    if nb_arts:
        nb_cache = tf_neighbors_cache()
        tid = movie_data.get('tmdb_id')
        rec = {}
        try:
            rec = nb_cache.get(f"movie:{int(float(tid))}") or {}
        except (TypeError, ValueError):
            rec = {}
        for endpoint, art in nb_arts.items():
            enc, sup = tf.apply_neighbor_map(rec.get(endpoint) or [],
                                             art['index'], art['mean'], self_id=tid)
            if f'nb_{endpoint}_mean' in final_df.columns:
                final_df[f'nb_{endpoint}_mean'] = enc
            if f'nb_{endpoint}_n' in final_df.columns:
                final_df[f'nb_{endpoint}_n'] = sup

    extras = tf.movie_numeric_extras(pd.DataFrame([{
        'revenue': movie_data.get('revenue'), 'budget': movie_data.get('budget'),
        'vote_count': movie_data.get('vote_count'),
        'released': movie_data.get('released'),
        'rating_date': movie_data.get('rating_date'),
    }]))
    for c in extras.columns:
        if c in final_df.columns:
            final_df[c] = float(extras[c].iloc[0])

    # Richer text — MUST match the training recipe in process_features() above,
    # which is now guaranteed by sharing _movie_header/_movie_body rather than by
    # keeping two hand-copied f-strings in sync.
    text_content = _movie_text(movie_data)
    transformer_model = SentenceTransformer(state['sentence_transformer'])
    embedding = encode_pooled_single(_movie_header(movie_data), _movie_body(movie_data),
                                     transformer_model)
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
                'keywords': row.get('keywords', ''),
                'collection_name': row.get('collection_name'),
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

    # Same franchise is the strongest possible "shared" signal — say it first.
    tc, sc = target_movie.get('collection_name'), similar_movie.get('collection_name')
    if tc and sc and str(tc) == str(sc):
        shared.insert(0, f"Franchise ({tc})")

    # TMDB keywords are far more specific than genre ("time loop", "heist",
    # "found family"), so they explain a semantic match much better.
    tk = {k.strip().lower() for k in parse_list(target_movie.get('keywords', []))}
    sk = {k.strip().lower() for k in parse_list(similar_movie.get('keywords', []))}
    common_kw = {k for k in (tk & sk) if k}
    if common_kw:
        shared.append(f"Themes ({', '.join(sorted(common_kw)[:4])})")

    if shared:
        return " & ".join(shared)
    
    return "Similar plot/vibe (Semantic Match)"

if __name__ == "__main__":
    process_features()
