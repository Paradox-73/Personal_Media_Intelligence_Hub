import pandas as pd
import numpy as np
import ast
import re
import joblib
import sys
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common import taste_features as tf
from src.common.text_embedding import encode_pooled, encode_pooled_single
from src.shows.season_data import attach_season_narrative, SEASON_NARRATIVE_COL

def _show_header(raw):
    """The per-item prefix repeated on every window. One definition shared by
    training, the scoring pipeline and the Oracle, which previously built three
    different strings -- the Oracle's dropped Creator and Cast entirely, so a show
    scored live landed in a different region of the space than the same show did in
    training."""
    def g(k, n=None):
        v = raw.get(k, '')
        v = '' if v is None or (isinstance(v, float) and v != v) else str(v)
        return v[:n] if n else v
    return ("Title: " + (g('name') or 'Unknown') +
            ". Network: " + (g('primary_network') or g('network') or 'Unknown') +
            ". Creator: " + g('created_by', 80) +
            ". Cast: " + g('actors', 120) + ".")


def _show_body(raw):
    """Show overview, then the season-by-season narrative if one is attached.

    A TMDB show `overview` is a MEDIAN OF ONE SENTENCE -- the thinnest text in the
    whole project. `season_narrative` (see src/shows/season_data.py) concatenates
    the per-season overviews in order, which for the 69% of rated shows that have
    it raises the median from 33 words to 111 AND gives the encoder something with
    an actual beginning/middle/end to read, rather than a one-line logline.

    Callers MUST have run `attach_season_narrative` on the frame first. If they
    have not, the column is simply absent and this degrades to the old overview-only
    text -- which is silent skew, so every caller is wired to attach it.
    """
    def g(k):
        v = raw.get(k, '')
        return '' if v is None or (isinstance(v, float) and v != v) else str(v).strip()
    return " ".join(x for x in (g('overview'), g(SEASON_NARRATIVE_COL)) if x)


def _show_text(raw):
    return f"{_show_header(raw)} {_show_body(raw)}"


# Smoothing prior for the entity encodings. Movies moved to m=2 (measured best on
# 1062 films). TV keeps m=10 -- the weak prior does NOT transfer to a 173-row domain
# where most creators have a single credit, so a light prior leaves the encoding
# sitting on one other show's rating. Prior sweep, 10 registry folds, deployed stack,
# only the four encoded columns varying:
#     m=2   MAE 0.7023  R2 0.2455      m=10  MAE 0.7168  R2 0.1994
#     m=3   MAE 0.6936  R2 0.2343      m=20  MAE 0.6879  R2 0.2443
#     m=5   MAE 0.6936  R2 0.2219      OFF   MAE 0.6792  R2 0.2406
# Nothing beats dropping the person encodings outright, and the whole spread sits
# inside TV's fold-reshuffle noise (~0.02 MAE), so this stays at the original value
# rather than chasing a winner out of noise. Revisit with a bigger show library.
TE_PRIOR = 10

# TMDB item-item neighbours are OFF for TV. They help movies (1062 rows, a title's
# 20 neighbours contain 2.6 rated shows on average) but the TV library is too small
# and too sparse for them: 60% of shows have ZERO rated `similar` neighbours and the
# mean support is 0.6, so the column is a near-constant with occasional spikes.
# Block ablation on the 10 registry folds with the deployed TV stack:
#     old matrix                       MAE 0.6879  R2 0.2194
#     + crowd block                    MAE 0.6792  R2 0.2480
#     + prodco                         MAE 0.6908  R2 0.2306
#     + recommendations only           MAE 0.6792  R2 0.2705
#     + similar only                   MAE 0.6792  R2 0.2455
#     + crowd + prodco        <-- kept MAE 0.6763  R2 0.2742
#     + crowd + prodco + recs          MAE 0.6936  R2 0.2393
#     + everything                     MAE 0.6936  R2 0.2044
# Each block helps alone; stacking them on 173 rows overfits. Revisit once the
# show library is materially larger.
USE_NEIGHBORS = False

# The tvdb_score / tvmaze_rating / tvmaze_weight / critic_avg_5 crowd block and the
# production-company encoding. Motivated (TV had only 2 crowd columns to movies' 5)
# and it helped in every 10-fold arm -- but it does NOT survive the authoritative
# 50-fold registry benchmark with the deployed stack:
#     original TV matrix                 MAE 0.6561  R2 0.2929
#     m=2  + crowd + prodco              MAE 0.6734  R2 0.2855
#     m=10 + crowd + prodco              MAE 0.6792  R2 0.2630
# Set by the measurement below, not by preference. TV is 173 rows; every arm here
# sits within its ~0.02 fold-reshuffle noise, and the edge-penalty alphas retune per
# configuration (alpha_lo swung 0.296 -> 0.044 -> 0.412), so the honest reading is
# "no demonstrated gain" rather than "the crowd block is harmful". Left OFF because
# the published number should be the best measured one. Flip to True and re-run the
# 50-fold TV benchmark once the show library is large enough to resolve 0.02 MAE.
USE_CROWD_BLOCK = False


def _neighbors_cache():
    """TMDB item-item graph; {} if not fetched (features fall back to the mean)."""
    import json
    p = config.CACHE_DIR / "tmdb_neighbors.json"
    if not p.exists():
        print("   ⚠️ TMDB neighbour cache missing -- run "
              "`python -m src.common.tmdb_neighbors --kind tv`.")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}

def safe_eval(val):
    if pd.isna(val) or val == "": return []
    try:
        if isinstance(val, list): return val
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(val).split(',') if x.strip()]

def get_primary_network(val):
    if pd.isna(val) or val == "": return "Other"
    try:
        if "[" in str(val):
            val_list = ast.literal_eval(str(val))
            if val_list and len(val_list) > 0:
                return val_list[0].strip()
        return str(val).split(',')[0].strip()
    except:
        return "Other"

def run_feature_engineering():
    print("⚙️ STARTING TV SHOW FEATURE ENGINEERING (MiniLM Upgrade)...")

    if not config.TV_SHOWS_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.TV_SHOWS_ENRICHED_DATA_PATH}")
        return

    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    
    # Filter for Training (Must have a user rating)
    df = df[df['user_rating'].notna()].copy()
    print(f"   Input Data: {len(df)} rated shows.")

    # --- 1. Numericals ---
    df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
    
    # Calculate medians BEFORE filling NaNs (Critical for Model State)
    median_vote = df['vote_average'].median()
    if pd.isna(median_vote): median_vote = 7.0 # Fallback
    
    median_year = df['year'].median()
    if pd.isna(median_year): median_year = 2010 # Fallback

    # Fill NaNs
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(median_vote)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(median_year)
    df['season_count'] = pd.to_numeric(df['number_of_seasons'], errors='coerce').fillna(1)

    # Multi-critic: add IMDb as a second crowd score (TV previously used only vote_average)
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce')
    median_imdb = df['imdb_rating'].median()
    if pd.isna(median_imdb): median_imdb = 7.5
    df['imdb_rating'] = df['imdb_rating'].fillna(median_imdb)
    df['imdb_votes_log'] = np.log1p(pd.to_numeric(df['imdb_votes'], errors='coerce').fillna(0))
    # Runtime + episode count (present in enriched, previously unused)
    median_runtime = pd.to_numeric(df['runtime'], errors='coerce').median()
    if pd.isna(median_runtime): median_runtime = 45.0
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(median_runtime)
    median_episodes = pd.to_numeric(df['number_of_episodes'], errors='coerce').median()
    if pd.isna(median_episodes): median_episodes = 10.0
    df['episode_count'] = pd.to_numeric(df['number_of_episodes'], errors='coerce').fillna(median_episodes)
    # Awards -> wins / nominations counts (like Movies)
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)

    # --- 1b. Creator / Cast / Writer target encodings (the Movies-style signal TV lacked) ---
    # Previously creator/cast/writer only lived in the text embedding (weak). Movies gets
    # large signal from leakage-safe, Bayesian-smoothed, out-of-fold target encodings of
    # director/actors; TV's high bias is partly the absence of this. Add the same here.
    from sklearn.model_selection import KFold

    def parse_people(val):
        if pd.isna(val) or str(val).strip() == "":
            return []
        s = str(val).strip()
        if s.startswith('['):
            try:
                lst = ast.literal_eval(s)
                return [str(x).strip() for x in lst if str(x).strip()]
            except (ValueError, SyntaxError):
                pass
        return [x.strip() for x in s.split(',') if x.strip()]

    def target_encode_list(frame, list_col, target='user_rating', m=10):
        """Leakage-safe OOF Bayesian-smoothed target encoding over a list-valued column."""
        global_mean = frame[target].mean()
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        encoded = pd.Series(index=frame.index, dtype=float)
        tmp = frame[[list_col, target]].copy()

        def item_mean(items, mapping, g):
            if not items:
                return g
            return float(np.mean([mapping.get(i, g) for i in items]))

        for tr_idx, va_idx in kf.split(tmp):
            ft = tmp.iloc[tr_idx].explode(list_col)
            stats = ft.groupby(list_col)[target].agg(['count', 'mean'])
            mapping = ((stats['count'] * stats['mean'] + m * global_mean)
                       / (stats['count'] + m)).to_dict()
            encoded.iloc[va_idx] = tmp.iloc[va_idx][list_col].apply(
                lambda x: item_mean(x, mapping, global_mean))
        return encoded.fillna(global_mean)

    def global_te_mapping(frame, list_col, target='user_rating', m=10):
        g = frame[target].mean()
        ex = frame[[list_col, target]].explode(list_col)
        stats = ex.groupby(list_col)[target].agg(['count', 'mean'])
        return ((stats['count'] * stats['mean'] + m * g) / (stats['count'] + m)).to_dict(), g

    df['creator_list'] = df['created_by'].apply(parse_people)
    df['actors_list'] = df['actors'].apply(parse_people)
    df['writer_list'] = df['writer'].apply(parse_people)
    # m=TE_PRIOR (2), not 10. With 173 shows the median creator has one credit, so
    # the old prior smoothed these onto the global mean and the columns did nothing.
    df['creator_encoded'] = target_encode_list(df, 'creator_list', m=TE_PRIOR)
    df['actors_encoded'] = target_encode_list(df, 'actors_list', m=TE_PRIOR)
    df['writer_encoded'] = target_encode_list(df, 'writer_list', m=TE_PRIOR)
    # Production company was fetched (99% coverage) and never encoded; leave-one-out
    # rho 0.352 on the movie side, and TV has far fewer signals to spare.
    df['prodco_list'] = df.get('production_companies',
                               pd.Series('', index=df.index)).apply(parse_people)
    df['prodco_encoded'] = target_encode_list(df, 'prodco_list', m=TE_PRIOR)

    # --- 2. Vibe Features (Network) ---
    df['primary_network'] = df['network'].apply(get_primary_network)
    
    # Determine top networks
    top_networks = df['primary_network'].value_counts().nlargest(10).index.tolist()
    
    df['network_clean'] = df['primary_network'].apply(lambda x: x if x in top_networks else "Other")
    network_dummies = pd.get_dummies(df['network_clean'], prefix='net')

    # Country one-hots (top countries by frequency; previously unused)
    df['primary_country'] = (df['country'].fillna('').astype(str)
                             .str.replace(r"[\[\]'\"<>]", "", regex=True)
                             .str.split(',').str[0].str.strip())
    top_countries = [c for c in df['primary_country'].value_counts().nlargest(8).index
                     if c and c.lower() not in ('nan', 'none', '')]
    df['country_clean'] = df['primary_country'].apply(lambda x: x if x in top_countries else 'Other')
    country_dummies = pd.get_dummies(df['country_clean'], prefix='ctry')

    # Adult Content
    df['is_adult'] = df['age_rating'].apply(lambda x: 1 if str(x) in ['TV-MA', 'R', '18+'] else 0)

    # --- 3. Genres ---
    df['genres_list'] = df['genres'].apply(safe_eval)
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres_list'])
    
    # Create temp dataframe for frequency calculation
    temp_genre_df = pd.DataFrame(genres_encoded, columns=[f"g_{c}" for c in mlb.classes_], index=df.index)
    
    # Remove rare genres to reduce noise
    min_freq = 0.05 * len(df)
    kept_genres = temp_genre_df.columns[temp_genre_df.sum() >= min_freq].tolist()
    genre_df = temp_genre_df[kept_genres]
    
    print(f"   Selected {len(kept_genres)} common genres.")

    # --- 4. Upgraded Text Representation (Sentence Transformers) ---
    print("   Generating MiniLM embeddings for TV shows...")
    # Header + overview embedded separately: sentence-windowed, read in full, and
    # L2-normalised the same way every other domain now is. TV overviews are short so
    # nothing was truncated -- this exists so training, predict_ratings and the Oracle
    # share ONE text recipe instead of three hand-copied ones.
    # Season narrative must be on the frame BEFORE records are built, or _show_body
    # silently falls back to the one-line overview.
    df = attach_season_narrative(df)
    _records = df.to_dict('records')
    df['text_header'] = [_show_header(r) for r in _records]
    _body = pd.Series([_show_body(r) for r in _records], index=df.index)
    df['text_content'] = df['text_header'] + " " + _body

    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)

    print("   Applying PCA (15 components) to embeddings...")
    pca = PCA(n_components=15)
    X_text_pca = pca.fit_transform(text_embeddings)
    pca_cols = [f'pca_{i}' for i in range(15)]
    text_df = pd.DataFrame(X_text_pca, columns=pca_cols, index=df.index)

    # --- 4.5 The crowd block TV never had, plus TMDB item-item neighbours ---
    # Gated: see USE_CROWD_BLOCK at the top of this file for the measurements.
    # Movies get five crowd columns (imdb, metascore, RT, tmdb, critic_avg_5); TV had
    # only imdb_rating (rho 0.551) and vote_average. tvdb_score / tvmaze_rating /
    # tvmaze_weight sat unread in enriched_data.csv at 95-100% coverage. Crowd
    # features are the ones that actually carry this model, so starving TV of them
    # is the likeliest cause of its worst-in-project val R2 (0.26).
    extra_df = (tf.show_numeric_extras(df).reset_index(drop=True)
                if USE_CROWD_BLOCK else pd.DataFrame(index=range(len(df))))

    y_rating = df['user_rating'].astype(float).values
    nb_cache = _neighbors_cache()
    tmdb_ids = pd.to_numeric(df.get('tmdb_id'), errors='coerce').tolist()
    nb_frames, nb_artifacts = {}, {}
    if USE_NEIGHBORS:
        for endpoint in ('similar', 'recommendations'):
            lists = tf.load_neighbor_lists(nb_cache, 'tv', tmdb_ids, endpoint)
            enc, sup, art = tf.oof_neighbor_encode(tmdb_ids, lists, y_rating)
            nb_frames[f'nb_{endpoint}_mean'] = enc
            nb_frames[f'nb_{endpoint}_n'] = sup
            nb_artifacts[endpoint] = art
    nb_df = pd.DataFrame(nb_frames, index=range(len(df))).reset_index(drop=True)

    # --- 5. Combine ---
    feature_cols = ['year', 'vote_average', 'imdb_rating', 'vote_count_log', 'imdb_votes_log',
                    'season_count', 'episode_count', 'runtime', 'is_adult',
                    'total_wins', 'total_nominations',
                    'creator_encoded', 'actors_encoded', 'writer_encoded']
    if USE_CROWD_BLOCK:
        feature_cols.append('prodco_encoded')
    X = pd.concat([df[feature_cols].reset_index(drop=True), extra_df, nb_df,
                   genre_df.reset_index(drop=True),
                   network_dummies.reset_index(drop=True),
                   country_dummies.reset_index(drop=True),
                   text_df.reset_index(drop=True)], axis=1)
    
    # --- 6. Targets & Save Data ---
    y_reg = df['user_rating']
    
    # Classification Target (for legacy compatibility)
    user_median = df['user_rating'].median()
    y_class = (df['user_rating'] >= user_median).astype(int)

    # NEW: Ordinal target mapping (10 buckets: 0 to 9)
    mapping = {0.5: 0, 1.0: 1, 1.5: 2, 2.0: 3, 2.5: 4, 3.0: 5, 3.5: 6, 4.0: 7, 4.5: 8, 5.0: 9}
    y_ord = df['user_rating'].map(mapping).fillna(5).astype(int)

    output = X.copy()
    output['target_reg'] = y_reg
    output['target_class'] = y_class
    output['target_ordinal'] = y_ord
    
    output.to_csv(config.TV_SHOWS_TRAINING_DATA_PATH, index=False)
    
    # --- 7. Save Artifacts ---
    artifacts = {
        'features_columns': X.columns.tolist(),
        'median_values': {
            'year': median_year,
            'vote_average': median_vote,
            'imdb_rating': median_imdb,
            'vote_count_log': df['vote_count_log'].median(),
            'imdb_votes_log': df['imdb_votes_log'].median(),
            'season_count': df['season_count'].median(),
            'episode_count': median_episodes,
            'runtime': median_runtime,
        },
        'top_networks': top_networks,
        'top_countries': top_countries,
        'target_encodings': {
            'creator': global_te_mapping(df, 'creator_list', m=TE_PRIOR)[0],
            'actors': global_te_mapping(df, 'actors_list', m=TE_PRIOR)[0],
            'writer': global_te_mapping(df, 'writer_list', m=TE_PRIOR)[0],
            'prodco': global_te_mapping(df, 'prodco_list', m=TE_PRIOR)[0],
            'global_mean': float(df['user_rating'].mean()),
        },
        # {endpoint: {"index": {tmdb_id: rating}, "mean": float}} -- serve side
        # rebuilds the neighbour means from the same index the training columns used.
        'neighbor_artifacts': nb_artifacts,
        'te_prior': TE_PRIOR,
        'mlb_genres': mlb,
        'kept_genres': kept_genres,
        'sentence_transformer': 'all-MiniLM-L6-v2',
        'pca': pca,
        'pca_cols': pca_cols
    }
    
    config.TV_SHOWS_PREPROCESSOR_STATE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.TV_SHOWS_PREPROCESSOR_STATE)
    
    print(f"✅ Features processed & Artifacts saved.")
    print(f"   Shape: {X.shape}")
    print(f"   Artifacts path: {config.TV_SHOWS_PREPROCESSOR_STATE}")

if __name__ == "__main__":
    run_feature_engineering()