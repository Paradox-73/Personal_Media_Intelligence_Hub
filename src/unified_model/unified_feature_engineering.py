import pandas as pd
import numpy as np
import joblib
import ast
import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

from src.common.text_embedding import encode_pooled, encode_pooled_single
from src.common import taste_features as tf
from src.common.wikipedia_plots import plot_for
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.parsing import (  # canonical shared parsers
    parse_list, sanitize_col, clean_money, canon_genre, canon_genres,
    genres_in_multiple_domains)
# categorize_rating lives in src/common/parsing.py. Three copies of it had drifted;
# two used a bare `'R' in r` test that read 'Not Rated'/'Unrated'/'Approved' as
# Adult, mislabelling 12.4% of the movie library.
from src.common.parsing import categorize_rating, norm_language, clean_year

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

def _s(v, n=None):
    """NaN/None-safe string, optionally truncated."""
    if v is None or (isinstance(v, float) and v != v):
        return ''
    out = str(v)
    return out[:n] if n else out


def unified_header(meta):
    """The per-item prefix repeated on every embedding window.

    ONE definition for all three paths -- the batch builder, single-item inference
    and the batch scorer in predict_unified_ratings. The scorer used to build its own
    string ("Title: X. Lead Creative: D. <overview>") which carried no Genre and no
    Tags slot at all, so every scored row landed somewhere the fitted PCA had never
    seen. That is the kind of drift this function exists to make impossible.
    """
    g = re.sub(r"[\[\]']", "", _s(meta.get('genre') or meta.get('categories')))[:80]
    t = re.sub(r"[\[\]']", "", _s(meta.get('tags') or meta.get('keywords')))[:80]
    return f"Title: {_s(meta.get('title'), 120)}. Genre: {g}. Tags: {t}."


def unified_body(meta):
    blurb = _s(meta.get('overview') or meta.get('description'))
    mt = _s(meta.get('media_type'))
    title = _s(meta.get('title'))
    # The year keeps remakes from sharing a plot (Ghostbusters 1984/2016).
    year = next((y for y in (clean_year(meta.get(k)) for k in
                             ('year', 'released', 'first_air_date', 'publishedDate'))
                 if y), None)
    return plot_for(mt, title, blurb, year=year) if mt and title else blurb


# --- HELPERS ---
class SkipMusic(Exception):
    """Raised to skip the music branch deliberately (see build_universal_dataset)."""




# (column, how many entities to read per item). `director` is the shared creator
# column every domain is renamed into; `studio` is the publisher/network/label axis.
UNIFIED_ENCODE = [("director", 3), ("studio", 1)]

# Smoothing prior for the entity encodings (see src/common/taste_features.py).
# Matches the books pipeline so the two are comparable.
TE_PRIOR = 2


_ST_CACHE = {}


def _get_st(name):
    """Cache the sentence transformer. It was being re-loaded from disk on every
    single-item call, which made each Oracle prediction take seconds."""
    if name not in _ST_CACHE:
        _ST_CACHE[name] = SentenceTransformer(name)
    return _ST_CACHE[name]


def _present(meta, *keys):
    """First numerically-parseable value among `keys`, else NaN.

    Absent MUST mean NaN, not 0. `build_universal_dataset` median-imputes every
    numeric column, so a book — which carries no metascore / RT / vote_average /
    popularity — trains with those columns at the GLOBAL median (69, 81, 7.24,
    9.18). Reading them as 0 here handed the model a vector no book in training
    ever had and crushed book/game predictions by ~1.7 stars.
    """
    for k in keys:
        v = meta.get(k)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        v = pd.to_numeric(str(v).replace('%', '').replace(',', ''), errors='coerce')
        if pd.notna(v):
            return float(v)
    return np.nan


def transform_single_media(meta, state, media_type='movie'):
    """
    Transforms a single item's metadata into the unified model's input columns.
    The Oracle AND the batch scorer (predict_unified_ratings) both go through here,
    so there is one serving path.

    Every step here must mirror `build_universal_dataset` exactly — the model only
    ever saw vectors that function produced. `scripts/audit_inference_parity.py`
    diffs the two paths on the real library and fails the pipeline if they drift.
    Only the blocks the model reads are built: critic_avg_5, creator/studio
    encodings, the shared genre axis and the aligned text block.
    """
    # Scores — NaN when the domain genuinely has no such field (see _present).
    imdb = _present(meta, 'imdb_rating')
    meta_score = _present(meta, 'metascore')
    rt = _present(meta, 'rotten_tomatoes_rating')
    va = _present(meta, 'vote_average')

    # Normalize Critic Avg. The batch builder uses DataFrame.mean(axis=1), which
    # skips NaN, and computes it BEFORE median-imputation — so a book's value is
    # its Hardcover rating alone (ir_100/20), not that rating diluted by zeros.
    ir_100 = imdb * 10 if media_type in ['movie', 'tv'] else imdb * 20
    _parts = [p for p in (ir_100, meta_score, rt, va * 10) if pd.notna(p)]
    critic_avg = (float(np.mean(_parts)) / 100 * 5) if _parts else np.nan
    if pd.isna(critic_avg):
        critic_avg = (state.get('median_values') or {}).get('critic_avg_5', np.nan)
    data = {'critic_avg_5': critic_avg}

    # Creator / studio entity encodings, applied through the SAME artifacts the batch
    # builder saved. The batch path writes out-of-fold values (target held out of its
    # own row); inference legitimately applies the full-data map.
    #
    # Aliases exist because each Oracle hands us its domain's native field name; the
    # backbone renames them to `director`/`studio`, but single-item callers may not.
    _ALIASES = {
        'director': ('director', 'authors', 'author', 'created_by', 'creators',
                     'developers', 'developer', 'artists', 'creator'),
        'studio': ('studio', 'publisher', 'publishers', 'network', 'networks',
                   'production_companies'),
    }
    _arts = state.get('te_artifacts') or {}
    for col, limit in state.get('unified_encode', []):
        art = _arts.get(col)
        if not art:
            continue
        raw_val = next((meta.get(k) for k in _ALIASES.get(col, (col,))
                        if meta.get(k) not in (None, '', [])), None)
        enc, sup = tf.apply_entity_map(raw_val, art['map'], art.get('counts', {}),
                                       art['mean'], art.get('limit', limit))
        name = 'creator' if col == 'director' else col
        data[f'{name}_encoded'] = enc
        data[f'{name}_n'] = sup

    # NLP — chunk-pooled, genre-templated text through the same `encode_pooled` the
    # builder calls, so the header template and the window pooling cannot drift.
    # `unified_body` needs media_type to find the item's Wikipedia plot; callers pass
    # it as an argument, not in the dict, and without it every inference fell back to
    # the short API blurb (Dune Part Two: 305 characters here vs the builder's 4,185).
    meta = dict(meta)
    meta.setdefault('media_type', media_type)
    emb = encode_pooled_single(unified_header(meta), unified_body(meta),
                               _get_st(state['sentence_transformer']))
    text_vec = np.asarray(emb, dtype=float)
    if state.get('aligner') is not None:
        text_vec = state['aligner'].transform(text_vec, [media_type])
    _tcols = getattr(state.get('aligner'), 'out_cols', None)         or [f'emb_{i}' for i in range(text_vec.shape[1])]
    X_text = pd.DataFrame(text_vec, columns=_tcols)

    # Genres — one call, the same one the batch builder makes.
    genres = canon_genres(meta.get('genre') or meta.get('categories') or [])
    known = set(state['mlb_genre'].classes_)
    X_genre = pd.DataFrame(state['mlb_genre'].transform([[g for g in genres if g in known]]),
                           columns=[f"gen_{sanitize_col(c)}" for c in state['mlb_genre'].classes_])

    X_final = pd.concat([pd.DataFrame([data]).astype(float), X_genre, X_text], axis=1)
    X_final.columns = [sanitize_col(c) for c in X_final.columns]

    # model_columns (written by the trainer) is what the models read: pca_0..9 in
    # place of the CSV's emb_* block. Older states fall back to training_columns.
    _model_cols = state.get('model_columns', state['training_columns'])
    # Never zero-fill: a model column this path does not build is a serving bug
    # (the batch scorer once served 21 of 32 columns as 0 that way).
    missing = [c for c in _model_cols if c not in X_final.columns]
    if missing:
        raise ValueError(f"transform_single_media does not build {len(missing)} model "
                         f"column(s): {missing[:12]}")
    return X_final[_model_cols]

def build_universal_dataset():
    print("🌍 Building Universal Media Dataset (Movies + TV Shows + Games + Books + Music)...")
    
    # 1. Load Movies
    df_m = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_m['media_type'] = 'movie'
    # Shared TAG channel. Every domain contributes its own fine-grained tag
    # vocabulary here so the slot has the same shape everywhere -- filling it for
    # only some domains would let the embedding separate domains by text shape,
    # the exact confound the length-normalisation below was added to remove.
    df_m['tags'] = df_m.get('keywords', '')
    
    # 2. Load TV Shows
    df_s = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=['user_rating'])
    df_s['media_type'] = 'tv'
    # TV carries the thinnest text of any domain here -- a TMDB show overview runs a
    # median of ONE sentence, against ~9 for a game blurb. Appending the per-season
    # overviews in order (src/shows/season_data.py) raises the median from 33 words
    # to 111 for the 69% of rated shows the season sheet covers, and unlike a logline
    # it has a beginning/middle/end the chunk-pooled encoder can actually read.
    # Folded into `overview` so it flows through `unified_body` with no special case,
    # which keeps the batch builder, the batch scorer and transform_single_media on
    # one text recipe.
    try:
        from src.shows.season_data import attach_season_narrative, SEASON_NARRATIVE_COL
        df_s = attach_season_narrative(df_s)
        _sn = df_s[SEASON_NARRATIVE_COL].fillna('').astype(str).str.strip()
        df_s['overview'] = (df_s['overview'].fillna('').astype(str).str.strip()
                            + " " + _sn).str.strip()
        print(f"      TV season narrative attached to {(_sn != '').sum()}/{len(df_s)} shows")
    except Exception as e:
        print(f"      ⚠️ season narrative unavailable ({e}); using show overview only")
    # TMDB keywords + TVDB tags; TVDB's are thematic ("Workplace", "Boss/Employee")
    # where TMDB's are topical, so the two are complementary.
    df_s['tags'] = (df_s.get('keywords', pd.Series('', index=df_s.index)).fillna('')
                    + ", "
                    + df_s.get('tvdb_tags', pd.Series('', index=df_s.index)).fillna('')
                    ).str.strip(' ,')
    df_s = df_s.rename(columns={
        'name': 'title', 
        'created_by': 'director', 
        'genres': 'genre', 
        'age_rating': 'rated'
    })

    # 3. Load Games
    try:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except UnicodeDecodeError:
        df_g = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    
    df_g = df_g.rename(columns={
        'name': 'title',
        'publishers': 'studio',
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
    df_g['tags'] = df_g.get('tags', '')
    df_g['year'] = df_g['released'].apply(clean_year)
    # Convert 'I' to NaN for numeric handling but keep for the universal dataset
    df_g['user_rating'] = pd.to_numeric(df_g['user_rating'], errors='coerce')

    # 4. Load Books
    try:
        df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except UnicodeDecodeError:
        df_b = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    
    df_b = df_b.rename(columns={
        'my_rating': 'user_rating',
        'publisher': 'studio',
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
    # Hardcover's cached_tags carries Mood ("reflective, dark, tense") and Tag
    # ("Plot driven", "Strong Character Development") alongside Genre; ingestion
    # used to read only Genre. src/books/enrich_extended.py now pulls the rest,
    # so Books contributes to the shared tag slot like every other domain.
    df_b['tags'] = (df_b.get('moods', pd.Series('', index=df_b.index)).fillna('')
                    + ", "
                    + df_b.get('hc_tags', pd.Series('', index=df_b.index)).fillna('')
                    ).str.strip(' ,')
    df_b['year'] = df_b['released'].apply(clean_year)
    df_b['user_rating'] = pd.to_numeric(df_b['user_rating'], errors='coerce')

    # 5. Load Music -- DELIBERATELY EXCLUDED from the unified dataset.
    #
    # Music was fully de-rated on 2026-07-11: no star ratings, handled as an
    # unsupervised taste-profile / similarity engine, and the old dual headline
    # "Unified (full pool, +music PU pseudo-labels)" was removed in favour of a
    # single `Unified (rated)` row. Until then this exclusion happened BY ACCIDENT
    # -- a rename collision (duration_ms/mb_length_ms) crashed the music branch and
    # the exception was swallowed as a warning. That bug is fixed below, so the
    # skip is now an explicit decision instead of a silent failure.
    #
    # Do NOT flip this on casually: including music puts 3,837 unrated rows in the
    # build (5,208 total) and refits the shared text PCA over them, which moves
    # `pca_*` for every other domain and invalidates the benchmarks, entity bridges
    # and the pre-registered transfer verdict. The Latent Space Explorer already
    # shows music through its own saved PCA layer, so nothing is lost visually.
    INCLUDE_MUSIC = False
    try:
        if not INCLUDE_MUSIC:
            raise SkipMusic("music is de-rated; excluded from the unified dataset")
        df_mu = pd.read_csv(config.MUSIC_ENRICHED_DATA_PATH)
        
        df_mu = df_mu.rename(columns={
            'name': 'title',
            'artists': 'director',
            'artist_genres': 'genre',
            'rating': 'user_rating',
            'release_year': 'year',
            'popularity': 'imdb_rating',
        })
        df_mu['media_type'] = 'music'
        # lastfm_tags has the best coverage of any music tag field (98.9%).
        df_mu['tags'] = df_mu.get('lastfm_tags', '')
        # The CSV carries BOTH duration_ms (Spotify) and mb_length_ms
        # (MusicBrainz). Renaming mb_length_ms -> duration_ms produced TWO columns
        # of that name, so df_mu['duration_ms'] returned a DataFrame and the whole
        # music load died with "Cannot set a DataFrame with multiple columns to the
        # single column runtime" -- silently dropping all 3,837 tracks from the
        # unified dataset. Coalesce instead: Spotify first, MusicBrainz as fallback.
        _dur = pd.to_numeric(df_mu.get('duration_ms'), errors='coerce')
        if 'mb_length_ms' in df_mu.columns:
            _dur = _dur.fillna(pd.to_numeric(df_mu['mb_length_ms'], errors='coerce'))
        df_mu['duration_ms'] = _dur
        df_mu['runtime'] = _dur.fillna(0) / 60000

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
        # Music was fully de-rated on 2026-07-11: it has no star ratings and is
        # handled as an unsupervised taste-profile / similarity engine, so the CSV
        # carries no rating column at all. It therefore contributes NO rows to the
        # unified RATED dataset by design -- the old "full pool + music
        # pseudo-labels" headline was deliberately removed. Set the column
        # explicitly so this is a stated decision rather than a swallowed KeyError.
        df_mu['user_rating'] = pd.to_numeric(
            df_mu.get('user_rating', pd.Series(np.nan, index=df_mu.index)),
            errors='coerce')
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
        # Unconditional -- see the note in movies/feature_engineering.py: a
        # `dtype == 'object'` gate no longer fires for Arrow-backed `str` columns,
        # and imdb_votes arrives comma-grouped ("7,13,701") for 70% of films.
        df[c] = df[c].astype('string').str.replace(',', '', regex=False)
        numeric_series = pd.to_numeric(df[c], errors='coerce')
        median_val = numeric_series.median() if not numeric_series.isna().all() else 0
        df[c] = numeric_series.fillna(median_val)

    # NLP
    # Normalise BEFORE picking the top 3, or the shared language axis cannot bridge
    # domains at all: TMDB gives ISO codes ('en') and OMDb/Hardcover give names
    # ('English'), so the matrix ended up with lang_en (movies/TV) AND lang_English
    # (books) as two orthogonal columns for ONE language -- the exact opposite of a
    # cross-domain bridge. Hindi was split 'hi'/'Hindi' the same way.
    df['language'] = df['language'].apply(norm_language)
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

    print("   Generating text embeddings (chunk-pooled, genre-templated)...")
    # The description is read IN FULL, not cut to its first 40 words. The old cut was
    # positional: it kept ~100% of a movie plot but ~23% of a game blurb, and the part
    # it kept is exactly where each source puts its boilerplate. `encode_pooled` splits
    # the description into sentence windows that fit the model, embeds each with the
    # same Title/Genre/Tags header, then averages the unit window vectors and
    # re-normalizes. One window and eight both come back unit length, so a short movie
    # plot and a long game blurb stay comparable WITHOUT throwing text away — which is
    # what removes the length confound now.
    if 'tags' not in df.columns:
        df['tags'] = ''
    _records = df.to_dict('records')
    df['txt_header'] = [unified_header(r) for r in _records]
    _desc = pd.Series([unified_body(r) for r in _records], index=df.index)
    # Kept for inspection/debugging: what a single-window item looks like end to end.
    df['txt'] = df['txt_header'] + " " + _desc
    transformer = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['txt_header'].tolist(), _desc.tolist(),
                                    transformer, show_progress_bar=True)
    
    # Persist the FULL 384-d matrix, not just the 10 PCA components the model consumes.
    # PCA-10 keeps the global axes and discards the local neighbourhood structure, which
    # is exactly the structure cross-domain retrieval needs: in 384-d the Scott Pilgrim
    # film's eight nearest neighbours are the show and all six book volumes, and that
    # survives stripping every title token from the text, so it is thematic rather than
    # name matching. Row i here is row i of the feature table (source_id == df.index).
    _emb_path = Path(config.UNIFIED_TRAINING_DATA_PATH).with_name('text_embeddings.npy')
    np.save(_emb_path, text_embeddings.astype(np.float32))
    print(f"      saved full text embeddings -> {_emb_path.name} {text_embeddings.shape}")

    # --- DOMAIN CENTROID ALIGNMENT (REMOVED: Handle in trainer per-fold to avoid leakage) ---
    # Centering here on full dataset leaks test info.
    # We'll use the raw embeddings for PCA and let the trainer apply DomainAligner.
    # THE FULL 384-d EMBEDDING, not a 10-component PCA of it.
    #
    # Compressing to pca_0..pca_9 is why the unified model could not find similar
    # items across domains. PCA keeps the directions of greatest GLOBAL variance and
    # discards local neighbourhood structure -- which is exactly what nearest-neighbour
    # retrieval runs on. Measured on this corpus: in the full 384-d space the Scott
    # Pilgrim film's top neighbours are the TV series and all six book volumes
    # (cos 0.69-0.74, and it survives scrubbing every title token, so it is thematic
    # rather than name matching). In pca_0..9 the same query returns Ted, The Studio
    # and Ferris Bueller's Day Off. 374 of 384 dimensions were being thrown away
    # before the model ever saw the text.
    #
    # Named `emb_*`, not `pca_*`, because they are no longer principal components --
    # and app/pages/12_Latent_Space_Explorer.py already treats `emb_` as a vibe prefix.
    X_text = pd.DataFrame(text_embeddings,
                          columns=[f'emb_{i}' for i in range(text_embeddings.shape[1])],
                          index=df.index)
    pca = None   # kept in `state` as None so old artifacts unpickle predictably

    # --- CREATOR / STUDIO ENTITY ENCODINGS (the shared "who made it" axis) ---
    #
    # Every domain's creator was ALREADY normalised into one `director` column
    # above (movie director / tv created_by / game developers / book authors /
    # music artists) -- and then thrown away. Nothing ever read it, so the unified
    # model could not know that the person who made this also made six other things
    # you rated 5. That is the single largest reason the unified book slice lost to
    # the standalone books model, which has had `authors_encoded` all along.
    #
    # Encoded out-of-fold with the same helper the books/games pipelines use, so the
    # target never leaks into its own row and the smoothing prior handles the long
    # tail of one-off creators.
    print("   Encoding shared creator/studio entities...")
    if 'studio' not in df.columns:
        df['studio'] = ''
    # A show's network / a film's production company is the same KIND of signal as a
    # game publisher or a book imprint, so they share one column.
    for alt in ('production', 'production_companies', 'networks', 'network'):
        if alt in df.columns:
            df['studio'] = df['studio'].fillna('').astype(str).replace('nan', '')
            df['studio'] = df['studio'].where(df['studio'].str.strip() != '',
                                              df[alt].fillna('').astype(str))
    # `creator` is the UNION of everyone credited with making the thing, not just the
    # top-billed one. Scott Pilgrim is the case that forces this: Bryan Lee O'Malley
    # is the books' author and the show's created_by, but on the FILM he is a writer
    # while the director is Edgar Wright. Encoding `director` alone leaves the film
    # outside his creator set and the book->film link never fires.
    if 'writer' in df.columns:
        df['director'] = (df['director'].fillna('').astype(str).str.strip(" []")
                          + ", "
                          + df['writer'].fillna('').astype(str).str.strip(" []")
                          ).str.strip(' ,')

    # NaN, not 0: _fit_entity_map skips unrated rows; a 0 would count as a 0-star vote.
    y_uni = pd.to_numeric(df['user_rating'], errors='coerce').astype(float).values
    te_frames, te_artifacts = {}, {}
    for col, limit in UNIFIED_ENCODE:
        if col not in df.columns:
            continue
        enc, sup, art = tf.oof_entity_encode(df[col].tolist(), y_uni,
                                             m=TE_PRIOR, limit=limit)
        name = 'creator' if col == 'director' else col
        te_frames[f'{name}_encoded'] = enc
        te_frames[f'{name}_n'] = sup
        te_artifacts[col] = art
    X_entity = pd.DataFrame(te_frames, index=df.index)
    print(f"      creator coverage: {(df['director'].fillna('').astype(str).str.strip() != '').mean():.1%}"
          f"   studio coverage: {(df['studio'].fillna('').astype(str).str.strip() != '').mean():.1%}")

    # Genres. `canon_genres` now owns the whole vocabulary problem: it expands TV's
    # compound labels ("Sci-Fi & Fantasy" -> Science Fiction + Fantasy), folds
    # non-English and variant spellings onto one label ("Aventure"/"Aventura" ->
    # Adventure, "Juvenile Fiction" -> Family) and drops non-genre noise that arrives
    # in Hardcover's free-text category field ("Mice", "Spanish", "General").
    df['gen_list'] = df['genre'].apply(canon_genres)

    # Then keep ONLY genres carried by two or more domains. A single-domain label is a
    # domain indicator, not a shared feature: "Fiction" sits on 87% of books and on
    # nothing else, and "Shooter"/"Platformer"/"Arcade" are games-only. Feeding those to
    # a model whose thesis is cross-domain generalisation lets it recover domain
    # identity from the very column meant to bridge domains -- the same failure as the
    # split lang_en / lang_English one-hot. This takes the axis from 183 columns to ~17
    # real ones.
    _shared = genres_in_multiple_domains(
        {mt: g['gen_list'].tolist() for mt, g in df.groupby('media_type')})
    df['gen_list'] = df['gen_list'].apply(lambda gl: [g for g in gl if g in _shared])
    print(f"      shared genre axis: {len(_shared)} genres in >=2 domains "
          f"({', '.join(sorted(_shared))})")

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
    # The full 384-d description embedding and the critic score, PLUS the two blocks
    # that were measured to carry real signal: the creator/studio target encodings and
    # the SHARED genre axis.
    #
    # History, so this is not re-litigated blind. These blocks were dropped on
    # 2026-08-29 because 183 genre one-hots crowded 10 PCA text dims out of a
    # 244-column matrix and several columns encoded domain identity. Two things have
    # changed since: the text side is now 384 columns rather than 10, so it can no
    # longer be crowded out; and the genre axis is filtered to genres appearing in two
    # or more domains, which removes the domain-indicator problem at its source rather
    # than by deleting the whole block. Dropping them cost unified MAE 0.5025 -> 0.5561.
    #
    # What is still NOT restored, deliberately: language, country and MPAA one-hots
    # (near-pure domain indicators -- MPAA exists for film/TV and not for books), and
    # the music-affinity block (music is de-rated, 0 rows).
    #
    # CAVEAT that must travel with any metric computed on this matrix: the
    # creator/studio encodings come from `oof_entity_encode`, which runs its OWN
    # internal KFold unrelated to the frozen 50-fold registry. 48% of train rows get an
    # encoding computed from a test row's target, and the value is computed once and
    # reused across all 50 folds. Every unified MAE/R2 that includes these columns is
    # therefore optimistic by an unmeasured amount. Fixing that means re-encoding inside
    # each registry fold, which is not done here.
    X_final = pd.concat([df[['critic_avg_5']].reset_index(drop=True),
                         X_entity.reset_index(drop=True),
                         X_genre.reset_index(drop=True),
                         X_text.reset_index(drop=True)], axis=1)
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
        'te_artifacts': te_artifacts,
        'unified_encode': UNIFIED_ENCODE,
        'te_prior': TE_PRIOR,
        # Raw creator/studio cells by source_id, so comprehensive_evaluator can
        # re-encode them inside each registry fold from training rows only.
        'entity_raw': {col: df[col].tolist() for col, _ in UNIFIED_ENCODE if col in df.columns},
        'top_languages': top_3_langs,
        # Needed by transform_single_media to rebuild the ctry_* one-hot block the
        # same way this builder does; without it inference emitted an all-zero block.
        'top_countries': top_countries,
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
