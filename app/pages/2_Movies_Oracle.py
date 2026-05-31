import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import ast
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Import ingestion & engineering tools
from src.movies.ingestion import search_movies_by_query, get_movie_metadata
from src.movies.feature_engineering import find_similar_movies, explain_similarity
from sentence_transformers import SentenceTransformer
import re

st.set_page_config(page_title="The Oracle", page_icon="🔮", layout="wide")

# --- PARITY TRANSFORMATION ---
def transform_for_parity(movie_data, state):
    """
    Replicates the EXACT logic in src/movies/predict_ratings.py 
    to ensure parity between batch and real-time predictions.
    """
    df = pd.DataFrame([movie_data])
    
    # 1. Numerical Cleaning
    if 'rotten_tomatoes_rating' in df.columns:
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].astype(str).str.replace('%', '', regex=False)
        df['rotten_tomatoes_rating'] = pd.to_numeric(df['rotten_tomatoes_rating'], errors='coerce')
        
    def clean_money_local(x):
        if pd.isna(x): return 0
        if isinstance(x, (int, float)): return x
        x = str(x).replace('$', '').replace(',', '').strip()
        try: return float(x)
        except: return 0

    df['box_office_clean'] = df['box_office'].apply(clean_money_local)
    df['box_office_log'] = np.log1p(df['box_office_clean'])
    
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    
    df['imdb_rating_100'] = pd.to_numeric(df['imdb_rating'], errors='coerce') * 10
    df['vote_average_100'] = pd.to_numeric(df['vote_average'], errors='coerce') * 10
    
    # Critic scores mean ignores NaNs (matches batch script)
    critic_scores = df[['imdb_rating_100', 'metascore', 'rotten_tomatoes_rating', 'vote_average_100']]
    df['critic_avg_100'] = critic_scores.mean(axis=1)
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5
    
    # FIXED: Ensure imdb_votes is present (matching training and batch fix)
    if 'imdb_votes' not in df.columns and 'vote_count' in df.columns:
        df.rename(columns={'vote_count': 'imdb_votes'}, inplace=True)
    
    # 2. Fill Medians (Numerical alignment)
    for col, med_val in state['median_values'].items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(med_val)
        else:
            df[col] = med_val
            
    # 3. Text Embeddings (String construction must be exact)
    df['text_content'] = "Title: " + df['title'].fillna('Unknown').astype(str) + \
                         ". Directed by: " + df['director'].fillna('Unknown').astype(str) + \
                         ". " + df['plot'].fillna('').astype(str)
    
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = transformer_model.encode(df['text_content'].tolist())
    pca_vec = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(pca_vec, columns=[f'pca_{i}' for i in range(pca_vec.shape[1])], index=df.index)

    # 4. One-Hot Features
    def map_language_local(lang_str):
        if pd.isna(lang_str): return 'Other'
        langs = [l.strip() for l in str(lang_str).split(',')]
        for lang in langs:
            if lang in state['top_languages']: return lang
        return 'Other'
    df['language_cleaned'] = df['language'].apply(map_language_local)
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')
    
    def parse_list_local(x):
        if isinstance(x, list): return x
        try: return ast.literal_eval(str(x))
        except: return []
    df['genre_list'] = df['genre'].apply(parse_list_local)
    genre_encoded = state['mlb_genre'].transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in state['mlb_genre'].classes_], index=df.index)

    def categorize_rating_local(r):
        r = str(r).upper()
        if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
        if 'PG' in r or 'TV-14' in r: return 'Teen'
        return 'General'
    df['mpaa_cat'] = df['rated'].apply(categorize_rating_local)
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')

    # 5. Final Alignment
    X_temp = pd.concat([df[list(state['median_values'].keys())], X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_temp.columns = [re.sub(r"[\[\]<']", "", str(col)) for col in X_temp.columns]
    X_temp = X_temp.loc[:, ~X_temp.columns.duplicated()]

    X_final = pd.DataFrame(0.0, index=df.index, columns=state['training_columns'])
    common_cols = list(set(X_temp.columns) & set(state['training_columns']))
    X_final[common_cols] = X_temp[common_cols]

    return X_final

st.header("🔮 The Oracle")
st.markdown("Predict your rating for any movie using a Stacking Ensemble, Ordinal Probability Engine, and Semantic Text Embeddings.")

# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

def get_ultimate_movie_metadata(tmdb_id):
    """Refactored to use core ingestion logic for parity with batch processing."""
    metadata = get_movie_metadata(None, None, tmdb_id=tmdb_id)
    if not metadata or not metadata.get('title'):
        return {'processing_status': 'error'}
    metadata['processing_status'] = 'success'
    return metadata


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    artifacts = {'stacking': None, 'xgb': None, 'svr': None, 'cat': None, 'ordinal': None, 'clf': None, 'state': None}
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    
    try:
        artifacts['stacking'] = joblib.load(ensemble_dir / "stacking_ensemble_regressor.joblib")
        artifacts['xgb'] = joblib.load(ensemble_dir / "xgb_base_regressor.joblib")
        artifacts['svr'] = joblib.load(ensemble_dir / "svr_base_regressor.joblib")
        artifacts['cat'] = joblib.load(ensemble_dir / "catboost_base_regressor.joblib")
        artifacts['ordinal'] = joblib.load(ensemble_dir / "ordinal_classifier.joblib")
        artifacts['state'] = joblib.load(config.PREPROCESSOR_STATE)
        
        if config.MODEL_CLASSIFIER.exists():
            artifacts['clf'] = joblib.load(config.MODEL_CLASSIFIER)
            
    except FileNotFoundError as e:
        st.warning(f"Could not load all artifacts. Missing: {e.filename}")
        
    return artifacts

artifacts = load_artifacts()
state = artifacts['state']

if not artifacts['stacking']:
    st.error("❌ Stacking model not found! Run `python src/movies/advanced_movie_model_trainer.py` first.")
    st.stop()


# --- INPUTS (Search Functionality) ---
if 'search_query_input' not in st.session_state:
    st.session_state['search_query_input'] = ""
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = []
if 'selected_movie_id' not in st.session_state:
    st.session_state['selected_movie_id'] = None
if 'selected_movie_raw_data' not in st.session_state:
    st.session_state['selected_movie_raw_data'] = None
if 'last_search_executed' not in st.session_state:
    st.session_state['last_search_executed'] = ""

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Movie", value=st.session_state['search_query_input'], key="movie_search_input")
    st.session_state['search_query_input'] = search_term

    if search_term and search_term != st.session_state['last_search_executed']:
        st.session_state['last_search_executed'] = search_term
        st.session_state['search_results'] = search_movies_by_query(search_term)
        st.session_state['selected_movie_id'] = None 
        st.session_state['selected_movie_raw_data'] = None

    if st.session_state['search_results']:
        st.write("Search Results:")
        for movie in st.session_state['search_results']:
            col_img, col_info, col_button = st.columns([1, 4, 1])
            with col_img:
                if movie['poster_path']:
                    st.image(f"https://image.tmdb.org/t/p/w92{movie['poster_path']}")
            with col_info:
                st.markdown(f"**{movie['title']}** ({movie['year']})")
            with col_button:
                if st.button("Select", key=f"sel_{movie['id']}"):
                    st.session_state['selected_movie_id'] = movie['id']
                    st.session_state['search_results'] = [] 
                    st.rerun() 

# --- PROCESSING & UI LAYOUT ---
if st.session_state['selected_movie_id'] and not st.session_state['selected_movie_raw_data']:
    with st.spinner(f"Fetching Ultimate Metadata (TMDB + OMDB)..."):
        raw_data = get_ultimate_movie_metadata(st.session_state['selected_movie_id'])
        if raw_data.get('processing_status') == 'success':
            st.session_state['selected_movie_raw_data'] = raw_data
        else:
            st.error(f"Failed to fetch details.")
            st.session_state['selected_movie_id'] = None 

if st.session_state['selected_movie_raw_data']:
    raw_data = st.session_state['selected_movie_raw_data']
    st.divider()
    
    # Split UI into 3 columns
    c_img, c_meta, c_pred = st.columns([1, 2, 2])
    
    with c_img:
        if raw_data.get('poster'):
            st.image(f"https://image.tmdb.org/t/p/w500{raw_data['poster']}")
            
    with c_meta:
        st.subheader(f"{raw_data.get('title')} ({raw_data.get('year')})")
        dirs = parse_list(raw_data.get('director'))
        st.write(f"**Director:** {', '.join(dirs)}")
        st.write(f"**Rotten Tomatoes:** {raw_data.get('rotten_tomatoes_rating', 'N/A')}")
        st.write(f"**Metascore:** {raw_data.get('metascore', 'N/A')}")
        st.caption(raw_data.get('overview'))
        
        consult_btn = st.button("Consult the Oracle 🔮", type="primary", use_container_width=True)

    with c_pred:
        if consult_btn:
            with st.spinner("Analyzing Embeddings & Ensembles..."):
                # Use parity transformation to match batch script exactly
                input_df = transform_for_parity(raw_data, state)
                
                # 1. Base Models & Stacking Predictions
                def round_half(x): return np.round(np.clip(x, 0, 5) * 2) / 2
                
                v_stack = round_half(artifacts['stacking'].predict(input_df)[0])
                v_svr = round_half(artifacts['svr'].predict(input_df)[0])
                v_cat = round_half(artifacts['cat'].predict(input_df)[0])
                v_xgb = round_half(artifacts['xgb'].predict(input_df)[0])
                
                # 2. Ordinal Classifier Probabilities & Expected Value
                ord_probs = None
                v_ordinal_ev = None
                if artifacts['ordinal']:
                    ord_probs = artifacts['ordinal'].predict_proba(input_df)[0]
                    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
                    v_ordinal_ev = round_half(np.sum(ord_probs * bucket_vals))
                
                # 3. Classifier Verdict (Tiers)
                verdict = "N/A"
                if artifacts['clf']:
                    try:
                        p_class = artifacts['clf'].predict(input_df)[0]
                        if p_class == 0: verdict = "Skip it ❌"
                        elif p_class == 1: verdict = "Watchable 🍿"
                        elif p_class == 2: verdict = "Masterpiece ⭐"
                    except: pass
                
                st.markdown("### The Verdict")
                
                c_stack, c_ord, c_tier = st.columns(3)
                hover_text = f"SVR: {v_svr:.1f}\nCatBoost: {v_cat:.1f}\nXGBoost: {v_xgb:.1f}"
                c_stack.metric(label="Stacking Ensemble", value=f"⭐ {v_stack:.1f}", help=hover_text)
                if v_ordinal_ev is not None:
                    c_ord.metric(label="Ordinal (EV)", value=f"⭐ {v_ordinal_ev:.1f}", help="Expected Value from the probability distribution.")
                c_tier.metric(label="Classifier Tier", value=verdict, help="0: Skip | 1: Watchable | 2: Masterpiece")

                # Confidence Engine Chart
                if ord_probs is not None:
                    st.markdown("#### Confidence Engine (Ordinal Spread)")
                    st.caption("Probability distribution across all 10 rating buckets.")
                    bucket_labels = [f"{i:.1f}" for i in np.arange(0.5, 5.5, 0.5)]
                    prob_df = pd.DataFrame({
                        "Rating Bucket": bucket_labels,
                        "Probability (%)": ord_probs * 100
                    })
                    st.bar_chart(data=prob_df.set_index("Rating Bucket"), use_container_width=True, height=150)


    # --- SIMILAR MOVIES (SEMANTIC TEXT MATCH) ---
    if consult_btn:
        st.divider()
        st.subheader("💡 Semantic & Metadata Matches")
        st.caption("Movies you've already rated with similar plot vibes (Sentence Embeddings), directors, and genres.")
        
        similar_movies = find_similar_movies(raw_data, input_df, state, top_n=3)
        
        if similar_movies:
            cols = st.columns(len(similar_movies))
            for i, (col, movie) in enumerate(zip(cols, similar_movies)):
                with col:
                    st.markdown(f"**{i+1}. {movie['title']} ({movie['year']})**")
                    st.markdown(f"*Similarity Match: {movie['score']*100:.1f}%*")
                    explanation = explain_similarity(raw_data, movie)
                    st.caption(f"Shared: {explanation}")
        else:
            st.info("No similar movies found in your watch history.")