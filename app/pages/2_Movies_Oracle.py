import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import ast
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Import ingestion & engineering tools
from src.movies.ingestion import search_movies_by_query, fetch_movie_details_by_tmdb_id, get_omdb_data
from src.movies.feature_engineering import transform_single_movie, find_similar_movies, explain_similarity

st.set_page_config(page_title="The Oracle", page_icon="🔮", layout="wide")

st.header("🔮 The Oracle")
st.markdown("Predict your rating for any movie using a Stacking Ensemble, NLP Plot Embeddings, and an aggressive Classifier.")

# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

def get_ultimate_movie_metadata(tmdb_id):
    """
    Fetches data from TMDB AND OMDB to ensure the model has 
    Box Office, Rotten Tomatoes, and Metascore data.
    """
    tmdb_data = fetch_movie_details_by_tmdb_id(tmdb_id)
    if tmdb_data.get('processing_status') != 'success':
        return tmdb_data
        
    imdb_id = tmdb_data.get('imdb_id')
    if imdb_id:
        omdb_data = get_omdb_data(imdb_id)
        if omdb_data:
            tmdb_data['rotten_tomatoes_rating'] = omdb_data.get('rotten_tomatoes_rating')
            tmdb_data['metascore'] = omdb_data.get('metascore')
            tmdb_data['awards'] = omdb_data.get('awards')
            tmdb_data['rated'] = omdb_data.get('rated')
            tmdb_data['plot'] = omdb_data.get('plot', '')
            
            if not tmdb_data.get('box_office') and omdb_data.get('box_office'):
                tmdb_data['box_office'] = omdb_data.get('box_office')
                
    return tmdb_data


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    artifacts = {'stacking': None, 'xgb': None, 'svr': None, 'cat': None, 'clf': None, 'state': None}
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    
    try:
        artifacts['stacking'] = joblib.load(ensemble_dir / "stacking_ensemble_regressor.joblib")
        artifacts['xgb'] = joblib.load(ensemble_dir / "xgb_base_regressor.joblib")
        artifacts['svr'] = joblib.load(ensemble_dir / "svr_base_regressor.joblib")
        artifacts['cat'] = joblib.load(ensemble_dir / "catboost_base_regressor.joblib")
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
    
    # Split UI into 3 columns: Poster | Context | Predictions
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
                input_df = transform_single_movie(raw_data, state)
                
                # 1. Base Models & Stacking Predictions
                def round_half(x): return np.round(np.clip(x, 0, 5) * 2) / 2
                
                v_stack = round_half(artifacts['stacking'].predict(input_df)[0])
                v_svr = round_half(artifacts['svr'].predict(input_df)[0])
                v_cat = round_half(artifacts['cat'].predict(input_df)[0])
                v_xgb = round_half(artifacts['xgb'].predict(input_df)[0])
                
                # 2. Classifier Verdict (Tiers)
                verdict = "N/A"
                if artifacts['clf']:
                    try:
                        p_class = artifacts['clf'].predict(input_df)[0]
                        if p_class == 0: verdict = "Skip it ❌"
                        elif p_class == 1: verdict = "Watchable 🍿"
                        elif p_class == 2: verdict = "Masterpiece ⭐"
                    except: pass
                
                st.markdown("### The Verdict")
                
                # Tooltip for individual models
                hover_text = f"SVR: {v_svr:.1f}\nCatBoost: {v_cat:.1f}\nXGBoost: {v_xgb:.1f}"
                
                st.metric(
                    label="Oracle Score (Stacking Ensemble)", 
                    value=f"⭐ {v_stack:.1f} / 5.0", 
                    help=hover_text
                )
                
                st.metric(
                    label="Classifier Tier", 
                    value=verdict, 
                    help="0: Skip (<= 2.5) | 1: Watchable (3.0-3.5) | 2: Masterpiece (>= 4.0)"
                )

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