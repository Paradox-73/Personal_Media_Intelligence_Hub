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
# Import ingestion tools
from src.movies.ingestion import search_movies_by_query, fetch_movie_details_by_tmdb_id, get_omdb_data
from src.movies.feature_engineering import transform_single_movie, find_similar_movies, explain_similarity

st.set_page_config(page_title="The Oracle", page_icon="🔮")

st.header("🔮 The Oracle")
st.markdown("Predict your rating for any movie (Uses TMDB + OMDB Data).")

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
    # 1. Fetch TMDB (Base Data)
    tmdb_data = fetch_movie_details_by_tmdb_id(tmdb_id)
    if tmdb_data.get('processing_status') != 'success':
        return tmdb_data
        
    # 2. Fetch OMDB (Rich Data)
    imdb_id = tmdb_data.get('imdb_id')
    if imdb_id:
        omdb_data = get_omdb_data(imdb_id) # Assumes this exists in ingestion.py
        if omdb_data:
            # Merge OMDB fields that TMDB often misses
            tmdb_data['rotten_tomatoes_rating'] = omdb_data.get('rotten_tomatoes_rating')
            tmdb_data['metascore'] = omdb_data.get('metascore')
            tmdb_data['awards'] = omdb_data.get('awards')
            tmdb_data['rated'] = omdb_data.get('rated') # MPAA Rating
            
            # Prefer OMDB Box Office if TMDB is empty
            if not tmdb_data.get('box_office') and omdb_data.get('box_office'):
                tmdb_data['box_office'] = omdb_data.get('box_office')
                
    return tmdb_data

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        reg = joblib.load(config.MODEL_REGRESSOR)
        state = joblib.load(config.PREPROCESSOR_STATE)
        
        # Classifier is optional
        clf = None
        if os.path.exists(config.MODEL_CLASSIFIER):
            clf = joblib.load(config.MODEL_CLASSIFIER)
            
        return reg, clf, state
    except FileNotFoundError:
        return None, None, None

regressor, classifier, state = load_artifacts()

if regressor is None:
    st.error("❌ Models not found! Run `python src/movies/model_trainer.py` first.")
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
                    poster_url = f"https://image.tmdb.org/t/p/w92{movie['poster_path']}"
                    st.image(poster_url)
            with col_info:
                st.markdown(f"**{movie['title']}** ({movie['year']})")
            with col_button:
                if st.button("Select", key=f"sel_{movie['id']}"):
                    st.session_state['selected_movie_id'] = movie['id']
                    st.session_state['search_results'] = [] 
                    st.rerun() 

# Process selected movie
if st.session_state['selected_movie_id'] and not st.session_state['selected_movie_raw_data']:
    with st.spinner(f"Fetching Ultimate Metadata (TMDB + OMDB)..."):
        # USE THE NEW ULTIMATE FETCHER
        raw_data = get_ultimate_movie_metadata(st.session_state['selected_movie_id'])
        if raw_data.get('processing_status') == 'success':
            st.session_state['selected_movie_raw_data'] = raw_data
        else:
            st.error(f"Failed to fetch details.")
            st.session_state['selected_movie_id'] = None 

if st.session_state['selected_movie_raw_data']:
    raw_data = st.session_state['selected_movie_raw_data']
    st.markdown(f"**Selected:** {raw_data.get('title')} ({raw_data.get('year')})")
    
    if st.button("Consult the Oracle", key="consult_button"):
        with st.spinner("Analyzing Directors, Writers, Box Office, and Vibes..."):
            # 2. PREPROCESS
            input_df = transform_single_movie(raw_data, state)

            # 3. PREDICT
            pred_score = regressor.predict(input_df)[0]
            # Clip to 0-5 scale
            final_score = np.clip(pred_score, 0, 5)
            
            # Classification Verdict
            verdict = "N/A"
            confidence = 0
            if classifier:
                try:
                    pred_probs = classifier.predict_proba(input_df)[0]
                    # Assuming class 1 is "Good"
                    if len(pred_probs) > 1:
                        verdict = "Watch it! 🍿" if pred_probs[1] > 0.5 else "Skip it ❌"
                        confidence = max(pred_probs) * 100
                except: pass

            # 4. DISPLAY
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Rating", f"⭐ {final_score:.1f}/5.0")
            c2.metric("Verdict", verdict)
            if classifier: c3.metric("Confidence", f"{confidence:.1f}%")
            
            # Context
            st.subheader(f"{raw_data.get('title')}")
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                if raw_data.get('poster'):
                    st.image(f"https://image.tmdb.org/t/p/w500{raw_data['poster']}")
            with col_txt:
                dirs = parse_list(raw_data.get('director'))
                st.write(f"**Director:** {', '.join(dirs)}")
                st.write(f"**Rotten Tomatoes:** {raw_data.get('rotten_tomatoes_rating', 'N/A')}")
                st.write(f"**Metascore:** {raw_data.get('metascore', 'N/A')}")
                st.caption(raw_data.get('overview'))

            # 5. SIMILARITY (Fixed Argument Error)
            st.subheader("💡 Analysis based on similar movies:")
            # FIX: Use top_n instead of n
            similar_movies = find_similar_movies(raw_data, input_df, state, top_n=3)

            if similar_movies:
                for i, movie in enumerate(similar_movies):
                    st.write(f"**{i+1}. {movie['title']} ({movie['year']})**")
                    explanation = explain_similarity(raw_data, movie)
                    st.caption(f"Why: {explanation}")
            else:
                st.info("No similar movies found.")