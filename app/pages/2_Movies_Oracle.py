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

def predict_rating_with_similarity(target_movie_uri, artifacts, top_n=15):
    """
    Predicts a movie rating based on weighted average of similar rated movies.
    This is for on-the-fly prediction in the Oracle.
    """
    sim_matrix = artifacts.get('similarity_matrix')
    uri_to_idx = artifacts.get('movie_uri_to_idx')
    user_ratings = artifacts.get('user_ratings') # FIX: Correct key

    if sim_matrix is None or uri_to_idx is None or user_ratings is None:
        return None, "Similarity model artifacts are missing or incomplete."

    if target_movie_uri not in uri_to_idx:
         return None, "This movie was not in the set used to build the similarity model, so a prediction cannot be made."

    target_idx = uri_to_idx[target_movie_uri]
    sim_scores = sim_matrix[target_idx]
    
    idx_to_uri = {i: uri for uri, i in uri_to_idx.items()}

    # Calculate weighted average
    numerator = 0
    denominator = 0
    count = 0
    
    # Sort similarities to find top N
    sorted_sim_indices = np.argsort(sim_scores)[::-1]
    
    rated_uris = set(user_ratings.index)

    for idx in sorted_sim_indices:
        if count >= top_n:
            break
        
        # Skip itself
        if idx == target_idx:
            continue

        uri = idx_to_uri.get(idx)
        if uri and uri in rated_uris:
            rating = user_ratings.loc[uri, 'user_rating']
            similarity = sim_scores[idx]
            
            numerator += similarity * rating
            denominator += similarity
            count += 1
            
    if denominator == 0:
        avg_rating = user_ratings['user_rating'].mean()
        return np.round(avg_rating * 2) / 2, "No similar rated movies found; returning average rating."

    prediction = np.round((numerator / denominator) * 2) / 2
    return prediction, f"Based on {count} similar rated movies."


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    artifacts = {'reg': None, 'clf': None, 'state': None, 'sim': None}
    try:
        artifacts['reg'] = joblib.load(config.MODEL_REGRESSOR)
        artifacts['state'] = joblib.load(config.PREPROCESSOR_STATE)
        
        if config.MODEL_CLASSIFIER.exists():
            artifacts['clf'] = joblib.load(config.MODEL_CLASSIFIER)
            
        if config.SIMILARITY_MATRIX.exists():
            artifacts['sim'] = joblib.load(config.SIMILARITY_MATRIX)
            
    except FileNotFoundError as e:
        st.warning(f"Could not load all artifacts. Missing: {e.filename}")
        
    return artifacts

artifacts = load_artifacts()
regressor, classifier, state, similarity_artifacts = artifacts['reg'], artifacts['clf'], artifacts['state'], artifacts['sim']


if regressor is None:
    st.error("❌ Main XGBoost model not found! Run `python src/movies/model_trainer.py` first.")
    st.stop()
if similarity_artifacts is None:
    st.warning("🔮 Similarity model not found. Predictions will be from the main model only. Run `python src/movies/similarity_model_trainer.py`")


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
            # 1. PREPROCESS for XGBoost
            input_df = transform_single_movie(raw_data, state)

            # 2. PREDICT with XGBoost
            pred_score_xgb = regressor.predict(input_df)[0]
            final_score_xgb = np.round(np.clip(pred_score_xgb, 0, 5) * 2) / 2
            
            # 3. PREDICT with Similarity Model
            sim_pred, sim_reason = None, "Not available."
            if similarity_artifacts:
                if raw_data.get('letterboxd_uri'):
                    sim_pred, sim_reason = predict_rating_with_similarity(
                        raw_data['letterboxd_uri'], similarity_artifacts
                    )
                else:
                    sim_reason = "Letterboxd URI missing in movie data, cannot use similarity model."
                    st.warning(sim_reason)
            
            # Classification Verdict
            verdict = "N/A"
            confidence = 0
            if classifier:
                try:
                    pred_probs = classifier.predict_proba(input_df)[0]
                    if len(pred_probs) > 1:
                        verdict = "Watch it! 🍿" if pred_probs[1] > 0.5 else "Skip it ❌"
                        confidence = max(pred_probs) * 100
                except: pass

            # 4. DISPLAY
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("XGBoost Prediction", f"⭐ {final_score_xgb:.1f}/5.0")
            if sim_pred is not None:
                c2.metric("Similarity Prediction", f"🤝 {sim_pred:.1f}/5.0")
                st.caption(f"Similarity model reason: {sim_reason}")
            else:
                c2.metric("Similarity Prediction", "N/A")
                st.caption(f"Similarity model reason: {sim_reason}")

            c3.metric("Verdict", verdict)
            
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

            # 5. SIMILARITY ANALYSIS
            st.subheader("💡 Analysis based on similar movies (XGBoost Features):")
            similar_movies = find_similar_movies(raw_data, input_df, state, top_n=3)

            if similar_movies:
                for i, movie in enumerate(similar_movies):
                    st.write(f"**{i+1}. {movie['title']} ({movie['year']})**")
                    explanation = explain_similarity(raw_data, movie)
                    st.caption(f"Why: {explanation}")
            else:
                st.info("No similar movies found.")
