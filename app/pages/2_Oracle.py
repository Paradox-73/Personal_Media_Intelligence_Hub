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
# Ensure data_ingestion has the updated functions from the previous fix
from src.data_ingestion import fetch_fresh_data, search_movies_by_query, fetch_movie_details_by_tmdb_id
from src.feature_engineering import transform_single_movie, find_similar_movies, explain_similarity

st.set_page_config(page_title="The Oracle", page_icon="🔮")

st.header("🔮 The Oracle")
st.markdown("Predict your rating for any movie.")

# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        reg = joblib.load(config.MODEL_REGRESSOR)
        clf = joblib.load(config.MODEL_CLASSIFIER)
        state = joblib.load(config.PREPROCESSOR_STATE)
        return reg, clf, state
    except FileNotFoundError:
        return None, None, None

regressor, classifier, state = load_artifacts()

if regressor is None:
    st.error("❌ Models not found! Run `python src/model_trainer.py` first.")
    st.stop()

# --- INPUTS (Search Functionality) ---
# Initialize session state variables
if 'search_query_input' not in st.session_state:
    st.session_state['search_query_input'] = ""
if 'search_results' not in st.session_state:
    st.session_state['search_results'] = []
if 'selected_movie_id' not in st.session_state:
    st.session_state['selected_movie_id'] = None
if 'selected_movie_raw_data' not in st.session_state:
    st.session_state['selected_movie_raw_data'] = None
# This will track the last query that actually triggered a search
if 'last_search_executed' not in st.session_state:
    st.session_state['last_search_executed'] = ""

search_col, _ = st.columns([3, 1])
with search_col:
    # Use key for text_input to automatically manage its value in session_state
    search_term = st.text_input("Search for a Movie", value=st.session_state['search_query_input'], key="movie_search_input")

    # Update search_query_input with current value for next run's default
    st.session_state['search_query_input'] = search_term

    # Trigger search only if the search term has changed and is not empty
    if search_term and search_term != st.session_state['last_search_executed']:
        st.session_state['last_search_executed'] = search_term
        st.session_state['search_results'] = search_movies_by_query(search_term)
        st.session_state['selected_movie_id'] = None # Reset selection on new search
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
                st.write(f"Director: {movie['director']}")
            with col_button:
                # Use a unique key for each button to prevent issues
                # Incorporate search_query_input into the key for uniqueness across different search results
                if st.button("Select", key=f"select_movie_{movie['id']}_{st.session_state['search_query_input'].replace(' ', '_')}"):
                    st.session_state['selected_movie_id'] = movie['id']
                    st.session_state['search_results'] = [] # Clear results after selection
                    st.rerun() # Rerun to process selection

# Process selected movie
raw_data = None
if st.session_state['selected_movie_id'] and not st.session_state['selected_movie_raw_data']:
    with st.spinner(f"Fetching details for selected movie (ID: {st.session_state['selected_movie_id']})..."):
        raw_data = fetch_movie_details_by_tmdb_id(st.session_state['selected_movie_id'])
        if raw_data.get('processing_status') == 'success':
            st.session_state['selected_movie_raw_data'] = raw_data
        else:
            st.error(f"Failed to fetch details for movie ID {st.session_state['selected_movie_id']}.")
            st.session_state['selected_movie_id'] = None # Clear selection on failure

if st.session_state['selected_movie_raw_data']:
    raw_data = st.session_state['selected_movie_raw_data']
    st.markdown(f"**Selected Movie:** {raw_data.get('title')} ({raw_data.get('year')})")
    
    if st.button("Consult the Oracle", key="consult_button"):
        with st.spinner("Preparing prediction..."):
            # 2. PREPROCESS (Single Row Inference)
            input_df = transform_single_movie(raw_data, state)

            # 3. PREDICT
            pred_score = regressor.predict(input_df)[0]
            pred_probs = classifier.predict_proba(input_df)[0]
            
            # Round logic (0.5 steps)
            final_score = round(pred_score * 2) / 2
            final_score = max(0.5, min(5.0, final_score)) # Clamp 0.5-5.0
            
            # Verdict
            classes = ["Bad 🤮", "Ok 😐", "Great 🤩"]
            verdict = classes[np.argmax(pred_probs)]
            confidence = max(pred_probs) * 100

            # 4. DISPLAY PREDICTION
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Rating", f"⭐ {final_score}/5.0")
            c2.metric("Verdict", verdict)
            c3.metric("Confidence", f"{confidence:.1f}%")
            
            # Movie Card for Input Movie
            st.subheader(f"{raw_data.get('title')} ({raw_data.get('year')})")
            col_img, col_txt = st.columns([1, 2])
            with col_img:
                if raw_data.get('poster'):
                    # Append base URL for TMDB images
                    poster_url = f"https://image.tmdb.org/t/p/w500{raw_data['poster']}"
                    st.image(poster_url)
            with col_txt:
                dirs = parse_list(raw_data.get('director'))
                acts = parse_list(raw_data.get('actors'))
                genres = parse_list(raw_data.get('genre'))

                st.write(f"**Director:** {', '.join(dirs)}")
                st.write(f"**Cast:** {', '.join(acts[:5])}")
                st.write(f"**Genre:** {', '.join(genres)}")
                st.caption(raw_data.get('overview'))
                
                # --- FIX: Safe Box Office Processing ---
                bo_raw = raw_data.get('box_office', 0)
                try:
                    # Remove '$' and ',' then convert to float
                    bo_val = float(str(bo_raw).replace('$', '').replace(',', ''))
                except (ValueError, TypeError):
                    bo_val = 0
                
                st.markdown(f"**Runtime:** {raw_data.get('runtime')}m | **Box Office:** ${bo_val:,.0f}")

            # 5. DISPLAY SIMILAR MOVIES AND EXPLANATIONS
            st.subheader("💡 You might also like...")
            similar_movies = find_similar_movies(raw_data, input_df, state, n=3) # Get top 3 similar movies

            if similar_movies:
                for i, movie in enumerate(similar_movies):
                    st.write(f"**{i+1}. {movie['title']} ({movie['year']})** - Similarity: {movie['similarity']:.2f}")
                    explanation = explain_similarity(raw_data, movie['raw_data'], state)
                    st.caption(explanation)
            else:
                st.info("No similar movies found in your enriched data.")