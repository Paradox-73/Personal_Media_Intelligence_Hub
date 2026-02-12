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
from src.music.ingestion import search_music_by_query, fetch_music_details_by_id
from src.music.feature_engineering import transform_single_music, find_similar_music, explain_similarity_music

st.set_page_config(page_title="Music Oracle", page_icon="🔮")

st.header("🔮 The Music Oracle")
st.markdown("Predict your rating for any music track.")

# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        reg = joblib.load(config.MUSIC_MODEL_REGRESSOR)
        clf = joblib.load(config.MUSIC_MODEL_CLASSIFIER)
        state = joblib.load(config.MUSIC_PREPROCESSOR_STATE)
        return reg, clf, state
    except FileNotFoundError:
        return None, None, None

regressor, classifier, state = load_artifacts()

if regressor is None:
    st.error("❌ Music models not found! Run ingestion and training for Music first.")
    st.stop()

# --- INPUTS (Search Functionality) ---
if 'search_query_input_music' not in st.session_state:
    st.session_state['search_query_input_music'] = ""
if 'search_results_music' not in st.session_state:
    st.session_state['search_results_music'] = []
if 'selected_music_id' not in st.session_state:
    st.session_state['selected_music_id'] = None
if 'selected_music_raw_data' not in st.session_state:
    st.session_state['selected_music_raw_data'] = None
if 'last_search_executed_music' not in st.session_state:
    st.session_state['last_search_executed_music'] = ""

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Music Track", value=st.session_state['search_query_input_music'], key="music_search_input")

    st.session_state['search_query_input_music'] = search_term

    if search_term and search_term != st.session_state['last_search_executed_music']:
        st.session_state['last_search_executed_music'] = search_term
        # For music, search_music_by_query is a placeholder, will always return empty
        st.session_state['search_results_music'] = search_music_by_query(search_term) 
        st.session_state['selected_music_id'] = None
        st.session_state['selected_music_raw_data'] = None

    if st.session_state['search_results_music']: # This will likely be empty for now
        st.write("Search Results:")
        for music_item in st.session_state['search_results_music']:
            col_info, col_button = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{music_item['title']}** by {music_item['artist']} ({music_item['year']})")
            with col_button:
                if st.button("Select", key=f"select_music_{music_item['id']}_{st.session_state['search_query_input_music'].replace(' ', '_')}"):
                    st.session_state['selected_music_id'] = music_item['id']
                    st.session_state['search_results_music'] = []
                    st.rerun()
    else:
        st.info("Music search functionality is currently a placeholder. Please input data via CSV.")

# Process selected music - for now, we simulate selection or expect manual input for oracle
# Since API search is placeholder, we'll allow manual entry for oracle if no search results.
if not st.session_state['selected_music_raw_data']:
    st.subheader("Or Enter Manually:")
    manual_title = st.text_input("Title")
    manual_artist = st.text_input("Artist")
    manual_year = st.number_input("Year", min_value=1900, max_value=2100, value=2000)
    manual_genres = st.text_input("Genres (comma-separated)")
    
    if st.button("Use Manual Input for Oracle"):
        if manual_title and manual_artist and manual_year:
            st.session_state['selected_music_raw_data'] = {
                'title': manual_title,
                'artist': manual_artist,
                'year': manual_year,
                'genre': [g.strip() for g in manual_genres.split(',')] if manual_genres else []
            }
            st.rerun()
        else:
            st.warning("Please fill in Title, Artist, and Year for manual input.")

if st.session_state['selected_music_raw_data']:
    raw_data = st.session_state['selected_music_raw_data']
    st.markdown(f"**Selected Track:** {raw_data.get('title')} by {raw_data.get('artist')} ({raw_data.get('year')})")
    
    if st.button("Consult the Oracle", key="consult_button_music"):
        with st.spinner("Preparing prediction..."):
            input_df = transform_single_music(raw_data, state)

            pred_score = regressor.predict(input_df)[0]
            pred_probs = classifier.predict_proba(input_df)[0]
            
            final_score = round(pred_score * 2) / 2
            final_score = max(0.5, min(5.0, final_score))
            
            classes = ["Bad 🤮", "Ok 😐", "Great 🤩"]
            verdict = classes[np.argmax(pred_probs)]
            confidence = max(pred_probs) * 100

            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Rating", f"⭐ {final_score}/5.0")
            c2.metric("Verdict", verdict)
            c3.metric("Confidence", f"{confidence:.1f}%")
            
            st.subheader(f"{raw_data.get('title')} by {raw_data.get('artist')} ({raw_data.get('year')})")
            col_info = st.columns([1])[0]
            with col_info:
                artists = parse_list(raw_data.get('artist'))
                genres = parse_list(raw_data.get('genre'))

                st.write(f"**Artist:** {', '.join(artists)}")
                if genres:
                    st.write(f"**Genre:** {', '.join(genres)}")

            st.subheader("💡 You might also like...")
            similar_music = find_similar_music(raw_data, input_df, state, n=3)

            if similar_music:
                for i, music_item in enumerate(similar_music):
                    st.write(f"**{i+1}. {music_item['title']}** by {music_item['artist']} ({music_item['year']}) - Similarity: {music_item['similarity']:.2f}")
                    explanation = explain_similarity_music(raw_data, music_item['raw_data'], state)
                    st.caption(explanation)
            else:
                st.info("No similar music found in your enriched data.")