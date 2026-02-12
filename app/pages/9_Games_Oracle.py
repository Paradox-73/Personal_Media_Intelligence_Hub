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
from src.games.ingestion import search_games_by_query, fetch_game_details_by_id
from src.games.feature_engineering import transform_single_game, find_similar_games, explain_similarity_games

st.set_page_config(page_title="Games Oracle", page_icon="🔮")

st.header("🔮 The Games Oracle")
st.markdown("Predict your rating for any game.")

# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        reg = joblib.load(config.GAMES_MODEL_REGRESSOR)
        clf = joblib.load(config.GAMES_MODEL_CLASSIFIER)
        state = joblib.load(config.GAMES_MODEL_PREPROCESSOR_STATE)
        return reg, clf, state
    except FileNotFoundError:
        return None, None, None

regressor, classifier, state = load_artifacts()

if regressor is None:
    st.error("❌ Games models not found! Run ingestion and training for Games first.")
    st.stop()

# --- INPUTS (Search Functionality) ---
if 'search_query_input_game' not in st.session_state:
    st.session_state['search_query_input_game'] = ""
if 'search_results_game' not in st.session_state:
    st.session_state['search_results_game'] = []
if 'selected_game_id' not in st.session_state:
    st.session_state['selected_game_id'] = None
if 'selected_game_raw_data' not in st.session_state:
    st.session_state['selected_game_raw_data'] = None
if 'last_search_executed_game' not in st.session_state:
    st.session_state['last_search_executed_game'] = ""

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Game", value=st.session_state['search_query_input_game'], key="game_search_input")

    st.session_state['search_query_input_game'] = search_term

    if search_term and search_term != st.session_state['last_search_executed_game']:
        st.session_state['last_search_executed_game'] = search_term
        # For games, search_games_by_query is a placeholder, will always return empty
        st.session_state['search_results_game'] = search_games_by_query(search_term) 
        st.session_state['selected_game_id'] = None
        st.session_state['selected_game_raw_data'] = None

    if st.session_state['search_results_game']: # This will likely be empty for now
        st.write("Search Results:")
        for game_item in st.session_state['search_results_game']:
            col_info, col_button = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{game_item['title']}** ({game_item['platform']}, {game_item['year']})")
            with col_button:
                if st.button("Select", key=f"select_game_{game_item['id']}_{st.session_state['search_query_input_game'].replace(' ', '_')}"):
                    st.session_state['selected_game_id'] = game_item['id']
                    st.session_state['search_results_game'] = []
                    st.rerun()
    else:
        st.info("Game search functionality is currently a placeholder. Please input data via CSV.")

# Process selected game - for now, we simulate selection or expect manual input for oracle
if not st.session_state['selected_game_raw_data']:
    st.subheader("Or Enter Manually:")
    manual_title = st.text_input("Title", key="game_manual_title")
    manual_platform = st.text_input("Platform", key="game_manual_platform")
    manual_developer = st.text_input("Developer", key="game_manual_developer")
    manual_year = st.number_input("Year", min_value=1950, max_value=2100, value=2000, key="game_manual_year")
    manual_genres = st.text_input("Genres (comma-separated)", key="game_manual_genres")
    
    if st.button("Use Manual Input for Oracle", key="game_manual_button"):
        if manual_title and manual_platform and manual_year:
            st.session_state['selected_game_raw_data'] = {
                'title': manual_title,
                'platform': manual_platform,
                'developer': [d.strip() for d in manual_developer.split(',')] if manual_developer else [],
                'year': manual_year,
                'genre': [g.strip() for g in manual_genres.split(',')] if manual_genres else []
            }
            st.rerun()
        else:
            st.warning("Please fill in Title, Platform, and Year for manual input.")

if st.session_state['selected_game_raw_data']:
    raw_data = st.session_state['selected_game_raw_data']
    st.markdown(f"**Selected Game:** {raw_data.get('title')} ({raw_data.get('platform')}, {raw_data.get('year')})")
    
    if st.button("Consult the Oracle", key="consult_button_game"):
        with st.spinner("Preparing prediction..."):
            input_df = transform_single_game(raw_data, state)

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
            
            st.subheader(f"{raw_data.get('title')} ({raw_data.get('platform')}, {raw_data.get('year')})")
            col_info = st.columns([1])[0]
            with col_info:
                developers = parse_list(raw_data.get('developer'))
                genres = parse_list(raw_data.get('genre'))

                st.write(f"**Platform:** {raw_data.get('platform')}")
                if developers:
                    st.write(f"**Developer:** {', '.join(developers)}")
                if genres:
                    st.write(f"**Genre:** {', '.join(genres)}")

            st.subheader("💡 You might also like...")
            similar_games = find_similar_games(raw_data, input_df, state, n=3)

            if similar_games:
                for i, game_item in enumerate(similar_games):
                    st.write(f"**{i+1}. {game_item['title']}** ({game_item['platform']}, {game_item['year']}) - Similarity: {game_item['similarity']:.2f}")
                    explanation = explain_similarity_games(raw_data, game_item['raw_data'], state)
                    st.caption(explanation)
            else:
                st.info("No similar games found in your enriched data.")