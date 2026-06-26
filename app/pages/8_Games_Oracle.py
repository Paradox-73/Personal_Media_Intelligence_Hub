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
from src.unified_model.unified_oracle import predict_unified

st.set_page_config(page_title="The Oracle", page_icon="🔮", layout="wide")

st.header("🔮 The Games Oracle")
st.markdown("Predict your rating for any game using a Local Regressor, a Tiered Classifier, and Semantic Text Embeddings.")


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
        meta_path = Path(config.GAMES_MODEL_REGRESSOR).parent / "model_meta.joblib"
        meta = joblib.load(meta_path) if meta_path.exists() else {}
        return reg, clf, state, meta
    except FileNotFoundError:
        return None, None, None, {}


regressor, classifier, state, meta = load_artifacts()

if regressor is None:
    st.error("❌ Games models not found! Run ingestion and training for Games first.")
    st.stop()

# --- INPUTS (Search Functionality) ---
for k, v in {'search_query_input_game': "", 'search_results_game': [], 'selected_game_id': None,
             'selected_game_raw_data': None, 'last_search_executed_game': ""}.items():
    st.session_state.setdefault(k, v)

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Game", value=st.session_state['search_query_input_game'], key="game_search_input")
    st.session_state['search_query_input_game'] = search_term

    if search_term and search_term != st.session_state['last_search_executed_game']:
        st.session_state['last_search_executed_game'] = search_term
        st.session_state['search_results_game'] = search_games_by_query(search_term)
        st.session_state['selected_game_id'] = None
        st.session_state['selected_game_raw_data'] = None

    if st.session_state['search_results_game']:
        st.write("Search Results:")
        for game_item in st.session_state['search_results_game']:
            col_img, col_info, col_button = st.columns([1, 4, 1])
            with col_img:
                if game_item.get('poster_path'):
                    st.image(game_item['poster_path'])
            with col_info:
                st.markdown(f"**{game_item['title']}** ({game_item.get('platform', '?')}, {game_item.get('year', '?')})")
            with col_button:
                if st.button("Select", key=f"sel_game_{game_item['id']}"):
                    st.session_state['selected_game_id'] = game_item['id']
                    st.session_state['search_results_game'] = []
                    st.rerun()
    elif search_term:
        st.info("No results — try another title, or use manual entry below.")

# Manual-entry fallback (collapsed so the main flow mirrors the Movies Oracle)
if not st.session_state['selected_game_raw_data']:
    with st.expander("✍️ Or enter details manually"):
        manual_title = st.text_input("Title", key="game_manual_title")
        manual_platform = st.text_input("Platform", key="game_manual_platform")
        manual_developer = st.text_input("Developer", key="game_manual_developer")
        manual_year = st.number_input("Year", min_value=1950, max_value=2100, value=2000, key="game_manual_year")
        manual_genres = st.text_input("Genres (comma-separated)", key="game_manual_genres")
        if st.button("Use Manual Input", key="game_manual_button"):
            if manual_title and manual_platform and manual_year:
                st.session_state['selected_game_raw_data'] = {
                    'title': manual_title, 'name': manual_title, 'platform': manual_platform,
                    'developer': [d.strip() for d in manual_developer.split(',')] if manual_developer else [],
                    'year': manual_year,
                    'genre': [g.strip() for g in manual_genres.split(',')] if manual_genres else [],
                }
                st.rerun()
            else:
                st.warning("Please fill in Title, Platform, and Year.")

# Fetch full details for the selected search result (RAWG)
if st.session_state['selected_game_id'] and not st.session_state['selected_game_raw_data']:
    with st.spinner("Fetching game details from RAWG..."):
        details = fetch_game_details_by_id(st.session_state['selected_game_id'])
        if details:
            st.session_state['selected_game_raw_data'] = details
        else:
            st.error("Could not fetch details for that game.")
            st.session_state['selected_game_id'] = None

# --- PROCESSING & UI LAYOUT ---
if st.session_state['selected_game_raw_data']:
    raw_data = st.session_state['selected_game_raw_data']
    st.divider()

    c_img, c_meta, c_pred = st.columns([1, 2, 2])

    with c_img:
        poster = raw_data.get('poster_path') or raw_data.get('cover')
        if poster:
            st.image(poster)

    with c_meta:
        st.subheader(f"{raw_data.get('title')} ({raw_data.get('year')})")
        st.write(f"**Platform:** {raw_data.get('platform', 'N/A')}")
        devs = parse_list(raw_data.get('developer') or raw_data.get('developers'))
        if devs:
            st.write(f"**Developer:** {', '.join(map(str, devs))}")
        st.write(f"**Metacritic:** {raw_data.get('metacritic', 'N/A')}")
        genres = parse_list(raw_data.get('genre') or raw_data.get('genres'))
        if genres:
            st.write(f"**Genre:** {', '.join(map(str, genres))}")
        st.caption(raw_data.get('description_raw') or raw_data.get('overview') or "")
        consult_btn = st.button("Consult the Oracle 🔮", type="primary", use_container_width=True)

    with c_pred:
        if consult_btn:
            with st.spinner("Analyzing Embeddings & Models..."):
                input_df = transform_single_game(raw_data, state)

                def round_half(x): return max(0.5, min(5.0, np.round(np.clip(x, 0, 5) * 2) / 2))

                final_score = round_half(regressor.predict(input_df)[0])
                pred_probs = classifier.predict_proba(input_df)[0]
                classes = ["Skip it ❌", "Watchable 🎮", "Must-Play ⭐"]
                verdict = classes[int(np.argmax(pred_probs))]
                confidence = float(np.max(pred_probs)) * 100

                st.markdown("### The Verdict")
                c1, c2, c3 = st.columns(3)
                q80 = meta.get('conformal_width_80')
                if q80:
                    c1.metric("Predicted Rating", f"⭐ {final_score:.1f}", f"±{q80:.2f} (80% CI)", delta_color="off")
                else:
                    c1.metric("Predicted Rating", f"⭐ {final_score:.1f}")
                c2.metric("Verdict", verdict)
                c3.metric("Confidence", f"{confidence:.1f}%")
                if q80:
                    st.caption(f"Sparse domain — 80% confident the true rating is between "
                               f"{max(0.5, final_score - q80):.1f} and {min(5.0, final_score + q80):.1f}.")

                # Unified cross-domain model (separate, 397-feature shared space)
                u = predict_unified(raw_data, 'game')
                if u is not None:
                    st.metric("🌐 Unified Model (cross-domain)", f"⭐ {np.round(u * 2) / 2:.1f}",
                              help="Independent prediction from the 397-feature Unified Model trained across "
                                   "movies, TV, games, books & music — a cross-domain sanity check against the "
                                   "local Games model above.")

                # SHAP Explainability
                import shap
                import matplotlib.pyplot as plt
                st.markdown("#### Why this prediction?")
                st.caption("SHAP waterfall showing the top features driving this specific verdict.")
                try:
                    explainer = shap.TreeExplainer(classifier)
                    shap_values = explainer(input_df)
                    target_class = int(np.argmax(pred_probs))
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[:, :, target_class][0], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.info(f"Feature explanation currently unavailable for this item. ({e})")

                # Confidence Engine (tier probability spread)
                st.markdown("#### Confidence Engine (Tier Spread)")
                st.caption("Probability across the three taste tiers.")
                prob_df = pd.DataFrame({"Tier": ["Skip", "Watchable", "Must-Play"],
                                        "Probability (%)": pred_probs * 100})
                st.bar_chart(data=prob_df.set_index("Tier"), use_container_width=True, height=150)

    # --- SIMILAR GAMES (SEMANTIC + METADATA MATCH) ---
    if consult_btn:
        st.divider()
        st.subheader("💡 Semantic & Metadata Matches")
        st.caption("Games you've already rated with similar vibes (Sentence Embeddings), developers, and genres.")
        similar_games = find_similar_games(raw_data, input_df, state, n=3)
        if similar_games:
            cols = st.columns(len(similar_games))
            for i, (col, game_item) in enumerate(zip(cols, similar_games)):
                with col:
                    st.markdown(f"**{i+1}. {game_item['title']} ({game_item.get('platform', '?')}, {game_item.get('year', '?')})**")
                    st.markdown(f"*Similarity Match: {game_item['similarity']*100:.1f}%*")
                    st.caption(f"Shared: {explain_similarity_games(raw_data, game_item['raw_data'], state)}")
        else:
            st.info("No similar games found in your enriched data.")
