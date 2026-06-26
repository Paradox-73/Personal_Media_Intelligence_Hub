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
from src.books.ingestion import search_books_by_query, fetch_book_details_by_id
from src.books.feature_engineering import transform_single_book, find_similar_books, explain_similarity_books
from src.unified_model.unified_oracle import predict_unified

st.set_page_config(page_title="The Oracle", page_icon="🔮", layout="wide")

st.header("🔮 The Books Oracle")
st.markdown("Predict your rating for any book using a Local Regressor, a Tiered Classifier, and Semantic Text Embeddings.")


# --- HELPERS ---
def parse_list(x):
    if isinstance(x, list): return x
    try: return ast.literal_eval(str(x))
    except: return []


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        reg = joblib.load(config.BOOKS_MODEL_REGRESSOR)
        clf = joblib.load(config.BOOKS_MODEL_CLASSIFIER)
        state = joblib.load(config.BOOKS_PREPROCESSOR_STATE)
        meta_path = Path(config.BOOKS_MODEL_REGRESSOR).parent / "model_meta.joblib"
        meta = joblib.load(meta_path) if meta_path.exists() else {}
        return reg, clf, state, meta
    except FileNotFoundError:
        return None, None, None, {}


regressor, classifier, state, meta = load_artifacts()

if regressor is None:
    st.error("❌ Books models not found! Run ingestion and training for Books first.")
    st.stop()

# --- INPUTS (Search Functionality) ---
for k, v in {'search_query_input_book': "", 'search_results_book': [], 'selected_book_id': None,
             'selected_book_raw_data': None, 'last_search_executed_book': "", 'selected_book_doc': None}.items():
    st.session_state.setdefault(k, v)

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Book", value=st.session_state['search_query_input_book'], key="book_search_input")
    st.session_state['search_query_input_book'] = search_term

    if search_term and search_term != st.session_state['last_search_executed_book']:
        st.session_state['last_search_executed_book'] = search_term
        st.session_state['search_results_book'] = search_books_by_query(search_term)
        st.session_state['selected_book_id'] = None
        st.session_state['selected_book_raw_data'] = None

    if st.session_state['search_results_book']:
        st.write("Search Results:")
        for book_item in st.session_state['search_results_book']:
            col_img, col_info, col_button = st.columns([1, 4, 1])
            with col_img:
                if book_item.get('poster_path'):
                    st.image(book_item['poster_path'])
            with col_info:
                st.markdown(f"**{book_item['title']}** by {book_item.get('author', '?')} ({book_item.get('year', '?')})")
            with col_button:
                if st.button("Select", key=f"sel_book_{book_item['id']}"):
                    st.session_state['selected_book_id'] = book_item['id']
                    st.session_state['selected_book_doc'] = book_item
                    st.session_state['search_results_book'] = []
                    st.rerun()
    elif search_term:
        st.info("No results — try another title, or use manual entry below.")

# Manual-entry fallback (collapsed so the main flow mirrors the Movies Oracle)
if not st.session_state['selected_book_raw_data']:
    with st.expander("✍️ Or enter details manually"):
        manual_title = st.text_input("Title", key="book_manual_title")
        manual_author = st.text_input("Author", key="book_manual_author")
        manual_publisher = st.text_input("Publisher", key="book_manual_publisher")
        manual_year = st.number_input("Publication Year", min_value=1000, max_value=2100, value=2000, key="book_manual_year")
        manual_genres = st.text_input("Genres (comma-separated)", key="book_manual_genres")
        if st.button("Use Manual Input", key="book_manual_button"):
            if manual_title and manual_author and manual_year:
                st.session_state['selected_book_raw_data'] = {
                    'title': manual_title,
                    'author': [a.strip() for a in manual_author.split(',')] if manual_author else [],
                    'publisher': manual_publisher, 'year': manual_year,
                    'genre': [g.strip() for g in manual_genres.split(',')] if manual_genres else [],
                }
                st.rerun()
            else:
                st.warning("Please fill in Title, Author, and Publication Year.")

# Fetch full details for the selected search result (Open Library / Hardcover)
if st.session_state['selected_book_id'] and not st.session_state['selected_book_raw_data']:
    with st.spinner("Fetching book details..."):
        details = fetch_book_details_by_id(
            st.session_state['selected_book_id'],
            fallback=st.session_state.get('selected_book_doc'))
        if details:
            st.session_state['selected_book_raw_data'] = details
        else:
            st.error("Could not fetch details for that book.")
            st.session_state['selected_book_id'] = None

# --- PROCESSING & UI LAYOUT ---
if st.session_state['selected_book_raw_data']:
    raw_data = st.session_state['selected_book_raw_data']
    st.divider()

    c_img, c_meta, c_pred = st.columns([1, 2, 2])

    with c_img:
        poster = raw_data.get('thumbnail') or raw_data.get('poster_path')
        if poster:
            st.image(poster)

    with c_meta:
        st.subheader(f"{raw_data.get('title')} ({raw_data.get('year', 'Unknown')})")
        authors = parse_list(raw_data.get('author') or raw_data.get('authors'))
        st.write(f"**Author:** {', '.join(map(str, authors)) if authors else 'N/A'}")
        if raw_data.get('publisher'):
            st.write(f"**Publisher:** {raw_data.get('publisher')}")
        st.write(f"**Avg Rating:** {raw_data.get('averageRating', 'N/A')}")
        genres = parse_list(raw_data.get('genre') or raw_data.get('categories'))
        if genres:
            st.write(f"**Genre:** {', '.join(map(str, genres))}")
        st.caption(raw_data.get('description') or raw_data.get('overview') or "")
        consult_btn = st.button("Consult the Oracle 🔮", type="primary", use_container_width=True)

    with c_pred:
        if consult_btn:
            with st.spinner("Analyzing Embeddings & Models..."):
                input_df = transform_single_book(raw_data, state)

                def round_half(x): return max(0.5, min(5.0, np.round(np.clip(x, 0, 5) * 2) / 2))

                final_score = round_half(regressor.predict(input_df)[0])
                pred_probs = classifier.predict_proba(input_df)[0]
                classes = ["Skip it ❌", "Worth a Read 📖", "Must-Read ⭐"]
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
                u = predict_unified(raw_data, 'book')
                if u is not None:
                    st.metric("🌐 Unified Model (cross-domain)", f"⭐ {np.round(u * 2) / 2:.1f}",
                              help="Independent prediction from the 397-feature Unified Model trained across "
                                   "movies, TV, games, books & music — a cross-domain sanity check against the "
                                   "local Books model above.")

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
                prob_df = pd.DataFrame({"Tier": ["Skip", "Worth a Read", "Must-Read"],
                                        "Probability (%)": pred_probs * 100})
                st.bar_chart(data=prob_df.set_index("Tier"), use_container_width=True, height=150)

    # --- SIMILAR BOOKS (SEMANTIC + METADATA MATCH) ---
    if consult_btn:
        st.divider()
        st.subheader("💡 Semantic & Metadata Matches")
        st.caption("Books you've already rated with similar vibes (Sentence Embeddings), authors, and genres.")
        similar_books = find_similar_books(raw_data, input_df, state, n=3)
        if similar_books:
            cols = st.columns(len(similar_books))
            for i, (col, book_item) in enumerate(zip(cols, similar_books)):
                with col:
                    st.markdown(f"**{i+1}. {book_item['title']} ({book_item.get('year', '?')})**")
                    st.markdown(f"*Similarity Match: {book_item['similarity']*100:.1f}%*")
                    st.caption(f"Shared: {explain_similarity_books(raw_data, book_item['raw_data'], state)}")
        else:
            st.info("No similar books found in your enriched data.")
