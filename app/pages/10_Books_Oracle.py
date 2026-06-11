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

st.set_page_config(page_title="Books Oracle", page_icon="🔮")

st.header("🔮 The Books Oracle")
st.markdown("Predict your rating for any book.")

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
if 'search_query_input_book' not in st.session_state:
    st.session_state['search_query_input_book'] = ""
if 'search_results_book' not in st.session_state:
    st.session_state['search_results_book'] = []
if 'selected_book_id' not in st.session_state:
    st.session_state['selected_book_id'] = None
if 'selected_book_raw_data' not in st.session_state:
    st.session_state['selected_book_raw_data'] = None
if 'last_search_executed_book' not in st.session_state:
    st.session_state['last_search_executed_book'] = ""

search_col, _ = st.columns([3, 1])
with search_col:
    search_term = st.text_input("Search for a Book", value=st.session_state['search_query_input_book'], key="book_search_input")

    st.session_state['search_query_input_book'] = search_term

    if search_term and search_term != st.session_state['last_search_executed_book']:
        st.session_state['last_search_executed_book'] = search_term
        # For books, search_books_by_query is a placeholder, will always return empty
        st.session_state['search_results_book'] = search_books_by_query(search_term) 
        st.session_state['selected_book_id'] = None
        st.session_state['selected_book_raw_data'] = None

    if st.session_state['search_results_book']: # This will likely be empty for now
        st.write("Search Results:")
        for book_item in st.session_state['search_results_book']:
            col_info, col_button = st.columns([4, 1])
            with col_info:
                st.markdown(f"**{book_item['title']}** by {book_item['author']} ({book_item['year']})")
            with col_button:
                if st.button("Select", key=f"select_book_{book_item['id']}_{st.session_state['search_query_input_book'].replace(' ', '_')}"):
                    st.session_state['selected_book_id'] = book_item['id']
                    st.session_state['search_results_book'] = []
                    st.rerun()
    else:
        st.info("Book search functionality is currently a placeholder. Please input data via CSV.")

# Process selected book - for now, we simulate selection or expect manual input for oracle
if not st.session_state['selected_book_raw_data']:
    st.subheader("Or Enter Manually:")
    manual_title = st.text_input("Title", key="book_manual_title")
    manual_author = st.text_input("Author", key="book_manual_author")
    manual_publisher = st.text_input("Publisher", key="book_manual_publisher")
    manual_year = st.number_input("Publication Year", min_value=1000, max_value=2100, value=2000, key="book_manual_year")
    manual_genres = st.text_input("Genres (comma-separated)", key="book_manual_genres")
    
    if st.button("Use Manual Input for Oracle", key="book_manual_button"):
        if manual_title and manual_author and manual_year:
            st.session_state['selected_book_raw_data'] = {
                'title': manual_title,
                'author': [a.strip() for a in manual_author.split(',')] if manual_author else [],
                'publisher': manual_publisher,
                'year': manual_year,
                'genre': [g.strip() for g in manual_genres.split(',')] if manual_genres else []
            }
            st.rerun()
        else:
            st.warning("Please fill in Title, Author, and Publication Year for manual input.")

if st.session_state['selected_book_raw_data']:
    raw_data = st.session_state['selected_book_raw_data']
    st.markdown(f"**Selected Book:** {raw_data.get('title')} by {raw_data.get('author')} ({raw_data.get('year')})")
    
    if st.button("Consult the Oracle", key="consult_button_book"):
        with st.spinner("Preparing prediction..."):
            input_df = transform_single_book(raw_data, state)

            pred_score = regressor.predict(input_df)[0]
            pred_probs = classifier.predict_proba(input_df)[0]
            
            final_score = round(pred_score * 2) / 2
            final_score = max(0.5, min(5.0, final_score))
            
            classes = ["Bad 🤮", "Ok 😐", "Great 🤩"]
            verdict = classes[np.argmax(pred_probs)]
            confidence = max(pred_probs) * 100

            st.divider()
            c1, c2, c3 = st.columns(3)
            
            # Display score with conformal interval if available
            q80 = meta.get('conformal_width_80')
            if q80:
                c1.metric("Predicted Rating", f"⭐ {final_score}/5.0", f"±{q80:.2f} (80% CI)", delta_color="off")
                st.caption(f"Note: This is a sparse domain. The Oracle is 80% confident the true rating is between {max(0.5, final_score-q80):.1f} and {min(5.0, final_score+q80):.1f}.")
            else:
                c1.metric("Predicted Rating", f"⭐ {final_score}/5.0")
                
            c2.metric("Verdict", verdict)
            c3.metric("Confidence", f"{confidence:.1f}%")

            import shap
            import matplotlib.pyplot as plt
            st.markdown("#### Why this prediction?")
            try:
                # Use the classifier for explanation
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer(input_df)
                fig, ax = plt.subplots(figsize=(10, 5))
                # For multi-class, shap_values is a list. We take the class with max prob.
                target_class = np.argmax(pred_probs)
                shap.plots.waterfall(shap_values[:, :, target_class][0], max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.info(f"SHAP explanation unavailable. {e}")

            st.subheader(f"{raw_data.get('title')} ({raw_data.get('year', 'Unknown')})")
            col_info = st.columns([1])[0]
            with col_info:
                authors = parse_list(raw_data.get('author'))
                genres = parse_list(raw_data.get('genre'))

                st.write(f"**Author:** {', '.join(authors)}")
                if raw_data.get('publisher'):
                    st.write(f"**Publisher:** {raw_data.get('publisher')}")
                if genres:
                    st.write(f"**Genre:** {', '.join(genres)}")

            st.subheader("💡 You might also like...")
            similar_books = find_similar_books(raw_data, input_df, state, n=3)

            if similar_books:
                for i, book_item in enumerate(similar_books):
                    st.write(f"**{i+1}. {book_item['title']}** by {book_item['author']} ({book_item['year']}) - Similarity: {book_item['similarity']:.2f}")
                    explanation = explain_similarity_books(raw_data, book_item['raw_data'], state)
                    st.caption(explanation)
            else:
                st.info("No similar books found in your enriched data.")