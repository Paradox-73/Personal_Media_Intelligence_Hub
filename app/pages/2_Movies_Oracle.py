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
from src.movies.feature_engineering import (find_similar_movies, explain_similarity,
                                            transform_single_movie)
from src.unified_model.unified_oracle import predict_unified
from sentence_transformers import SentenceTransformer
import re

# Register the custom estimator under __main__ so joblib can unpickle the stacking /
# ordinal models — they reference OrdinalExpectedValueRegressor under __main__ (the
# trainer's entrypoint), which is app/main.py inside Streamlit.
from src.movies.advanced_movie_model_trainer import OrdinalExpectedValueRegressor as _OEVR
setattr(sys.modules['__main__'], 'OrdinalExpectedValueRegressor', _OEVR)
from src.movies.advanced_movie_model_trainer import MeanMetaRegressor as _MMR
setattr(sys.modules['__main__'], 'MeanMetaRegressor', _MMR)
from src.movies.advanced_movie_model_trainer import AnchoredRegressor as _AR
setattr(sys.modules['__main__'], 'AnchoredRegressor', _AR)

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="The Oracle", page_icon="🔮", layout="wide")
apply_mobile_css()

# --- PARITY TRANSFORMATION ---
def transform_for_parity(movie_data, state):
    """Featurise exactly like training.

    This used to be a third hand-maintained copy of the featurisation logic and it
    had drifted badly from the trained model: it built the OLD text recipe
    (Title+Director+Plot, while the model's PCA was fit on the richer
    Title+Genre+Director+Writer+Cast+Tagline+plot+overview), it never produced the
    three *_encoded target-encoding columns at all (so they sat at 0 for every
    Oracle prediction, off the ~3.24 scale the model was trained on), and its list
    parser only understood "['A','B']" so comma-string cells collapsed to one item.
    It now delegates to the single canonical implementation, which the batch
    predictor and the watch-next queue also use.
    """
    return transform_single_movie(movie_data, state)


def align_to_model(X, model):
    """Reindex to whatever column set THIS estimator was actually fit on.

    The ensemble members can be refit at different times; aligning per model
    means a stale member degrades instead of raising "feature names should match".
    """
    m = getattr(model, "clf", model)
    names = getattr(m, "feature_names_in_", None)
    if names is None and hasattr(m, "named_steps"):
        for step in m.named_steps.values():
            if hasattr(step, "feature_names_in_"):
                names = step.feature_names_in_
                break
    if names is None:
        try:
            names = m.get_booster().feature_names
        except Exception:
            names = None
    if names is None and getattr(m, "feature_names_", None):
        names = m.feature_names_
    if names is None:
        # A fitted StackingRegressor exposes no feature_names_in_ of its own, so the
        # old code passed X straight through and any vintage mismatch surfaced as a
        # raw "feature_names mismatch" dump from inside XGBoost. Fall back to the
        # column set its base estimators were actually fit on.
        for attr in ("estimators_", "named_estimators_"):
            members = getattr(m, attr, None)
            if not members:
                continue
            for est in (members.values() if hasattr(members, "values") else members):
                inner = getattr(est, "base_", est)
                n = getattr(inner, "feature_names_in_", None)
                if n is None:
                    try:
                        n = inner.get_booster().feature_names
                    except Exception:  # noqa: BLE001
                        n = None
                if n is not None:
                    names = n
                    break
            if names is not None:
                break
    if names is None:
        return X
    return X.reindex(columns=list(names), fill_value=0)


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

    # Stamp the artifacts' mtimes so a retrain that happens while this app is
    # running can be detected. @st.cache_resource keeps the models loaded for the
    # life of the process, so after `run_pipeline.py` the cached models and the
    # freshly written preprocessor state are different vintages, and the mismatch
    # used to surface as an unreadable XGBoost feature-name dump.
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    stamps = [config.PREPROCESSOR_STATE, ensemble_dir / "stacking_ensemble_regressor.joblib"]
    artifacts['_mtimes'] = tuple(p.stat().st_mtime for p in stamps if p.exists())
    return artifacts


artifacts = load_artifacts()

_ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
_now = tuple(p.stat().st_mtime for p in
             [config.PREPROCESSOR_STATE, _ensemble_dir / "stacking_ensemble_regressor.joblib"]
             if p.exists())
if artifacts.get('_mtimes') and _now != artifacts['_mtimes']:
    st.warning("♻️ The models on disk were retrained after this page loaded. "
               "Reloading them so predictions match the current feature set.")
    load_artifacts.clear()
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
    do_search = st.button("🔍 Search", key="movie_search_btn", use_container_width=True)

    if search_term and (do_search or search_term != st.session_state['last_search_executed']):
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
        if raw_data.get('collection_name'):
            st.write(f"**Franchise:** {raw_data['collection_name']}")
        st.caption(raw_data.get('overview'))

        # TMDB keywords — finer-grained "what it's about" than genre.
        kws = parse_list(raw_data.get('keywords'))
        if kws:
            chips = " ".join(
                f"<span style='background:#2C343F;color:#99AABB;padding:2px 8px;"
                f"border-radius:10px;font-size:0.75em;margin:2px;display:inline-block'>"
                f"{k}</span>" for k in kws[:14])
            st.markdown(f"**Keywords:** {chips}", unsafe_allow_html=True)
        
        consult_btn = st.button("Consult the Oracle 🔮", type="primary", use_container_width=True)

    with c_pred:
        if consult_btn:
            with st.spinner("Analyzing Embeddings & Ensembles..."):
                # Use parity transformation to match batch script exactly
                input_df = transform_for_parity(raw_data, state)
                
                # 1. Base Models & Stacking Predictions
                def round_half(x): return np.round(np.clip(x, 0, 5) * 2) / 2
                
                v_stack = round_half(artifacts['stacking'].predict(align_to_model(input_df, artifacts['stacking']))[0])
                v_svr = round_half(artifacts['svr'].predict(align_to_model(input_df, artifacts['svr']))[0])
                v_cat = round_half(artifacts['cat'].predict(align_to_model(input_df, artifacts['cat']))[0])
                v_xgb = round_half(artifacts['xgb'].predict(align_to_model(input_df, artifacts['xgb']))[0])
                
                # 2. Ordinal Classifier Probabilities & Expected Value
                ord_probs = None
                v_ordinal_ev = None
                if artifacts['ordinal']:
                    ord_probs = artifacts['ordinal'].predict_proba(align_to_model(input_df, artifacts['ordinal']))[0]
                    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
                    v_ordinal_ev = round_half(np.sum(ord_probs * bucket_vals))
                
                # 3. Classifier Verdict (Tiers)
                verdict = "N/A"
                if artifacts['clf']:
                    try:
                        p_class = artifacts['clf'].predict(align_to_model(input_df, artifacts['clf']))[0]
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

                # Unified cross-domain model (separate, 397-feature shared space)
                u = predict_unified(raw_data, 'movie')
                if u is not None:
                    st.metric("🌐 Unified Model (cross-domain)", f"⭐ {np.round(u * 2) / 2:.1f}",
                              help="Independent prediction from the 397-feature Unified Model trained across "
                                   "movies, TV, games, books & music — useful as a cross-domain sanity check "
                                   "against the domain-specific stacking ensemble above.")

                # SHAP Explainability
                import shap
                import matplotlib.pyplot as plt
                
                st.markdown("#### Why this prediction?")
                st.caption("SHAP waterfall showing the top features driving this specific verdict.")
                
                try:
                    # Use the base XGBoost model for explanation
                    explainer = shap.TreeExplainer(artifacts['xgb'])
                    shap_values = explainer(input_df)
                    
                    # Create a SHAP waterfall plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.info(f"Feature explanation currently unavailable for this item. ({e})")

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