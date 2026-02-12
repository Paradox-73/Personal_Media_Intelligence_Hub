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

# Import ingestion logic
from src.shows.ingestion import tmdb_tv, get_tv_show_metadata_ultimate

# --- PAGE SETUP ---
st.set_page_config(page_title="TV Shows Oracle", page_icon="📺", layout="wide")

st.title("📺 The TV Show Oracle")
st.markdown("""
    **I've analyzed your TV viewing history.** Search for a show you haven't watched, and I'll predict your rating based on its 
    metadata (genres, runtime, network, creators, etc.).
""")

# --- LOAD ARTIFACTS (ROBUST) ---
@st.cache_resource
def load_artifacts():
    reg = None
    clf = None
    preprocessor = None
    
    # 1. Load Regressor (CRITICAL)
    if os.path.exists(config.TV_SHOWS_MODEL_REGRESSOR):
        reg = joblib.load(config.TV_SHOWS_MODEL_REGRESSOR)
    else:
        st.error(f"❌ Critical Error: Regressor model not found at {config.TV_SHOWS_MODEL_REGRESSOR}")
    
    # 2. Load State (CRITICAL)
    if os.path.exists(config.TV_SHOWS_PREPROCESSOR_STATE):
        preprocessor = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    else:
        st.error(f"❌ Critical Error: Preprocessor state not found at {config.TV_SHOWS_PREPROCESSOR_STATE}")

    # 3. Load Classifier (OPTIONAL - For 'Verdict')
    if os.path.exists(config.TV_SHOWS_MODEL_CLASSIFIER):
        try:
            clf = joblib.load(config.TV_SHOWS_MODEL_CLASSIFIER)
        except:
            pass # Ignore if fails, it's optional

    return reg, clf, preprocessor

regressor, classifier, state = load_artifacts()

# Stop if critical files are missing
if regressor is None or state is None:
    st.warning("⚠️ Please run `src/shows/feature_engineering.py` and `src/shows/model_trainer.py` to generate the missing files.")
    st.stop()

# --- HELPER: TRANSFORM LIVE DATA ---
def transform_single_row(raw_data, state):
    """
    Transforms a single dictionary of raw TV show data into the model's feature vector.
    """
    # 1. Create DataFrame
    df = pd.DataFrame([raw_data])
    
    # 2. Extract Artifacts
    mlb_genres = state['mlb_genres']
    median_values = state['median_values']
    cat_cols = state['cat_cols']
    final_cols = state['features_columns']
    
    # 3. Numericals (Fill with Training Medians)
    for col, med_val in median_values.items():
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(med_val)

    # 4. Genres
    def clean_genres(val):
        if isinstance(val, list): return val
        return []
    
    if 'genres' not in df.columns: df['genres'] = [[]]
    df['genres'] = df['genres'].apply(clean_genres)
    
    # Handle case where genres might be empty or new
    try:
        genres_encoded = mlb_genres.transform(df['genres'])
        genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c}" for c in mlb_genres.classes_], index=df.index)
    except:
        # Fallback if genre transformation fails
        genres_df = pd.DataFrame(columns=[f"genre_{c}" for c in mlb_genres.classes_])

    # 5. Categoricals
    df_encoded_cats = pd.get_dummies(df[cat_cols], dummy_na=True)

    # 6. Combine
    X_single = pd.concat([df[list(median_values.keys())], genres_df, df_encoded_cats], axis=1)

    # 7. Align with Training Columns (Fill missing with 0)
    # This ensures the model gets exactly the columns it expects
    X_final = pd.DataFrame(0, index=np.arange(len(df)), columns=final_cols)
    
    # Update values where columns match
    common_cols = list(set(X_final.columns) & set(X_single.columns))
    X_final[common_cols] = X_single[common_cols]
    
    return X_final

# --- SEARCH UI ---
col_search, col_space = st.columns([2, 1])
with col_search:
    query = st.text_input("Search TMDB:", placeholder="e.g. Severance")

if query:
    # 1. Search TMDB (Live)
    with st.spinner(f"Searching TMDB for '{query}'..."):
        try:
            results = tmdb_tv.search(query)
        except Exception as e:
            st.error(f"TMDB Error: {e}")
            results = []
        
    if not results:
        st.warning("No results found.")
    else:
        st.write(f"Found {len(results)} results:")
        st.markdown("---")
        
        # FIX: Convert results to a list before slicing
        display_results = list(results)[:3] 
        
        # Display Results
        for res in display_results:
            r_id = res.id
            r_name = getattr(res, 'name', 'Unknown')
            r_date = getattr(res, 'first_air_date', '')
            r_year = r_date[:4] if r_date else '????'
            r_overview = getattr(res, 'overview', '')
            r_poster = getattr(res, 'poster_path', None)

            c1, c2, c3 = st.columns([1, 3, 1])
            
            with c1:
                if r_poster:
                    st.image(f"https://image.tmdb.org/t/p/w200{r_poster}")
                else:
                    st.empty()
            
            with c2:
                st.subheader(f"{r_name} ({r_year})")
                st.caption(r_overview[:200] + "..." if len(r_overview) > 200 else r_overview)
            
            with c3:
                st.write("") 
                st.write("")
                if st.button(f"🔮 Analyze", key=r_id):
                    
                    # --- DEEP ANALYSIS ---
                    with st.spinner("Fetching full metadata & Running models..."):
                        # 1. Get Ultimate Metadata
                        full_meta = get_tv_show_metadata_ultimate(
                            title=r_name,
                            year=int(r_year) if r_year.isdigit() else None,
                            tmdb_id=r_id
                        )
                        
                        # 2. Transform Data
                        X_input = transform_single_row(full_meta, state)
                        
                        # 3. Predict
                        # Regression (1-10)
                        pred_score = regressor.predict(X_input)[0]
                        pred_score = np.clip(pred_score, 1, 10)
                        
                        # Classification (Like/Dislike) - Optional
                        confidence_str = ""
                        verdict_str = "N/A"
                        
                        if classifier:
                            try:
                                pred_probs = classifier.predict_proba(X_input)[0]
                                # Assuming index 1 is "Like" (True)
                                prob_like = pred_probs[1]
                                is_liked = prob_like > 0.5
                                
                                verdict_str = "Likely to Watch" if is_liked else "Pass"
                                confidence_str = f"{max(pred_probs)*100:.0f}% Conf."
                            except:
                                verdict_str = "Unknown"

                        # 4. Display Prediction
                        st.markdown("---")
                        st.success(f"Analysis Complete for **{r_name}**!")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Predicted Rating", f"{pred_score:.1f} / 10")
                        if classifier:
                            m2.metric("Verdict", verdict_str, delta=confidence_str)
                        else:
                            m2.metric("Verdict", "Model Missing")
                            
                        m3.metric("Seasons", full_meta.get('number_of_seasons', 'N/A'))
                        
                        # Explainability Data
                        st.caption("Prediction Context:")
                        st.json({
                            "Genres": full_meta.get('genres'),
                            "Runtime": f"{full_meta.get('runtime')} mins",
                            "Network": full_meta.get('network'),
                            "Age Rating": full_meta.get('age_rating')
                        })
            
            st.markdown("---")