import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import tmdb_tv, get_tv_show_metadata_ultimate

st.set_page_config(page_title="TV Shows Oracle", page_icon="📺", layout="wide")
st.title("📺 The TV Show Oracle")

# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    if not config.TV_SHOWS_MODEL_REGRESSOR.exists(): return None, None, None
    reg = joblib.load(config.TV_SHOWS_MODEL_REGRESSOR)
    state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    clf = None
    if config.TV_SHOWS_MODEL_CLASSIFIER.exists():
        try: clf = joblib.load(config.TV_SHOWS_MODEL_CLASSIFIER)
        except: pass
    return reg, clf, state

regressor, classifier, state = load_artifacts()
if not regressor:
    st.error("Models missing. Run feature_engineering.py")
    st.stop()

# --- TRANSFORM LOGIC ---
def transform_single_row(raw_data, state):
    # 1. Defaults
    final_cols = state['features_columns']
    X_final = pd.DataFrame(0, index=[0], columns=final_cols)
    
    # 2. Extract Medians (Matches new feature_engineering.py structure)
    medians = state.get('median_values', {})
    
    # 3. Fill Numerics
    try: y = int(raw_data.get('first_air_date', '0')[:4])
    except: y = medians.get('year', 2020)
    
    # Inputs from metadata are usually 0-10 (vote_average), model learns to map to 0-5
    v_avg = float(raw_data.get('vote_average', medians.get('vote_average', 7.0)))
    v_count = float(raw_data.get('vote_count', 0))
    v_count_log = np.log1p(v_count)
    n_seasons = float(raw_data.get('number_of_seasons', 1))

    if 'year' in X_final.columns: X_final['year'] = y
    if 'vote_average' in X_final.columns: X_final['vote_average'] = v_avg
    if 'vote_count_log' in X_final.columns: X_final['vote_count_log'] = v_count_log
    if 'season_count' in X_final.columns: X_final['season_count'] = n_seasons

    # 4. Network & Adult
    raw_net = str(raw_data.get('network', 'Other')).split(',')[0].strip()
    if raw_net in state['top_networks']:
        col = f"net_{raw_net}"
        if col in X_final.columns: X_final[col] = 1
    else:
        if 'net_Other' in X_final.columns: X_final['net_Other'] = 1
        
    age = str(raw_data.get('age_rating', ''))
    if age in ['TV-MA', 'R', '18+'] and 'is_adult' in X_final.columns:
        X_final['is_adult'] = 1

    # 5. Genres
    genres = raw_data.get('genres', [])
    if isinstance(genres, str): genres = [genres]
    
    # Use 'mlb_genres' key
    mlb = state.get('mlb_genres')
    if mlb and hasattr(mlb, 'transform'):
        try:
            gen_vec = mlb.transform([genres])
            for i, cls in enumerate(mlb.classes_):
                col = f"g_{cls}"
                if col in X_final.columns: X_final[col] = gen_vec[0][i]
        except: pass

    # 6. NLP
    tfidf = state.get('tfidf_model')
    if tfidf:
        txt = str(raw_data.get('overview', ''))
        try:
            vec = tfidf.transform([txt]).toarray()[0]
            names = tfidf.get_feature_names_out()
            for i, word in enumerate(names):
                col = f"txt_{word}"
                if col in X_final.columns: X_final[col] = vec[i]
        except: pass

    return X_final

# --- UI ---
query = st.text_input("Search TV Show:", placeholder="e.g. Severance")
if query:
    with st.spinner("Searching..."):
        results = tmdb_tv.search(query)
        
    if results:
        st.write(f"Found {len(results)} results:")
        for res in list(results)[:3]:
            # Layout: Poster | Info | Action
            c1, c2, c3 = st.columns([1, 3, 1])
            
            with c1:
                if res.poster_path:
                    st.image(f"https://image.tmdb.org/t/p/w200{res.poster_path}")
                else:
                    st.empty()
            
            with c2:
                year = getattr(res, 'first_air_date', '????')[:4]
                st.subheader(f"{res.name} ({year})")
                st.caption(getattr(res, 'overview', '')[:200] + "...")
            
            with c3:
                st.write("")
                if st.button("🔮 Predict", key=res.id):
                    # Fetch Ultimate Data (credits, ratings, etc)
                    with st.spinner("Analyzing..."):
                        full_meta = get_tv_show_metadata_ultimate(res.name, getattr(res, 'id', None))
                        
                        X_input = transform_single_row(full_meta, state)
                        pred = regressor.predict(X_input)[0]
                        
                        # --- FIX: Scale 0-5 ---
                        pred = np.clip(pred, 0, 5)
                        
                        st.divider()
                        st.metric("Oracle Rating", f"⭐ {pred:.1f} / 5.0")
                        
                        st.json({
                            "Network": full_meta.get('network'),
                            "Seasons": full_meta.get('number_of_seasons'),
                            "Genres": full_meta.get('genres')
                        })
            st.divider()