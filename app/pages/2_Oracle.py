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
from src.data_ingestion import fetch_fresh_data

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

# --- INPUTS ---
col1, col2 = st.columns([3, 1])
with col1:
    title = st.text_input("Movie Title", "Dune")
with col2:
    year = st.number_input("Year", 1900, 2030, 2021)

if st.button("Consult the Oracle"):
    with st.spinner("Fetching data from TMDB & OMDB..."):
        # 1. FETCH DATA (Using the NEW function name)
        # We create a dummy row because the function expects a dictionary
        dummy_row = {'Name': title, 'Year': year, 'Letterboxd URI': '', 'Rating': 0}
        
        raw_data = fetch_fresh_data(dummy_row, {}, {})
        
        if raw_data.get('processing_status') != 'success':
            st.error(f"Movie not found! (TMDB Status: {raw_data.get('processing_status')})")
            st.stop()

        # 2. PREPROCESS (Single Row Inference)
        # We must align this SINGLE movie with the 91 columns the model knows.
        
        # A. Initialize empty row with all training columns set to 0
        training_cols = state['training_columns']
        # Remove targets from list if they exist (just in case)
        feature_cols = [c for c in training_cols if c not in ['target_reg', 'target_class']]
        
        input_df = pd.DataFrame(0, index=[0], columns=feature_cols)
        
        # B. Fill Numeric Features
        num_map = {
            'year': raw_data.get('year'),
            'runtime': raw_data.get('runtime'),
            'imdb_rating': raw_data.get('imdb_rating'),
            'metascore': raw_data.get('metascore'),
            'vote_average': raw_data.get('vote_average'),
            'popularity': raw_data.get('popularity')
        }
        
        # Handle Rotten Tomatoes (string "87%" -> float 87.0)
        rt = raw_data.get('rotten_tomatoes_rating')
        if isinstance(rt, str) and '%' in rt:
            rt = float(rt.replace('%', ''))
        num_map['rotten_tomatoes_rating'] = rt
        
        for col, val in num_map.items():
            if col in input_df.columns:
                # Fill with value, or median if missing
                if val is None or val == '' or str(val).lower() == 'nan':
                    input_df.loc[0, col] = state['median_values'].get(col, 0)
                else:
                    input_df.loc[0, col] = float(val)

        # C. Categorical: Director
        dirs = parse_list(raw_data.get('director'))
        primary_dir = dirs[0] if dirs else 'Unknown'
        # Check validity
        if primary_dir not in state['valid_directors']:
            primary_dir = 'Other_Director'
        
        dir_col = f"dir_{primary_dir}"
        if dir_col in input_df.columns:
            input_df.loc[0, dir_col] = 1

        # D. Categorical: Actor
        acts = parse_list(raw_data.get('actors'))
        primary_act = acts[0] if acts else 'Unknown'
        if primary_act not in state['valid_actors']:
            primary_act = 'Other_Actor'
            
        act_col = f"act_{primary_act}"
        if act_col in input_df.columns:
            input_df.loc[0, act_col] = 1
            
        # E. Categorical: Genres (Multi-hot)
        genres = parse_list(raw_data.get('genre'))
        # Transform using the saved MultiLabelBinarizer would be complex for 1 row alignment
        # Easier: manual set
        for g in genres:
            gen_col = f"gen_{g}"
            if gen_col in input_df.columns:
                input_df.loc[0, gen_col] = 1
                
        # F. Text Features (PCA)
        text = (str(raw_data.get('overview', '')) + " " + str(raw_data.get('tagline', ''))).strip()
        if text:
            tfidf_vec = state['tfidf'].transform([text])
            pca_vec = state['pca'].transform(tfidf_vec.toarray())
            
            for i in range(pca_vec.shape[1]):
                col_name = f"pca_{i}"
                if col_name in input_df.columns:
                    input_df.loc[0, col_name] = pca_vec[0][i]

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

        # 4. DISPLAY
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Rating", f"⭐ {final_score}/5.0")
        c2.metric("Verdict", verdict)
        c3.metric("Confidence", f"{confidence:.1f}%")
        
        # Movie Card
        st.subheader(f"{raw_data.get('title')} ({raw_data.get('year')})")
        col_img, col_txt = st.columns([1, 2])
        with col_img:
            if raw_data.get('poster'):
                # Append base URL for TMDB images
                poster_url = f"https://image.tmdb.org/t/p/w500{raw_data['poster']}"
                st.image(poster_url)
        with col_txt:
            st.write(f"**Director:** {', '.join(dirs)}")
            st.write(f"**Cast:** {', '.join(acts[:5])}")
            st.write(f"**Genre:** {', '.join(genres)}")
            st.caption(raw_data.get('overview'))
            st.markdown(f"**Runtime:** {raw_data.get('runtime')}m | **Box Office:** ${raw_data.get('box_office', 0):,.0f}")