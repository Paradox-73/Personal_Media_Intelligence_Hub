import streamlit as st
import joblib
import numpy as np
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.music.model_training import Oracle
from src.unified_model.unified_oracle import predict_unified
import src.music.config as music_config

st.set_page_config(page_title="Music Oracle", page_icon="🎧", layout="wide")

st.title("🎧 The Music Oracle")
st.markdown("Discover how closely a track aligns with your sonic profile and find similar gems in your library.")

@st.cache_resource
def get_oracle():
    try:
        return Oracle()
    except Exception as e:
        st.error(f"Failed to initialize Oracle: {e}")
        return None

oracle = get_oracle()

if oracle is None:
    st.warning("Oracle not ready. Please run the music pipeline and model_training.py first.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔮 Predict Rating", "🧬 Find Similar", "🌱 Discover New"])

with tab1:
    st.subheader("Predict Your Affinity")
    st.markdown("Select a track from your library to see what the Oracle thinks of it.")
    
    # Let user select a track from library. Build a robust id->label map: any NaN
    # name/artist would otherwise make format_func return a float and crash the
    # selectbox ("bad argument type for built-in operation").
    track_options = oracle.df[['track_id', 'name', 'artists']].copy()
    track_options['track_id'] = track_options['track_id'].astype(str)
    track_options['name'] = track_options['name'].fillna('Unknown').astype(str)
    track_options['artists'] = track_options['artists'].fillna('Unknown').astype(str)
    track_options['display'] = track_options['name'] + " — " + track_options['artists']
    track_options = track_options.drop_duplicates(subset='track_id')
    track_ids = track_options['track_id'].tolist()
    label_map = dict(zip(track_options['track_id'], track_options['display']))

    selected_track_id = st.selectbox("Select a Track",
                                     options=track_ids,
                                     format_func=lambda x: label_map.get(str(x), str(x)))
    
    if st.button("Consult the Oracle for Prediction"):
        with st.spinner("Analyzing..."):
            pred = oracle.predict_rating(selected_track_id)
            row = oracle.df[oracle.df['track_id'] == selected_track_id].iloc[0]
            
            st.divider()
            c1, c2 = st.columns(2)
            c1.metric("Predicted Rating", f"{pred:.2f} / 5.0")
            c2.metric("Implicit Library Rating", f"{row['rating']:.2f} / 5.0")

            # Independent cross-domain prediction from the Unified Model, shown
            # separately as a sanity check against the local music affinity model.
            u = predict_unified(row.to_dict(), 'music')
            if u is not None:
                st.metric("🌐 Unified Model (cross-domain)", f"⭐ {np.round(u * 2) / 2:.1f}",
                          help="Independent prediction from the 397-feature Unified Model trained "
                               "across all domains — a cross-domain sanity check against the local "
                               "music affinity model above. (For music the Unified slice is trained "
                               "on PU pseudo-labels, so treat this as a curiosity, not a taste verdict.)")

            if pred > 4.5:
                st.success("A masterpiece in your ears.")
            elif pred > 3.5:
                st.info("Solid rotation material.")
            else:
                st.warning("Maybe just a phase?")

with tab2:
    st.subheader("Similar Tracks in Your Library")
    
    selected_track_id_sim = st.selectbox("Select a Seed Track",
                                         options=track_ids,
                                         format_func=lambda x: label_map.get(str(x), str(x)),
                                         key="sim_select")
    
    n_sim = st.slider("Number of similar tracks", 5, 20, 10)
    
    if st.button("Find Similar Gems"):
        with st.spinner("Calculating sonic similarity..."):
            similar_df = oracle.similar_to(selected_track_id_sim, n=n_sim)
            
            st.write("### Recommended from your library:")
            for _, r in similar_df.iterrows():
                # We can't easily get the 'explanation' string here without modifying Oracle
                # But we can display the basic info
                st.write(f"• **{r['name']}** — {r['artists']}  (Rating: {r['rating']:.2f})")

with tab3:
    st.subheader("Discover New Music")
    st.markdown("Uses ReccoBeats to find tracks outside your library based on seeds.")
    
    seed_ids_input = st.text_input("Enter Spotify Track IDs (comma separated)", 
                                   placeholder="e.g., 3n3Ppam7vgaVa1iaRUc9Lp, 7ouMYWpwJ422jRcDASZB7P")
    
    n_disco = st.slider("Number of tracks to discover", 5, 30, 15)
    
    if st.button("Discover New Sounds"):
        if not seed_ids_input:
            st.error("Please enter at least one Spotify Track ID.")
        else:
            with st.spinner("Scanning the musical multiverse..."):
                seeds = [s.strip() for s in seed_ids_input.split(',')]
                discovery = oracle.discover(seeds, n=n_disco)
                
                if discovery:
                    st.write("### The Oracle recommends:")
                    for obj in discovery:
                        title = obj.get('trackTitle') or obj.get('name')
                        artists = obj.get('artists')
                        if isinstance(artists, list):
                            artists = ", ".join(a.get('name', "") for a in artists)
                        st.write(f"• **{title}** — {artists}")
                else:
                    st.info("No discoveries found or ReccoBeats unavailable.")
