import streamlit as st
import joblib
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MUSIC_TASTE_PROFILE
from src.music.feature_engineering import get_audio_embedding # Reuse extraction logic

st.set_page_config(page_title="Music Oracle", page_icon="🎧")

st.title("🎧 The Music Oracle")
st.markdown("Discover how closely a new track aligns with your sonic profile.")

@st.cache_resource
def load_taste_profile():
    try:
        return joblib.load(MUSIC_TASTE_PROFILE)
    except FileNotFoundError:
        return None

taste_centroid = load_taste_profile()

if taste_centroid is None:
    st.warning("Taste Profile not found. Please run the data pipeline and model_trainer.py first.")
    st.stop()

# --- Search Interface ---
with st.form("oracle_form"):
    col1, col2 = st.columns(2)
    with col1:
        track_input = st.text_input("Track Name", placeholder="e.g., Midnight City")
    with col2:
        artist_input = st.text_input("Artist Name", placeholder="e.g., M83")
    
    submit = st.form_submit_button("Consult the Oracle")

if submit and track_input and artist_input:
    with st.spinner("The Oracle is listening to the track..."):
        # Use a temporary ID for the oracle query
        new_embedding = get_audio_embedding(track_input, artist_input, "oracle_temp")
        
        if new_embedding:
            # Calculate Cosine Similarity
            v = np.array(new_embedding)
            c = taste_centroid
            
            cosine_sim = np.dot(v, c) / (np.linalg.norm(v) * np.linalg.norm(c))
            
            # Convert -1 to 1 range into a 0% to 100% match score
            match_percentage = ((cosine_sim + 1) / 2) * 100
            
            st.divider()
            st.subheader(f"Results for: {track_input} by {artist_input}")
            
            st.metric(label="Sonic Alignment Score", value=f"{match_percentage:.1f}%")
            
            if match_percentage > 85:
                st.success("Perfect Match: This hits right at the core of your musical taste.")
            elif match_percentage > 65:
                st.info("Strong Vibe: You'll likely enjoy this. It shares major elements with your library.")
            elif match_percentage > 40:
                st.warning("Fringe Territory: A good track to expand your horizons, but outside your usual rotation.")
            else:
                st.error("Dissonance: Mathematically opposed to your standard library.")
                
        else:
            st.error("Failed to analyze the audio. The track might not be available or exceeds duration limits.")