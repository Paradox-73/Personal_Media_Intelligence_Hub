import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import config
# FIX: Import the correct function name
from src.data_ingestion import fetch_fresh_data 

st.set_page_config(page_title="Letterboxd Oracle", page_icon="🎬", layout="wide")

st.title("🎬 Personal Letterboxd Recommendation Engine")

st.markdown("""
### Welcome to your Personal Movie Intelligence Hub.

* **📊 Dashboard:** Analyze your taste, finding patterns in directors, genres, and box office.
* **🔮 The Oracle:** Predict how you will rate a movie *before* you watch it.
""")

# Sidebar controls
st.sidebar.header("Configuration")
if st.sidebar.button("Reload Cache"):
    st.cache_data.clear()
    st.success("Cache Cleared!")