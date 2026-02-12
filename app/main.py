import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Personal Media Intelligence Hub", page_icon="🧠", layout="wide")

st.title("🧠 Personal Media Intelligence Hub")

st.markdown("""
### Welcome! Explore your taste across various media domains.

**Movies 🎬**
*   **1. Movies Dashboard:** Analyze your movie-watching patterns.
*   **2. Movies Oracle:** Predict your rating for any movie before you watch it.

**TV Shows 📺**
*   **3. TV Shows Dashboard:** Analyze your TV show preferences.
*   **7. TV Shows Oracle:** Predict your rating for any TV show.

**Music 🎵**
*   **4. Music Dashboard:** Analyze your music listening habits.
*   **8. Music Oracle:** Predict your rating for any music track.

**Games 🎮**
*   **5. Games Dashboard:** Analyze your gaming patterns.
*   **9. Games Oracle:** Predict your rating for any game.

**Books 📚**
*   **6. Books Dashboard:** Analyze your reading habits.
*   **10. Books Oracle:** Predict your rating for any book.
""")

# Sidebar controls
st.sidebar.header("Configuration")
if st.sidebar.button("Reload Cache"):
    st.cache_data.clear()
    st.success("Cache Cleared!")