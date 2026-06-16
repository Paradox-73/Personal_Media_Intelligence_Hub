import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

st.set_page_config(page_title="Personal Media Intelligence Hub", page_icon="🧠", layout="wide")

st.title("🧠 Personal Media Intelligence Hub")


# --- CROSS-MEDIA WATCH / CONSUMPTION METRICS ---
def _num(series):
    return pd.to_numeric(series, errors='coerce')


@st.cache_data
def load_library_stats():
    """Aggregate top-line consumption metrics across every media domain."""
    stats = {}

    # Movies: count + total watch hours (runtime)
    try:
        dm = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        hrs = _num(dm['runtime']).sum() / 60
        stats['movies'] = {'count': len(dm), 'sub': f"~{int(hrs):,} hrs watched"}
    except Exception:
        stats['movies'] = None

    # TV Shows: count + total watch hours (episodes x runtime, default 45m)
    try:
        ds = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        est_rt = _num(ds['runtime']).fillna(45)
        hrs = (_num(ds['number_of_episodes']) * est_rt).sum() / 60
        stats['shows'] = {'count': len(ds), 'sub': f"~{int(hrs):,} hrs watched"}
    except Exception:
        stats['shows'] = None

    # Music: count + listening hours + lyric words
    try:
        path = config.MUSIC_ENRICHED_DATA_PATH if config.MUSIC_ENRICHED_DATA_PATH.exists() else config.MUSIC_FULL_VIEW_PATH
        mu = pd.read_csv(path)
        length_col = 'mb_length_ms' if 'mb_length_ms' in mu.columns else ('duration_ms' if 'duration_ms' in mu.columns else None)
        hrs = _num(mu[length_col]).sum() / 3.6e6 if length_col else 0
        words = int(_num(mu['lyric_word_count']).fillna(0).sum()) if 'lyric_word_count' in mu.columns else 0
        sub = f"~{int(hrs):,} hrs" if hrs else f"{len(mu):,} tracks"
        if words:
            sub += f" · {words/1e6:.1f}M lyric words"
        stats['music'] = {'count': len(mu), 'sub': sub}
    except Exception:
        stats['music'] = None

    # Books: count + total pages (falls back to word estimate; data may be sparse)
    try:
        bk = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        pages = int(_num(bk['pageCount']).fillna(0).sum()) if 'pageCount' in bk.columns else 0
        words = 0
        if 'description' in bk.columns:
            words = int(bk['description'].fillna('').apply(lambda x: len(str(x).split())).sum())
        if pages:
            sub = f"~{pages:,} pages read"
        elif words:
            sub = f"~{words:,} words"
        else:
            sub = "length data N/A"
        stats['books'] = {'count': len(bk), 'sub': sub}
    except Exception:
        stats['books'] = None

    # Games: count + rough hours-to-beat (RAWG playtime)
    try:
        gm = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        hrs = _num(gm['playtime']).fillna(0).sum() if 'playtime' in gm.columns else 0
        sub = f"~{int(hrs):,} hrs to beat" if hrs else "playtime N/A"
        stats['games'] = {'count': len(gm), 'sub': sub}
    except Exception:
        stats['games'] = None

    return stats


def _metric_card(icon, label, count, sub):
    st.markdown(f"""
    <div style="background-color:#2C343F; padding:14px 16px; border-radius:8px; border-bottom:3px solid #40bcf4;">
        <div style="color:#99AABB; font-size:0.8em; text-transform:uppercase; font-weight:bold; letter-spacing:0.05em;">{icon} {label}</div>
        <div style="color:#FFFFFF; font-size:1.7em; font-weight:600;">{count:,}</div>
        <div style="color:#00e054; font-size:0.85em;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


stats = load_library_stats()
st.subheader("📊 Your Library at a Glance")
cols = st.columns(5)
spec = [
    ('🎬', 'Movies', 'movies'), ('📺', 'TV Shows', 'shows'), ('🎵', 'Music', 'music'),
    ('📚', 'Books', 'books'), ('🎮', 'Games', 'games'),
]
for col, (icon, label, key) in zip(cols, spec):
    with col:
        s = stats.get(key)
        if s:
            _metric_card(icon, label, s['count'], s['sub'])
        else:
            _metric_card(icon, label, 0, "no data")

st.divider()

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