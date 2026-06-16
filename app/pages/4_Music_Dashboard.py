import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Music Intelligence", page_icon="🎵", layout="wide")

# Spotify Brand Colors
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"

# Custom CSS for Spotify vibe
st.markdown(f"""
    <style>
    .main {{
        background-color: {SPOTIFY_BLACK};
        color: {SPOTIFY_WHITE};
    }}
    .stMetric {{
        background-color: #282828;
        padding: 15px;
        border-radius: 10px;
    }}
    div[data-testid="stExpander"] {{
        background-color: #282828;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

DARK_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))

# --- HELPER FUNCTIONS ---

def parse_genres(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none', 'unknown'):
        return []
    if s.startswith('[') and s.endswith(']'):
        try:
            return [str(g).strip() for g in ast.literal_eval(s)]
        except Exception:
            pass
    # Genres are usually comma/semicolon separated
    return [g.strip() for g in s.replace(';', ',').split(',') if g.strip() and g.strip().lower() != 'unknown']

def primary_artist(x):
    # Keep the full credit intact ("Tyler, The Creator" must not be split on the comma).
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() not in ('nan', 'none', 'unknown') else None

def get_frequent_items(series, top_n=10):
    all_items = [i for i in series if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=['Item', 'Count'])

def get_frequent_genres(series, top_n=12):
    all_items = []
    for row in series:
        if isinstance(row, list):
            all_items.extend(row)
    all_items = [i for i in all_items if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=['Genre', 'Count'])

# --- LOAD DATA ---
@st.cache_data
def load_data():
    paths_to_try = [config.MUSIC_ENRICHED_DATA_PATH, config.MUSIC_FULL_VIEW_PATH]
    df = None
    for path in paths_to_try:
        if path.exists():
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue

    if df is None:
        return None

    try:
        # Clean Numerics
        df['release_year'] = pd.to_numeric(df.get('release_year'), errors='coerce').fillna(0).astype(int)
        df['rating'] = pd.to_numeric(df.get('rating'), errors='coerce')
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

        # Track length now comes from MusicBrainz (mb_length_ms); fall back to legacy duration_ms
        length_col = 'mb_length_ms' if 'mb_length_ms' in df.columns else ('duration_ms' if 'duration_ms' in df.columns else None)
        if length_col:
            df['duration_min'] = pd.to_numeric(df[length_col], errors='coerce') / 60000
        else:
            df['duration_min'] = np.nan

        # Lyric-derived numeric features
        for c in ['lyric_word_count', 'lyric_unique_ratio', 'lyric_line_count',
                  'lyric_sentiment', 'lyric_pos', 'lyric_neu', 'lyric_neg']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Filter out Year 0 (unknown release)
        df = df[df['release_year'] > 0]

        # Primary artist for grouping (avoid splitting "Tyler, The Creator")
        artist_src = 'artists' if 'artists' in df.columns else 'mb_canonical_artist'
        df['primary_artist'] = df[artist_src].apply(primary_artist)

        # Genres: prefer Spotify artist_genres, fall back to MusicBrainz genres/tags
        genre_src = None
        for cand in ['artist_genres', 'mb_genres', 'mb_tags', 'genre']:
            if cand in df.columns:
                genre_src = cand
                break
        df['genre_list'] = df[genre_src].apply(parse_genres) if genre_src else [[] for _ in range(len(df))]

        df['decade'] = (df['release_year'] // 10) * 10
        return df
    except Exception as e:
        import traceback
        st.error(f"Error processing music data: {e}")
        st.code(traceback.format_exc())
        return None

df = load_data()

if df is None:
    st.error("❌ Music Data not found. Run the music pipeline first.")
    st.stop()

# --- HEADER ---
st.title("🎵 Music Intelligence")
st.markdown(f"**Sonic Profile Analysis** | {len(df):,} tracks in library")

# 1. TOP METRICS
has_lyrics = 'lyric_word_count' in df.columns
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Tracks", f"{len(df):,}")
with c2:
    total_hrs = df['duration_min'].sum() / 60
    st.metric("Library Duration", f"{int(total_hrs):,} hrs" if total_hrs else "N/A")
with c3:
    if 'lyrics_found' in df.columns:
        cov = df['lyrics_found'].astype(str).str.lower().isin(['true', '1', '1.0']).mean() * 100
        st.metric("Lyrics Coverage", f"{cov:.0f}%")
    else:
        st.metric("Lyrics Coverage", "N/A")
with c4:
    if has_lyrics:
        st.metric("Total Lyric Words", f"{int(df['lyric_word_count'].fillna(0).sum()):,}")
    else:
        st.metric("Total Lyric Words", "N/A")

st.divider()

# 2. LYRICAL LANDSCAPE (replaces Spotify audio features, which are no longer ingested)
if has_lyrics:
    st.subheader("📝 The Lyrical Landscape")
    with st.expander("ℹ️ What do these lyric features mean?"):
        st.markdown("""
        *   **Sentiment**: VADER compound score of the lyrics, from -1.0 (dark/aggressive) to +1.0 (positive/uplifting).
        *   **Word Count**: Total words in the track's lyrics — a proxy for lyrical density.
        *   **Lexical Richness**: Ratio of unique words to total words. Higher = more varied vocabulary.
        """)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        # Which genres carry the most positive vs darkest lyrics. (VADER's raw pos/neu/neg
        # split is ~78% neutral and uninformative, so we lean on genres instead.)
        if 'lyric_sentiment' in df.columns and df['lyric_sentiment'].notna().any():
            gexp = df.explode('genre_list')
            gexp = gexp[gexp['genre_list'].astype(str).str.len().gt(0) & gexp['lyric_sentiment'].notna()]
            gstats = gexp.groupby('genre_list').agg(
                Tracks=('lyric_sentiment', 'size'), Sentiment=('lyric_sentiment', 'mean')
            ).reset_index()
            gstats = gstats[gstats['Tracks'] >= 10].sort_values('Sentiment')
            if not gstats.empty:
                show = pd.concat([gstats.head(8), gstats.tail(8)]).drop_duplicates('genre_list')
                show['Sentiment'] = show['Sentiment'].round(3)
                fig_g = px.bar(show, x='Sentiment', y='genre_list', orientation='h', color='Sentiment',
                               title="Genres by Avg Lyric Sentiment", hover_data={'Tracks': True},
                               color_continuous_scale=['#E22134', '#535353', SPOTIFY_GREEN])
                fig_g.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None},
                                    coloraxis_showscale=False, **DARK_LAYOUT)
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption("Green = genres whose lyrics skew positive; red = genres that skew darker (min 10 tracks).")
            else:
                st.info("Not enough genre-tagged tracks for a sentiment breakdown.")
    with col_b:
        if 'lyric_sentiment' in df.columns and df['lyric_sentiment'].notna().any():
            fig_dist = px.histogram(df.dropna(subset=['lyric_sentiment']), x='lyric_sentiment', nbins=40,
                                    title="Lyric Sentiment Distribution",
                                    labels={'lyric_sentiment': 'Lyric Sentiment (-1 dark → +1 positive)'},
                                    color_discrete_sequence=[SPOTIFY_GREEN])
            fig_dist.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_dist, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        if 'lyric_word_count' in df.columns:
            fig_wc = px.histogram(df.dropna(subset=['lyric_word_count']), x='lyric_word_count', nbins=40,
                                  title="Lyrical Density (Words per Track)",
                                  color_discrete_sequence=[SPOTIFY_GREEN])
            fig_wc.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_wc, use_container_width=True)
    with col_d:
        if 'lyric_unique_ratio' in df.columns:
            fig_lex = px.histogram(df.dropna(subset=['lyric_unique_ratio']), x='lyric_unique_ratio', nbins=40,
                                   title="Lexical Richness (Unique-word Ratio)",
                                   color_discrete_sequence=['#1ED760'])
            fig_lex.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_lex, use_container_width=True)

    st.divider()

# 3. ARTISTS & GENRES
col_art, col_gen = st.columns(2)

with col_art:
    st.subheader("🎤 Top Artists")
    freq_art = get_frequent_items(df['primary_artist'], 12)
    fig_art = px.bar(freq_art, x='Count', y='Item', orientation='h',
                     title="Most Frequent Artists",
                     color='Count', color_continuous_scale=['#1DB954', '#1ED760'])
    fig_art.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None}, **DARK_LAYOUT)
    st.plotly_chart(fig_art, use_container_width=True)

with col_gen:
    st.subheader("🎼 Genre Distribution")
    freq_gen = get_frequent_genres(df['genre_list'], 10)
    if not freq_gen.empty:
        fig_gen = px.pie(freq_gen, values='Count', names='Genre', hole=0.5,
                         title="Top Genres",
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_gen.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_gen, use_container_width=True)
    else:
        st.info("No genre metadata available.")

st.divider()

# 4. TIME TRAVEL
st.subheader("📅 Release Era")
c_time1, c_time2 = st.columns(2)

with c_time1:
    df_year = df.groupby('release_year')['rating'].count().reset_index(name='Count')
    fig_year = px.area(df_year, x='release_year', y='Count', title="Tracks by Release Year",
                       color_discrete_sequence=[SPOTIFY_GREEN])
    fig_year.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_year, use_container_width=True)

with c_time2:
    if 'popularity' in df.columns and df['popularity'].notna().any():
        fig_pop = px.histogram(df, x="popularity", nbins=20, title="Popularity Distribution (Spotify Score)",
                               color_discrete_sequence=[SPOTIFY_GREEN])
        fig_pop.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_pop, use_container_width=True)
    else:
        dec = df.groupby('decade')['rating'].count().reset_index(name='Count')
        fig_dec = px.bar(dec, x='decade', y='Count', title="Tracks by Decade",
                         color_discrete_sequence=[SPOTIFY_GREEN])
        fig_dec.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_dec, use_container_width=True)

# 5. DATA EXPLORER
with st.expander("📂 View Library Data"):
    explorer_cols = [c for c in ['name', 'artists', 'release_year', 'rating', 'popularity',
                                 'duration_min', 'lyric_word_count', 'lyric_sentiment']
                     if c in df.columns]
    st.dataframe(df[explorer_cols].sort_values('rating', ascending=False), use_container_width=True)
