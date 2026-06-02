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

# --- HELPER FUNCTIONS ---

def parse_genres(x):
    if pd.isna(x): return []
    s = str(x).strip()
    if not s: return []
    if s.startswith('[') and s.endswith(']'):
        try: return ast.literal_eval(s)
        except: pass
    # Genres are usually comma separated
    return [g.strip() for g in s.replace(';', ',').split(',') if g.strip()]

def get_frequent_items(df, col_name, top_n=10):
    all_items = [i for i in df[col_name] if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=[col_name, 'Count'])

def get_frequent_genres(df, col_name, top_n=10):
    all_items = []
    for row in df[col_name]:
        if isinstance(row, list): all_items.extend(row)
        else: all_items.append(row)
    all_items = [i for i in all_items if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=[col_name, 'Count'])

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
            except: continue
    
    if df is None: return None
        
    try:
        # Clean Numerics
        df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce').fillna(0).astype(int)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['duration_min'] = df['duration_ms'] / 60000
        
        # Filter out Year 0
        df = df[df['release_year'] > 0]
            
        # Parse Genres
        if 'artist_genres' in df.columns:
            df['genre_list'] = df['artist_genres'].apply(parse_genres)
        elif 'genre' in df.columns:
            df['genre_list'] = df['genre'].apply(parse_genres)
            
        df['decade'] = (df['release_year'] // 10) * 10
        return df
    except Exception as e:
        return None

df = load_data()

if df is None:
    st.error("❌ Music Data not found. Run the music pipeline first.")
    st.stop()

# --- HEADER ---
st.title("🎵 Music Intelligence")
st.markdown(f"**Sonic Profile Analysis** | {len(df)} tracks in library")

# 1. TOP METRICS
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Avg Rating", f"{df['rating'].mean():.2f} ★")
with c2:
    st.metric("Mean Energy", f"{df['energy'].mean():.2f}" if 'energy' in df.columns else "N/A")
with c3:
    st.metric("Explicit Ratio", f"{(df['explicit'].mean()*100):.1f}%" if 'explicit' in df.columns else "N/A")
with c4:
    st.metric("Library Duration", f"{int(df['duration_min'].sum() / 60)} hrs")

st.divider()

# 2. THE SONIC LANDSCAPE (Audio Features)
st.subheader("🔊 The Sonic Landscape")

with st.expander("ℹ️ What do these audio features mean?"):
    st.markdown("""
    *   **Acousticness**: Confidence measure of whether the track is acoustic (no electrical amplification).
    *   **Danceability**: Suitability for dancing based on tempo, rhythm stability, and beat strength.
    *   **Energy**: Intensity and activity; fast, loud, and noisy tracks score higher.
    *   **Instrumentalness**: Likelihood the track contains no vocals (Ooh/Aah are treated as instrumental).
    *   **Liveness**: Probability the track was recorded with a live audience.
    *   **Speechiness**: Detection of spoken words; values > 0.66 are likely entirely spoken (podcasts/poetry).
    *   **Valence**: Musical positiveness; high scores feel Happy/Cheerful, low scores feel Sad/Depressed/Angry.
    """)

feat_cols = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness', 'valence']
available_feats = [c for c in feat_cols if c in df.columns]

if available_feats:
    col_radar, col_dist = st.columns([1, 1.5])
    
    with col_radar:
        # Average Features Radar
        avg_feats = df[available_feats].mean()
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=avg_feats.values,
            theta=[c.capitalize() for c in available_feats],
            fill='toself',
            line_color=SPOTIFY_GREEN,
            name='Library Average'
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Average Audio Fingerprint",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=SPOTIFY_WHITE)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_dist:
        # Mood Map (Valence vs Energy Density)
        if 'valence' in df.columns and 'energy' in df.columns:
            # Removed marginals to fix Plotly ValueError incompatibility
            fig_density = px.density_heatmap(df, x='valence', y='energy', 
                                            nbinsx=30, nbinsy=30,
                                            color_continuous_scale='Viridis',
                                            title="Mood Density (Positivity vs Intensity)",
                                            labels={'valence': 'Positivity (Valence)', 'energy': 'Intensity (Energy)'})
            fig_density.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))
            st.plotly_chart(fig_density, use_container_width=True)

st.divider()

# 3. ARTISTS & GENRES
col_art, col_gen = st.columns(2)

with col_art:
    st.subheader("🎤 Top Artists")
    # Using primary_artist to avoid splitting names like "Tyler, The Creator"
    freq_art = get_frequent_items(df, 'primary_artist', 12)
    fig_art = px.bar(freq_art, x='Count', y='primary_artist', orientation='h',
                     title="Most Frequent Artists (Primary)",
                     color='Count', color_continuous_scale=['#1DB954', '#1ED760'])
    fig_art.update_layout(yaxis={'categoryorder':'total ascending'}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))
    st.plotly_chart(fig_art, use_container_width=True)

with col_gen:
    st.subheader("🎼 Genre Distribution")
    freq_gen = get_frequent_genres(df, 'genre_list', 10)
    fig_gen = px.pie(freq_gen, values='Count', names='genre_list', hole=0.5,
                     title="Top Genres",
                     color_discrete_sequence=px.colors.sequential.Greens_r)
    fig_gen.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))
    st.plotly_chart(fig_gen, use_container_width=True)

st.divider()

# 4. TIME TRAVEL
st.subheader("📅 Release Era")
c_time1, c_time2 = st.columns(2)

with c_time1:
    df_year = df.groupby('release_year')['rating'].count().reset_index()
    fig_year = px.area(df_year, x='release_year', y='rating', title="Tracks by Release Year",
                       color_discrete_sequence=[SPOTIFY_GREEN])
    fig_year.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))
    st.plotly_chart(fig_year, use_container_width=True)

with c_time2:
    # Popularity distribution
    fig_pop = px.histogram(df, x="popularity", nbins=20, title="Popularity Distribution (Spotify Score)",
                          color_discrete_sequence=[SPOTIFY_GREEN])
    fig_pop.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))
    st.plotly_chart(fig_pop, use_container_width=True)

# 5. DATA EXPLORER
with st.expander("📂 View Library Data"):
    st.dataframe(df[['name', 'artists', 'release_year', 'rating', 'popularity', 'energy', 'valence']].sort_values('rating', ascending=False), use_container_width=True)
