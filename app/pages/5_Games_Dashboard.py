import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import sys
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Games Dashboard", page_icon="🎮", layout="wide")

# --- HELPER FUNCTIONS ---

def parse_list_col(x):
    if pd.isna(x): return []
    try:
        if '[' not in str(x) and ',' in str(x):
            return [i.strip() for i in str(x).split(',')]
        return ast.literal_eval(str(x))
    except: return [str(x)]

def get_frequent_items(df, col_name, top_n=10):
    """Counts unique items."""
    all_items = []
    for row in df[col_name]: all_items.extend(row)
    all_items = [i for i in all_items if i and i.lower() not in ['unknown', 'nan']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=[col_name, 'Count'])

def get_highly_rated_items(df, col_name, top_n=10, min_count=3):
    """Calculates average rating per item (e.g. Developer)."""
    df_exploded = df.explode(col_name)
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan])]
    
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    
    stats = stats[stats['count'] >= min_count]
    return stats.sort_values('avg_rating', ascending=False).head(top_n)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        
        # Clean Numerics
        for c in ['year']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Parse Lists
        for c in ['platform', 'developer', 'genre']: # Adjusted for Games
            if c in df.columns: df[f'{c}_list'] = df[c].apply(parse_list_col)
            
        df['decade'] = (df['year'] // 10) * 10
        return df
    except FileNotFoundError: return None

import numpy as np
df = load_data()

if df is None:
    st.error("❌ Games Data not found. Run ingestion for Games first.")
    st.stop()

st.title("🎮 Personal Games Intelligence")

# 1. METRICS
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Games", len(df))
c2.metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")
c3.metric("Avg Year", f"{int(df['year'].mean())}")
c4.metric("Liked Games", f"{df['is_liked'].sum()}")

st.divider()

# 2. TIME
st.subheader("⏳ Time Analysis")
c1, c2 = st.columns(2)
with c1:
    df_decade = df.groupby('decade')['user_rating'].count().reset_index()
    fig = px.bar(df_decade, x='decade', y='user_rating', title="Games per Decade")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(df, x="year", nbins=20, title="Year Distribution", color_discrete_sequence=['red'])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. PEOPLE/COMPANIES (Developer View)
st.subheader("👨‍💻 Developer Analysis")

if 'developer_list' in df.columns:
    c_freq, c_rate = st.columns(2)
    
    with c_freq:
        st.caption(f"**Most Played Developers**")
        freq = get_frequent_items(df, 'developer_list')
        fig = px.bar(freq, x='Count', y='developer_list', orientation='h', color='Count')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c_rate:
        st.caption(f"**Highest Rated Developers** (Min 3 Games)")
        rated = get_highly_rated_items(df, 'developer_list', min_count=3)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y='developer_list', orientation='h', 
                         color='avg_rating', color_continuous_scale='Viridis',
                         range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data (need >3 games per developer)")

st.divider()

# 4. Platforms & Genres
st.subheader("🕹️ Platforms & Genres")

c_plat, c_genre = st.columns(2)

with c_plat:
    if 'platform_list' in df.columns:
        plat = get_frequent_items(df, 'platform_list', 8)
        fig = px.pie(plat, values='Count', names='platform_list', title="Top Platforms", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

with c_genre:
    if 'genre_list' in df.columns:
        gen = get_frequent_items(df, 'genre_list', 8)
        fig = px.pie(gen, values='Count', names='genre_list', title="Top Genres", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
