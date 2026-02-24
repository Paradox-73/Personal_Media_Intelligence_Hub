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

st.set_page_config(page_title="TV Analytics Dashboard", page_icon="📺", layout="wide")

# --- HELPER FUNCTIONS ---

def clean_percentage(x):
    if pd.isna(x): return None
    x = str(x).replace('%', '').strip()
    try: return float(x)
    except: return None

def parse_list_col(x):
    if pd.isna(x): return []
    try:
        # Check if it looks like a list string
        if isinstance(x, str) and x.startswith('[') and x.endswith(']'):
            return ast.literal_eval(x)
        # Handle simple comma strings
        if isinstance(x, str) and ',' in x:
            return [i.strip() for i in x.split(',')]
        return [str(x)]
    except: return [str(x)]

def get_frequent_items(df, col_name, top_n=10):
    """Counts unique items."""
    all_items = []
    for row in df[col_name]: 
        if isinstance(row, list):
            all_items.extend(row)
    all_items = [i for i in all_items if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=[col_name, 'Count'])

def get_highly_rated_items(df, col_name, top_n=10, min_count=2):
    """Calculates average rating per item."""
    df_exploded = df.explode(col_name)
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan])]
    
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    
    stats = stats[stats['count'] >= min_count]
    return stats.sort_values('avg_rating', ascending=False).head(top_n)

def calculate_entropy(series):
    """Calculates the Shannon entropy for a series of lists."""
    import math
    all_items = [item for sublist in series for item in sublist]
    if not all_items:
        return 0
    
    counts = Counter(all_items)
    total_items = len(all_items)
    
    entropy = 0.0
    for count in counts.values():
        probability = count / total_items
        entropy -= probability * math.log2(probability)
        
    return entropy

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        # Load TV Shows Data
        df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        
        # --- FIX: Handle Missing 'is_liked' ---
        if 'is_liked' not in df.columns:
            # Infer 'Liked' status: If rating >= 4.0 or missing rating
            df['is_liked'] = df['user_rating'].apply(lambda x: 1 if pd.notnull(x) and x >= 4.0 else 0)
        
        # Clean Numerics
        numeric_cols = ['year', 'number_of_episodes', 'number_of_seasons', 
                        'vote_average', 'vote_count', 'imdb_rating', 'user_rating']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Parse Lists
        list_cols = ['genres', 'created_by', 'actors', 'production_companies', 'country']
        for c in list_cols:
            if c in df.columns: df[f'{c}_list'] = df[c].apply(parse_list_col)
            
        return df
    except FileNotFoundError: return None

import numpy as np
df = load_data()

if df is None:
    st.error("❌ TV Show Data not found. Run ingestion first.")
    st.stop()

st.title("📺 TV Show Intelligence")

# 1. METRICS
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Shows", len(df))
c2.metric("Avg Rating", f"{df['user_rating'].mean():.2f}")

# Calculate Total Episodes Watched (Estimate)
total_eps = df['number_of_episodes'].sum()
c3.metric("Total Episodes", f"{int(total_eps):,}")

# 'Liked' Metric (Safe Access)
c4.metric("Liked Shows (4+ ⭐)", f"{int(df['is_liked'].sum())}")

# Total Watch Time (Rough Estimate: 45 min per ep average if runtime missing)
# Prefer 'runtime' column if it exists, else default 40
if 'runtime' in df.columns:
    df['est_runtime'] = df['runtime'].fillna(40)
else:
    df['est_runtime'] = 40
    
total_minutes = (df['number_of_episodes'] * df['est_runtime']).sum()
c5.metric("Hours Watched", f"{int(total_minutes / 60):,} hrs")

if 'genres_list' in df.columns:
    entropy = calculate_entropy(df['genres_list'])
    c6.metric("Taste Diversity", f"{entropy:.2f}", help="A measure of genre diversity (Shannon Entropy). A higher value means your taste is more varied; a lower value means it's more specific.")

st.divider()

# 2. STATUS & FORMAT
st.subheader("📡 Status & Format")
c1, c2 = st.columns(2)

with c1:
    if 'status' in df.columns:
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        fig = px.pie(status_counts, values='Count', names='Status', title="Show Status", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if 'type' in df.columns:
        type_counts = df['type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        fig = px.bar(type_counts, x='Type', y='Count', title="Scripted vs Reality", color='Type')
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. CREATORS & CAST
st.subheader("👥 Cast & Creators")

tab_creat, tab_act, tab_net = st.tabs(["✍️ Creators", "🎭 Actors", "📺 Networks"])

def render_tab(col_list, label, min_c=2):
    if col_list not in df.columns:
        st.info(f"No {label} data available.")
        return

    c_freq, c_rate = st.columns(2)
    with c_freq:
        st.caption(f"**Most Watched {label}**")
        freq = get_frequent_items(df, col_list)
        if not freq.empty:
            fig = px.bar(freq, x='Count', y=col_list, orientation='h', color='Count')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            
    with c_rate:
        st.caption(f"**Highest Rated {label}** (Min {min_c} Shows)")
        rated = get_highly_rated_items(df, col_list, min_count=min_c)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y=col_list, orientation='h', 
                         color='avg_rating', color_continuous_scale='Viridis', range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

with tab_creat: render_tab('created_by_list', "Creators")
with tab_act: render_tab('actors_list', "Actors", min_c=3) # Actors appear in many shows, raise threshold

with tab_net:
    # Network is usually a single string, not a list, but let's check
    if 'network' in df.columns:
        net_counts = df['network'].value_counts().head(10).reset_index()
        net_counts.columns = ['Network', 'Count']
        fig = px.bar(net_counts, x='Count', y='Network', orientation='h', title="Top Networks")
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4. RATINGS ANALYSIS
st.subheader("📈 Ratings & Seasons")

t1, t2 = st.tabs(["Ratings Distribution", "Long vs Short Shows"])

with t1:
    fig = px.histogram(df, x="user_rating", nbins=10, title="Your Rating Distribution", color_discrete_sequence=['teal'])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

with t2:
    if 'number_of_seasons' in df.columns:
        fig = px.scatter(df, x="number_of_seasons", y="user_rating", 
                         hover_data=['name'], trendline="ols",
                         title="Do you prefer longer shows?",
                         labels={'number_of_seasons': 'Total Seasons'})
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 5. GENRES
st.subheader("🏷️ Genre Analysis")
if 'genres_list' in df.columns:
    df_gen = df.explode('genres_list')
    gen_stats = df_gen.groupby('genres_list').agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index().sort_values('count', ascending=False)
    
    # Filter small genres
    gen_stats = gen_stats[gen_stats['count'] > 1]
    
    fig = px.treemap(gen_stats, path=['genres_list'], values='count', color='avg_rating',
                     color_continuous_scale='RdBu', title="Genre Size vs. Rating")
    st.plotly_chart(fig, use_container_width=True)