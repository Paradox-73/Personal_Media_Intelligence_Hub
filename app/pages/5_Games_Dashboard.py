import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Games Dashboard", page_icon="🎮", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #14171C;
        color: #E0E0E0;
        font-family: 'Helvetica', sans-serif;
    }
    h1, h2, h3, h4 { color: #40bcf4 !important; }
    .stMetric { background-color: #2C343F; padding: 15px; border-radius: 8px; border: 1px solid #556678; }
</style>
""", unsafe_allow_html=True)

HUB_PALETTE = ["#ff8000", "#00e054", "#40bcf4", "#556678", "#2C343F"]

@st.cache_data
def load_data():
    try:
        # Load enriched data
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        # Load predictions if available
        pred_path = config.BASE_DIR / "results" / "games" / "game_predictions_detailed.csv"
        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            # Merge or use predicted column if it exists in enriched? 
            # In my current setup, predict_ratings.py saves a new CSV.
            # I'll join them on name and platform.
            df = pd.merge(df, df_pred[['name', 'platform_from_text', 'predicted_rating']], 
                          on=['name', 'platform_from_text'], how='left')
        
        # Cleanup
        df['my_rating'] = pd.to_numeric(df['my_rating'], errors='coerce')
        df['metacritic'] = pd.to_numeric(df['metacritic'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

st.title("🎮 Game Intelligence Dashboard")

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Games", len(df))
with c2: st.metric("Avg My Rating", f"{df['my_rating'].mean():.2f}")
with c3: st.metric("Avg Metacritic", f"{df['metacritic'].dropna().mean():.1f}")
with c4: 
    if 'predicted_rating' in df.columns:
        mae = (df['my_rating'] - df['predicted_rating']).abs().mean()
        st.metric("Model MAE", f"{mae:.3f}")

# Layout
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("⭐ My Rating Distribution")
    rating_counts = df['my_rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    fig = px.bar(rating_counts.sort_values('Rating'), x='Rating', y='Count', 
                 color_discrete_sequence=[HUB_PALETTE[1]])
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🖥️ Platform Breakdown")
    plat_counts = df['platform_from_text'].value_counts().reset_index()
    plat_counts.columns = ['Platform', 'Count']
    fig = px.pie(plat_counts, values='Count', names='Platform', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Detailed Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Genres", "Developers", "Predictions", "Full List"])

with tab1:
    st.subheader("🏷️ Top Genres")
    # Explode genres
    df['genre_list'] = df['genres'].fillna('').str.split(', ')
    df_genres = df.explode('genre_list')
    genre_counts = df_genres['genre_list'].value_counts().reset_index().head(15)
    genre_counts.columns = ['Genre', 'Count']
    fig = px.bar(genre_counts, x='Count', y='Genre', orientation='h', color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("👨‍💻 Top Developers")
    df['dev_list'] = df['developers'].fillna('').str.split(', ')
    df_devs = df.explode('dev_list')
    dev_counts = df_devs['dev_list'].value_counts().reset_index().head(15)
    dev_counts.columns = ['Developer', 'Count']
    fig = px.bar(dev_counts, x='Count', y='Developer', orientation='h', color='Count', color_continuous_scale='Magma')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔮 Model Accuracy: Actual vs Predicted")
    if 'predicted_rating' in df.columns:
        fig = px.scatter(df, x='my_rating', y='predicted_rating', hover_data=['name'], 
                         trendline="ols", color='my_rating', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/games/predict_ratings.py` to see model performance.")

with tab4:
    st.subheader("📋 All Games")
    display_cols = ['name', 'platform_from_text', 'released', 'my_rating', 'metacritic', 'genres']
    if 'predicted_rating' in df.columns:
        display_cols.append('predicted_rating')
    st.dataframe(df[display_cols].sort_values('my_rating', ascending=False), use_container_width=True)

st.divider()
st.subheader("🏆 Top Rated Games")
top_games = df.sort_values('my_rating', ascending=False).head(10)
for _, game in top_games.iterrows():
    c1, c2 = st.columns([1, 5])
    with c1:
        if pd.notna(game['cover']):
            st.image(game['cover'], width=150)
    with c2:
        st.markdown(f"### {game['name']} ({game['released']})")
        st.markdown(f"**Platform:** {game['platform_from_text']} | **My Rating:** {game['my_rating']} | **Metacritic:** {game['metacritic']}")
        st.markdown(f"**Genres:** {game['genres']}")
        st.write(f"{str(game['description_raw'])[:300]}...")
    st.divider()
