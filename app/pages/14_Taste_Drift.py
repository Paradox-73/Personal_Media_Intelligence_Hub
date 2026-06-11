import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path
from scipy.stats import entropy

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Taste Drift Analysis", page_icon="📈", layout="wide")

st.header("📈 Taste Drift & Entropy Analysis")
st.markdown("""
How does your taste evolve over time? This dashboard tracks the **Shannon Entropy** of your media consumption. 
High entropy means you are exploring diverse genres; low entropy means you are focusing on a specific niche.
""")

@st.cache_data
def load_unified_data():
    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        return None
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    return df

df = load_unified_data()

if df is not None:
    # 1. Rolling Entropy over Genres
    # Genre columns start with 'gen_'
    genre_cols = [c for c in df.columns if c.startswith('gen_')]
    
    # Sort by date
    df = df.sort_values('rating_date')
    
    # Set date as index for rolling
    df_temp = df.set_index('rating_date').sort_index()
    
    # Compute rolling sums for all genres
    rolling_genre_sums = df_temp[genre_cols].rolling('90D').sum()
    
    def calculate_entropy_from_row(row):
        total = row.sum()
        if total == 0: return 0
        dist = row / total
        return entropy(dist)

    # Apply entropy calculation to each row of rolling sums
    df['entropy'] = rolling_genre_sums.apply(calculate_entropy_from_row, axis=1).values
    
    fig = px.line(df, x='rating_date', y='entropy', 
                  title="90-Day Rolling Taste Entropy (Genre Diversity)",
                  labels={'rating_date': 'Date', 'entropy': 'Shannon Entropy'},
                  template="plotly_white")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. Cumulative Average Rating Drift
    # We need target_reg to be numeric
    df['target_reg'] = pd.to_numeric(df['target_reg'], errors='coerce')
    df['mean_rating_90d'] = df.set_index('rating_date')['target_reg'].rolling('90D').mean().values
    
    fig2 = px.line(df, x='rating_date', y='mean_rating_90d', 
                   title="90-Day Rolling Average Rating (Taste Mood)",
                   labels={'rating_date': 'Date', 'mean_rating_90d': 'Avg Rating'},
                   color_discrete_sequence=['green'],
                   template="plotly_white")
    
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.error("Unified training data not found.")
