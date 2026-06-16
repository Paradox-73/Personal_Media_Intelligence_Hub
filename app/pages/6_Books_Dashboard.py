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

st.set_page_config(page_title="Books Dashboard", page_icon="📚", layout="wide")

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
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
        # Load predictions if available
        pred_path = config.BASE_DIR / "results" / "books" / "book_predictions_detailed.csv"
        if pred_path.exists():
            df_pred = pd.read_csv(pred_path)
            # The enriched export currently leaves 'authors'/'categories' empty, but the
            # predictions file carries them. Merge on title only (authors dtypes differ and
            # would otherwise raise a float64-vs-object merge error), then backfill metadata.
            df['title'] = df['title'].astype(str)
            df_pred['title'] = df_pred['title'].astype(str)
            pred_cols = [c for c in ['title', 'authors', 'categories', 'predicted_rating'] if c in df_pred.columns]
            df_pred = df_pred[pred_cols].drop_duplicates(subset='title', keep='first')
            df = pd.merge(df, df_pred, on='title', how='left', suffixes=('', '_pred'))
            # Coalesce metadata: prefer enriched value, fall back to prediction file
            for col in ['authors', 'categories']:
                pred_col = f'{col}_pred'
                if pred_col in df.columns:
                    df[col] = df[col].where(df[col].notna(), df[pred_col])
                    df.drop(columns=[pred_col], inplace=True)

        # Cleanup
        df['my_rating'] = pd.to_numeric(df['my_rating'], errors='coerce')
        df['averageRating'] = pd.to_numeric(df['averageRating'], errors='coerce')
        df['pageCount'] = pd.to_numeric(df['pageCount'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

st.title("📚 Book Intelligence Dashboard")

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Books", len(df))
with c2: st.metric("Avg My Rating", f"{df['my_rating'].mean():.2f}")
with c3: st.metric("Avg Hardcover Rating", f"{df['averageRating'].dropna().mean():.1f}")
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
                 color_discrete_sequence=[HUB_PALETTE[2]])
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("📖 Page Count Distribution")
    fig = px.histogram(df.dropna(subset=['pageCount']), x='pageCount', nbins=20,
                       color_discrete_sequence=[HUB_PALETTE[0]])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Detailed Analysis Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Categories", "Authors", "Predictions", "Full List"])

with tab1:
    st.subheader("🏷️ Top Categories")
    # Explode categories
    df['cat_list'] = df['categories'].fillna('').str.split(', ')
    df_cats = df.explode('cat_list')
    cat_counts = df_cats['cat_list'].value_counts().reset_index().head(15)
    cat_counts.columns = ['Category', 'Count']
    fig = px.bar(cat_counts, x='Count', y='Category', orientation='h', color='Count', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("✍️ Top Authors")
    df['auth_list'] = df['authors'].fillna('').str.split(', ')
    df_auths = df.explode('auth_list')
    auth_counts = df_auths['auth_list'].value_counts().reset_index().head(15)
    auth_counts.columns = ['Author', 'Count']
    fig = px.bar(auth_counts, x='Count', y='Author', orientation='h', color='Count', color_continuous_scale='Magma')
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🔮 Model Accuracy: Actual vs Predicted")
    if 'predicted_rating' in df.columns:
        fig = px.scatter(df, x='my_rating', y='predicted_rating', hover_data=['title'], 
                         trendline="ols", color='my_rating', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run `python src/books/predict_ratings.py` to see model performance.")

with tab4:
    st.subheader("📋 All Books")
    display_cols = ['title', 'authors', 'publishedDate', 'my_rating', 'pageCount', 'categories']
    if 'predicted_rating' in df.columns:
        display_cols.append('predicted_rating')
    st.dataframe(df[display_cols].sort_values('my_rating', ascending=False), use_container_width=True)

st.divider()
st.subheader("🏆 Top Rated Books")
top_books = df.sort_values('my_rating', ascending=False).head(10)
for _, book in top_books.iterrows():
    c1, c2 = st.columns([1, 5])
    with c1:
        if pd.notna(book['thumbnail']):
            st.image(book['thumbnail'], width=150)
    with c2:
        st.markdown(f"### {book['title']} ({book['publishedDate']})")
        st.markdown(f"**Authors:** {book['authors']} | **My Rating:** {book['my_rating']} | **Hardcover Rating:** {book['averageRating']}")
        st.markdown(f"**Categories:** {book['categories']} | **Pages:** {book['pageCount']}")
        st.write(f"{str(book['description'])[:500]}...")
    st.divider()
