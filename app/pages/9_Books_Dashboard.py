import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import sys
import re
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

# --- Derived helpers ---
df['pub_year'] = pd.to_numeric(df['publishedDate'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce')
df['n_authors'] = df['authors'].fillna('').apply(lambda s: len([a for a in str(s).split(',') if a.strip()]))
n_unique_authors = (
    df['authors'].fillna('').str.split(',').explode().str.strip().replace('', np.nan).dropna().nunique()
)

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Total Books", len(df))
with c2: st.metric("Avg My Rating", f"{df['my_rating'].mean():.2f}")
with c3: st.metric("Avg Hardcover Rating", f"{df['averageRating'].dropna().mean():.1f}")
with c4: st.metric("Unique Authors", int(n_unique_authors))

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Categories", "Authors", "Taste vs Critics", "Reading Profile", "Full List"])

with tab1:
    st.subheader("🏷️ Top Categories")
    # Explode categories
    df['cat_list'] = df['categories'].fillna('').str.split(', ')
    df_cats = df.explode('cat_list')
    cat_counts = df_cats['cat_list'].value_counts().reset_index().head(15)
    cat_counts.columns = ['Category', 'Count']
    fig = px.bar(cat_counts, x='Count', y='Category', orientation='h', color='Count', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("✍️ Top Authors")
    df['auth_list'] = df['authors'].fillna('').str.split(', ')
    df_auths = df.explode('auth_list')
    auth_counts = df_auths['auth_list'].value_counts().reset_index().head(15)
    auth_counts.columns = ['Author', 'Count']
    fig = px.bar(auth_counts, x='Count', y='Author', orientation='h', color='Count', color_continuous_scale='Magma')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("🆚 You vs. the Crowd")
    st.caption("How your ratings compare to Hardcover's community average — are you harsher or kinder than the crowd?")
    crit = df.dropna(subset=['my_rating', 'averageRating']).copy()
    crit = crit[crit['averageRating'] > 0]
    if not crit.empty:
        # Hardcover ratings are on a ~5 scale already; align defensively
        crit['critic_5'] = crit['averageRating'].where(crit['averageRating'] <= 5, crit['averageRating'] / 2)
        crit['delta'] = crit['my_rating'] - crit['critic_5']

        cc1, cc2 = st.columns([2, 1])
        with cc1:
            fig = px.scatter(crit, x='critic_5', y='my_rating', hover_data=['title'],
                             color='delta', color_continuous_scale='RdYlGn',
                             labels={'critic_5': 'Hardcover Avg (5★)', 'my_rating': 'My Rating'})
            fig.add_shape(type='line', x0=0.5, y0=0.5, x1=5, y1=5,
                          line=dict(color=HUB_PALETTE[3], dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
        with cc2:
            st.metric("Avg agreement gap", f"{crit['delta'].mean():+.2f}★",
                      help="Positive = you rate higher than the crowd on average.")
            st.markdown("**🔥 Most over-rated (vs crowd)**")
            for _, r in crit.nlargest(3, 'delta')[['title', 'delta']].iterrows():
                st.caption(f"{r['title']} ({r['delta']:+.1f})")
            st.markdown("**❄️ Most under-rated (vs crowd)**")
            for _, r in crit.nsmallest(3, 'delta')[['title', 'delta']].iterrows():
                st.caption(f"{r['title']} ({r['delta']:+.1f})")
    else:
        st.info("Not enough overlapping Hardcover ratings to compare.")

with tab4:
    st.subheader("📚 Reading Profile")
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("#### 📅 Books by Publication Decade")
        dec = df.dropna(subset=['pub_year']).copy()
        if not dec.empty:
            dec['decade'] = (dec['pub_year'] // 10 * 10).astype(int).astype(str) + "s"
            decade_counts = dec['decade'].value_counts().reset_index()
            decade_counts.columns = ['Decade', 'Count']
            fig = px.bar(decade_counts.sort_values('Decade'), x='Decade', y='Count',
                         color_discrete_sequence=[HUB_PALETTE[1]])
            st.plotly_chart(fig, use_container_width=True)
    with p2:
        st.markdown("#### 📏 Do you rate longer books higher?")
        sc = df.dropna(subset=['pageCount', 'my_rating'])
        sc = sc[sc['pageCount'] > 0]
        if not sc.empty:
            fig = px.scatter(sc, x='pageCount', y='my_rating', hover_data=['title'],
                             trendline="ols", color_discrete_sequence=[HUB_PALETTE[2]],
                             labels={'pageCount': 'Pages', 'my_rating': 'My Rating'})
            st.plotly_chart(fig, use_container_width=True)
            corr = sc['pageCount'].corr(sc['my_rating'])
            st.caption(f"Page-count ↔ rating correlation: **{corr:+.2f}**")

    st.markdown("#### ⭐ Average Rating by Top Category")
    df['cat_list2'] = df['categories'].fillna('').str.split(', ')
    cat_rt = df.explode('cat_list2')
    cat_rt = cat_rt[cat_rt['cat_list2'].str.strip() != '']
    top_cats = cat_rt['cat_list2'].value_counts().head(10).index
    cat_rt = cat_rt[cat_rt['cat_list2'].isin(top_cats)]
    agg = cat_rt.groupby('cat_list2')['my_rating'].agg(['mean', 'count']).reset_index()
    agg.columns = ['Category', 'Avg Rating', 'Count']
    fig = px.bar(agg.sort_values('Avg Rating'), x='Avg Rating', y='Category', orientation='h',
                 color='Avg Rating', color_continuous_scale='RdYlGn', hover_data=['Count'])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("📋 All Books")
    display_cols = ['title', 'authors', 'publishedDate', 'my_rating', 'pageCount', 'categories']
    if 'predicted_rating' in df.columns:
        display_cols.append('predicted_rating')
    st.dataframe(df[display_cols].sort_values('my_rating', ascending=False), use_container_width=True)

st.divider()

# --- DESCRIPTION VIBE CHECK (themes from book descriptions, like the Movies plot vibe check) ---
st.subheader("🧠 Description Vibe Check")
st.caption("Dominant themes extracted from your book descriptions. Hover to see your average rating "
           "for books containing each theme (color = your average rating).")

if 'description' in df.columns:
    _stops = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'where', 'when', 'how', 'who',
        'which', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'at', 'by', 'for', 'from', 'in', 'into',
        'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without', 'he', 'him',
        'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs', 'i', 'me', 'my', 'mine',
        'you', 'your', 'yours', 'we', 'us', 'our', 'ours', 'after', 'before', 'while', 'during', 'since',
        'until', 'through', 'about', 'against', 'between', 'will', 'must', 'only', 'other', 'one', 'two',
        'new', 'back', 'most', 'just', 'more', 'there', 'soon', 'now', 'then', 'than', 'so', 'all', 'her',
        # book-specific noise
        'book', 'books', 'novel', 'story', 'stories', 'tale', 'series', 'author', 'reader', 'readers',
        'edition', 'page', 'pages', 'bestseller', 'bestselling', 'york', 'times', 'life', 'world', 'man',
        'woman', 'young', 'first', 'years', 'year', 'find', 'finds', 'must',
    ])
    _word_ratings = {}
    for _, _row in df.dropna(subset=['description', 'my_rating']).iterrows():
        _words = set(w.lower() for w in re.findall(r'\w+', str(_row['description']))
                     if len(w) > 3 and w.lower() not in _stops)
        for _w in _words:
            _word_ratings.setdefault(_w, []).append(_row['my_rating'])
    _word_stats = [{'Word': w, 'Count': len(r), 'Avg_Rating': float(np.mean(r))}
                   for w, r in _word_ratings.items() if len(r) >= 2]
    if _word_stats:
        _wdf = pd.DataFrame(_word_stats).sort_values('Count', ascending=False).head(50)
        _wdf['Avg_Rating'] = _wdf['Avg_Rating'].round(3)
        fig = px.treemap(_wdf, path=['Word'], values='Count', color='Avg_Rating',
                         hover_data=['Avg_Rating'], range_color=[1, 5],
                         color_continuous_scale='RdYlGn',
                         title="Dominant Themes in Your Books (Color = Your Avg Rating)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough description text to extract recurring themes yet.")

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
