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

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# --- HELPER FUNCTIONS ---

def clean_percentage(x):
    if pd.isna(x): return None
    x = str(x).replace('%', '').strip()
    try: return float(x)
    except: return None

def clean_currency(x):
    if pd.isna(x): return None
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return None

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
    """Calculates average rating per item (e.g. Director)."""
    # Explode the list column so each director gets their own row
    df_exploded = df.explode(col_name)
    
    # Filter valid
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan])]
    
    # Group
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    
    # Filter min count
    stats = stats[stats['count'] >= min_count]
    return stats.sort_values('avg_rating', ascending=False).head(top_n)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(config.ENRICHED_DATA_PATH)
        
        # Clean Numerics
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_percentage)
        df['box_office'] = df['box_office'].apply(clean_currency)
        for c in ['runtime', 'year', 'vote_average', 'imdb_rating', 'metascore']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        
        # Parse Lists
        for c in ['genre', 'actors', 'writer', 'director', 'production', 'country', 'language']:
            if c in df.columns: df[f'{c}_list'] = df[c].apply(parse_list_col)
            
        df['decade'] = (df['year'] // 10) * 10
        return df
    except FileNotFoundError: return None

import numpy as np # Import needed for nan check inside helper
df = load_data()

if df is None:
    st.error("❌ Data not found. Run ingestion first.")
    st.stop()

st.title("📊 Personal Movie Intelligence")

# 1. METRICS
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Movies", len(df))
c2.metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")
c3.metric("Avg Runtime", f"{int(df['runtime'].mean())} min")
c4.metric("Liked Movies", f"{df['is_liked'].sum()}")
c5.metric("Total Watch Time", f"{int(df['runtime'].sum() / 60)} hrs")

st.divider()

# 2. TIME & LENGTH
st.subheader("⏳ Time & Duration")
c1, c2 = st.columns(2)
with c1:
    df_decade = df.groupby('decade')['user_rating'].count().reset_index()
    fig = px.bar(df_decade, x='decade', y='user_rating', title="Movies per Decade")
    st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.histogram(df, x="runtime", nbins=20, title="Runtime Distribution", color_discrete_sequence=['teal'])
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. PEOPLE (Dual View)
st.subheader("👥 Cast & Crew Analysis")

tab_dir, tab_act, tab_wri = st.tabs(["🎬 Directors", "🎭 Actors", "✍️ Writers"])

def render_people_tab(col_list, label):
    c_freq, c_rate = st.columns(2)
    
    with c_freq:
        st.caption(f"**Most Watched {label}**")
        freq = get_frequent_items(df, col_list)
        fig = px.bar(freq, x='Count', y=col_list, orientation='h', color='Count')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    with c_rate:
        st.caption(f"**Highest Rated {label}** (Min 3 Movies)")
        rated = get_highly_rated_items(df, col_list, min_count=3)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y=col_list, orientation='h', 
                         color='avg_rating', color_continuous_scale='Viridis',
                         range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data (need >3 movies per person)")

with tab_dir: render_people_tab('director_list', "Directors")
with tab_act: render_people_tab('actors_list', "Actors")
with tab_wri: render_people_tab('writer_list', "Writers")

st.divider()

# 4. SCORES & FINANCIALS (Intuitive)
st.subheader("💰 Scores & Financials")

t1, t2, t3 = st.tabs(["Critic Alignment", "Box Office Impact", "Global Stats"])

with t1:
    st.markdown("#### Do you agree with Critics?")
    c_rt, c_meta = st.columns(2)
    
    # Intuitive Binning for RT
    if 'rotten_tomatoes_rating' in df.columns:
        df['rt_bin'] = pd.cut(df['rotten_tomatoes_rating'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
        rt_agg = df.groupby('rt_bin')['user_rating'].mean().reset_index()
        
        with c_rt:
            fig = px.bar(rt_agg, x='rt_bin', y='user_rating', 
                         title="Your Avg Rating by Rotten Tomatoes Score",
                         labels={'rt_bin': 'Rotten Tomatoes Range', 'user_rating': 'Your Avg Rating'},
                         range_y=[0, 5], color='user_rating', color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)

    if 'metascore' in df.columns:
        df['meta_bin'] = pd.cut(df['metascore'], bins=[0, 20, 40, 60, 80, 100], labels=['Red (0-39)', 'Yellow (40-60)', 'Green (61-80)', 'Universal (81-100)', 'Perfect (100)'])
        meta_agg = df.groupby('meta_bin')['user_rating'].mean().reset_index()
        
        with c_meta:
            fig = px.bar(meta_agg, x='meta_bin', y='user_rating',
                         title="Your Avg Rating by Metacritic Score",
                         labels={'meta_bin': 'Metascore Range', 'user_rating': 'Your Avg Rating'},
                         range_y=[0, 5])
            st.plotly_chart(fig, use_container_width=True)

with t2:
    if 'box_office' in df.columns:
        # Bin Box Office into readable tiers
        bins = [0, 1e6, 50e6, 100e6, 500e6, 1e9, 10e9]
        labels = ['Indie (<$1M)', 'Moderate ($1M-$50M)', 'Hit ($50M-$100M)', 'Blockbuster ($100M-$500M)', 'Mega-Blockbuster ($500M-$1B)', 'Historical (>$1B)']
        df['bo_tier'] = pd.cut(df['box_office'], bins=bins, labels=labels)
        
        bo_agg = df.groupby('bo_tier')['user_rating'].mean().reset_index()
        
        fig = px.bar(bo_agg, x='bo_tier', y='user_rating', 
                     title="Do you prefer Blockbusters or Indies?",
                     labels={'bo_tier': 'Box Office Tier', 'user_rating': 'Your Avg Rating'},
                     color='user_rating', range_y=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Box Office data missing.")

with t3:
    c_ctry, c_lang = st.columns(2)
    with c_ctry:
        cnt = get_frequent_items(df, 'country_list', 8)
        st.plotly_chart(px.bar(cnt, x='country_list', y='Count', title="Top Countries"), use_container_width=True)
    with c_lang:
        lng = get_frequent_items(df, 'language_list', 8)
        st.plotly_chart(px.pie(lng, values='Count', names='language_list', title="Languages"), use_container_width=True)

st.divider()

# 5. VIBE CHECK (Better Filters)
st.subheader("🧠 Plot Vibe Check")

if 'overview' in df.columns:
    text = " ".join(df['overview'].dropna().astype(str))
    
    # EXTENDED Stopword List
    stops = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 
        'where', 'when', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
        'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 
        'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours',
        'after', 'before', 'while', 'during', 'since', 'until', 'through', 'about', 'against', 'between',
        'movie', 'film', 'story', 'plot', 'character', 'life', 'world', 'find', 'finds', 'one', 'two',
        'into', 'back', 'new', 'young', 'man', 'woman', 'family', 'friend', 'friends'
    ])
    
    words = [w.lower() for w in re.findall(r'\w+', text) if len(w) > 3]
    filtered_words = [w for w in words if w not in stops]
    
    common = Counter(filtered_words).most_common(50)
    wdf = pd.DataFrame(common, columns=['Word', 'Count'])
    
    fig = px.treemap(wdf, path=['Word'], values='Count', title="Dominant Themes in Your Movies", color='Count')
    st.plotly_chart(fig, use_container_width=True)
    
st.divider()

# 8. ADVANCED INSIGHTS
st.subheader("🕵️ Advanced Insights")

t_main, t_studio, t_season = st.tabs(["Mainstream vs. Niche", "Studio Preferences", "Release Seasonality"])

with t_main:
    # Clean IMDb Votes (remove commas)
    if 'imdb_votes' in df.columns:
        df['votes_clean'] = df['imdb_votes'].astype(str).str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        
        fig = px.scatter(
            df, x="votes_clean", y="user_rating",
            color="user_rating", hover_data=['title'],
            title="The Hipster Index: Popularity vs Your Rating",
            labels={'votes_clean': 'IMDb Vote Count (Log Scale)'},
            log_x=True, # Log scale is crucial here as votes vary wildly
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Movies on the **left** are obscure/hidden gems. Movies on the **right** are mainstream.")

with t_studio:
    if 'production_list' in df.columns:
        # We use the helper function 'get_highly_rated_items' we defined earlier
        # Filter for studios you've watched at least 3 times
        top_studios = get_highly_rated_items(df, 'production_list', top_n=10, min_count=3)
        
        if not top_studios.empty:
            fig = px.bar(top_studios, x='avg_rating', y='production_list', 
                         orientation='h', title="Your Favorite Studios (Avg Rating)",
                         color='avg_rating', range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Watch more movies from specific studios to see trends here!")

with t_season:
    if 'released' in df.columns:
        # Extract Month
        df['release_date_dt'] = pd.to_datetime(df['released'], errors='coerce')
        df['month'] = df['release_date_dt'].dt.month_name()
        
        # Order months correctly
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        
        season_agg = df.groupby('month')['user_rating'].mean().reindex(month_order).reset_index()
        
        fig = px.line(season_agg, x='month', y='user_rating', markers=True,
                      title="Average Rating by Release Month",
                      range_y=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Do you like Summer Blockbusters (July) or Oscar Bait (December)?")

# 9. OSCAR ANALYSIS
if 'awards' in df.columns:
    st.divider()
    st.subheader("🏆 The Oscar Effect")
    
    # Simple text check for "Oscar"
    df['is_oscar_winner'] = df['awards'].fillna('').apply(lambda x: 'Oscar' in str(x))
    
    # Compare ratings
    oscar_agg = df.groupby('is_oscar_winner')['user_rating'].mean().reset_index()
    oscar_agg['Label'] = oscar_agg['is_oscar_winner'].map({True: 'Oscar Winner/Nominee', False: 'No Oscars'})
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(oscar_agg, x='Label', y='user_rating', color='Label',
                     title="Do you prefer Oscar movies?", range_y=[0, 5])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        # Show your highest rated Oscar movies
        oscar_movies = df[df['is_oscar_winner']].sort_values('user_rating', ascending=False).head(5)
        st.write("**Your Top Oscar Movies:**")
        for _, m in oscar_movies.iterrows():
            st.write(f"★ {m['user_rating']} - **{m['title']}**")
            
st.divider()

# 8. HOT TAKES & POPULARITY
st.subheader("🌶️ Hot Takes & Popularity")

c1, c2 = st.columns(2)

with c1:
    # 1. HOT TAKES (IMDb Edition)
    st.markdown("#### 🔥 Your Hot Takes (vs IMDb)")
    st.caption("Where your opinion splits from the public (IMDb)")
    
    if 'imdb_rating' in df.columns:
        # Normalize IMDb (0-10) to User Scale (0-5)
        df['imdb_norm'] = df['imdb_rating'] / 2
        df['controversy'] = df['user_rating'] - df['imdb_norm']
        
        # You loved it, IMDb hated it
        hot_takes_good = df.sort_values('controversy', ascending=False).head(3)
        
        # You hated it, IMDb loved it
        hot_takes_bad = df.sort_values('controversy', ascending=True).head(3)
        
        st.write("**Movies you defended:** (You > IMDb)")
        for _, row in hot_takes_good.iterrows():
            diff = row['controversy']
            st.write(f"📈 **{row['title']}** (+{diff:.1f})")
            
        st.write("**Movies you roasted:** (You < IMDb)")
        for _, row in hot_takes_bad.iterrows():
            diff = abs(row['controversy'])
            st.write(f"📉 **{row['title']}** ({diff:.1f})")
    else:
        st.info("IMDb ratings missing.")

with c2:
    # 2. POPULARITY VS RATING
    st.markdown("#### 🌟 Popularity vs Quality")
    
    if 'popularity' in df.columns:
        fig = px.scatter(
            df, x="popularity", y="user_rating",
            color="user_rating",
            hover_data=['title'],
            title="TMDB Popularity vs Your Rating",
            labels={'popularity': 'Viral Popularity Score'},
            log_x=True, # Popularity is exponential
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Right = Viral/Trending. Left = Niche/Older.")

st.divider()

# 9. SENTIMENT ANALYSIS (TextBlob)
st.subheader("🧠 Emotional Sentiment Analysis")

try:
    from textblob import TextBlob
    
    # Combine Overview + Tagline
    df['full_text'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    
    # Calculate Polarity (-1 to +1)
    # -1 = Negative/Tragic, +1 = Positive/Happy
    df['sentiment'] = df['full_text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if len(str(x)) > 5 else 0)
    
    t_sent, t_corr = st.tabs(["Happy vs. Sad Movies", "Does Mood Affect Rating?"])
    
    with t_sent:
        # Bin sentiment into readable labels
        bins = [-1, -0.2, 0.2, 1]
        labels = ['Tragic/Dark 🌑', 'Neutral/Serious 😐', 'Happy/Uplifting ☀️']
        df['mood'] = pd.cut(df['sentiment'], bins=bins, labels=labels)
        
        mood_counts = df['mood'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        
        c_pie, c_list = st.columns([1, 2])
        
        with c_pie:
            fig = px.pie(mood_counts, values='Count', names='Mood', title="Movie Moods", hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
        with c_list:
            # Show examples
            st.write("**Darkest Movies:**")
            darkest = df.sort_values('sentiment').head(3)
            for _, r in darkest.iterrows():
                st.write(f"🌑 **{r['title']}** ({r['sentiment']:.2f})")
                
            st.write("**Happiest Movies:**")
            happiest = df.sort_values('sentiment', ascending=False).head(3)
            for _, r in happiest.iterrows():
                st.write(f"☀️ **{r['title']}** (+{r['sentiment']:.2f})")

    with t_corr:
        # Scatter Sentiment vs Rating
        fig = px.scatter(
            df, x="sentiment", y="user_rating",
            color="user_rating",
            hover_data=['title'],
            title="Do you prefer Tragedies or Comedies?",
            labels={'sentiment': 'Sentiment (-1 Tragic to +1 Happy)'},
            range_x=[-1, 1],
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If the line goes UP, you like Happy movies. If DOWN, you prefer Dark movies.")

except ImportError:
    st.warning("⚠️ TextBlob library not found. Run `pip install textblob` to see Sentiment Analysis.")