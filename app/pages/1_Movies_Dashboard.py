import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import ast
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    /* Font Fallbacks */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #14171C;
        color: #E0E0E0;
        font-family: 'Graphik', 'Helvetica', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #2C343F;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #40bcf4 !important;
        font-family: 'Graphik', 'Helvetica', sans-serif;
    }
    
    .stMetric {
        background-color: #2C343F;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #556678;
    }
    
    div[data-testid="metric-container"] label {
        color: #00e054 !important;
    }
    
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ff8000 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #14171C;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #E0E0E0;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ff8000 !important;
        border-bottom-color: #ff8000 !important;
    }
    
    hr {
        border-top: 1px solid #556678;
    }
</style>
""", unsafe_allow_html=True)

# Define palette for plotly
HUB_PALETTE = ["#ff8000", "#00e054", "#40bcf4", "#556678", "#2C343F"]

# Set default Plotly template
pio.templates["hub_theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Graphik, Helvetica, sans-serif"),
        colorway=HUB_PALETTE,
        xaxis=dict(gridcolor="#2C343F", zerolinecolor="#2C343F", tickfont=dict(color="#556678")),
        yaxis=dict(gridcolor="#2C343F", zerolinecolor="#2C343F", tickfont=dict(color="#556678")),
    )
)
pio.templates.default = "hub_theme"

# --- HELPER FUNCTIONS ---

def render_stars(rating, max_rating=5):
    """Renders Letterboxd-style stars."""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    stars = "★" * full_stars + ("½" if half_star else "")
    return f"<span style='color:#00e054; font-family: courier; font-size: 1.2em;'>{stars}</span>"

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

def get_frequent_items_with_avg(df, col_name, top_n=10):
    df_exp = df.explode(col_name)
    df_exp = df_exp[~df_exp[col_name].isin(['Unknown', 'nan', np.nan, ''])]
    stats = df_exp.groupby(col_name).agg(
        Count=('user_rating', 'size'), 
        avg_rating=('user_rating', 'mean')
    ).reset_index()
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('Count', ascending=False).head(top_n)

def get_highly_rated_items(df, col_name, top_n=10, min_count=3):
    df_exploded = df.explode(col_name)
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan, ''])]
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    stats = stats[stats['count'] >= min_count]
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('avg_rating', ascending=False).head(top_n)

def calculate_entropy(series):
    """Calculates Shannon Entropy, robustly handling both lists and scalar values."""
    import math
    all_items = []
    
    for val in series.dropna():
        if isinstance(val, list):
            valid_items = [str(i).strip() for i in val if str(i).strip().lower() not in ['unknown', 'nan', 'none', '']]
            all_items.extend(valid_items)
        else:
            if str(val).strip().lower() not in ['unknown', 'nan', 'none', '']:
                all_items.append(str(val).strip())
                
    if not all_items: return 0
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
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_percentage)
        df['box_office'] = df['box_office'].apply(clean_currency)
        for c in ['runtime', 'year', 'vote_average', 'imdb_rating', 'metascore']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        
        for c in ['genre', 'actors', 'writer', 'director', 'production', 'country', 'language']:
            if c in df.columns: df[f'{c}_list'] = df[c].apply(parse_list_col)
            
        df['decade'] = (df['year'] // 10) * 10
        return df
    except FileNotFoundError: return None

df = load_data()

if df is None:
    st.error("❌ Data not found. Run ingestion first.")
    st.stop()

st.title("📊 Personal Movie Intelligence")

# 1. METRICS & RATINGS HISTOGRAM
c1, c2, c3, c4, c5, c6 = st.columns(6)

def lb_metric(label, value, help_text=None):
    tooltip = f'title="{help_text}"' if help_text else ""
    cursor = "help" if help_text else "default"
    st.markdown(f"""
    <div {tooltip} style="background-color: #2C343F; padding: 10px 15px; border-radius: 4px; border-bottom: 2px solid #556678; cursor: {cursor};">
        <div style="color: #99AABB; font-size: 0.8em; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;">{label}</div>
        <div style="color: #FFFFFF; font-size: 1.8em; font-weight: 500;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

with c1: lb_metric("Total Movies", len(df))
with c2: lb_metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")
with c3: lb_metric("Avg Runtime", f"{int(df['runtime'].mean())} min")
with c4: lb_metric("Liked Movies", f"{df['is_liked'].sum()}")

# Calculate Watch Time Tooltip
total_hours = df['runtime'].sum() / 60
total_days = total_hours / 24
total_months = total_days / 30.44
watch_time_tooltip = f"{total_days:.1f} days"
if total_days >= 30:
    watch_time_tooltip += f" (~{total_months:.1f} months)"

with c5: lb_metric("Total Watch Time", f"{int(total_hours)} hrs", help_text=watch_time_tooltip)
with c6:
    if 'genre_list' in df.columns:
        gen_entropy = calculate_entropy(df['genre_list'])
        lb_metric("Taste Diversity", f"{gen_entropy:.2f}", 
                  help_text="Shannon Entropy. Max ~4.5.")

st.markdown("#### Your Rating Distribution")
rating_counts = df['user_rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Count']
# Letterboxd signature rating chart is green
fig_ratings = px.bar(rating_counts.sort_values('Rating'), x='Rating', y='Count', 
                     text='Count', color_discrete_sequence=['#00e054'])
fig_ratings.update_traces(textposition='outside', marker_line_width=1, marker_line_color="#14171C")
fig_ratings.update_layout(xaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5))
st.plotly_chart(fig_ratings, use_container_width=True)

st.divider()

# 2. TIME, LENGTH & SEASONALITY
st.subheader("⏳ Time & Duration")
t_dec, t_run, t_sea = st.tabs(["Decades", "Runtime", "Release Season"])

with t_dec:
    dec_ent = calculate_entropy(df['decade'])
    st.caption(f"**Decade Diversity (Entropy):** {dec_ent:.2f}")
    
    df_decade = df.groupby('decade').agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
    df_decade['avg_rating'] = df_decade['avg_rating'].round(3)
    fig = px.bar(df_decade, x='decade', y='Count', hover_data={'avg_rating': ':.3f'}, title="Movies per Decade", color_discrete_sequence=[HUB_PALETTE[0]])
    fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
    st.plotly_chart(fig, use_container_width=True)

with t_run:
    bins = [0, 60, 80, 100, 120, 140, 160, 180, 999]
    labels = ['<60m', '60-79m', '80-99m', '100-119m', '120-139m', '140-159m', '160-179m', '180m+']
    df['runtime_bin'] = pd.cut(df['runtime'], bins=bins, labels=labels, right=False)
    rt_agg = df.groupby('runtime_bin', observed=True).agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
    rt_agg['avg_rating'] = rt_agg['avg_rating'].round(3)
    
    fig = px.bar(rt_agg, x='runtime_bin', y='Count', hover_data={'avg_rating': ':.3f'}, 
                 title="Runtime Distribution", color_discrete_sequence=[HUB_PALETTE[1]])
    fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
    st.plotly_chart(fig, use_container_width=True)

with t_sea:
    if 'released' in df.columns:
        df['release_date_dt'] = pd.to_datetime(df['released'], errors='coerce')
        df['month'] = df['release_date_dt'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        season_agg = df.groupby('month')['user_rating'].mean().reindex(month_order).reset_index()
        season_agg['user_rating'] = season_agg['user_rating'].round(3)
        fig = px.bar(season_agg, x='month', y='user_rating',
                      title="Average Rating by Release Month", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[2]])
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. PEOPLE & STUDIOS
st.subheader("👥 Cast, Crew & Studios Analysis")
tab_dir, tab_act, tab_wri, tab_stu = st.tabs(["🎬 Directors", "🎭 Actors", "✍️ Writers", "🏢 Studios"])

def render_people_tab(col_list, label, min_count_threshold):
    ent = calculate_entropy(df[col_list])
    st.caption(f"**{label} Diversity (Entropy):** {ent:.2f}")
    
    c_freq, c_rate = st.columns(2)
    with c_freq:
        st.caption(f"**Most Watched {label}**")
        freq = get_frequent_items_with_avg(df, col_list)
        fig = px.bar(freq, x='Count', y=col_list, orientation='h', color='Count', hover_data={'avg_rating': ':.3f'}, color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    with c_rate:
        st.caption(f"**Highest Rated {label}** (Min {min_count_threshold} Movies)")
        rated = get_highly_rated_items(df, col_list, min_count=min_count_threshold)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y=col_list, orientation='h', color='avg_rating', 
                         color_continuous_scale=[[0, HUB_PALETTE[1]], [1, HUB_PALETTE[2]]], range_x=[0, 5], hover_data={'count': True})
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"Not enough data (need >{min_count_threshold-1} movies per person)")

with tab_dir: render_people_tab('director_list', "Directors", 3)
with tab_act: render_people_tab('actors_list', "Actors", 7) 
with tab_wri: render_people_tab('writer_list', "Writers", 3)
with tab_stu: render_people_tab('production_list', "Studios", 8)

st.divider()

# 4. SCORES & FINANCIALS
st.subheader("💰 Scores & Consensus Alignment")
t1, t2, t3 = st.tabs(["Critic Alignment", "Box Office Impact", "Global Stats"])

with t1:
    st.markdown("#### Do you agree with Critics & Audiences?")
    c_rt, c_meta = st.columns(2)
    c_imdb, c_tmdb = st.columns(2)
    
    if 'rotten_tomatoes_rating' in df.columns:
        df['rt_bin'] = pd.cut(df['rotten_tomatoes_rating'], bins=[0, 20, 40, 60, 80, 100], labels=['0-20%', '21-40%', '41-60%', '61-80%', '81-100%'])
        rt_agg = df.groupby('rt_bin', observed=True)['user_rating'].mean().reset_index()
        rt_agg['user_rating'] = rt_agg['user_rating'].round(3)
        with c_rt:
            fig = px.bar(rt_agg, x='rt_bin', y='user_rating', title="Avg Rating by Rotten Tomatoes", range_y=[0, 5], color='user_rating', color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)

    if 'metascore' in df.columns:
        df['meta_bin'] = pd.cut(df['metascore'], bins=[0, 20, 40, 60, 80, 100], labels=['0-39', '40-60', '61-80', '81-100', '100'])
        meta_agg = df.groupby('meta_bin', observed=True)['user_rating'].mean().reset_index()
        meta_agg['user_rating'] = meta_agg['user_rating'].round(3)
        with c_meta:
            fig = px.bar(meta_agg, x='meta_bin', y='user_rating', title="Avg Rating by Metascore", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[2]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
            
    if 'imdb_rating' in df.columns:
        df['imdb_bin'] = pd.cut(df['imdb_rating'], bins=[0, 4, 6, 7, 8, 10], labels=['<4.0', '4.0-6.0', '6.0-7.0', '7.0-8.0', '>8.0'])
        imdb_agg = df.groupby('imdb_bin', observed=True)['user_rating'].mean().reset_index()
        imdb_agg['user_rating'] = imdb_agg['user_rating'].round(3)
        with c_imdb:
            fig = px.bar(imdb_agg, x='imdb_bin', y='user_rating', title="Avg Rating by IMDb", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[1]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)

    if 'vote_average' in df.columns:
        df['tmdb_bin'] = pd.cut(df['vote_average'], bins=[0, 4, 6, 7, 8, 10], labels=['<4.0', '4.0-6.0', '6.0-7.0', '7.0-8.0', '>8.0'])
        tmdb_agg = df.groupby('tmdb_bin', observed=True)['user_rating'].mean().reset_index()
        tmdb_agg['user_rating'] = tmdb_agg['user_rating'].round(3)
        with c_tmdb:
            fig = px.bar(tmdb_agg, x='tmdb_bin', y='user_rating', title="Avg Rating by TMDB Votes", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[0]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)

with t2:
    if 'box_office' in df.columns:
        bins = [0, 1e6, 50e6, 100e6, 500e6, 1e9, 10e9]
        labels = ['Indie (<$1M)', 'Moderate ($1M-$50M)', 'Hit ($50M-$100M)', 'Blockbuster ($100M-$500M)', 'Mega-Blockbuster ($500M-$1B)', 'Historical (>$1B)']
        df['bo_tier'] = pd.cut(df['box_office'], bins=bins, labels=labels)
        bo_agg = df.groupby('bo_tier', observed=True)['user_rating'].mean().reset_index()
        bo_agg['user_rating'] = bo_agg['user_rating'].round(3)
        fig = px.bar(bo_agg, x='bo_tier', y='user_rating', title="Do you prefer Blockbusters or Indies?", color='user_rating', range_y=[0, 5], color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[1]]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Box Office data missing.")

with t3:
    c_ctry, c_lang = st.columns(2)
    with c_ctry:
        ctry_ent = calculate_entropy(df['country_list'])
        st.caption(f"**Country Diversity (Entropy):** {ctry_ent:.2f}")
        cnt = get_frequent_items_with_avg(df, 'country_list', 8)
        fig = px.bar(cnt, x='country_list', y='Count', hover_data={'avg_rating': ':.3f'}, title="Top Countries", color_discrete_sequence=[HUB_PALETTE[2]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    with c_lang:
        lang_ent = calculate_entropy(df['language_list'])
        st.caption(f"**Language Diversity (Entropy):** {lang_ent:.2f}")
        lng = get_frequent_items_with_avg(df, 'language_list', 8)
        fig = px.pie(lng, values='Count', names='language_list', hover_data={'avg_rating': ':.3f'}, title="Languages", color_discrete_sequence=HUB_PALETTE)
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 5. VIBE CHECK
st.subheader("🧠 Plot Vibe Check")
st.caption("Dominant plot themes extracted from Plot + Overview + Tagline. Hover to see your average rating for movies with these themes.")

if 'overview' in df.columns and 'plot' in df.columns:
    df['vibe_text'] = df['overview'].fillna('') + " " + df['tagline'].fillna('') + " " + df['plot'].fillna('')
    stops = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'where', 'when', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours',
        'after', 'before', 'while', 'during', 'since', 'until', 'through', 'about', 'against', 'between',
        'movie', 'film', 'story', 'plot', 'character', 'life', 'world', 'find', 'finds', 'one', 'two',
        'into', 'back', 'new', 'must', 'will', 'only', 'other', 'city', 'years', 'year', 'himself', 'high', 'most', 'just', 'more', 'there', 'soon',
    ])
    
    word_ratings = {}
    for _, row in df.dropna(subset=['vibe_text', 'user_rating']).iterrows():
        text = str(row['vibe_text'])
        words = set([w.lower() for w in re.findall(r'\w+', text) if len(w) > 3 and w.lower() not in stops])
        for w in words:
            if w not in word_ratings: word_ratings[w] = []
            word_ratings[w].append(row['user_rating'])
            
    word_stats = []
    for w, ratings in word_ratings.items():
        word_stats.append({'Word': w, 'Count': len(ratings), 'Avg_Rating': np.mean(ratings)})
        
    wdf = pd.DataFrame(word_stats).sort_values('Count', ascending=False).head(50)
    wdf['Avg_Rating'] = wdf['Avg_Rating'].round(3)
    
    fig = px.treemap(wdf, path=['Word'], values='Count', color='Count', hover_data=['Avg_Rating'], title="Dominant Themes in Your Movies", color_continuous_scale=[[0, HUB_PALETTE[2]], [1, HUB_PALETTE[1]]])
    st.plotly_chart(fig, use_container_width=True)
    
st.divider()

# 9. OSCAR ANALYSIS
if 'awards' in df.columns:
    st.subheader("🏆 The Oscar Effect")
    df['is_oscar_winner'] = df['awards'].fillna('').str.contains(r'Won\s.*Oscar', flags=re.IGNORECASE, regex=True)
    oscar_agg = df.groupby('is_oscar_winner')['user_rating'].mean().reset_index()
    oscar_agg['user_rating'] = oscar_agg['user_rating'].round(3)
    oscar_agg['Label'] = oscar_agg['is_oscar_winner'].map({True: 'Oscar Winner', False: 'Not a Winner'})
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(oscar_agg, x='Label', y='user_rating', color='Label', title="Do you prefer Actual Oscar Winners?", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[1], HUB_PALETTE[2]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        oscar_movies = df[df['is_oscar_winner']].sort_values('user_rating', ascending=False).head(5)
        st.write("**Your Top Oscar-Winning Movies:**")
        for _, m in oscar_movies.iterrows():
            st.markdown(f"{render_stars(m['user_rating'])} **{m['title']}**", unsafe_allow_html=True)
            
st.divider()

# 10. HOT TAKES
st.subheader("🌶️ Your Hot Takes (vs General Consensus)")
st.caption("Where your opinion wildly splits from the average critic & audience consensus (Sorted by Magnitude).")

# Calculate unified Critic Average, explicitly ignoring zeroes
df['imdb_100'] = df['imdb_rating'] * 10
df['va_100'] = df['vote_average'] * 10
critic_cols = ['imdb_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']

# Replace 0s with NaNs so they don't drag down the average
df[critic_cols] = df[critic_cols].replace(0, np.nan)
df['critic_avg_100'] = df[critic_cols].mean(axis=1, skipna=True) 
df['critic_norm'] = (df['critic_avg_100'] / 100) * 5

# Drop movies with absolutely no critic scores
ht_df = df.dropna(subset=['user_rating', 'critic_norm']).copy()
ht_df['controversy'] = ht_df['user_rating'] - ht_df['critic_norm']
ht_df['magnitude'] = ht_df['controversy'].abs()

# Filters: Only defend movies critics rated <= 3.5. Only roast movies critics rated >= 2.5
hot_takes_good = ht_df[(ht_df['controversy'] > 0) & (ht_df['critic_norm'] <= 2.9)].sort_values('magnitude', ascending=False).head(10)
hot_takes_bad = ht_df[(ht_df['controversy'] < 0) & (ht_df['critic_norm'] >= 3.5)].sort_values('magnitude', ascending=False).head(10)

c1, c2 = st.columns(2)
with c1:
    st.write("**Movies you defended (You > Critics):**")
    for _, row in hot_takes_good.iterrows():
        st.markdown(f"📈 **{row['title']}** (Diff: +{row['magnitude']:.2f})  \n<span style='font-size:0.95em; color:#99AABB;'>{render_stars(row['user_rating'])} | Critic: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)
with c2:
    st.write("**Movies you roasted (You < Critics):**")
    for _, row in hot_takes_bad.iterrows():
        st.markdown(f"📉 **{row['title']}** (Diff: -{row['magnitude']:.2f})  \n<span style='font-size:0.95em; color:#99AABB;'>{render_stars(row['user_rating'])} | Critic: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)

st.divider()

# 11. GENRE & MATURITY ANALYSIS
st.subheader("🎭 Genre & Maturity Analysis")
st.caption("Identify which specific categories drive your ratings up or down.")

t_genre, t_rated = st.tabs(["Genre Influence", "Maturity (MPAA) Ratings"])

with t_genre:
    if 'genre_list' in df.columns:
        gen_ent = calculate_entropy(df['genre_list'])
        st.caption(f"**Genre Diversity (Entropy):** {gen_ent:.2f}")
        
        df_genre_exp = df.explode('genre_list')
        df_genre_exp = df_genre_exp[~df_genre_exp['genre_list'].isin(['Unknown', 'nan', np.nan, ''])]
        
        genre_stats = df_genre_exp.groupby('genre_list').agg(
            Count=('user_rating', 'size'),
            Avg_Rating=('user_rating', 'mean'),
            Mode_Rating=('user_rating', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        ).reset_index()
        genre_stats['Avg_Rating'] = genre_stats['Avg_Rating'].round(3)
        genre_stats = genre_stats.sort_values('Count', ascending=False)
        
        cg1, cg2 = st.columns([1, 1.2])
        with cg1:
            st.write("**Genre Breakdown**")
            
            # 1. Count Chart (Popularity)
            fig_count = px.bar(
                genre_stats.head(15), 
                x='Count', 
                y='genre_list', 
                orientation='h', 
                title="Most Watched Genres",
                color='Count', 
                color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[1]]],
                text='Count'
            )
            fig_count.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig_count.update_traces(textposition='outside')
            st.plotly_chart(fig_count, use_container_width=True)

            # 2. Avg Rating Chart (Quality)
            top_rated_genres = genre_stats[genre_stats['Count'] >= 20].sort_values('Avg_Rating', ascending=False).head(15)
            fig_avg = px.bar(
                top_rated_genres, 
                x='Avg_Rating', 
                y='genre_list', 
                orientation='h', 
                title="Highest Rated Genres (Min 20 Movies)",
                color='Avg_Rating', 
                color_continuous_scale=[[0, HUB_PALETTE[4]], [1, HUB_PALETTE[0]]],
                range_x=[0, 5],
                text='Avg_Rating'
            )
            fig_avg.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig_avg.update_traces(textposition='inside', texttemplate='%{text:.2f}')
            st.plotly_chart(fig_avg, use_container_width=True)
            
        with cg2:
            st.write("**Rating Spread per Genre (Influence)**")
            st.caption("Look at the median line (inside the box) to see true genre influence.")
            valid_genres = genre_stats[genre_stats['Count'] >= 5]['genre_list']
            filtered_genre_df = df_genre_exp[df_genre_exp['genre_list'].isin(valid_genres)]
            fig = px.box(filtered_genre_df, x="genre_list", y="user_rating", color="genre_list", color_discrete_sequence=HUB_PALETTE)
            fig.update_layout(showlegend=False, xaxis_title="Genre", yaxis_title="Your Rating", height=740, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

with t_rated:
    if 'rated' in df.columns:
        df['rated'] = df['rated'].fillna('Unrated')
        rated_ent = calculate_entropy(df['rated'])
        st.caption(f"**Maturity Rating Diversity (Entropy):** {rated_ent:.2f}")
        
        rated_stats = df.groupby('rated').agg(
            Count=('user_rating', 'size'),
            Avg_Rating=('user_rating', 'mean')
        ).reset_index()
        rated_stats['Avg_Rating'] = rated_stats['Avg_Rating'].round(3)
        rated_stats = rated_stats.sort_values('Count', ascending=False)
        
        cr1, cr2 = st.columns(2)
        with cr1:
            fig = px.bar(rated_stats, x='rated', y='Count', hover_data={'Avg_Rating': ':.3f'}, title="Maturity Rating Breakdown", color_discrete_sequence=[HUB_PALETTE[2]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
        with cr2:
            fig = px.box(df, x="rated", y="user_rating", color="rated", title="Rating Spread per Maturity Level", color_discrete_sequence=HUB_PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# 12. SENTIMENT ANALYSIS
st.subheader("🧠 Emotional Sentiment Analysis")
st.caption("Upgraded to use VADER NLP Engine. Contextualizes genre and plot to score movies from -1.0 (Tragic/Dark) to +1.0 (Happy/Wholesome).")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    
    df['sentiment_text'] = "Genres: " + df['genre'].fillna('') + ". Plot: " + df['overview'].fillna('') + " " + df['tagline'].fillna('') + " " + df.get('plot', pd.Series(dtype=str)).fillna('')
    df['sentiment'] = df['sentiment_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if len(str(x)) > 5 else 0)
    
    t_sent, t_corr = st.tabs(["High Conflict vs. Wholesome", "Does Mood Affect Rating?"])
    
    with t_sent:
        bins = [-1, -0.25, 0.25, 1]
        labels = ['High Conflict/Tragic 🌑', 'Neutral/Balanced 😐', 'Wholesome/Triumphant ☀️']
        df['mood'] = pd.cut(df['sentiment'], bins=bins, labels=labels)
        mood_counts = df['mood'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        
        c_pie, c_list = st.columns([1, 2])
        with c_pie:
            fig = px.pie(mood_counts, values='Count', names='Mood', title="VADER Sentiment Split", hole=0.4, color_discrete_sequence=[HUB_PALETTE[3], HUB_PALETTE[2], HUB_PALETTE[1]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
            
        with c_list:
            st.write("**Most 'Negative/Conflict' Plots:**")
            darkest = df.sort_values('sentiment').head(3)
            for _, r in darkest.iterrows():
                st.write(f"🌑 **{r['title']}** ({r['sentiment']:.2f})")
                
            st.write("**Most 'Positive/Wholesome' Plots:**")
            happiest = df.sort_values('sentiment', ascending=False).head(3)
            for _, r in happiest.iterrows():
                st.write(f"☀️ **{r['title']}** (+{r['sentiment']:.2f})")

    with t_corr:
        fig = px.scatter(
            df, x="sentiment", y="user_rating",
            color="user_rating", hover_data=['title'],
            title="VADER Lexicon Score vs. Your Rating",
            labels={'sentiment': 'Sentiment (-1 Conflict to +1 Wholesome)'},
            range_x=[-1, 1], trendline="ols",
            color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If the line goes UP, you prefer 'wholesome' plots. If DOWN, you prefer plots with high conflict/action keywords.")

except ImportError:
    st.warning("⚠️ vaderSentiment library not found. Run `pip install vaderSentiment` to see Sentiment Analysis.")