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
sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from src import config
from geo_utils import aggregate_country_counts, make_world_map
from milestones import render_milestones

st.set_page_config(page_title="TV Intelligence Dashboard", page_icon="📺", layout="wide")

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
    if pd.isna(rating): return ""
    full_stars = int(rating)
    half_star = 1 if (rating - full_stars) >= 0.5 else 0
    stars = "★" * full_stars + ("½" if half_star else "")
    return f"<span style='color:#00e054; font-family: courier; font-size: 1.2em;'>{stars}</span>"

def parse_list_col(x):
    if pd.isna(x): return []
    try:
        if isinstance(x, str) and '[' not in x and ',' in x:
            return [i.strip() for i in x.split(',')]
        return ast.literal_eval(str(x))
    except: return [str(x)]

# Canonicalize language labels so ISO codes and full names collapse together
# (e.g. "en"/"English", "hi"/"Hindi", "ja"/"Japanese" all map to one label).
LANGUAGE_MAP = {
    'en': 'English', 'eng': 'English', 'english': 'English',
    'hi': 'Hindi', 'hin': 'Hindi', 'hindi': 'Hindi',
    'ja': 'Japanese', 'jpn': 'Japanese', 'japanese': 'Japanese',
    'ko': 'Korean', 'kor': 'Korean', 'korean': 'Korean',
    'fr': 'French', 'fra': 'French', 'french': 'French',
    'es': 'Spanish', 'spa': 'Spanish', 'spanish': 'Spanish',
    'de': 'German', 'ger': 'German', 'deu': 'German', 'german': 'German',
    'it': 'Italian', 'ita': 'Italian', 'italian': 'Italian',
    'ru': 'Russian', 'rus': 'Russian', 'russian': 'Russian',
    'pt': 'Portuguese', 'por': 'Portuguese', 'portuguese': 'Portuguese',
    'nl': 'Dutch', 'dut': 'Dutch', 'dutch': 'Dutch',
    'zh': 'Chinese', 'cn': 'Chinese', 'zho': 'Chinese', 'chinese': 'Chinese',
    'mandarin': 'Chinese', 'cantonese': 'Chinese',
    'pa': 'Punjabi', 'punjabi': 'Punjabi',
    'bn': 'Bengali', 'ben': 'Bengali', 'bengali': 'Bengali',
    'ta': 'Tamil', 'tamil': 'Tamil',
    'te': 'Telugu', 'telugu': 'Telugu',
    'mr': 'Marathi', 'marathi': 'Marathi',
    'sv': 'Swedish', 'swedish': 'Swedish',
    'no': 'Norwegian', 'nb': 'Norwegian', 'norwegian': 'Norwegian',
    'th': 'Thai', 'thai': 'Thai',
    'uk': 'Ukrainian', 'ukrainian': 'Ukrainian',
    'la': 'Latin', 'latin': 'Latin',
}

def normalize_language(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return val
    key = str(val).strip().lower()
    if key in ('', 'nan', 'none', 'unknown'):
        return str(val).strip()
    return LANGUAGE_MAP.get(key, str(val).strip().title())

def get_frequent_items_with_avg(df, col_name, top_n=10):
    df_exp = df.explode(col_name)
    df_exp = df_exp[~df_exp[col_name].isin(['Unknown', 'nan', np.nan, ''])]
    stats = df_exp.groupby(col_name).agg(
        Count=('user_rating', 'size'), 
        avg_rating=('user_rating', 'mean')
    ).reset_index()
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('Count', ascending=False).head(top_n)

def get_highly_rated_items(df, col_name, top_n=10, min_count=2):
    df_exploded = df.explode(col_name)
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan, ''])]
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    stats = stats[stats['count'] >= min_count]
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('avg_rating', ascending=False).head(top_n)

def get_lowest_rated_items(df, col_name, top_n=10, min_count=2):
    df_exploded = df.explode(col_name)
    df_exploded = df_exploded[~df_exploded[col_name].isin(['Unknown', 'nan', np.nan, ''])]
    stats = df_exploded.groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'),
        count=('user_rating', 'count')
    ).reset_index()
    stats = stats[stats['count'] >= min_count]
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('avg_rating', ascending=True).head(top_n)

def calculate_entropy(series):
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
        df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        
        # Clean Numerics
        numeric_cols = ['year', 'number_of_episodes', 'number_of_seasons', 
                        'vote_average', 'vote_count', 'imdb_rating', 'user_rating', 'runtime']
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        # Parse Lists
        list_cols = ['genres', 'created_by', 'actors', 'production_companies', 'country', 'writer']
        for c in list_cols:
            if c in df.columns: df[f'{c}_list'] = df[c].apply(parse_list_col)

        # Collapse equivalent language labels (en == English, ja == Japanese, etc.)
        if 'language' in df.columns:
            df['language'] = df['language'].apply(normalize_language)

        df['decade'] = (df['year'] // 10) * 10
        
        # Handle is_liked
        if 'is_liked' not in df.columns:
            df['is_liked'] = df['user_rating'].apply(lambda x: 1 if pd.notnull(x) and x >= 4.0 else 0)
            
        return df
    except FileNotFoundError: return None

df = load_data()

if df is None:
    st.error("❌ TV Show Data not found. Run ingestion first.")
    st.stop()

st.title("📊 Personal TV Intelligence")

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

with c1: lb_metric("Total Shows", len(df))
with c2: lb_metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")

total_eps = df['number_of_episodes'].sum()
with c3: lb_metric("Total Episodes", f"{int(total_eps):,}")
with c4: lb_metric("Liked Shows", f"{df['is_liked'].sum()}")

# Calculate Watch Time Tooltip
# Use runtime if available, else default 45 min
df['est_runtime'] = df['runtime'].fillna(45)
total_hours = (df['number_of_episodes'] * df['est_runtime']).sum() / 60
total_days = total_hours / 24
total_months = total_days / 30.44
watch_time_tooltip = f"{total_days:.1f} days (~{total_months:.1f} months)"

with c5: lb_metric("Total Watch Time", f"{int(total_hours):,} hrs", help_text=watch_time_tooltip)
with c6:
    if 'genres_list' in df.columns:
        gen_entropy = calculate_entropy(df['genres_list'])
        lb_metric("Taste Diversity", f"{gen_entropy:.2f}", help_text="Shannon Entropy of Genres.")

st.markdown("#### Your Rating Distribution")
rating_counts = df['user_rating'].value_counts().reset_index()
rating_counts.columns = ['Rating', 'Count']
fig_ratings = px.bar(rating_counts.sort_values('Rating'), x='Rating', y='Count', 
                     text='Count', color_discrete_sequence=['#00e054'])
fig_ratings.update_traces(textposition='outside', marker_line_width=1, marker_line_color="#14171C")
fig_ratings.update_layout(xaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5))
st.plotly_chart(fig_ratings, use_container_width=True)

st.divider()

# 1b. MILESTONE SHOWS (the 1st / 50th / 100th / … show you logged)
# TV ratings carry only a year watched (no precise log date), so the logged
# sequence is taken by year watched, then entry order within a year.
shows_logged = df.copy()
if 'year_watched' in shows_logged.columns:
    shows_logged['_yw'] = pd.to_numeric(shows_logged['year_watched'], errors='coerce')
    shows_logged['_orig'] = range(len(shows_logged))
    shows_logged = shows_logged.sort_values(['_yw', '_orig'], kind='stable',
                                            na_position='last').reset_index(drop=True)


def _yw_sub(row):
    yw = row.get('year_watched')
    return f"watched {int(yw)}" if pd.notna(pd.to_numeric(yw, errors='coerce')) else None


render_milestones(shows_logged, 'name', 'poster_path', "🏁 Milestone Shows (logged)",
                  sub_fn=_yw_sub)

st.divider()

# 2. TIME, LENGTH & SEASONALITY
st.subheader("⏳ Time & Duration")
t_dec, t_run, t_sea = st.tabs(["Decades", "Seasons & Episodes", "Release Season"])

with t_dec:
    dec_ent = calculate_entropy(df['decade'])
    st.caption(f"**Decade Diversity (Entropy):** {dec_ent:.2f}")
    df_decade = df.groupby('decade').agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
    df_decade['avg_rating'] = df_decade['avg_rating'].round(3)
    fig = px.bar(df_decade, x='decade', y='Count', hover_data={'avg_rating': ':.3f'}, title="Shows per Decade", color_discrete_sequence=[HUB_PALETTE[0]])
    st.plotly_chart(fig, use_container_width=True)

with t_run:
    c_sea, c_eps = st.columns(2)
    with c_sea:
        fig = px.histogram(df, x="number_of_seasons", title="Distribution of Seasons", color_discrete_sequence=[HUB_PALETTE[1]])
        st.plotly_chart(fig, use_container_width=True)
    with c_eps:
        fig = px.scatter(df, x="number_of_seasons", y="user_rating", hover_data=['name'], trendline="ols",
                         title="Seasons vs Your Rating", color_discrete_sequence=[HUB_PALETTE[2]])
        st.plotly_chart(fig, use_container_width=True)

with t_sea:
    if 'first_air_date' in df.columns:
        df['air_date_dt'] = pd.to_datetime(df['first_air_date'], errors='coerce')
        df['month'] = df['air_date_dt'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        season_agg = df.groupby('month')['user_rating'].mean().reindex(month_order).reset_index()
        season_agg['user_rating'] = season_agg['user_rating'].round(3)
        fig = px.bar(season_agg, x='month', y='user_rating', title="Avg Rating by Premiere Month", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[3]])
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 3. PEOPLE & NETWORKS
st.subheader("👥 Cast, Creators & Networks")
tab_cre, tab_act, tab_wri, tab_net = st.tabs(["✍️ Creators", "🎭 Actors", "✍️ Writers", "📡 Networks"])

def render_people_tab(col_list, label, min_count_threshold=2):
    ent = calculate_entropy(df[col_list])
    st.caption(f"**{label} Diversity (Entropy):** {ent:.2f}")
    c_freq, c_high, c_low = st.columns(3)
    with c_freq:
        st.caption(f"**Most Watched {label}**")
        freq = get_frequent_items_with_avg(df, col_list)
        fig = px.bar(freq, x='Count', y=col_list, orientation='h', color='Count', color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]])
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with c_high:
        st.caption(f"**Highest Rated {label}** (Min {min_count_threshold} Shows)")
        rated = get_highly_rated_items(df, col_list, min_count=min_count_threshold)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y=col_list, orientation='h', color='avg_rating', 
                         color_continuous_scale=[[0, HUB_PALETTE[1]], [1, HUB_PALETTE[2]]], range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    with c_low:
        st.caption(f"**Lowest Rated {label}** (Min {min_count_threshold} Shows)")
        low_rated = get_lowest_rated_items(df, col_list, min_count=min_count_threshold)
        if not low_rated.empty:
            fig = px.bar(low_rated, x='avg_rating', y=col_list, orientation='h', color='avg_rating', 
                         color_continuous_scale=[[0, "#ff4b4b"], [1, HUB_PALETTE[0]]], range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder':'total descending'})
            st.plotly_chart(fig, use_container_width=True)

with tab_cre: render_people_tab('created_by_list', "Creators", 2)
with tab_act: render_people_tab('actors_list', "Actors", 3)
with tab_wri: render_people_tab('writer_list', "Writers", 2)
with tab_net:
    if 'network' in df.columns:
        net_stats = df.groupby('network').agg(Count=('user_rating', 'size'), Avg_Rating=('user_rating', 'mean')).reset_index()
        net_stats = net_stats.sort_values('Count', ascending=False).head(15)
        fig = px.bar(net_stats, x='Count', y='network', orientation='h', color='Avg_Rating', 
                     title="Top Networks/Platforms", color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4. SCORES & CONSENSUS
st.subheader("📈 Scores & Consensus Alignment")
t1, t2 = st.tabs(["Critic Alignment", "Global Stats"])

with t1:
    st.markdown("#### How often do you agree with IMDb & TMDB?")
    st.caption("Each source is rescaled to your 0–5 scale, then compared show-by-show. "
               "**Same** = within 0.5★ · **Close** = within 1★ · **Far** = more than 1★ apart.")

    critic_specs = []
    if 'imdb_rating' in df.columns:
        critic_specs.append(('IMDb', df['imdb_rating'] / 2.0))     # 0-10 -> 0-5
    if 'vote_average' in df.columns:
        critic_specs.append(('TMDB', df['vote_average'] / 2.0))    # 0-10 -> 0-5

    rows, summary = [], []
    for label, norm in critic_specs:
        comp = pd.DataFrame({'user': df['user_rating'], 'critic': pd.to_numeric(norm, errors='coerce')})
        comp = comp[comp['critic'] > 0].dropna()
        if comp.empty:
            continue
        d = (comp['user'] - comp['critic']).abs()
        same = int((d <= 0.5).sum())
        close = int(((d > 0.5) & (d <= 1.0)).sum())
        far = int((d > 1.0).sum())
        rows += [{'Critic': label, 'Agreement': 'Same', 'Count': same},
                 {'Critic': label, 'Agreement': 'Close', 'Count': close},
                 {'Critic': label, 'Agreement': 'Far', 'Count': far}]
        mode_bucket = max([('Same', same), ('Close', close), ('Far', far)], key=lambda x: x[1])[0]
        summary.append({'Critic': label, '% Same': round(100 * same / len(comp), 1),
                        'Avg |diff|': round(float(d.mean()), 2), 'Most common': mode_bucket, 'Shows': len(comp)})

    if rows:
        agg_df = pd.DataFrame(rows)
        order = [s['Critic'] for s in sorted(summary, key=lambda s: (-s['% Same'], s['Avg |diff|']))]
        fig = px.bar(agg_df, x='Critic', y='Count', color='Agreement', barmode='group',
                     title="Agreement Breakdown by Source",
                     category_orders={'Critic': order, 'Agreement': ['Same', 'Close', 'Far']},
                     color_discrete_map={'Same': HUB_PALETTE[1], 'Close': HUB_PALETTE[2], 'Far': HUB_PALETTE[0]})
        st.plotly_chart(fig, use_container_width=True)

        sdf = pd.DataFrame(summary).sort_values(['% Same', 'Avg |diff|'], ascending=[False, True]).reset_index(drop=True)
        sdf.index += 1
        sdf.index.name = 'Rank'
        st.markdown("**Who you agree with most** — ranked by % Same, then smallest average difference:")
        st.dataframe(sdf, use_container_width=True)
    else:
        st.info("No critic scores available to compare.")

with t2:
    st.markdown("#### 🌍 Your TV World Map")
    st.caption("Countries shaded by how many shows you've watched from there (log-scaled so smaller countries stay visible). Hover for the exact count and your average rating.")
    geo_agg, geo_unmapped = aggregate_country_counts(df, 'country_list', 'user_rating')
    if not geo_agg.empty:
        map_fig = make_world_map(geo_agg, low_color="#2C343F", high_color=HUB_PALETTE[2])
        st.plotly_chart(map_fig, use_container_width=True)
        if geo_unmapped:
            st.caption(f"Not placed on the map (unrecognized region): {', '.join(geo_unmapped)}")
    else:
        st.info("No country data available to map.")

    c_ctry, c_lang = st.columns(2)
    with c_ctry:
        cnt = get_frequent_items_with_avg(df, 'country_list', 8)
        fig = px.bar(cnt, x='country_list', y='Count', title="Top Countries of Origin", color_discrete_sequence=[HUB_PALETTE[2]])
        st.plotly_chart(fig, use_container_width=True)
    with c_lang:
        if 'language' in df.columns:
            lang_stats = df['language'].value_counts().head(8).reset_index()
            fig = px.pie(lang_stats, values='count', names='language', title="Languages", color_discrete_sequence=HUB_PALETTE)
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# 5. VIBE CHECK
st.subheader("🧠 Plot Vibe Check")
st.caption("Dominant plot themes extracted from Overview + Tagline.")

if 'overview' in df.columns:
    df['vibe_text'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    # Same stopword set as the Movies Plot Vibe Check (with TV-specific terms added).
    stops = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'where', 'when', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours',
        'after', 'before', 'while', 'during', 'since', 'until', 'through', 'about', 'against', 'between',
        'movie', 'film', 'story', 'plot', 'character', 'life', 'world', 'find', 'finds', 'one', 'two',
        'back', 'new', 'must', 'will', 'only', 'other', 'city', 'years', 'year', 'himself', 'high', 'most', 'just', 'more', 'there', 'soon',
        'show', 'series', 'episode', 'episodes', 'season', 'seasons',
    ])
    word_ratings = {}
    for _, row in df.dropna(subset=['vibe_text', 'user_rating']).iterrows():
        words = set([w.lower() for w in re.findall(r'\w+', str(row['vibe_text'])) if len(w) > 3 and w.lower() not in stops])
        for w in words:
            if w not in word_ratings: word_ratings[w] = []
            word_ratings[w].append(row['user_rating'])
    word_stats = [{'Word': w, 'Count': len(r), 'Avg_Rating': np.mean(r)} for w, r in word_ratings.items()]
    wdf = pd.DataFrame(word_stats).sort_values('Count', ascending=False).head(50)
    fig = px.treemap(wdf, path=['Word'], values='Count', color='Count', hover_data=['Avg_Rating'], title="Dominant Themes in Your TV Shows", color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 6. HOT TAKES
st.subheader("🌶️ Your Hot Takes")
df['imdb_norm'] = df['imdb_rating'] / 2
df['tmdb_norm'] = df['vote_average'] / 2
df['critic_norm'] = df[['imdb_norm', 'tmdb_norm']].mean(axis=1)
ht_df = df.dropna(subset=['user_rating', 'critic_norm']).copy()
ht_df['controversy'] = ht_df['user_rating'] - ht_df['critic_norm']
ht_df['magnitude'] = ht_df['controversy'].abs()

c1, c2 = st.columns(2)
with c1:
    st.write("**Shows you defended (You > Critics):**")
    for _, row in ht_df[ht_df['controversy'] > 0].sort_values('magnitude', ascending=False).head(5).iterrows():
        st.markdown(f"📈 **{row['name']}** (Diff: +{row['magnitude']:.2f})  \n<span style='color:#99AABB;'>{render_stars(row['user_rating'])} | Public: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)
with c2:
    st.write("**Shows you roasted (You < Critics):**")
    for _, row in ht_df[ht_df['controversy'] < 0].sort_values('magnitude', ascending=False).head(5).iterrows():
        st.markdown(f"📉 **{row['name']}** (Diff: -{row['magnitude']:.2f})  \n<span style='color:#99AABB;'>{render_stars(row['user_rating'])} | Public: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)

st.divider()

# 7. GENRE & MATURITY
st.subheader("🎭 Genre & Maturity Analysis")
t_gen, t_mat = st.tabs(["Genre Influence", "Maturity Ratings"])

with t_gen:
    if 'genres_list' in df.columns:
        df_gen_exp = df.explode('genres_list')
        gen_stats = df_gen_exp.groupby('genres_list').agg(Count=('user_rating', 'size'), Avg_Rating=('user_rating', 'mean')).reset_index().sort_values('Count', ascending=False)
        cg1, cg2 = st.columns([1, 1.2])
        with cg1:
            fig = px.bar(gen_stats.head(15), x='Count', y='genres_list', orientation='h', title="Most Watched Genres", color='Avg_Rating', color_continuous_scale='Viridis')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with cg2:
            valid_genres = gen_stats[gen_stats['Count'] >= 3]['genres_list']
            fig = px.box(df_gen_exp[df_gen_exp['genres_list'].isin(valid_genres)], x="genres_list", y="user_rating", title="Rating Spread per Genre", color_discrete_sequence=HUB_PALETTE)
            st.plotly_chart(fig, use_container_width=True)

with t_mat:
    if 'age_rating' in df.columns:
        mat_stats = df.groupby('age_rating').agg(Count=('user_rating', 'size'), Avg_Rating=('user_rating', 'mean')).reset_index().sort_values('Count', ascending=False)
        cm1, cm2 = st.columns(2)
        with cm1:
            fig = px.bar(mat_stats, x='age_rating', y='Count', title="Maturity Rating Breakdown", color_discrete_sequence=[HUB_PALETTE[2]])
            st.plotly_chart(fig, use_container_width=True)
        with cm2:
            fig = px.box(df, x="age_rating", y="user_rating", title="Rating Spread per Maturity Level", color_discrete_sequence=HUB_PALETTE)
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# 8. SENTIMENT ANALYSIS
st.subheader("🧠 Emotional Sentiment Analysis")
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    df['sentiment'] = df['overview'].fillna('').apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if len(str(x)) > 5 else 0)
    
    cs1, cs2 = st.columns([1, 2])
    with cs1:
        df['mood'] = pd.cut(df['sentiment'], bins=[-1, -0.25, 0.25, 1], labels=['Dark/Tragic', 'Neutral', 'Wholesome'])
        mood_counts = df['mood'].value_counts().reset_index()
        fig = px.pie(mood_counts, values='count', names='mood', title="Sentiment Split", hole=0.4, color_discrete_sequence=[HUB_PALETTE[3], HUB_PALETTE[2], HUB_PALETTE[1]])
        st.plotly_chart(fig, use_container_width=True)
    with cs2:
        fig = px.scatter(df, x="sentiment", y="user_rating", color="user_rating", hover_data=['name'], title="VADER Sentiment vs Your Rating", range_x=[-1, 1], trendline="ols", color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
except ImportError:
    st.warning("⚠️ vaderSentiment library not found. Run `pip install vaderSentiment` to see Sentiment Analysis.")

st.divider()

# 9. AWARDS
if 'awards' in df.columns:
    st.subheader("🏆 Award Recognition")
    df['has_awards'] = df['awards'].fillna('').apply(lambda x: 1 if len(str(x)) > 5 else 0)
    awd_agg = df.groupby('has_awards')['user_rating'].mean().reset_index()
    awd_agg['Label'] = awd_agg['has_awards'].map({1: 'Award Nominee/Winner', 0: 'No Major Awards'})
    fig = px.bar(awd_agg, x='Label', y='user_rating', title="Do you prefer Award-Winning Shows?", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[1], HUB_PALETTE[2]])
    st.plotly_chart(fig, use_container_width=True)
