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
from src.shows.season_data import load_seasons, attach_watched_counts
from geo_utils import aggregate_country_counts, make_world_map
from milestones import render_milestones
from ridgeline import genre_rating_ridgeline
from genre_spread import genre_spread_figure, SPREAD_CAPTION
from rating_distribution import render_rating_distribution

from critic_delta import render_critic_delta
from mobile import apply_mobile_css
st.set_page_config(page_title="TV Intelligence Dashboard", page_icon="📺", layout="wide")
apply_mobile_css()

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

        # Episode/watch-time totals must count only the seasons actually watched --
        # number_of_episodes is every season TMDB knows about, including ones that
        # aired after you stopped and ones not yet released.
        df = attach_watched_counts(df)

        return df
    except FileNotFoundError: return None


@st.cache_data
def load_season_data():
    return load_seasons()

df = load_data()
seasons = load_season_data()

if df is None:
    st.error("❌ TV Show Data not found. Run ingestion first.")
    st.stop()

st.title("📊 Personal TV Intelligence")

# Season-wise toggle. Show-level treats each show as one row with its stored
# rating; season-wise expands to one row per season you rated, which is the
# honest view for the 29 shows whose quality moved season to season.
has_seasons = not seasons.empty and seasons['watched'].any()
season_view = False
if has_seasons:
    season_view = st.toggle(
        "Season-wise",
        value=False,
        help="Break multi-season shows into the individual seasons you rated, "
             "instead of one averaged row per show.")
else:
    st.caption("Season sheet not found — showing show-level figures only.")

s_watched = seasons[seasons['watched']] if has_seasons else pd.DataFrame()

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

if season_view:
    with c1: lb_metric("Seasons Rated", f"{len(s_watched):,}",
                       help_text=f"Across {s_watched['tmdb_id'].nunique()} shows.")
    with c2: lb_metric("Avg Season Rating", f"{s_watched['season_rating'].mean():.2f}")
else:
    with c1: lb_metric("Total Shows", len(df))
    with c2: lb_metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")

# Watched seasons only — see src/shows/season_data.py.
total_eps = df['episodes_watched'].sum()
skipped_eps = pd.to_numeric(df['number_of_episodes'], errors='coerce').fillna(0).sum() - total_eps
with c3: lb_metric("Episodes Watched", f"{int(total_eps):,}",
                   help_text=f"{int(skipped_eps):,} episodes from seasons you did not "
                             f"watch are excluded.")
with c4: lb_metric("Liked Shows", f"{df['is_liked'].sum()}")

total_hours = df['hours_watched'].sum()
total_days = total_hours / 24
total_months = total_days / 30.44
watch_time_tooltip = f"{total_days:.1f} days (~{total_months:.1f} months), watched seasons only"

with c5: lb_metric("Total Watch Time", f"{int(total_hours):,} hrs", help_text=watch_time_tooltip)
with c6:
    if 'genres_list' in df.columns:
        gen_entropy = calculate_entropy(df['genres_list'])
        lb_metric("Taste Diversity", f"{gen_entropy:.2f}", help_text="Shannon Entropy of Genres.")

st.markdown("#### Rating Distribution" + (" (per season)" if season_view else ""))
rating_src = s_watched['season_rating'] if season_view else df['user_rating']
# Critic scores are per show, so they stay show-level even in the season view.
render_rating_distribution(
    rating_src,
    critic_sources=[
        ('IMDb', df.get('imdb_rating'), 10),
        ('TMDB', df.get('vote_average'), 10),
        ('TVmaze', df.get('tvmaze_rating'), 10),
    ],
    key="tv_dist", user_color='#00e054', critic_color='#40bcf4',
    style_fn=lambda f: f.update_traces(marker_line_width=1, marker_line_color="#14171C"))

if season_view:
    # The payoff of the season view: where a show's quality actually moved.
    st.markdown("#### Season-to-Season Movement")
    multi = s_watched[s_watched['tmdb_id'].isin(
        s_watched['tmdb_id'].value_counts()[lambda x: x > 1].index)]
    if not multi.empty:
        swing = (multi.groupby('show_name')['season_rating']
                 .agg(['min', 'max', 'mean', 'count']))
        swing['swing'] = (swing['max'] - swing['min']).round(2)
        swing = swing[swing['swing'] > 0].sort_values('swing', ascending=False)
        cA, cB = st.columns([3, 2])
        with cA:
            fig_sw = px.line(
                multi[multi['show_name'].isin(swing.head(8).index)].sort_values(
                    ['show_name', 'season_number']),
                x='season_number', y='season_rating', color='show_name', markers=True,
                labels={'season_number': 'Season', 'season_rating': 'Your rating',
                        'show_name': 'Show'},
                title="Biggest swings across seasons")
            fig_sw.update_layout(xaxis=dict(tickmode='linear', dtick=1))
            st.plotly_chart(fig_sw, use_container_width=True)
        with cB:
            st.dataframe(
                swing.reset_index()[['show_name', 'count', 'min', 'max', 'swing']]
                     .rename(columns={'show_name': 'Show', 'count': 'Seasons',
                                      'min': 'Worst', 'max': 'Best', 'swing': 'Swing'}),
                hide_index=True, use_container_width=True, height=420)
        st.caption(f"{len(swing)} of {multi['tmdb_id'].nunique()} multi-season shows "
                   f"you rated moved by at least half a star between seasons — that "
                   f"spread is what a single show-level rating averages away.")

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
st.subheader("🏷️ Theme Keywords")
st.caption("TMDB keywords (topical) plus TVDB tags (thematic — 'Workplace', "
           "'Boss/Employee'), both finer-grained than genre.")
if 'keywords' in df.columns and df['keywords'].notna().any():
    kw_rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get('user_rating')):
            continue
        seen = set()
        for src in ('keywords', 'tvdb_tags'):
            val = r.get(src)
            if pd.isna(val):
                continue
            for k in str(val).split(','):
                k = k.strip()
                if k and k.lower() not in seen:
                    seen.add(k.lower())
                    kw_rows.append({'keyword': k, 'rating': r['user_rating'],
                                    'name': r.get('name'), 'source': src})
    kw_df = pd.DataFrame(kw_rows)
    if not kw_df.empty:
        MIN_N = 3  # smaller library than movies, so a lower bar
        agg = (kw_df.groupby('keyword')
               .agg(Shows=('rating', 'size'), AvgRating=('rating', 'mean'))
               .query('Shows >= @MIN_N').sort_values('AvgRating', ascending=False))
        agg['AvgRating'] = agg['AvgRating'].round(2)
        overall = df['user_rating'].mean()
        c_hi, c_lo = st.columns(2)
        for col, sub, title, colour in (
                (c_hi, agg.head(10), f"Themes you rate highest (≥{MIN_N} shows)", HUB_PALETTE[1]),
                (c_lo, agg.tail(10), f"Themes you rate lowest (≥{MIN_N} shows)", HUB_PALETTE[0])):
            with col:
                d = sub.reset_index().sort_values('AvgRating')
                fig = px.bar(d, x='AvgRating', y='keyword', orientation='h',
                             hover_data=['Shows'], title=title,
                             color_discrete_sequence=[colour])
                fig.add_vline(x=overall, line_dash="dot", line_color="#99AABB")
                fig.update_layout(yaxis_title=None, height=380)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("##### 🔎 Search by Keyword")
        all_kws = sorted(kw_df['keyword'].unique(), key=str.lower)
        st.caption(f"Search any of your {len(all_kws):,} show keywords and see every "
                   f"show carrying it, with your rating.")
        q = st.text_input("Keyword", placeholder="e.g. workplace, time travel, heist",
                          key="tv_kw_search").strip().lower()
        if q:
            matches = [k for k in all_kws if q in k.lower()]
            if not matches:
                st.warning(f"No keyword matches “{q}”.")
            else:
                pick = matches[0] if len(matches) == 1 else st.selectbox(
                    f"{len(matches)} matching keywords", matches, key="tv_kw_pick")
                hits = (kw_df[kw_df['keyword'] == pick]
                        .merge(df[['name', 'year', 'poster_path', 'seasons_watched']],
                               on='name', how='left')
                        .drop_duplicates('name')
                        .sort_values('rating', ascending=False))
                avg = hits['rating'].mean()
                m1, m2, m3 = st.columns(3)
                with m1: lb_metric("Shows", len(hits))
                with m2: lb_metric("Your Avg", f"{avg:.2f}")
                with m3: lb_metric("vs Overall", f"{avg - df['user_rating'].mean():+.2f}",
                                   help_text=f"Library average {df['user_rating'].mean():.2f}.")
                show = hits[['name', 'year', 'rating', 'seasons_watched']].rename(
                    columns={'name': 'Show', 'year': 'Year', 'rating': 'Your Rating',
                             'seasons_watched': 'Seasons Watched'})
                show['Year'] = pd.to_numeric(show['Year'], errors='coerce').astype('Int64')
                st.dataframe(show, use_container_width=True, hide_index=True,
                             height=min(520, 60 + 35 * len(show)))
                posters = hits[hits['poster_path'].notna()].head(10)
                if not posters.empty:
                    st.caption("Highest rated first")
                    for col, (_, r) in zip(st.columns(len(posters)), posters.iterrows()):
                        with col:
                            st.image(f"https://image.tmdb.org/t/p/w185{r['poster_path']}")
                            st.caption(f"{r['rating']:.1f}★")
        else:
            st.info("Type a keyword above to see every show that carries it.")

        with st.expander(f"All {len(agg)} keywords on ≥{MIN_N} shows"):
            st.dataframe(agg.reset_index(), use_container_width=True, hide_index=True)
else:
    st.info("Keywords not ingested yet — run `python -m src.shows.enrich_extended --apply`.")

st.divider()

st.subheader("⏳ Time & Duration")
t_dec, t_run, t_sea = st.tabs(["Over Time", "Seasons & Episodes", "Release Season"])

with t_dec:
    # Continuous by year, not decade buckets. In season view each season is placed
    # in ITS OWN air year -- a show running 2013-2021 genuinely spans two decades,
    # which a single show-level first-air year hides.
    if season_view:
        src = s_watched.dropna(subset=['air_year']).copy()
        src['_year'] = src['air_year'].astype(int)
        src['_rating'] = src['season_rating']
        unit, title = "seasons", "Seasons watched per year"
    else:
        src = df.dropna(subset=['year']).copy()
        src['_year'] = src['year'].astype(int)
        src['_rating'] = src['user_rating']
        unit, title = "shows", "Shows per year (by first air year)"

    per_year = (src.groupby('_year')
                .agg(Count=('_rating', 'size'), avg_rating=('_rating', 'mean'))
                .reindex(range(int(src['_year'].min()), int(src['_year'].max()) + 1),
                         fill_value=0)
                .reset_index().rename(columns={'index': '_year'}))
    per_year['avg_rating'] = per_year['avg_rating'].round(2)

    ent = calculate_entropy(src['_year'] // 10 * 10)
    st.caption(f"**Decade Diversity (Entropy):** {ent:.2f} — "
               f"{len(src):,} {unit} across {per_year['_year'].max() - per_year['_year'].min() + 1} years.")

    fig = px.area(per_year, x='_year', y='Count', title=title,
                  hover_data={'avg_rating': ':.2f'},
                  labels={'_year': 'Year', 'Count': unit.capitalize()},
                  color_discrete_sequence=[HUB_PALETTE[0]])
    fig.update_traces(mode='lines', line_shape='spline', line_width=2,
                      fillcolor='rgba(64,188,244,0.18)')
    rated = per_year[per_year['Count'] > 0]
    fig.add_scatter(x=rated['_year'], y=rated['avg_rating'], name='Avg rating',
                    yaxis='y2', mode='lines+markers', line=dict(width=1.5, dash='dot'),
                    marker=dict(size=4), line_color=HUB_PALETTE[2 % len(HUB_PALETTE)])
    fig.update_layout(
        yaxis2=dict(title='Avg rating', overlaying='y', side='right',
                    range=[0, 5], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified')
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

    st.divider()
    # --- Added: the same comparison at TITLE level, not just in aggregate. The
    # bars above say how OFTEN you agree; these two say WHERE you disagree.
    render_critic_delta(
        df=df,
        title_col='name',
        user_col='user_rating',
        critic_sources=[
            ('IMDb', df.get('imdb_rating'), 10),
            ('TMDB', df.get('vote_average'), 10),
            ('TVmaze', df.get('tvmaze_rating'), 10),
            # NOT tvdb_score: despite the name it holds vote COUNTS
            # (up to 9.1M in this library), not a 0-10 rating.
        ],
        key='tv_cd',
    )

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
            st.markdown("**Rating Spread per Genre**")
            st.caption(SPREAD_CAPTION.format(noun="show"))
            fig = genre_spread_figure(
                df_gen_exp, 'genres_list', 'user_rating', 'name', HUB_PALETTE,
                min_count=3, x_title="Genre", y_title="Your Rating",
                year_col='year', height=520, noun="show(s)")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough genre-tagged ratings to draw the spread.")

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

# RATING DISTRIBUTION BY GENRE (RIDGELINE)
st.subheader("🎚️ Rating Distribution by Genre")
st.caption("A ridgeline of how your scores are spread *within* each genre. Each curve is a "
           "smoothed distribution of your ratings; warmer curves are genres you rate higher on "
           "average, and the number on the right is how many shows fed that curve. Genres are "
           "stacked from your lowest- to highest-rated.")
if 'genres_list' in df.columns:
    _ridge_pairs = (df[['genres_list', 'user_rating']].explode('genres_list')
                    .rename(columns={'genres_list': 'genre', 'user_rating': 'rating'}))
    _fig_ridge = genre_rating_ridgeline(_ridge_pairs, rating_label="Your Rating",
                                        count_label="shows", min_count=4)
    if _fig_ridge is not None:
        st.plotly_chart(_fig_ridge, use_container_width=True)
    else:
        st.info("Not enough genre-tagged ratings to draw the ridgeline.")

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
