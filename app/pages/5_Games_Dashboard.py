import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import re
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

# Shared stopword set (same as movies Plot Vibe Check)
STOPS = set([
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'where', 'when', 'how', 'who', 'which', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without',
    'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
    'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours',
    'after', 'before', 'while', 'during', 'since', 'until', 'through', 'about', 'against', 'between',
    'game', 'games', 'play', 'player', 'players', 'story', 'plot', 'world', 'find', 'finds', 'one', 'two',
    'back', 'new', 'must', 'will', 'only', 'other', 'years', 'year', 'most', 'just', 'more', 'there', 'soon', 'their', 'also', 'series',
])


def calculate_entropy(series):
    import math
    all_items = []
    for val in series.dropna():
        if isinstance(val, list):
            all_items.extend([str(i).strip() for i in val if str(i).strip().lower() not in ['unknown', 'nan', 'none', '']])
        elif str(val).strip().lower() not in ['unknown', 'nan', 'none', '']:
            all_items.append(str(val).strip())
    if not all_items:
        return 0
    counts = Counter(all_items)
    total = len(all_items)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
        df['my_rating'] = pd.to_numeric(df['my_rating'], errors='coerce')
        df['metacritic'] = pd.to_numeric(df['metacritic'], errors='coerce')
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  # RAWG community rating (0-5)
        if 'playtime' in df.columns:
            df['playtime'] = pd.to_numeric(df['playtime'], errors='coerce')
        # Parse release year from DD-MM-YYYY style strings
        df['release_dt'] = pd.to_datetime(df['released'], errors='coerce', dayfirst=True)
        df['release_year'] = df['release_dt'].dt.year
        # Lists
        df['genre_list'] = df['genres'].fillna('').str.split(', ')
        df['dev_list'] = df['developers'].fillna('').str.split(', ')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


df = load_data()
if df is None:
    st.stop()

st.title("🎮 Game Intelligence Dashboard")

# --- METRICS (data-focused, no model stats) ---
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.metric("Total Games", len(df))
with c2:
    st.metric("Avg My Rating", f"{df['my_rating'].mean():.2f}")
with c3:
    st.metric("Avg Metacritic", f"{df['metacritic'].dropna().mean():.1f}")
with c4:
    if 'playtime' in df.columns and df['playtime'].notna().any():
        total_hrs = df['playtime'].fillna(0).sum()
        st.metric("Est. Hours to Beat", f"{int(total_hrs):,} hrs",
                  help="Sum of RAWG average completion time across your library.")
    else:
        st.metric("Est. Hours to Beat", "N/A")
with c5:
    st.metric("Genre Diversity", f"{calculate_entropy(df['genre_list']):.2f}",
              help="Shannon entropy of genres.")

# --- DISTRIBUTIONS ---
col_left, col_right = st.columns([1, 1])
with col_left:
    st.subheader("⭐ My Rating Distribution")
    rating_counts = df['my_rating'].value_counts().reset_index()
    rating_counts.columns = ['Rating', 'Count']
    fig = px.bar(rating_counts.sort_values('Rating'), x='Rating', y='Count',
                 text='Count', color_discrete_sequence=[HUB_PALETTE[1]])
    fig.update_traces(textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🖥️ Platform Breakdown")
    plat_counts = df['platform_from_text'].value_counts().reset_index()
    plat_counts.columns = ['Platform', 'Count']
    fig = px.pie(plat_counts, values='Count', names='Platform', color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- DETAILED ANALYSIS ---
tab_gen, tab_dev, tab_crit, tab_time, tab_era, tab_age, tab_vibe, tab_list = st.tabs(
    ["Genres", "Developers", "Critic Alignment", "Time to Beat", "Release Era", "Age Rating", "Vibe Check", "Full List"]
)

with tab_gen:
    st.subheader("🏷️ Genre Influence")
    df_genres = df.explode('genre_list')
    df_genres = df_genres[~df_genres['genre_list'].isin(['', 'nan', np.nan])]
    gen_stats = df_genres.groupby('genre_list').agg(
        Count=('my_rating', 'size'), Avg_Rating=('my_rating', 'mean')
    ).reset_index()
    gen_stats['Avg_Rating'] = gen_stats['Avg_Rating'].round(2)
    gen_stats = gen_stats.sort_values('Count', ascending=False)

    cg1, cg2 = st.columns([1, 1.2])
    with cg1:
        fig = px.bar(gen_stats.head(15), x='Count', y='genre_list', orientation='h',
                     title="Most Played Genres", color='Avg_Rating',
                     color_continuous_scale='Viridis', text='Count')
        fig.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None})
        st.plotly_chart(fig, use_container_width=True)
    with cg2:
        valid = gen_stats[gen_stats['Count'] >= 3]['genre_list']
        fig = px.box(df_genres[df_genres['genre_list'].isin(valid)], x='genre_list', y='my_rating',
                     title="Rating Spread per Genre (Min 3 Games)", color_discrete_sequence=HUB_PALETTE)
        fig.update_layout(showlegend=False, xaxis_title="Genre", yaxis_title="My Rating")
        st.plotly_chart(fig, use_container_width=True)

with tab_dev:
    st.subheader("👨‍💻 Developers")
    df_devs = df.explode('dev_list')
    df_devs = df_devs[~df_devs['dev_list'].isin(['', 'nan', np.nan])]
    dev_stats = df_devs.groupby('dev_list').agg(
        Count=('my_rating', 'size'), Avg_Rating=('my_rating', 'mean')
    ).reset_index()
    dev_stats['Avg_Rating'] = dev_stats['Avg_Rating'].round(2)

    cd1, cd2 = st.columns(2)
    with cd1:
        st.caption("**Most Played Developers**")
        top_dev = dev_stats.sort_values('Count', ascending=False).head(15)
        fig = px.bar(top_dev, x='Count', y='dev_list', orientation='h', color='Count',
                     color_continuous_scale='Magma')
        fig.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None})
        st.plotly_chart(fig, use_container_width=True)
    with cd2:
        st.caption("**Highest Rated Developers** (Min 2 Games)")
        rated_dev = dev_stats[dev_stats['Count'] >= 2].sort_values('Avg_Rating', ascending=False).head(15)
        if not rated_dev.empty:
            fig = px.bar(rated_dev, x='Avg_Rating', y='dev_list', orientation='h', color='Avg_Rating',
                         color_continuous_scale='Viridis', range_x=[0, 5])
            fig.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough repeat developers for a rating ranking.")

with tab_crit:
    st.subheader("🎯 Do you agree with the critics?")
    st.caption("How often your rating matches each source. **Same** = within 0.5★, **Close** = within 1★, **Far** = beyond 1★ (all normalised to your 0–5 scale).")

    critic_specs = []
    if 'metacritic' in df.columns:
        critic_specs.append(('Metacritic', df['metacritic'] / 20.0))   # 0-100 -> 0-5
    if 'rating' in df.columns:
        critic_specs.append(('RAWG Users', df['rating']))              # already 0-5

    rows, summary = [], []
    for label, norm in critic_specs:
        comp = pd.DataFrame({'user': df['my_rating'], 'critic': norm}).dropna()
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
        summary.append({'Critic': label, 'Avg |diff|': round(d.mean(), 2), 'Mode': mode_bucket,
                        '% Same': round(100 * same / len(comp), 1), 'N': len(comp)})

    if rows:
        agg_df = pd.DataFrame(rows)
        fig = px.bar(agg_df, x='Critic', y='Count', color='Agreement', barmode='group',
                     title="Agreement Breakdown",
                     color_discrete_map={'Same': HUB_PALETTE[1], 'Close': HUB_PALETTE[2], 'Far': HUB_PALETTE[0]})
        st.plotly_chart(fig, use_container_width=True)
        sdf = pd.DataFrame(summary).sort_values(['% Same', 'Avg |diff|'], ascending=[False, True]).reset_index(drop=True)
        sdf.index += 1
        st.markdown("**Ranking — who you agree with most** (by % Same, then lowest avg difference):")
        st.dataframe(sdf, use_container_width=True)
    else:
        st.info("No critic scores available to compare.")

with tab_time:
    st.subheader("⏱️ Time to Beat")
    if 'playtime' in df.columns and df['playtime'].notna().any():
        pt = df[df['playtime'] > 0]
        ct1, ct2 = st.columns(2)
        with ct1:
            fig = px.histogram(pt, x='playtime', nbins=20, title="Hours-to-Beat Distribution",
                               color_discrete_sequence=[HUB_PALETTE[0]])
            fig.update_layout(xaxis_title="Hours to beat (RAWG avg)")
            st.plotly_chart(fig, use_container_width=True)
        with ct2:
            fig = px.scatter(pt, x='playtime', y='my_rating', hover_data=['name'], trendline='ols',
                             title="Does a longer game earn a higher rating?",
                             color='my_rating', color_continuous_scale='RdYlGn')
            fig.update_layout(xaxis_title="Hours to beat", yaxis_title="My Rating")
            st.plotly_chart(fig, use_container_width=True)
        st.caption("**Longest games in your library:**")
        longest = pt.sort_values('playtime', ascending=False).head(8)[['name', 'playtime', 'my_rating']]
        longest.columns = ['Game', 'Hours to Beat', 'My Rating']
        st.dataframe(longest, use_container_width=True, hide_index=True)
    else:
        st.info("No RAWG playtime data available. Re-run `python src/games/ingestion.py` to backfill it.")

with tab_era:
    st.subheader("📅 Release Era")
    era = df.dropna(subset=['release_year'])
    if not era.empty:
        era_agg = era.groupby('release_year').agg(
            Count=('my_rating', 'size'), Avg_Rating=('my_rating', 'mean')
        ).reset_index()
        era_agg['Avg_Rating'] = era_agg['Avg_Rating'].round(2)
        fig = px.bar(era_agg, x='release_year', y='Count', hover_data={'Avg_Rating': ':.2f'},
                     title="Games by Release Year", color_discrete_sequence=[HUB_PALETTE[2]])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No parseable release dates.")

with tab_age:
    st.subheader("🔞 Age Rating (ESRB)")
    if 'age_rating' in df.columns:
        age = df.copy()
        age['age_rating'] = age['age_rating'].fillna('Unrated')
        age_stats = age.groupby('age_rating').agg(
            Count=('my_rating', 'size'), Avg_Rating=('my_rating', 'mean')
        ).reset_index().sort_values('Count', ascending=False)
        ca1, ca2 = st.columns(2)
        with ca1:
            fig = px.bar(age_stats, x='age_rating', y='Count', hover_data={'Avg_Rating': ':.2f'},
                         title="Games per ESRB Rating", color_discrete_sequence=[HUB_PALETTE[3]])
            st.plotly_chart(fig, use_container_width=True)
        with ca2:
            fig = px.box(age, x='age_rating', y='my_rating', title="Rating Spread per ESRB Level",
                         color_discrete_sequence=HUB_PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

with tab_vibe:
    st.subheader("🧠 Plot Vibe Check")
    st.caption("Dominant themes from game descriptions + tags. Color shows your average rating for games with each theme.")
    df['vibe_text'] = df['description_raw'].fillna('') + " " + df['tags'].fillna('')
    word_ratings = {}
    for _, row in df.dropna(subset=['my_rating']).iterrows():
        words = set(w.lower() for w in re.findall(r'\w+', str(row['vibe_text']))
                    if len(w) > 3 and w.lower() not in STOPS)
        for w in words:
            word_ratings.setdefault(w, []).append(row['my_rating'])
    word_stats = [{'Word': w, 'Count': len(r), 'Avg_Rating': round(np.mean(r), 2)}
                  for w, r in word_ratings.items() if len(r) >= 2]
    if word_stats:
        wdf = pd.DataFrame(word_stats).sort_values('Count', ascending=False).head(50)
        fig = px.treemap(wdf, path=['Word'], values='Count', color='Avg_Rating',
                         color_continuous_scale='RdYlGn', range_color=[1, 5],
                         title="Dominant Themes in Your Games")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough description text for a vibe check.")

with tab_list:
    st.subheader("📋 All Games")
    display_cols = [c for c in ['name', 'platform_from_text', 'released', 'my_rating', 'metacritic', 'playtime', 'genres'] if c in df.columns]
    st.dataframe(df[display_cols].sort_values('my_rating', ascending=False), use_container_width=True, hide_index=True)

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
        pt_txt = f" | **Hours to Beat:** {int(game['playtime'])}" if 'playtime' in df.columns and pd.notna(game['playtime']) and game['playtime'] > 0 else ""
        st.markdown(f"**Platform:** {game['platform_from_text']} | **My Rating:** {game['my_rating']} | **Metacritic:** {game['metacritic']}{pt_txt}")
        st.markdown(f"**Genres:** {game['genres']}")
        st.write(f"{str(game['description_raw'])[:300]}...")
    st.divider()
