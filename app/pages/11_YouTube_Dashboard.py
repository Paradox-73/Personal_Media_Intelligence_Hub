import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import ast
import sys
import re
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="YouTube Intelligence", page_icon="🟥", layout="wide")
apply_mobile_css()

# --- YOUTUBE RED & WHITE PALETTE ---
YT_RED = "#FF0000"
YT_DARK = "#CC0000"
YT_GREY = "#909090"
# Fixed colours for the Long form / Shorts split so the same series is the same
# colour in every stacked chart on the page.
FORMAT_COLORS = {"Long form": "#CC0000", "Shorts": "#FF8080"}

# channels.list snippet.country returns ISO-2; plotly's choropleth takes ISO-3 or
# country names. Covers the codes that actually appear in this library (76 of them);
# anything unmapped is reported rather than silently dropped.
ISO2_NAME = {
    "US": "United States", "GB": "United Kingdom", "IN": "India", "CA": "Canada",
    "AU": "Australia", "DE": "Germany", "FR": "France", "JP": "Japan",
    "KR": "South Korea", "BR": "Brazil", "RU": "Russia", "NL": "Netherlands",
    "SE": "Sweden", "NO": "Norway", "DK": "Denmark", "FI": "Finland",
    "ES": "Spain", "IT": "Italy", "PL": "Poland", "MX": "Mexico",
    "AR": "Argentina", "CL": "Chile", "CO": "Colombia", "PH": "Philippines",
    "ID": "Indonesia", "MY": "Malaysia", "SG": "Singapore", "TH": "Thailand",
    "VN": "Vietnam", "TR": "Turkey", "UA": "Ukraine", "IE": "Ireland",
    "NZ": "New Zealand", "ZA": "South Africa", "NG": "Nigeria", "EG": "Egypt",
    "PK": "Pakistan", "BD": "Bangladesh", "LK": "Sri Lanka", "NP": "Nepal",
    "CN": "China", "TW": "Taiwan", "HK": "Hong Kong", "IL": "Israel",
    "AE": "United Arab Emirates", "SA": "Saudi Arabia", "CH": "Switzerland",
    "AT": "Austria", "BE": "Belgium", "PT": "Portugal", "GR": "Greece",
    "CZ": "Czechia", "HU": "Hungary", "RO": "Romania", "BG": "Bulgaria",
    "HR": "Croatia", "RS": "Serbia", "SK": "Slovakia", "SI": "Slovenia",
    "EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "IS": "Iceland",
    "PE": "Peru", "VE": "Venezuela", "EC": "Ecuador", "UY": "Uruguay",
    "CR": "Costa Rica", "PA": "Panama", "DO": "Dominican Republic",
    "GT": "Guatemala", "MA": "Morocco", "KE": "Kenya", "GH": "Ghana",
    "JO": "Jordan", "QA": "Qatar", "KW": "Kuwait", "LB": "Lebanon",
}
# Red ramp (light -> dark) for categorical charts on a white background
RED_SEQUENCE = ["#FF0000", "#CC0000", "#FF4D4D", "#990000", "#FF8080", "#660000", "#E50914", "#FFB3B3"]
# White-to-red continuous scale for heatmaps / histograms
RED_SCALE = [[0.0, "#FFFFFF"], [0.15, "#FFD6D6"], [0.4, "#FF6B6B"], [0.7, "#FF0000"], [1.0, "#990000"]]

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = RED_SEQUENCE
px.defaults.color_continuous_scale = RED_SCALE

# --- PATHS ---
RAW_HISTORY = config.BASE_DIR / "data/raw/youtube/watch-history.json"
SUBSCRIPTIONS_PATH = config.BASE_DIR / "data/raw/youtube/subscriptions.csv"  # From Takeout
VIDEO_STATS_PATH = config.BASE_DIR / "data/processed/youtube/youtube_video_details.csv"
CHANNEL_STATS_PATH = config.BASE_DIR / "data/processed/youtube/youtube_channel_details.csv"

# Standard YouTube Category Map (snippet.categoryId -> human label)
CATEGORY_MAP = {
    '1': 'Film & Animation', '2': 'Autos & Vehicles', '10': 'Music', '15': 'Pets & Animals',
    '17': 'Sports', '18': 'Short Movies', '19': 'Travel & Events', '20': 'Gaming',
    '21': 'Videoblogging', '22': 'People & Blogs', '23': 'Comedy', '24': 'Entertainment',
    '25': 'News & Politics', '26': 'Howto & Style', '27': 'Education', '28': 'Science & Tech',
    '29': 'Nonprofits & Activism', '30': 'Movies', '31': 'Anime/Animation', '32': 'Action/Adventure',
    '33': 'Classics', '34': 'Comedy', '35': 'Documentary', '36': 'Drama', '37': 'Family',
    '38': 'Foreign', '39': 'Horror', '40': 'Sci-Fi/Fantasy', '41': 'Thriller', '42': 'Shorts',
    '43': 'Shows', '44': 'Trailers'
}

# --- HELPER FUNCTIONS ---
def clean_title(title):
    if pd.isna(title): return "Unknown"
    return str(title).removeprefix("Watched ").strip()

def extract_channel_info(subtitles):
    """Returns (channel_name, channel_url) from the history 'subtitles' field."""
    if isinstance(subtitles, str):
        try: subtitles = ast.literal_eval(subtitles)
        except: pass
    if isinstance(subtitles, list) and len(subtitles) > 0:
        return subtitles[0].get('name', 'Unknown'), subtitles[0].get('url', None)
    return 'Unknown', None

def extract_channel_id(url):
    """Channel URLs look like https://www.youtube.com/channel/UCxxxx ."""
    if isinstance(url, str) and '/channel/' in url:
        return url.split('/channel/')[1].split('/')[0]
    return None

def extract_video_id(url):
    if isinstance(url, str) and 'v=' in url:
        return url.split('v=')[1].split('&')[0]
    return None

def map_category(cat_id):
    """category_id arrives from CSV as a float-ish value; normalise to a label."""
    if pd.isna(cat_id): return 'Unknown'
    try:
        key = str(int(float(cat_id)))
    except (ValueError, TypeError):
        key = str(cat_id)
    return CATEGORY_MAP.get(key, 'Other')

def parse_iso_duration(iso_str):
    """Parses ISO 8601 duration (e.g. PT1H2M10S) to seconds."""
    if pd.isna(iso_str): return 0
    match = re.match(r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?', str(iso_str))
    if not match: return 0
    h = int(match.group('hours')) if match.group('hours') else 0
    m = int(match.group('minutes')) if match.group('minutes') else 0
    s = int(match.group('seconds')) if match.group('seconds') else 0
    return h * 3600 + m * 60 + s

def detect_short(row):
    """Robust Shorts detection."""
    title = str(row.get('title', '')).lower()
    url = str(row.get('titleUrl', '')).lower()
    channel = str(row.get('channel_name', '')).lower()
    if "/shorts/" in url: return True
    if "#shorts" in title: return True
    if "shorts" in channel: return True
    if row.get('duration_sec', 0) and 0 < row['duration_sec'] < 61: return True
    return False

def hour_label(h):
    """24h integer -> 12-hour clock label ('0' -> '12 AM', '13' -> '1 PM')."""
    h = int(h) % 24
    suffix = "AM" if h < 12 else "PM"
    display = h % 12
    return f"{12 if display == 0 else display} {suffix}"


HOUR_ORDER = [hour_label(h) for h in range(24)]


def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning (5 AM–12 PM)'
    elif 12 <= hour < 17: return 'Afternoon (12–5 PM)'
    elif 17 <= hour < 22: return 'Evening (5–10 PM)'
    else: return 'Night (10 PM–5 AM)'

def calculate_sessions(df, threshold_min=30):
    if df.empty: return pd.DataFrame()
    df = df.sort_values('time')
    df['time_diff'] = df['time'].diff().dt.total_seconds() / 60
    df['session_id'] = (df['time_diff'] > threshold_min).cumsum()
    sessions = df.groupby('session_id').agg(
        start_time=('time', 'min'),
        end_time=('time', 'max'),
        video_count=('time', 'count'),
        channels=('channel_name', lambda x: list(x))
    )
    sessions['duration_min'] = (sessions['end_time'] - sessions['start_time']).dt.total_seconds() / 60
    return sessions

# --- LOAD DATA ---
@st.cache_data
def load_all_data():
    data = {}

    # 1. Watch History (Required)
    if not RAW_HISTORY.exists():
        return None
    with open(RAW_HISTORY, 'r', encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))

    # Parse as UTC then drop tz (all Takeout timestamps are UTC) so period/date
    # grouping downstream doesn't emit tz-drop warnings.
    df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True).dt.tz_localize(None)
    df['video_title'] = df['title'].apply(clean_title)
    df[['channel_name', 'channel_url']] = df['subtitles'].apply(lambda x: pd.Series(extract_channel_info(x)))
    df['channel_id'] = df['channel_url'].apply(extract_channel_id)
    df['video_id'] = df['titleUrl'].apply(extract_video_id)
    # Community posts (/post/...) have no video_id; flag and exclude from video stats
    df['is_post'] = df['titleUrl'].astype(str).str.contains('/post/', na=False)
    n_posts = int(df['is_post'].sum())
    df = df[~df['is_post']]
    df = df[df['channel_name'] != 'Unknown']
    data['history'] = df
    data['n_posts'] = n_posts

    # 2. Enriched Video Stats (Optional)
    if VIDEO_STATS_PATH.exists():
        v_df = pd.read_csv(VIDEO_STATS_PATH)
        v_df['tags'] = v_df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).startswith('[') else [])
        # topicDetails.topicCategories -> list (schema v2)
        if 'topic_categories' in v_df.columns:
            v_df['topic_categories'] = v_df['topic_categories'].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).startswith('[') else [])
        # The API title is authoritative; Takeout titles carry a "Watched " prefix
        # and are absent for anything not in the history.
        if 'video_title' in v_df.columns:
            v_df = v_df.rename(columns={'video_title': 'api_title'})
        # Avoid clobbering the history-derived channel_id on merge
        if 'channel_id' in v_df.columns:
            v_df = v_df.drop(columns=['channel_id'])
        data['video_stats'] = v_df

    # 3. Enriched Channel Stats (Optional) -- normalise key to channel_id
    if CHANNEL_STATS_PATH.exists():
        c_df = pd.read_csv(CHANNEL_STATS_PATH)
        if 'Channel ID' in c_df.columns:
            c_df = c_df.rename(columns={'Channel ID': 'channel_id'})
        data['channel_stats'] = c_df

    # 4. Subscriptions (Optional) -- normalise Takeout column names
    if SUBSCRIPTIONS_PATH.exists():
        try:
            s_df = pd.read_csv(SUBSCRIPTIONS_PATH)
            s_df = s_df.rename(columns={
                'Channel ID': 'channel_id', 'Channel Id': 'channel_id',
                'Channel title': 'channel_title', 'Channel Title': 'channel_title',
                'Channel URL': 'channel_url', 'Channel Url': 'channel_url',
            })
            data['subs'] = s_df
        except Exception:
            pass

    return data

# --- MAIN EXECUTION ---
data = load_all_data()

if not data:
    st.error("❌ `watch-history.json` not found. Please put it in `data/raw/youtube/`.")
    st.stop()

df = data['history']
is_enriched = 'video_stats' in data
is_subbed = 'subs' in data

# MERGING DATA
if is_enriched:
    df = df.merge(data['video_stats'], on='video_id', how='left')
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    df['duration_sec'] = df['duration_iso'].apply(parse_iso_duration) if 'duration_iso' in df.columns else 0
    df['category'] = df['category_id'].apply(map_category) if 'category_id' in df.columns else 'Unknown'
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce', utc=True).dt.tz_localize(None)
        df['video_age_days'] = (df['time'] - df['published_at']).dt.total_seconds() / 86400
    # Engagement quality of the content you chose to watch
    if 'public_likes' in df.columns and 'public_views' in df.columns:
        df['like_view_ratio'] = (df['public_likes'] / df['public_views'].replace(0, pd.NA)) * 100
else:
    df['duration_sec'] = 0
    df['category'] = 'Unknown'
    df['tags'] = [[] for _ in range(len(df))]

# Join channel-level stats (subscriber counts, etc.) on the normalised channel_id
has_channel_stats = 'channel_stats' in data and 'channel_id' in df.columns
if has_channel_stats:
    df = df.merge(data['channel_stats'], on='channel_id', how='left')

# CALCULATED COLUMNS
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.day_name()
df['time_of_day'] = df['hour'].apply(get_time_of_day)
df['is_short'] = df.apply(detect_short, axis=1)

# --- DASHBOARD UI ---
st.title("🟥 YouTube Personal Intelligence")
if data.get('n_posts'):
    st.caption(f"Analysing **{len(df):,}** video watches "
               f"({df['time'].min():%b %Y} – {df['time'].max():%b %Y}). "
               f"Excluded {data['n_posts']:,} community-post entries.")

# --- SHORTS FILTER -----------------------------------------------------------
# Applies to every metric and chart below. `df_all` keeps the unfiltered frame so
# the charts whose whole point is comparing the two formats (the format pie, the
# stacked splits) still have both series to show.
#
# This has to run BEFORE rewatch counts and sessions are derived -- those are
# computed FROM df, so filtering afterwards would leave them describing the full
# history while every chart around them described the filtered slice.
df_all = df
_fmt = st.radio(
    "Content format", ["All", "Long-form only", "Shorts only"],
    horizontal=True, index=0,
    help="Filters every metric and chart on this page. Charts that compare the two "
         "formats always use the full history regardless of this setting.",
)
if _fmt == "Long-form only":
    df = df_all[~df_all['is_short']].copy()
elif _fmt == "Shorts only":
    df = df_all[df_all['is_short']].copy()
if _fmt != "All":
    st.caption(f"Showing **{len(df):,}** of {len(df_all):,} watches ({_fmt.lower()}).")
if df.empty:
    st.warning("No watches match that filter.")
    st.stop()

# Rewatch counts (history has one row per watch event) and sessions -- both derived
# from the FILTERED frame so they agree with everything else on the page.
rewatch = df['video_id'].value_counts()
sessions = calculate_sessions(df)

# 1. KEY METRICS ROW
st.subheader("📊 At a Glance")
total_views = len(df)
unique_videos = df['video_id'].nunique()
unique_channels = df['channel_id'].nunique()
# Shorts share is always measured against the FULL history -- inside a filtered
# view it would trivially read 0% or 100% and tell you nothing.
shorts_count = int(df_all['is_short'].sum())
shorts_pct = (shorts_count / len(df_all)) * 100 if len(df_all) > 0 else 0

r1 = st.columns(3)
r2 = st.columns(3)
r1[0].metric("Total Views", f"{total_views:,}")
r1[1].metric("Unique Videos", f"{unique_videos:,}", help="Distinct videos (excludes rewatches)")
r1[2].metric("Unique Channels", f"{unique_channels:,}")
r2[0].metric("Shorts Ratio", f"{shorts_pct:.1f}%")
if is_enriched:
    total_hours = int(df['duration_sec'].sum() / 3600)
    r2[1].metric("Actual Watch Time", f"{total_hours:,} hrs", help="Sum of video durations across all watches")
    top_cat = df[df['category'] != 'Unknown']['category'].mode()
    r2[2].metric("Top Category", top_cat[0] if len(top_cat) else "N/A")
else:
    est_hours = int(((total_views - shorts_count) * 10 + shorts_count * 1) / 60)
    r2[1].metric("Est. Watch Time", f"{est_hours:,} hrs", "Run enrich_yt.py for real data")
    r2[2].metric("Top Channel", df['channel_name'].mode()[0] if not df.empty else "N/A")

st.divider()

# --- TABS ---
tab_time, tab_content, tab_behavior, tab_stats, tab_comm, tab_deep = st.tabs([
    "📅 Temporal Habits",
    "🎬 Content & Genres",
    "🧠 Behavior & Sessions",
    "📈 Hipster Index (Stats)",
    "🤝 Loyalty & Subs",
    "🔬 Deep Metadata"
])

# === TAB 1: TEMPORAL ===
with tab_time:
    c_left, c_right = st.columns(2)
    with c_left:
        # Stacked: Shorts sit on top of long-form in ONE chart rather than two.
        monthly = (df_all.assign(_m=df_all['time'].dt.to_period("M").astype(str),
                                 Format=np.where(df_all['is_short'], 'Shorts', 'Long form'))
                   .groupby(['_m', 'Format']).size().reset_index(name='Views')
                   .rename(columns={'_m': 'Month'}))
        fig = px.bar(monthly, x='Month', y='Views', color='Format', barmode='stack',
                     title="Watch History Timeline",
                     color_discrete_map=FORMAT_COLORS,
                     category_orders={'Format': ['Long form', 'Shorts']})
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        heatmap_data['hour_lbl'] = heatmap_data['hour'].apply(hour_label)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data['count'], x=heatmap_data['hour_lbl'], y=heatmap_data['day_of_week'],
            colorscale=RED_SCALE,
            hovertemplate="%{y} %{x}<br>%{z} videos<extra></extra>"))
        fig.update_yaxes(categoryorder='array', categoryarray=days_order)
        fig.update_xaxes(categoryorder='array', categoryarray=HOUR_ORDER)
        fig.update_layout(title="Weekly Heatmap: When do you watch?")
        st.plotly_chart(fig, use_container_width=True)

    c_a, c_b = st.columns(2)
    with c_a:
        tod_counts = df['time_of_day'].value_counts().reset_index()
        tod_counts.columns = ['Period', 'Count']
        fig = px.pie(tod_counts, values='Count', names='Period', title="Activity by Time of Day", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c_b:
        # NEW: cumulative growth curve
        cum = df.sort_values('time').copy()
        cum['cumulative'] = range(1, len(cum) + 1)
        daily = cum.groupby(cum['time'].dt.date)['cumulative'].max().reset_index()
        fig = px.area(daily, x='time', y='cumulative', title="Cumulative Videos Watched")
        st.plotly_chart(fig, use_container_width=True)

    # NEW: busiest days
    daily_counts = df.groupby(df['time'].dt.date).size()
    if not daily_counts.empty:
        busiest = daily_counts.idxmax()
        st.info(f"📌 Busiest day: **{busiest}** with **{int(daily_counts.max())}** videos · "
                f"Daily average: **{daily_counts.mean():.1f}** · "
                f"Active on **{daily_counts.size}** distinct days.")

# === TAB 2: CONTENT ===
with tab_content:
    c_g1, c_g2 = st.columns(2)
    with c_g1:
        st.markdown("#### 📱 Format Preference")
        # Always the FULL history -- this chart exists to compare the two formats.
        type_counts = df_all['is_short'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        type_counts['Type'] = type_counts['Type'].map({True: 'Shorts', False: 'Long form'})
        fig = px.pie(type_counts, values='Count', names='Type', hole=0.5,
                     color='Type', color_discrete_map=FORMAT_COLORS)
        st.plotly_chart(fig, use_container_width=True)
    with c_g2:
        if is_enriched:
            st.markdown("#### 🎭 Top Categories (by Watch Time)")
            _cat = df_all[df_all['category'] != 'Unknown'].copy()
            _cat['Format'] = np.where(_cat['is_short'], 'Shorts', 'Long form')
            cat_stats = (_cat.groupby(['category', 'Format'])['duration_sec'].sum() / 3600
                         ).reset_index(name='Hours')
            top10 = (cat_stats.groupby('category')['Hours'].sum()
                     .nlargest(10).index.tolist())
            cat_stats = cat_stats[cat_stats['category'].isin(top10)]
            fig = px.bar(cat_stats, x='Hours', y='category', color='Format',
                         orientation='h', barmode='stack',
                         color_discrete_map=FORMAT_COLORS,
                         category_orders={'Format': ['Long form', 'Shorts']})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Run `python -m src.youtube.enrich_yt` to unlock genres.")

    # NEW: category mix over time (stacked area)
    if is_enriched:
        st.divider()
        st.markdown("#### 📈 How your taste shifted (monthly category mix)")
        cat_time = df[df['category'] != 'Unknown'].copy()
        cat_time['month'] = cat_time['time'].dt.to_period('M').astype(str)
        top_cats = cat_time['category'].value_counts().head(6).index
        cat_time = cat_time[cat_time['category'].isin(top_cats)]
        mix = cat_time.groupby(['month', 'category']).size().reset_index(name='count')
        fig = px.area(mix, x='month', y='count', color='category', title="Category Watch Counts Over Time")
        st.plotly_chart(fig, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        # Topic cloud from tags
        if is_enriched:
            st.markdown("#### 🏷️ Topic Cloud (Tags)")
            all_tags = [t.lower() for tags in df['tags'] if isinstance(tags, list) for t in tags]
            if all_tags:
                common = Counter(all_tags).most_common(20)
                tag_df = pd.DataFrame(common, columns=['Tag', 'Count'])
                fig = px.treemap(tag_df, path=['Tag'], values='Count')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No tags found.")
    with cc2:
        # NEW: content freshness
        if is_enriched and 'video_age_days' in df.columns:
            st.markdown("#### 🕰️ Content Freshness")
            st.caption("How old a video was when you watched it.")
            fresh = df[df['video_age_days'].between(0, 3650)]
            if not fresh.empty:
                same_week = (fresh['video_age_days'] <= 7).mean() * 100
                fig = px.histogram(fresh, x='video_age_days', nbins=50, log_y=True,
                                   labels={'video_age_days': 'Video age when watched (days)'})
                st.plotly_chart(fig, use_container_width=True)
                st.metric("Watched within a week of upload", f"{same_week:.0f}%")

# === TAB 3: BEHAVIOR ===
with tab_behavior:
    if not sessions.empty:
        c_b1, c_b2, c_b3 = st.columns(3)
        c_b1.metric("Avg Session Length", f"{int(sessions['duration_min'].mean())} min")
        c_b2.metric("Marathon Sessions (>1hr)", f"{len(sessions[sessions['duration_min'] > 60])}")
        c_b3.metric("Avg Videos / Session", f"{sessions['video_count'].mean():.1f}")

        st.markdown("#### 🐇 Session Length Distribution")
        fig = px.histogram(sessions, x='duration_min', nbins=50,
                           labels={'duration_min': 'Minutes'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to calculate session metrics.")

    b1, b2 = st.columns(2)
    with b1:
        # NEW: most rewatched videos
        st.markdown("#### 🔁 Most Rewatched")
        rw = rewatch[rewatch > 1].head(12)
        if not rw.empty:
            titles = (df.dropna(subset=['video_id'])
                        .drop_duplicates('video_id')
                        .set_index('video_id')['video_title'])
            rw_df = pd.DataFrame({'video_id': rw.index, 'Plays': rw.values})
            rw_df['Title'] = rw_df['video_id'].map(titles).fillna('Unknown').str.slice(0, 40)
            fig = px.bar(rw_df.sort_values('Plays'), x='Plays', y='Title', orientation='h')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rewatched videos found.")
    with b2:
        # NEW: channel exploration vs repetition (unique channels per month)
        st.markdown("#### 🧭 Discovery Rate")
        st.caption("Distinct channels watched each month.")
        explore = df.groupby(df['time'].dt.to_period('M').astype(str))['channel_id'].nunique().reset_index()
        explore.columns = ['Month', 'Unique Channels']
        fig = px.line(explore, x='Month', y='Unique Channels', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # Binge / loyalty channels
    st.markdown("#### ❤️ Binge-Worthy Channels")
    daily_watch = df.groupby(['channel_name', df['time'].dt.date]).size().reset_index(name='daily_count')
    binge_stats = daily_watch.groupby('channel_name')['daily_count'].mean().reset_index(name='avg_per_day')
    active_days = daily_watch.groupby('channel_name').size()
    valid_channels = active_days[active_days > 5].index
    binge_stats = binge_stats[binge_stats['channel_name'].isin(valid_channels)]
    if not binge_stats.empty:
        fig = px.bar(binge_stats.sort_values('avg_per_day', ascending=False).head(10),
                     x='avg_per_day', y='channel_name', orientation='h',
                     labels={'avg_per_day': 'Avg videos per active day'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# === TAB 4: STATISTICS (HIPSTER INDEX) ===
with tab_stats:
    if is_enriched and 'public_views' in df.columns:
        st.markdown("#### 🧐 The Hipster Index")
        st.caption("Do you watch mainstream viral videos or obscure content?")
        valid_views = df[pd.to_numeric(df['public_views'], errors='coerce') > 0].copy()
        valid_views['public_views'] = pd.to_numeric(valid_views['public_views'],
                                                    errors='coerce')

        c_h1, c_h2 = st.columns(2)
        with c_h1:
            if valid_views.empty:
                st.info("No public view counts available for these watches.")
            else:
                # Bin in LOG space, not linear-with-a-log-axis. px.histogram(log_x=True)
                # computes bin edges linearly and only then maps them onto a log axis:
                # the first edge is 0, log10(0) is undefined, and Plotly silently drops
                # every bar -- which is why this chart rendered empty.
                lv = np.log10(valid_views['public_views'])
                fig = px.histogram(x=lv, nbins=40,
                                   title="Public Views of Videos You Watched (log scale)",
                                   labels={'x': 'Views'},
                                   color_discrete_sequence=[YT_RED])
                ticks = [t for t in range(0, 11) if lv.min() - 1 <= t <= lv.max() + 1]
                fig.update_xaxes(
                    tickvals=ticks,
                    ticktext=[("1" if t == 0 else
                               f"{10**t:,.0f}" if t < 6 else
                               f"{10**(t-6):,.0f}M" if t < 9 else
                               f"{10**(t-9):,.0f}B") for t in ticks])
                fig.update_yaxes(title="Videos")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Median video you watch has "
                           f"**{valid_views['public_views'].median():,.0f}** views.")
        with c_h2:
            if has_channel_stats and 'subscriber_count' in df.columns:
                subs_data = df[df['subscriber_count'] > 0]
                fig = px.box(subs_data, y='subscriber_count', log_y=True,
                             title="Size of Creators You Watch (Subscribers)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Channel stats unavailable.")

        def categorize(v):
            if v < 10000: return 'Deep Niche (<10k)'
            if v < 100000: return 'Niche (10k-100k)'
            if v < 1000000: return 'Popular (100k-1M)'
            return 'Viral (>1M)'
        valid_views['tier'] = valid_views['public_views'].apply(categorize)
        tier_counts = valid_views['tier'].value_counts().reset_index()
        tier_counts.columns = ['tier', 'count']

        c_t1, c_t2 = st.columns(2)
        with c_t1:
            fig = px.pie(tier_counts, values='count', names='tier', title="Content Mainstream-ness")
            st.plotly_chart(fig, use_container_width=True)
        with c_t2:
            # NEW: engagement quality (like/view ratio)
            if 'like_view_ratio' in df.columns:
                lvr = df[df['like_view_ratio'].between(0, 30)]
                if not lvr.empty:
                    fig = px.histogram(lvr, x='like_view_ratio', nbins=40,
                                       title="Like-to-View Ratio of Watched Videos (%)",
                                       labels={'like_view_ratio': 'Likes per 100 views'})
                    st.plotly_chart(fig, use_container_width=True)
                    st.metric("Median like/view ratio", f"{lvr['like_view_ratio'].median():.2f}%")
    else:
        st.warning("⚠️ This tab requires the enrichment script. Run `python -m src.youtube.enrich_yt`.")

# === TAB 5: COMMUNITY (SUBS) ===
with tab_comm:
    # Channel concentration / loyalty (always available)
    st.markdown("#### 🏆 Top Channels")
    top_n = st.slider("How many channels?", 5, 25, 10, key="topn")
    _ch = df_all.assign(Format=np.where(df_all['is_short'], 'Shorts', 'Long form'))
    top_names = df_all['channel_name'].value_counts().head(top_n).index.tolist()
    cl1, cl2 = st.columns(2)
    with cl1:
        by_count = (_ch[_ch['channel_name'].isin(top_names)]
                    .groupby(['channel_name', 'Format']).size().reset_index(name='Videos'))
        fig = px.bar(by_count, x='Videos', y='channel_name', color='Format',
                     orientation='h', barmode='stack', title="By Video Count",
                     color_discrete_map=FORMAT_COLORS,
                     category_orders={'Format': ['Long form', 'Shorts']},
                     labels={'channel_name': 'Channel'})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    with cl2:
        if is_enriched:
            time_names = ((df_all.groupby('channel_name')['duration_sec'].sum() / 3600)
                          .nlargest(top_n).index.tolist())
            by_time = (_ch[_ch['channel_name'].isin(time_names)]
                       .groupby(['channel_name', 'Format'])['duration_sec'].sum() / 3600
                       ).reset_index(name='Hours')
            fig = px.bar(by_time, x='Hours', y='channel_name', color='Format',
                         orientation='h', barmode='stack', title="By Watch Time (hrs)",
                         color_discrete_map=FORMAT_COLORS,
                         category_orders={'Format': ['Long form', 'Shorts']},
                         labels={'channel_name': 'Channel'})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Loyalty concentration metric
    ch_counts = df['channel_name'].value_counts()
    if len(ch_counts) > 0:
        top10_share = ch_counts.head(10).sum() / ch_counts.sum() * 100
        st.info(f"🎯 Your **top 10 channels** account for **{top10_share:.0f}%** of everything you watch, "
                f"spread across **{len(ch_counts):,}** channels total.")

    st.divider()

    # Subscriptions analysis
    if is_subbed and 'channel_id' in data['subs'].columns:
        subs_df = data['subs']
        subbed_ids = set(subs_df['channel_id'].dropna().astype(str))
        watched_ids = set(df['channel_id'].dropna().astype(str))
        active = subbed_ids & watched_ids
        ghosts = subbed_ids - watched_ids

        st.markdown("#### 🤝 Subscription Utilization")
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            st.metric("Total Subscriptions", len(subbed_ids))
            st.metric("Active Subs", len(active), help="Subs you actually watched in this period")
            st.metric("Ghost Subs", len(ghosts), help="Subs you never watched a video from")
        with c_s2:
            if subbed_ids:
                fig = px.pie(values=[len(active), len(ghosts)], names=['Watched', 'Ghost/Inactive'],
                             title="Subscription Utilization", hole=0.4,
                             color_discrete_sequence=[YT_RED, YT_GREY])
                st.plotly_chart(fig, use_container_width=True)

        # NEW: how much of your watching comes from subscribed channels
        sub_share = df['channel_id'].astype(str).isin(subbed_ids).mean() * 100
        st.info(f"📡 **{sub_share:.0f}%** of your watches come from channels you're subscribed to "
                f"(the rest is algorithmic discovery / search).")

        title_col = 'channel_title' if 'channel_title' in subs_df.columns else subs_df.columns[-1]
        with st.expander("👻 View Ghost Subscriptions (clean up your feed!)"):
            ghost_list = subs_df[subs_df['channel_id'].astype(str).isin(ghosts)]
            st.dataframe(ghost_list[[c for c in [title_col, 'channel_url'] if c in ghost_list.columns]],
                         use_container_width=True)
    else:
        st.warning("⚠️ `subscriptions.csv` not found in `data/raw/youtube/`. Download it from Google Takeout.")

# === TAB 6: DEEP METADATA (schema v2 fields) ===
with tab_deep:
    st.caption("Everything below comes from parts of the YouTube Data API that cost "
               "**no extra quota** — `videos.list` is 1 unit per call however many "
               "parts you ask for. Run `python -m src.youtube.enrich_yt` to populate.")

    have = lambda c: c in df.columns and df[c].notna().any()  # noqa: E731

    # These come back from CSV as OBJECT columns holding real bools plus NaN for the
    # ~2% of videos the API no longer returns. .mean()/.map() cope, but boolean
    # MASKING on a column containing NA raises, so coerce once here.
    for _b in ('has_caption', 'made_for_kids', 'licensed_content', 'embeddable',
               'is_live_stream'):
        if _b in df.columns:
            df[_b] = df[_b].fillna(False).astype(bool)

    # --- 1. Topic categories: the real taxonomy -------------------------------
    if 'topic_categories' in df.columns:
        st.markdown("#### 🧭 What you actually watch (Wikipedia topics)")
        st.caption("`topicDetails` gives real subject tags. The numeric `categoryId` "
                   "has only ~32 buckets and lumps most of YouTube into 'Entertainment'.")
        topics = Counter()
        for lst in df['topic_categories'].dropna():
            if isinstance(lst, list):
                topics.update(lst)
        if topics:
            t_df = pd.DataFrame(topics.most_common(20), columns=['Topic', 'Videos'])
            c1, c2 = st.columns([2, 1])
            with c1:
                fig = px.bar(t_df.sort_values('Videos'), x='Videos', y='Topic',
                             orientation='h', title="Top 20 Topics",
                             color='Videos', color_continuous_scale=RED_SCALE)
                fig.update_layout(height=560, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                cat_n = df['category'].nunique() if 'category' in df.columns else 0
                st.metric("Distinct topics", f"{len(topics):,}")
                st.metric("Distinct categoryIds", f"{cat_n:,}")
                st.caption(f"The topic vocabulary is **{len(topics) / max(cat_n, 1):.0f}×** "
                           "richer than the category field the dashboard used before.")
        else:
            st.info("No topic data yet — re-run the enrichment script.")

    st.divider()

    # --- 2. Production quality / accessibility --------------------------------
    st.markdown("#### 🎛️ Production & Accessibility")
    q1, q2, q3 = st.columns(3)
    with q1:
        if have('definition'):
            d = df['definition'].str.upper().value_counts().reset_index()
            d.columns = ['Definition', 'Videos']
            fig = px.pie(d, values='Videos', names='Definition', hole=0.5,
                         title="HD vs SD", color_discrete_sequence=[YT_RED, YT_GREY])
            st.plotly_chart(fig, use_container_width=True)
    with q2:
        if have('has_caption'):
            c = df['has_caption'].map({True: 'Captioned', False: 'No captions'}).value_counts().reset_index()
            c.columns = ['Captions', 'Videos']
            fig = px.pie(c, values='Videos', names='Captions', hole=0.5,
                         title="Caption Availability",
                         color_discrete_sequence=[YT_RED, YT_GREY])
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{df['has_caption'].mean() * 100:.0f}% of what you watch is captioned.")
    with q3:
        if have('made_for_kids') or have('licensed_content'):
            bits = {}
            if have('made_for_kids'):
                bits['Made for kids'] = df['made_for_kids'].mean() * 100
            if have('licensed_content'):
                bits['Licensed (rights-held)'] = df['licensed_content'].mean() * 100
            if have('embeddable'):
                bits['Embeddable'] = df['embeddable'].mean() * 100
            b_df = pd.DataFrame({'Flag': list(bits), 'Percent': list(bits.values())})
            fig = px.bar(b_df, x='Percent', y='Flag', orientation='h',
                         title="Content Flags (% of watches)",
                         color_discrete_sequence=[YT_RED], range_x=[0, 100])
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 3. Language --------------------------------------------------------
    if have('audio_language'):
        st.markdown("#### 🗣️ Audio Language")
        lang = df['audio_language'].fillna('').astype(str).str.strip()
        lang = lang[lang != ''].str.split('-').str[0].str.lower().value_counts().head(12)
        if not lang.empty:
            l_df = lang.reset_index()
            l_df.columns = ['Language', 'Videos']
            fig = px.bar(l_df.sort_values('Videos'), x='Videos', y='Language',
                         orientation='h', title="Audio Language of Videos Watched",
                         color='Videos', color_continuous_scale=RED_SCALE)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            declared = (df['audio_language'].notna()
                        & (df['audio_language'].astype(str).str.strip() != '')).mean() * 100
            st.caption(f"Declared for {declared:.0f}% of watches — uploaders often "
                       "leave this field blank.")

    # --- 4. Live streams -----------------------------------------------------
    if have('is_live_stream'):
        st.markdown("#### 🔴 Livestreams vs Uploads")
        n_live = int(df['is_live_stream'].sum())
        lc1, lc2 = st.columns([1, 2])
        lc1.metric("Livestream watches", f"{n_live:,}",
                   f"{n_live / len(df) * 100:.1f}% of all")
        if n_live:
            live_month = (df[df['is_live_stream']]
                          .groupby(df['time'].dt.to_period('M').astype(str))
                          .size().reset_index(name='Livestreams'))
            live_month.columns = ['Month', 'Livestreams']
            with lc2:
                fig = px.bar(live_month, x='Month', y='Livestreams',
                             title="Livestream Watches Over Time",
                             color_discrete_sequence=[YT_RED])
                st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- 5. Channel geography + size ----------------------------------------
    cg1, cg2 = st.columns(2)
    with cg1:
        if have('channel_country'):
            st.markdown("#### 🌍 Where Your Creators Are")
            geo = (df.dropna(subset=['channel_country'])
                     .drop_duplicates('channel_id')['channel_country']
                     .value_counts().reset_index())
            geo.columns = ['Country', 'Channels']
            if not geo.empty:
                # The API returns ISO-2 ("US"); plotly's choropleth wants ISO-3 or
                # full names. pycountry isn't a dependency here, so map the codes we
                # actually see and fall back to a bar chart for anything unmapped —
                # a blank globe would look like a bug rather than missing data.
                geo['name'] = geo['Country'].map(ISO2_NAME)
                mapped = geo.dropna(subset=['name'])
                unmapped = int(geo['name'].isna().sum())
                if len(mapped) >= 5:
                    fig = px.choropleth(mapped, locations='name',
                                        locationmode='country names',
                                        color='Channels', color_continuous_scale=RED_SCALE,
                                        title="Creator Countries (distinct channels)",
                                        hover_name='name')
                    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    if unmapped:
                        st.caption(f"{unmapped} smaller country code(s) not shown on the map.")
                else:
                    fig = px.bar(geo.head(15).sort_values('Channels'),
                                 x='Channels', y='Country', orientation='h',
                                 title="Creator Countries (distinct channels)",
                                 color='Channels', color_continuous_scale=RED_SCALE)
                    fig.update_layout(coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                st.caption(f"{geo['Channels'].sum():,} channels across "
                           f"{len(geo)} countries. Top: **{geo.iloc[0]['Country']}** "
                           f"({geo.iloc[0]['Channels']:,}).")
    with cg2:
        if have('channel_created_at'):
            st.markdown("#### 📆 Creator Vintage")
            ch = df.drop_duplicates('channel_id').copy()
            ch['created'] = pd.to_datetime(ch['channel_created_at'], errors='coerce',
                                           utc=True).dt.tz_localize(None)
            ch = ch.dropna(subset=['created'])
            if not ch.empty:
                fig = px.histogram(ch, x=ch['created'].dt.year, nbins=20,
                                   title="When the Channels You Watch Were Created",
                                   labels={'x': 'Channel created'},
                                   color_discrete_sequence=[YT_RED])
                fig.update_yaxes(title="Channels")
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Median creator started in **{int(ch['created'].dt.year.median())}**.")

    # --- 6. Channel reach vs your attention ---------------------------------
    if have('channel_view_count') and have('subscriber_count'):
        st.markdown("#### 📊 Creator Reach vs Your Attention")
        agg = (df.groupby('channel_name')
                 .agg(your_watches=('video_id', 'size'),
                      subs=('subscriber_count', 'max'),
                      total_views=('channel_view_count', 'max'))
                 .reset_index())
        agg = agg[(agg['subs'] > 0) & (agg['total_views'] > 0)].nlargest(300, 'your_watches')
        if not agg.empty:
            fig = px.scatter(agg, x='subs', y='your_watches', size='total_views',
                             hover_name='channel_name', log_x=True, log_y=True,
                             title="Subscribers vs Times You Watched (bubble = channel lifetime views)",
                             labels={'subs': 'Subscribers', 'your_watches': 'Your watches'},
                             color='your_watches', color_continuous_scale=RED_SCALE)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Top-left = small creators you watch a lot. Bottom-right = huge "
                       "channels you barely touch.")

# RAW DATA
st.divider()
with st.expander("🔎 Search & Inspect Data"):
    search = st.text_input("Search (Title or Channel):")
    cols = ['time', 'video_title', 'channel_name', 'category', 'is_short']
    cols = [c for c in cols if c in df.columns]
    if search:
        mask = df['video_title'].str.contains(search, case=False, na=False) | \
               df['channel_name'].str.contains(search, case=False, na=False)
        shown = df[mask][cols].head(100).copy()
    else:
        shown = df[cols].head(100).copy()
    # 12-hour clock to match the rest of the page.
    if 'time' in shown.columns:
        shown['time'] = pd.to_datetime(shown['time']).dt.strftime('%d %b %Y, %I:%M %p')
    st.dataframe(shown, use_container_width=True)
