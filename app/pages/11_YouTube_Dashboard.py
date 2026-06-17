import streamlit as st
import pandas as pd
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

st.set_page_config(page_title="YouTube Intelligence", page_icon="🟥", layout="wide")

# --- YOUTUBE RED & WHITE PALETTE ---
YT_RED = "#FF0000"
YT_DARK = "#CC0000"
YT_GREY = "#909090"
# Red ramp (light -> dark) for categorical charts on a white background
RED_SEQUENCE = ["#FF0000", "#CC0000", "#FF4D4D", "#990000", "#FF8080", "#660000", "#E50914", "#FFB3B3"]
# White-to-red continuous scale for heatmaps / histograms
RED_SCALE = [[0.0, "#FFFFFF"], [0.15, "#FFD6D6"], [0.4, "#FF6B6B"], [0.7, "#FF0000"], [1.0, "#990000"]]

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = RED_SEQUENCE
px.defaults.color_continuous_scale = RED_SCALE

# --- PATHS ---
RAW_HISTORY = Path("data/raw/youtube/watch-history.json")
SUBSCRIPTIONS_PATH = Path("data/raw/youtube/subscriptions.csv")  # From Takeout
VIDEO_STATS_PATH = Path("data/processed/youtube/youtube_video_details.csv")
CHANNEL_STATS_PATH = Path("data/processed/youtube/youtube_channel_details.csv")

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
    return str(title).replace("Watched ", "").strip()

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

def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning (5-12)'
    elif 12 <= hour < 17: return 'Afternoon (12-17)'
    elif 17 <= hour < 22: return 'Evening (17-22)'
    else: return 'Night (22-5)'

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

# Rewatch counts (history has one row per watch event)
rewatch = df['video_id'].value_counts()

# SESSIONS
sessions = calculate_sessions(df)

# --- DASHBOARD UI ---
st.title("🟥 YouTube Personal Intelligence")
if data.get('n_posts'):
    st.caption(f"Analysing **{len(df):,}** video watches "
               f"({df['time'].min():%b %Y} – {df['time'].max():%b %Y}). "
               f"Excluded {data['n_posts']:,} community-post entries.")

# 1. KEY METRICS ROW
st.subheader("📊 At a Glance")
total_views = len(df)
unique_videos = df['video_id'].nunique()
unique_channels = df['channel_id'].nunique()
shorts_count = int(df['is_short'].sum())
shorts_pct = (shorts_count / total_views) * 100 if total_views > 0 else 0

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
tab_time, tab_content, tab_behavior, tab_stats, tab_comm = st.tabs([
    "📅 Temporal Habits",
    "🎬 Content & Genres",
    "🧠 Behavior & Sessions",
    "📈 Hipster Index (Stats)",
    "🤝 Loyalty & Subs"
])

# === TAB 1: TEMPORAL ===
with tab_time:
    c_left, c_right = st.columns(2)
    with c_left:
        monthly = df.groupby(df['time'].dt.to_period("M")).size().reset_index(name='Views')
        monthly['time'] = monthly['time'].astype(str)
        fig = px.bar(monthly, x='time', y='Views', title="Watch History Timeline")
        st.plotly_chart(fig, use_container_width=True)
    with c_right:
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data['count'], x=heatmap_data['hour'], y=heatmap_data['day_of_week'],
            colorscale=RED_SCALE))
        fig.update_yaxes(categoryorder='array', categoryarray=days_order)
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
        type_counts = df['is_short'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        type_counts['Type'] = type_counts['Type'].map({True: 'Shorts', False: 'Long Form'})
        fig = px.pie(type_counts, values='Count', names='Type', hole=0.5,
                     color_discrete_sequence=[YT_RED, YT_GREY])
        st.plotly_chart(fig, use_container_width=True)
    with c_g2:
        if is_enriched:
            st.markdown("#### 🎭 Top Categories (by Watch Time)")
            cat_stats = df[df['category'] != 'Unknown'].groupby('category')['duration_sec'].sum().reset_index()
            cat_stats['Hours'] = cat_stats['duration_sec'] / 3600
            fig = px.bar(cat_stats.sort_values('Hours', ascending=False).head(10),
                         x='Hours', y='category', orientation='h')
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
        valid_views = df[df['public_views'] > 0].copy()

        c_h1, c_h2 = st.columns(2)
        with c_h1:
            fig = px.histogram(valid_views, x='public_views', log_x=True, nbins=40,
                               title="Public Views of Videos You Watched (log)",
                               color_discrete_sequence=[YT_RED])
            st.plotly_chart(fig, use_container_width=True)
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
    by_count = df['channel_name'].value_counts().head(top_n).reset_index()
    by_count.columns = ['Channel', 'Videos']
    cl1, cl2 = st.columns(2)
    with cl1:
        fig = px.bar(by_count.sort_values('Videos'), x='Videos', y='Channel', orientation='h',
                     title="By Video Count")
        st.plotly_chart(fig, use_container_width=True)
    with cl2:
        if is_enriched:
            by_time = (df.groupby('channel_name')['duration_sec'].sum() / 3600).sort_values(ascending=False).head(top_n).reset_index()
            by_time.columns = ['Channel', 'Hours']
            fig = px.bar(by_time.sort_values('Hours'), x='Hours', y='Channel', orientation='h',
                         title="By Watch Time (hrs)")
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

# RAW DATA
st.divider()
with st.expander("🔎 Search & Inspect Data"):
    search = st.text_input("Search (Title or Channel):")
    cols = ['time', 'video_title', 'channel_name', 'category', 'is_short']
    cols = [c for c in cols if c in df.columns]
    if search:
        mask = df['video_title'].str.contains(search, case=False, na=False) | \
               df['channel_name'].str.contains(search, case=False, na=False)
        st.dataframe(df[mask][cols].head(100), use_container_width=True)
    else:
        st.dataframe(df[cols].head(100), use_container_width=True)
