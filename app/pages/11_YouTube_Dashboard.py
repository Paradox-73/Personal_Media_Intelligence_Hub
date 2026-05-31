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
from textblob import TextBlob 

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(page_title="YouTube Intelligence", page_icon="🟥", layout="wide")
# --- PATHS ---
RAW_HISTORY = Path("data/raw/youtube/watch-history.json")
SUBSCRIPTIONS_PATH = Path("data/raw/youtube/subscriptions.csv") # From Takeout
VIDEO_STATS_PATH = Path("data/processed/youtube/youtube_video_details.csv")
CHANNEL_STATS_PATH = Path("data/processed/youtube/youtube_channel_details.csv")

# Standard YouTube Category Map
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
    if isinstance(subtitles, str):
        try: subtitles = ast.literal_eval(subtitles)
        except: pass
    if isinstance(subtitles, list) and len(subtitles) > 0:
        return subtitles[0].get('name', 'Unknown'), subtitles[0].get('url', None)
    return 'Unknown', None

def parse_iso_duration(iso_str):
    """
    Parses ISO 8601 duration (e.g., PT1H2M10S) to seconds manually.
    """
    if pd.isna(iso_str): return 0
    iso_str = str(iso_str)
    
    # Simple regex parser
    match = re.match(
        r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?', 
        iso_str
    )
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
    # If API data is available (duration > 0), check if < 61s
    if 'duration_sec' in row and row['duration_sec'] > 0:
        if row['duration_sec'] < 61: return True
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
    if RAW_HISTORY.exists():
        with open(RAW_HISTORY, 'r', encoding='utf-8') as f:
            df = pd.DataFrame(json.load(f))
            
        # Parse Dates & Titles immediately
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df['video_title'] = df['title'].apply(clean_title)
        
        # Extract Channel & Video ID
        df[['channel_name', 'channel_url']] = df['subtitles'].apply(lambda x: pd.Series(extract_channel_info(x)))
        df['video_id'] = df['titleUrl'].apply(lambda x: str(x).split('v=')[1].split('&')[0] if isinstance(x, str) and 'v=' in x else None)
        
        df = df[df['channel_name'] != 'Unknown']
        data['history'] = df
    else:
        return None

    # 2. Enriched Video Stats (Optional)
    if VIDEO_STATS_PATH.exists():
        v_df = pd.read_csv(VIDEO_STATS_PATH)
        # Parse tags safely
        v_df['tags'] = v_df['tags'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and str(x).startswith('[') else [])
        data['video_stats'] = v_df

    # 3. Enriched Channel Stats (Optional)
    if CHANNEL_STATS_PATH.exists():
        data['channel_stats'] = pd.read_csv(CHANNEL_STATS_PATH)

    # 4. Subscriptions (Optional)
    if SUBSCRIPTIONS_PATH.exists():
        try:
            data['subs'] = pd.read_csv(SUBSCRIPTIONS_PATH)
        except: pass
        
    return data

# --- MAIN EXECUTION ---
data = load_all_data()

if not data:
    st.error("❌ `watch-history.json` not found. Please put it in the /data folder.")
    st.stop()

# MERGING DATA
df = data['history']
is_enriched = 'video_stats' in data
is_subbed = 'subs' in data

if is_enriched:
    df = df.merge(data['video_stats'], on='video_id', how='left')
    
    # --- CRITICAL FIX: CLEAN UP TAGS ---
    # Convert NaNs (floats) to empty lists so iteration doesn't crash
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    
    # --- FIX FOR DURATION ---
    if 'duration_iso' in df.columns:
        df['duration_sec'] = df['duration_iso'].apply(parse_iso_duration)
    elif 'duration_sec' not in df.columns:
        df['duration_sec'] = 0
        
    # Map Categories
    if 'category_id' in df.columns:
        df['category'] = df['category_id'].astype(str).replace(CATEGORY_MAP)
    else:
        df['category'] = 'Unknown'
else:
    df['duration_sec'] = 0
    df['category'] = 'Unknown'
    df['tags'] = [] # Ensure tags column exists as empty lists
    
if 'channel_stats' in data and 'Channel ID' in df.columns:
    df = df.merge(data['channel_stats'], on='Channel ID', how='left')

# CALCULATED COLUMNS
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.day_name()
df['time_of_day'] = df['hour'].apply(get_time_of_day)
df['is_short'] = df.apply(detect_short, axis=1)

# SESSIONS
sessions = calculate_sessions(df)


# --- DASHBOARD UI ---
st.title("🟥 YouTube Personal Intelligence")

# 1. KEY METRICS ROW
st.subheader("📊 At a Glance")
c1, c2, c3, c4 = st.columns(4)

total_videos = len(df)
shorts_count = df['is_short'].sum()
shorts_pct = (shorts_count / total_videos) * 100 if total_videos > 0 else 0

c1.metric("Total Views", f"{total_videos:,}")
c2.metric("Shorts Ratio", f"{shorts_pct:.1f}%")

if is_enriched:
    total_sec = df['duration_sec'].sum()
    hours = int(total_sec / 3600)
    c3.metric("Actual Watch Time", f"{hours:,} hrs", "Based on duration")
    top_genre = df['category'].mode()[0] if not df['category'].empty else "N/A"
    c4.metric("Top Genre", top_genre)
else:
    # Estimate
    est_hours = int(((total_videos - shorts_count) * 10 + (shorts_count * 1)) / 60)
    c3.metric("Est. Watch Time", f"{est_hours:,} hrs", "Estimated (Run API script for real data)")
    if not df.empty:
        c4.metric("Top Channel", df['channel_name'].mode()[0])

st.divider()

# --- TABS FOR DETAILED ANALYSIS ---
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
        # Timeline
        monthly = df.groupby(df['time'].dt.to_period("M")).size().reset_index(name='Views')
        monthly['time'] = monthly['time'].astype(str)
        fig = px.bar(monthly, x='time', y='Views', title="Watch History Timeline")
        st.plotly_chart(fig, use_container_width=True)
        
    with c_right:
        # Heatmap
        heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data['count'], x=heatmap_data['hour'], y=heatmap_data['day_of_week'],
            colorscale='Viridis'
        ))
        fig.update_yaxes(categoryorder='array', categoryarray=days_order)
        fig.update_layout(title="Weekly Heatmap: When do you watch?")
        st.plotly_chart(fig, use_container_width=True)

    # Time of Day Distribution
    tod_counts = df['time_of_day'].value_counts().reset_index()
    tod_counts.columns = ['Period', 'Count']
    fig = px.pie(tod_counts, values='Count', names='Period', title="Activity by Time of Day", hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

# === TAB 2: CONTENT ===
with tab_content:
    c_g1, c_g2 = st.columns(2)
    
    with c_g1:
        st.markdown("#### 📱 Format Preference")
        type_counts = df['is_short'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        type_counts['Type'] = type_counts['Type'].map({True: 'Shorts', False: 'Long Form'})
        fig = px.pie(type_counts, values='Count', names='Type', hole=0.5, color_discrete_sequence=['#FF4B4B', '#1E88E5'])
        st.plotly_chart(fig, use_container_width=True)

    with c_g2:
        if is_enriched and 'category' in df.columns:
            st.markdown("#### 🎭 Top Genres (By Time)")
            cat_stats = df.groupby('category')['duration_sec'].sum().reset_index()
            cat_stats['Hours'] = cat_stats['duration_sec'] / 3600
            fig = px.bar(cat_stats.sort_values('Hours', ascending=False).head(10), 
                         x='Hours', y='category', orientation='h', title="Most Watched Categories")
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("⚠️ Run `enrich_youtube.py` to see Genres.")
            st.markdown("#### 📺 Top Channels")
            top_ch = df['channel_name'].value_counts().head(10).reset_index()
            fig = px.bar(top_ch, x='count', y='channel_name', orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    # Tags Cloud
    if is_enriched and 'tags' in df.columns:
        st.divider()
        st.markdown("#### 🏷️ Topic Cloud")
        # --- SAFE ITERATION HERE ---
        all_tags = [t for tags in df['tags'] if isinstance(tags, list) for t in tags]
        
        if all_tags:
            common_tags = Counter(all_tags).most_common(20)
            tag_df = pd.DataFrame(common_tags, columns=['Tag', 'Count'])
            fig = px.treemap(tag_df, path=['Tag'], values='Count', title="Topics You Are Interested In")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No tags found in the enriched data.")

# === TAB 3: BEHAVIOR ===
with tab_behavior:
    c_b1, c_b2, c_b3 = st.columns(3)
    
    if not sessions.empty:
        avg_sess = sessions['duration_min'].mean()
        binge_sess = len(sessions[sessions['duration_min'] > 60])
        
        c_b1.metric("Avg Session Length", f"{int(avg_sess)} min")
        c_b2.metric("Marathon Sessions (>1hr)", f"{binge_sess}")
        c_b3.metric("Avg Videos per Session", f"{sessions['video_count'].mean():.1f}")
        
        st.markdown("#### 🐇 Session Analysis")
        fig = px.histogram(sessions, x='duration_min', nbins=50, 
                           title="How long do you usually sit and watch?",
                           labels={'duration_min': 'Minutes'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Loyalty / Binge Channels
        st.markdown("#### ❤️ Binge-Worthy Channels")
        daily_watch = df.groupby(['channel_name', df['time'].dt.date]).size().reset_index(name='daily_count')
        binge_stats = daily_watch.groupby('channel_name')['daily_count'].mean().reset_index(name='avg_per_day')
        active_days = daily_watch.groupby('channel_name').size()
        valid_channels = active_days[active_days > 5].index
        binge_stats = binge_stats[binge_stats['channel_name'].isin(valid_channels)]
        
        if not binge_stats.empty:
            fig = px.bar(binge_stats.sort_values('avg_per_day', ascending=False).head(10),
                         x='avg_per_day', y='channel_name', orientation='h')
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to calculate session metrics.")

# === TAB 4: STATISTICS (HIPSTER INDEX) ===
with tab_stats:
    if is_enriched and 'public_views' in df.columns:
        st.markdown("#### 🧐 The Hipster Index")
        st.caption("Do you watch mainstream viral videos or obscure content?")
        
        valid_views = df[df['public_views'] > 0].copy()
        
        c_h1, c_h2 = st.columns(2)
        
        with c_h1:
            # View Distribution
            fig = px.histogram(valid_views, x='public_views', log_y=True, nbins=40,
                               title="Public View Counts of Videos You Watched (Log Scale)",
                               color_discrete_sequence=['teal'])
            st.plotly_chart(fig, use_container_width=True)
            
        with c_h2:
            # Creator Size
            if 'subscriber_count' in df.columns:
                fig = px.box(df[df['subscriber_count'] > 0], y='subscriber_count', log_y=True,
                             title="Size of Creators You Watch (Subscribers)")
                st.plotly_chart(fig, use_container_width=True)
                
        # Mainstream vs Niche Pie
        def categorize(v):
            if v < 10000: return 'Deep Niche (<10k)'
            if v < 100000: return 'Niche (10k-100k)'
            if v < 1000000: return 'Popular (100k-1M)'
            return 'Viral (>1M)'
            
        valid_views['tier'] = valid_views['public_views'].apply(categorize)
        tier_counts = valid_views['tier'].value_counts().reset_index()
        fig = px.pie(tier_counts, values='count', names='tier', title="Content Mainstream-ness")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ This tab requires the enrichment script (`enrich_youtube.py`) to be run.")

# === TAB 5: COMMUNITY (SUBS) ===
with tab_comm:
    if is_subbed and 'subs' in data:
        subs_df = data['subs']
        
        # Check if we have IDs to map
        if 'Channel Id' in subs_df.columns and 'Channel ID' in df.columns:
            watched_ids = set(df['Channel ID'].dropna().unique())
            subbed_ids = set(subs_df['Channel ID'].unique())
            
            active = subbed_ids.intersection(watched_ids)
            ghosts = subbed_ids - watched_ids
            
            c_s1, c_s2 = st.columns(2)
            
            with c_s1:
                st.metric("Total Subscriptions", len(subbed_ids))
                st.metric("Active Subs", len(active), help="Subs you have actually watched in this history period")
                st.metric("Ghost Subs", len(ghosts), help="Subs you haven't watched a single video from")
                
            with c_s2:
                fig = px.pie(values=[len(active), len(ghosts)], names=['Watched', 'Ghost/Inactive'],
                             title="Subscription Utilization", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)
                
            with st.expander("👻 View Ghost Subscriptions (Clean up your feed!)"):
                ghost_list = subs_df[subs_df['Channel Id'].isin(ghosts)][['Channel Title', 'Channel Url']]
                st.dataframe(ghost_list)
        else:
            st.error("`Channel Id` column missing in subscriptions or history. Cannot match.")
    else:
        st.warning("⚠️ `subscriptions.csv` not found in /data folder. Download it from Google Takeout to see this analysis.")

# RAW DATA
st.divider()
with st.expander("🔎 Search & Inspect Data"):
    search = st.text_input("Search (Title or Channel):")
    if search:
        mask = df['video_title'].str.contains(search, case=False) | df['channel_name'].str.contains(search, case=False)
        st.dataframe(df[mask][['time', 'video_title', 'channel_name', 'is_short']].head(100))
    else:
        st.dataframe(df[['time', 'video_title', 'channel_name', 'is_short']].head(100))