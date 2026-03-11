import os
import json
import pandas as pd
import isodate
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pathlib import Path

# Load Environment Variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Paths
RAW_HISTORY_PATH = Path("data/raw/yt/watch-history.json") 
ENRICHED_VIDEO_PATH = Path("data/processed/youtube_video_details.csv")
ENRICHED_CHANNEL_PATH = Path("data/processed/youtube_channel_details.csv")

def get_video_id(url):
    if pd.isna(url): return None
    url = str(url)
    if "v=" in url: return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url: return url.split("youtu.be/")[1]
    return None

def fetch_video_details(youtube, video_ids):
    stats = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        try:
            response = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=','.join(batch)
            ).execute()
            
            for item in response['items']:
                stats.append({
                    'video_id': item['id'],
                    'channel_id': item['snippet']['channelId'],
                    'duration_iso': item['contentDetails']['duration'],
                    'tags': str(item['snippet'].get('tags', [])),
                    'description': item['snippet'].get('description', '')[:500], # Truncate to save space
                    'public_views': int(item['statistics'].get('viewCount', 0)),
                    'public_likes': int(item['statistics'].get('likeCount', 0)),
                    'comment_count': int(item['statistics'].get('commentCount', 0))
                })
        except Exception as e:
            print(f"Error batch {i}: {e}")
    return pd.DataFrame(stats)

def fetch_channel_details(youtube, channel_ids):
    stats = []
    # Deduplicate
    channel_ids = list(set(channel_ids))
    
    for i in range(0, len(channel_ids), 50):
        batch = channel_ids[i:i+50]
        try:
            response = youtube.channels().list(
                part="statistics,snippet",
                id=','.join(batch)
            ).execute()
            
            for item in response['items']:
                stats.append({
                    'channel_id': item['id'],
                    'channel_title': item['snippet']['title'],
                    'subscriber_count': int(item['statistics'].get('subscriberCount', 0)),
                    'video_count': int(item['statistics'].get('videoCount', 0))
                })
        except Exception as e:
            print(f"Error channel batch {i}: {e}")
    return pd.DataFrame(stats)

def main():
    if not API_KEY:
        print("❌ Error: YOUTUBE_API_KEY not found in .env")
        return

    youtube = build('youtube', 'v3', developerKey=API_KEY)

    print("📂 Loading history...")
    with open(RAW_HISTORY_PATH, 'r', encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))
    
    df['video_id'] = df['titleUrl'].apply(get_video_id)
    unique_vids = df['video_id'].dropna().unique().tolist()
    
    print(f"📡 Fetching details for {len(unique_vids)} videos...")
    vid_df = fetch_video_details(youtube, unique_vids)
    vid_df.to_csv(ENRICHED_VIDEO_PATH, index=False)
    
    # Extract unique channel IDs from the video details we just fetched
    if not vid_df.empty:
        unique_channels = vid_df['channel_id'].unique().tolist()
        print(f"📡 Fetching details for {len(unique_channels)} channels...")
        chan_df = fetch_channel_details(youtube, unique_channels)
        chan_df.to_csv(ENRICHED_CHANNEL_PATH, index=False)
        print("✅ Done! Data saved.")
    else:
        print("❌ No video details found.")

if __name__ == "__main__":
    main()