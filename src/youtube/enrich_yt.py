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
RAW_HISTORY_PATH = Path("data/raw/youtube/watch-history.json")
ENRICHED_VIDEO_PATH = Path("data/processed/youtube/youtube_video_details.csv")
ENRICHED_CHANNEL_PATH = Path("data/processed/youtube/youtube_channel_details.csv")

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
                snip = item['snippet']
                stats.append({
                    'video_id': item['id'],
                    'channel_id': snip['channelId'],
                    'category_id': snip.get('categoryId', ''),
                    'published_at': snip.get('publishedAt', ''),
                    'duration_iso': item['contentDetails']['duration'],
                    'tags': str(snip.get('tags', [])),
                    'description': snip.get('description', '')[:500], # Truncate to save space
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
                    # Match the existing youtube_channel_details.csv schema ("Channel ID")
                    'Channel ID': item['id'],
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

    # Ensure output directory exists
    ENRICHED_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("📂 Loading history...")
    with open(RAW_HISTORY_PATH, 'r', encoding='utf-8') as f:
        df = pd.DataFrame(json.load(f))

    df['video_id'] = df['titleUrl'].apply(get_video_id)
    unique_vids = df['video_id'].dropna().unique().tolist()

    # --- INCREMENTAL & MERGE-SAFE ---
    # The watch-history export is a rolling window, not a strict superset of past
    # exports, so we NEVER overwrite. We keep every previously-enriched row and
    # only fetch videos/channels we have not enriched before.
    if ENRICHED_VIDEO_PATH.exists():
        existing_vid_df = pd.read_csv(ENRICHED_VIDEO_PATH)
        # "Fully enriched" = present AND has the newer fields (category_id). Rows
        # written by older versions of this script are re-fetched to backfill them.
        if 'category_id' in existing_vid_df.columns:
            fully_enriched = set(
                existing_vid_df.loc[existing_vid_df['category_id'].notna(), 'video_id'].astype(str)
            )
        else:
            fully_enriched = set()
        print(f"🗂️  {len(existing_vid_df)} videos on disk "
              f"({len(fully_enriched)} fully enriched, preserved).")
    else:
        existing_vid_df = pd.DataFrame()
        fully_enriched = set()

    # Fetch anything in the current history OR already on disk that is not yet
    # fully enriched (this backfills new fields onto old rows in one pass).
    target_ids = set(str(v) for v in unique_vids)
    if not existing_vid_df.empty:
        target_ids |= set(existing_vid_df['video_id'].astype(str))
    new_vids = [v for v in target_ids if v not in fully_enriched]
    print(f"📡 Fetching details for {len(new_vids)} videos "
          f"({len(unique_vids)} in history; backfilling missing fields)...")

    new_vid_df = fetch_video_details(youtube, new_vids) if new_vids else pd.DataFrame()

    vid_df = pd.concat([existing_vid_df, new_vid_df], ignore_index=True)
    vid_df = vid_df.drop_duplicates(subset='video_id', keep='last')
    vid_df.to_csv(ENRICHED_VIDEO_PATH, index=False)
    print(f"✅ Videos: {len(existing_vid_df)} kept + {len(new_vid_df)} new "
          f"= {len(vid_df)} total -> {ENRICHED_VIDEO_PATH}")

    # --- CHANNELS (same merge-safe logic) ---
    if not vid_df.empty and 'channel_id' in vid_df.columns:
        all_channels = set(vid_df['channel_id'].dropna().astype(str))

        if ENRICHED_CHANNEL_PATH.exists():
            existing_chan_df = pd.read_csv(ENRICHED_CHANNEL_PATH)
            known_chans = set(existing_chan_df['Channel ID'].astype(str))
        else:
            existing_chan_df = pd.DataFrame()
            known_chans = set()

        new_channels = [c for c in all_channels if c not in known_chans]
        print(f"📡 Fetching details for {len(new_channels)} NEW channels...")

        new_chan_df = fetch_channel_details(youtube, new_channels) if new_channels else pd.DataFrame()

        chan_df = pd.concat([existing_chan_df, new_chan_df], ignore_index=True)
        chan_df = chan_df.drop_duplicates(subset='Channel ID', keep='last')
        chan_df.to_csv(ENRICHED_CHANNEL_PATH, index=False)
        print(f"✅ Channels: {len(existing_chan_df)} kept + {len(new_chan_df)} new "
              f"= {len(chan_df)} total -> {ENRICHED_CHANNEL_PATH}")
        print("✅ Done! Data saved.")
    else:
        print("❌ No video details found.")

if __name__ == "__main__":
    main()