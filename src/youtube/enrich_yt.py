"""Enrich the YouTube watch history with everything the Data API will give us.

Quota note that drives the whole design: **videos.list costs 1 unit per CALL,
regardless of how many `part`s you request.** 11,919 videos at 50 ids per call is
239 units against a 10,000/day quota — about 2.4%. So there is no reason to be
frugal with parts; every field below is effectively free.

Previously this fetched only `snippet,contentDetails,statistics` and kept 10 of the
available fields. It now requests every part a public API key is allowed to read and
keeps everything useful:

  snippet              title, channel, category, publishedAt, tags, description
                       (FULL, no longer truncated at 500 chars), default + audio
                       language, liveBroadcastContent, thumbnail
  contentDetails       duration, definition (hd/sd), caption availability,
                       licensedContent, projection (360/rectangular)
  statistics           views, likes, comments
  topicDetails         Wikipedia topic categories -- a far richer taxonomy than the
                       ~30 numeric categoryIds, and the single biggest thing the old
                       version left on the table
  status               privacy, license, embeddable, madeForKids
  liveStreamingDetails whether it was a stream, and its concurrent viewer peak
  recordingDetails     recording date where the uploader set it

Channels additionally pull `topicDetails` and `brandingSettings`, plus the channel's
total viewCount, country and creation date.

`videoCategories.list` (1 unit) is used to build the category map from the API
instead of the hand-maintained dict that had duplicate labels ("Comedy" at both 23
and 34) and movie-only ids.

Merge-safe and incremental: the watch-history export is a rolling window, not a
superset, so rows are never dropped. A SCHEMA_VERSION column marks rows written by
this version; anything older is re-fetched to backfill the new columns.

Run:  python -m src.youtube.enrich_yt
      python -m src.youtube.enrich_yt --limit 500     # smoke test
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

RAW_HISTORY_PATH = Path("data/raw/youtube/watch-history.json")
ENRICHED_VIDEO_PATH = Path("data/processed/youtube/youtube_video_details.csv")
ENRICHED_CHANNEL_PATH = Path("data/processed/youtube/youtube_channel_details.csv")
CATEGORY_MAP_PATH = Path("data/processed/youtube/youtube_category_map.json")

# Bump when the extracted columns change -- older rows are then re-fetched.
SCHEMA_VERSION = 2

VIDEO_PARTS = ("snippet,contentDetails,statistics,topicDetails,status,"
               "liveStreamingDetails,recordingDetails")
CHANNEL_PARTS = "statistics,snippet,topicDetails,brandingSettings"


def get_video_id(url):
    if pd.isna(url):
        return None
    url = str(url)
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None


def _topics(item):
    """topicDetails.topicCategories are Wikipedia URLs; keep the readable tail."""
    cats = (item.get("topicDetails") or {}).get("topicCategories") or []
    return [c.rsplit("/", 1)[-1].replace("_", " ") for c in cats]


def fetch_video_details(youtube, video_ids, verbose=True):
    rows = []
    total = len(video_ids)
    for i in range(0, total, 50):
        batch = video_ids[i:i + 50]
        try:
            resp = youtube.videos().list(part=VIDEO_PARTS, id=",".join(batch)).execute()
        except Exception as e:  # noqa: BLE001
            print(f"   ! batch {i}: {e}")
            continue
        for item in resp.get("items", []):
            snip = item.get("snippet", {}) or {}
            cd = item.get("contentDetails", {}) or {}
            stats = item.get("statistics", {}) or {}
            status = item.get("status", {}) or {}
            live = item.get("liveStreamingDetails", {}) or {}
            rec = item.get("recordingDetails", {}) or {}
            thumbs = snip.get("thumbnails", {}) or {}
            rows.append({
                "video_id": item["id"],
                "channel_id": snip.get("channelId", ""),
                "video_title": snip.get("title", ""),
                "channel_title_api": snip.get("channelTitle", ""),
                "category_id": snip.get("categoryId", ""),
                "published_at": snip.get("publishedAt", ""),
                "duration_iso": cd.get("duration", ""),
                "tags": str(snip.get("tags", [])),
                # Full description: the old version truncated at 500 chars.
                "description": snip.get("description", ""),
                "default_language": snip.get("defaultLanguage", ""),
                "audio_language": snip.get("defaultAudioLanguage", ""),
                "live_broadcast": snip.get("liveBroadcastContent", ""),
                "thumbnail_url": (thumbs.get("high") or thumbs.get("default") or {}).get("url", ""),
                "definition": cd.get("definition", ""),          # hd / sd
                "has_caption": str(cd.get("caption", "")).lower() == "true",
                "licensed_content": bool(cd.get("licensedContent", False)),
                "projection": cd.get("projection", ""),          # rectangular / 360
                "public_views": int(stats.get("viewCount", 0) or 0),
                "public_likes": int(stats.get("likeCount", 0) or 0),
                "comment_count": int(stats.get("commentCount", 0) or 0),
                "topic_categories": str(_topics(item)),
                "privacy_status": status.get("privacyStatus", ""),
                "license": status.get("license", ""),
                "embeddable": bool(status.get("embeddable", False)),
                "made_for_kids": bool(status.get("madeForKids", False)),
                "is_live_stream": bool(live),
                "live_concurrent_peak": int(live.get("concurrentViewers", 0) or 0),
                "recording_date": rec.get("recordingDate", ""),
                "schema_version": SCHEMA_VERSION,
            })
        if verbose and (i // 50) % 20 == 0:
            print(f"   ...{min(i + 50, total)}/{total} videos", flush=True)
    return pd.DataFrame(rows)


def fetch_channel_details(youtube, channel_ids, verbose=True):
    rows = []
    channel_ids = list(dict.fromkeys(channel_ids))
    total = len(channel_ids)
    for i in range(0, total, 50):
        batch = channel_ids[i:i + 50]
        try:
            resp = youtube.channels().list(part=CHANNEL_PARTS, id=",".join(batch)).execute()
        except Exception as e:  # noqa: BLE001
            print(f"   ! channel batch {i}: {e}")
            continue
        for item in resp.get("items", []):
            snip = item.get("snippet", {}) or {}
            stats = item.get("statistics", {}) or {}
            brand = (item.get("brandingSettings", {}) or {}).get("channel", {}) or {}
            rows.append({
                "Channel ID": item["id"],
                "channel_title": snip.get("title", ""),
                "subscriber_count": int(stats.get("subscriberCount", 0) or 0),
                "video_count": int(stats.get("videoCount", 0) or 0),
                "channel_view_count": int(stats.get("viewCount", 0) or 0),
                "channel_country": snip.get("country", ""),
                "channel_created_at": snip.get("publishedAt", ""),
                "custom_url": snip.get("customUrl", ""),
                "channel_topics": str(_topics(item)),
                "channel_keywords": brand.get("keywords", ""),
                "schema_version": SCHEMA_VERSION,
            })
        if verbose and (i // 50) % 10 == 0:
            print(f"   ...{min(i + 50, total)}/{total} channels", flush=True)
    return pd.DataFrame(rows)


def fetch_category_map(youtube, region="US"):
    """Authoritative categoryId -> title, instead of a hand-maintained dict."""
    try:
        resp = youtube.videoCategories().list(part="snippet", regionCode=region).execute()
        return {it["id"]: it["snippet"]["title"] for it in resp.get("items", [])}
    except Exception as e:  # noqa: BLE001
        print(f"   ! category map: {e}")
        return {}


def _needs_refresh(df):
    """Video ids whose row predates SCHEMA_VERSION (or has no version at all)."""
    if df.empty:
        return set()
    if "schema_version" not in df.columns:
        return set(df["video_id"].astype(str))
    # fillna(-1) matters: rows written before this column existed hold NaN, and
    # `NaN < SCHEMA_VERSION` is False, so without it every legacy row silently
    # counts as up to date and never gets backfilled.
    ver = pd.to_numeric(df["schema_version"], errors="coerce").fillna(-1)
    return set(df.loc[ver < SCHEMA_VERSION, "video_id"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap videos fetched (smoke test)")
    ap.add_argument("--region", default="US", help="region for the category map")
    args = ap.parse_args()

    if not API_KEY:
        print("[x] YOUTUBE_API_KEY not found in .env")
        return
    youtube = build("youtube", "v3", developerKey=API_KEY)
    ENRICHED_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading history...")
    with open(RAW_HISTORY_PATH, "r", encoding="utf-8") as f:
        hist = pd.DataFrame(json.load(f))
    hist["video_id"] = hist["titleUrl"].apply(get_video_id)
    unique_vids = hist["video_id"].dropna().astype(str).unique().tolist()

    # --- category map (1 quota unit) ---
    cmap = fetch_category_map(youtube, args.region)
    if cmap:
        CATEGORY_MAP_PATH.write_text(json.dumps(cmap, indent=1), encoding="utf-8")
        print(f"[ok] category map: {len(cmap)} categories -> {CATEGORY_MAP_PATH}")

    # --- videos ---
    if ENRICHED_VIDEO_PATH.exists():
        existing = pd.read_csv(ENRICHED_VIDEO_PATH)
    else:
        existing = pd.DataFrame()

    stale = _needs_refresh(existing)
    target = set(unique_vids)
    if not existing.empty:
        target |= set(existing["video_id"].astype(str))
    current = set() if existing.empty else (
        set(existing["video_id"].astype(str)) - stale)
    todo = sorted(target - current)
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(existing)} rows on disk, {len(stale)} at an older schema.")
    print(f"Fetching {len(todo)} videos "
          f"(~{-(-len(todo) // 50)} quota units of 10,000/day)...")
    new_vid = fetch_video_details(youtube, todo) if todo else pd.DataFrame()

    if not new_vid.empty:
        vid_df = pd.concat([existing, new_vid], ignore_index=True)
        vid_df = vid_df.drop_duplicates(subset="video_id", keep="last")
    else:
        vid_df = existing
    vid_df.to_csv(ENRICHED_VIDEO_PATH, index=False)
    print(f"[ok] videos: {len(vid_df)} total ({len(new_vid)} fetched) -> {ENRICHED_VIDEO_PATH}")

    # --- channels ---
    if not vid_df.empty and "channel_id" in vid_df.columns:
        all_chans = set(vid_df["channel_id"].dropna().astype(str)) - {""}
        if ENRICHED_CHANNEL_PATH.exists():
            ex_chan = pd.read_csv(ENRICHED_CHANNEL_PATH)
        else:
            ex_chan = pd.DataFrame()
        if ex_chan.empty or "schema_version" not in ex_chan.columns:
            chan_current = set()
        else:
            v = pd.to_numeric(ex_chan["schema_version"], errors="coerce")
            chan_current = set(ex_chan.loc[v >= SCHEMA_VERSION, "Channel ID"].astype(str))
        chan_todo = sorted(all_chans - chan_current)
        print(f"Fetching {len(chan_todo)} channels "
              f"(~{-(-len(chan_todo) // 50)} quota units)...")
        new_chan = fetch_channel_details(youtube, chan_todo) if chan_todo else pd.DataFrame()
        chan_df = (pd.concat([ex_chan, new_chan], ignore_index=True)
                   .drop_duplicates(subset="Channel ID", keep="last")
                   if not new_chan.empty else ex_chan)
        chan_df.to_csv(ENRICHED_CHANNEL_PATH, index=False)
        print(f"[ok] channels: {len(chan_df)} total ({len(new_chan)} fetched) "
              f"-> {ENRICHED_CHANNEL_PATH}")

    # --- coverage report ---
    got = set(vid_df["video_id"].astype(str))
    missing = set(unique_vids) - got
    print(f"\nCoverage: {len(got & set(unique_vids))}/{len(unique_vids)} history videos "
          f"enriched; {len(missing)} unavailable (deleted/private).")
    for c in ("topic_categories", "audio_language", "definition", "made_for_kids",
              "description", "video_title"):
        if c in vid_df.columns:
            s = vid_df[c].astype(str)
            nonempty = ((s.str.strip() != "") & (s != "[]") & (s != "nan")).sum()
            print(f"   {c:20} populated {nonempty}/{len(vid_df)} "
                  f"({nonempty / len(vid_df) * 100:.1f}%)")


if __name__ == "__main__":
    main()
