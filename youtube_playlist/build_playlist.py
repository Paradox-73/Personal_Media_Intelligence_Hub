"""Build a YouTube playlist from videos referenced in another playlist's descriptions.

Reads every video description in a source playlist, finds the other YouTube videos
it links to, and collects them into a new playlist.

Two extraction methods run, in order of preference:
  1. Direct video links (youtu.be/ID, youtube.com/watch?v=ID). Exact and free.
  2. Bullet lines ("... : <bullet> Some Title"), resolved with search. 100 quota units each.

The YouTube Data API gives 10,000 quota units per day by default. This script
tracks spend, stops before it runs out, and saves state so the next run resumes.

Usage:
  python build_playlist.py --dry-run     # inspect descriptions, ~free
  python build_playlist.py               # create playlist and add videos
  python build_playlist.py --budget 8000 # cap quota spend for this run
"""

import argparse
import json
import os
import re
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_SECRET_FILE = os.path.join(HERE, "client_secret.json")
TOKEN_FILE = os.path.join(HERE, "token.json")
STATE_FILE = os.path.join(HERE, "state.json")

SOURCE_PLAYLIST_ID = "PLxoHK1S7LhWRBvv5VLqkIYWYXT6_GnPiS"
NEW_PLAYLIST_TITLE = "CoryxKenshin SSS - Reacted To Animations & Shorts"
NEW_PLAYLIST_DESC = (
    "Autonomous compilation of original horror shorts and animations"
    " featured in Spooky Scary Sunday."
)

# Quota cost per API call. See developers.google.com/youtube/v3/determine_quota_cost
COST_LIST = 1
COST_VIDEOS = 1
COST_SEARCH = 100
COST_INSERT = 50
DEFAULT_BUDGET = 9500  # leave headroom under the 10,000/day default

VIDEO_URL_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^\s&]*&)*v=|shorts/|embed/|live/))"
    r"([A-Za-z0-9_-]{11})"
)


class Quota:
  """Tracks quota spend and refuses calls that would exceed the budget."""

  def __init__(self, budget):
    self.budget = budget
    self.spent = 0

  def can_afford(self, cost):
    return self.spent + cost <= self.budget

  def charge(self, cost):
    self.spent += cost

  def remaining(self):
    return self.budget - self.spent


def load_state():
  if os.path.exists(STATE_FILE):
    with open(STATE_FILE, "r", encoding="utf-8") as handle:
      return json.load(handle)
  return {"playlist_id": None, "added": [], "searched": {}, "unresolved": []}


def save_state(state):
  with open(STATE_FILE, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2)


def get_authenticated_service():
  """Authenticates and returns the YouTube API client. Caches the token."""
  credentials = None
  if os.path.exists(TOKEN_FILE):
    credentials = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

  if not credentials or not credentials.valid:
    if credentials and credentials.expired and credentials.refresh_token:
      credentials.refresh(Request())
    else:
      if not os.path.exists(CLIENT_SECRET_FILE):
        sys.exit(
            f"Missing {CLIENT_SECRET_FILE}.\n"
            "Create an OAuth client ID (type: Desktop app) in Google Cloud"
            " Console, enable the YouTube Data API v3, then download the JSON"
            " to that path."
        )
      flow = InstalledAppFlow.from_client_secrets_file(
          CLIENT_SECRET_FILE, SCOPES
      )
      credentials = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "w", encoding="utf-8") as handle:
      handle.write(credentials.to_json())

  return build("youtube", "v3", credentials=credentials)


def get_playlist_videos(youtube, playlist_id, quota):
  """Fetches all video details (title, description) from a source playlist."""
  videos = []
  request = youtube.playlistItems().list(
      part="snippet", playlistId=playlist_id, maxResults=50
  )

  while request:
    if not quota.can_afford(COST_LIST):
      print("  Quota budget reached while paging the source playlist.")
      break
    response = request.execute()
    quota.charge(COST_LIST)
    for item in response["items"]:
      snippet = item["snippet"]
      videos.append({
          "title": snippet["title"],
          "description": snippet.get("description", ""),
          "video_id": snippet.get("resourceId", {}).get("videoId"),
          "owner_channel_id": snippet.get("videoOwnerChannelId"),
      })
    request = youtube.playlistItems().list_next(request, response)

  return videos


def extract_video_ids(description):
  """Returns YouTube video IDs linked directly in a description."""
  return VIDEO_URL_RE.findall(description)


def extract_referenced_titles(description):
  """Parses video titles from bullet lines such as:

  'Winnie The Pooh (MeatCanyon):     - Winnie The Pooh And The Great Honey Tree'
  """
  titles = []
  for line in description.split("\n"):
    if "•" not in line:
      continue
    parts = line.split("•")
    if len(parts) < 2:
      continue
    clean_title = parts[1].strip()
    if clean_title and len(clean_title) > 3 and "ALTER" not in clean_title:
      titles.append(clean_title)
  return titles


def fetch_video_meta(youtube, video_ids, quota):
  """Returns {video_id: {title, channel_id, channel_title}} for the given IDs.

  Costs 1 quota unit per batch of 50, so this is cheap. It also drops IDs that
  are private, deleted, or otherwise unavailable.
  """
  meta = {}
  for start in range(0, len(video_ids), 50):
    batch = video_ids[start : start + 50]
    if not quota.can_afford(COST_VIDEOS):
      print("  Quota budget reached while reading video details.")
      break
    response = (
        youtube.videos()
        .list(part="snippet", id=",".join(batch), maxResults=50)
        .execute()
    )
    quota.charge(COST_VIDEOS)
    for item in response.get("items", []):
      snippet = item["snippet"]
      meta[item["id"]] = {
          "title": snippet.get("title", ""),
          "channel_id": snippet.get("channelId", ""),
          "channel_title": snippet.get("channelTitle", ""),
      }
  return meta


def get_existing_playlist_items(youtube, playlist_id, quota):
  """Returns video IDs already in the target playlist, so reruns do not duplicate."""
  existing = []
  request = youtube.playlistItems().list(
      part="contentDetails", playlistId=playlist_id, maxResults=50
  )
  while request:
    if not quota.can_afford(COST_LIST):
      break
    response = request.execute()
    quota.charge(COST_LIST)
    for item in response["items"]:
      existing.append(item["contentDetails"]["videoId"])
    request = youtube.playlistItems().list_next(request, response)
  return existing


def search_video_id(youtube, query, quota):
  """Searches YouTube for a query string and returns the top video ID."""
  if not quota.can_afford(COST_SEARCH):
    return None
  try:
    search_response = (
        youtube.search()
        .list(q=query, type="video", part="id,snippet", maxResults=1)
        .execute()
    )
    quota.charge(COST_SEARCH)
    items = search_response.get("items", [])
    if items:
      return items[0]["id"]["videoId"]
  except HttpError as error:
    quota.charge(COST_SEARCH)
    print(f"  Error searching for '{query}': {error}")
    raise
  return None


def create_playlist(youtube, title, description, quota):
  """Creates a new YouTube playlist and returns its ID."""
  playlist_body = {
      "snippet": {
          "title": title,
          "description": description,
          "tags": ["Spooky Scary Sunday", "Horror Shorts"],
      },
      "status": {"privacyStatus": "private"},
  }
  response = (
      youtube.playlists()
      .insert(part="snippet,status", body=playlist_body)
      .execute()
  )
  quota.charge(COST_INSERT)
  return response["id"]


def add_video_to_playlist(youtube, playlist_id, video_id, quota):
  """Adds a video ID to the specified playlist. Returns True on success."""
  try:
    youtube.playlistItems().insert(
        part="snippet",
        body={
            "snippet": {
                "playlistId": playlist_id,
                "resourceId": {"kind": "youtube#video", "videoId": video_id},
            }
        },
    ).execute()
    quota.charge(COST_INSERT)
    return True
  except HttpError as error:
    quota.charge(COST_INSERT)
    print(f"  Failed to add {video_id}: {error}")
    return False


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dry-run",
      action="store_true",
      help="Read the source playlist and report findings. Writes nothing.",
  )
  parser.add_argument(
      "--budget",
      type=int,
      default=DEFAULT_BUDGET,
      help=f"Quota units this run may spend (default {DEFAULT_BUDGET}).",
  )
  parser.add_argument(
      "--no-search",
      action="store_true",
      help="Use direct links only. Skips the 100-unit search fallback.",
  )
  parser.add_argument(
      "--playlist-id",
      default=SOURCE_PLAYLIST_ID,
      help="Source playlist ID to read.",
  )
  args = parser.parse_args()

  quota = Quota(args.budget)
  state = load_state()

  print("Authenticating with Google...")
  youtube = get_authenticated_service()

  print(f"Fetching videos from playlist: {args.playlist_id}")
  source_videos = get_playlist_videos(youtube, args.playlist_id, quota)
  print(f"Found {len(source_videos)} videos in source playlist.")

  source_ids = {v["video_id"] for v in source_videos if v["video_id"]}

  # The uploader of the source playlist. Their own videos are the reactions,
  # not the animations, so exclude them.
  owner_ids = {
      v["owner_channel_id"] for v in source_videos if v.get("owner_channel_id")
  }

  linked_ids = []
  bullet_titles = []
  for video in source_videos:
    linked_ids.extend(extract_video_ids(video["description"]))
    bullet_titles.extend(extract_referenced_titles(video["description"]))

  # Deduplicate, keep order, and drop links back into the source playlist itself.
  linked_ids = [
      vid
      for vid in dict.fromkeys(linked_ids)
      if vid not in source_ids
  ]
  bullet_titles = list(dict.fromkeys(bullet_titles))

  print(f"Direct video links found: {len(linked_ids)}")
  print(f"Bullet-style titles found: {len(bullet_titles)}")

  # Read titles and channels. Cheap, and it filters the reactor's own uploads.
  meta = fetch_video_meta(youtube, linked_ids, quota)
  unavailable = [vid for vid in linked_ids if vid not in meta]
  own_uploads = [
      vid for vid in linked_ids if meta.get(vid, {}).get("channel_id") in owner_ids
  ]
  linked_ids = [
      vid
      for vid in linked_ids
      if vid in meta and meta[vid]["channel_id"] not in owner_ids
  ]
  if unavailable:
    print(f"  Skipped {len(unavailable)} private/deleted videos.")
  if own_uploads:
    print(f"  Skipped {len(own_uploads)} of the reactor's own videos.")
  print(f"Animator videos to collect: {len(linked_ids)}")

  if args.dry_run:
    print("\n--- DRY RUN, nothing written ---")
    print(f"Quota spent so far: {quota.spent} units.")
    print("\nSample of the first 3 descriptions:")
    for video in source_videos[:3]:
      print(f"\n=== {video['title']} ===")
      print(video["description"][:600])
    print(f"\nFirst 25 videos that would be added:")
    for vid in linked_ids[:25]:
      info = meta[vid]
      print(f"  [{info['channel_title']}] {info['title']}")
    if bullet_titles:
      print(f"\nBullet titles needing search: {bullet_titles[:10]}")
    search_cost = len(bullet_titles) * COST_SEARCH
    insert_cost = (len(linked_ids) + len(bullet_titles)) * COST_INSERT
    total = search_cost + insert_cost
    print(
        f"\nEstimated full run: {search_cost} units of search"
        f" + {insert_cost} units of inserts = {total}."
    )
    print("Daily quota is 10,000 units unless you requested an increase.")
    if total > DEFAULT_BUDGET:
      days = -(-total // DEFAULT_BUDGET)
      print(f"That needs about {days} days. State file resumes automatically.")
    return

  # Create the target playlist once, then reuse it on later runs.
  if state["playlist_id"]:
    playlist_id = state["playlist_id"]
    print(f"Reusing existing playlist: {playlist_id}")
    already_in = set(get_existing_playlist_items(youtube, playlist_id, quota))
  else:
    print(f"Creating new playlist: '{NEW_PLAYLIST_TITLE}'...")
    playlist_id = create_playlist(
        youtube, NEW_PLAYLIST_TITLE, NEW_PLAYLIST_DESC, quota
    )
    state["playlist_id"] = playlist_id
    save_state(state)
    print(f"Created! Playlist ID: {playlist_id}")
    already_in = set()

  already_in.update(state["added"])

  # Resolve bullet titles to IDs with search, reusing cached results.
  resolved_from_search = []
  if bullet_titles and not args.no_search:
    for title in bullet_titles:
      if title in state["searched"]:
        cached = state["searched"][title]
        if cached:
          resolved_from_search.append(cached)
        continue
      if not quota.can_afford(COST_SEARCH + COST_INSERT):
        print("  Quota budget reached. Remaining titles wait for the next run.")
        break
      print(f"Searching for: '{title}'")
      try:
        video_id = search_video_id(youtube, title, quota)
      except HttpError:
        # An API error (quota spent, outage) is not "no match": leave this and
        # the remaining titles uncached so the next run searches them again.
        print("  Stopping searches. Remaining titles wait for the next run.")
        break
      state["searched"][title] = video_id
      save_state(state)
      if video_id:
        resolved_from_search.append(video_id)
      else:
        print(f"  No match for: {title}")
        state["unresolved"].append(title)

  to_add = [
      vid
      for vid in dict.fromkeys(linked_ids + resolved_from_search)
      if vid not in already_in
  ]
  print(f"\n{len(to_add)} new videos to add. Quota left: {quota.remaining()}.")

  added_count = 0
  for video_id in to_add:
    if not quota.can_afford(COST_INSERT):
      print("Quota budget reached. Run again tomorrow to continue.")
      break
    if add_video_to_playlist(youtube, playlist_id, video_id, quota):
      state["added"].append(video_id)
      added_count += 1
      label = meta.get(video_id, {}).get("title", video_id)
      print(f"  [{added_count}/{len(to_add)}] {label}")
      save_state(state)

  save_state(state)
  print(f"\nDone. Added {added_count} videos this run.")
  print(f"Playlist total tracked: {len(state['added'])} videos.")
  print(f"Quota spent: {quota.spent} of {args.budget} units.")
  print(f"Playlist: https://www.youtube.com/playlist?list={playlist_id}")


if __name__ == "__main__":
  main()
