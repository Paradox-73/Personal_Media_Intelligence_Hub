"""Audit recall: find every YouTube video link the builder may have missed.

Caches all source descriptions to descriptions.json so later audits cost 0 quota.
Total cost on a cold run: about 4 units.
"""

import json
import os
import re
import sys
import urllib.parse

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

HERE = os.path.dirname(os.path.abspath(__file__))
SCOPES = ["https://www.googleapis.com/auth/youtube.force-ssl"]
SOURCE_PLAYLIST_ID = "PLxoHK1S7LhWRBvv5VLqkIYWYXT6_GnPiS"
DESC_CACHE = os.path.join(HERE, "descriptions.json")

# The regex the builder uses.
STRICT_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^\s&]*&)*v=|shorts/|embed/|live/))"
    r"([A-Za-z0-9_-]{11})"
)

# Any token that mentions youtube at all. Used to hunt for shapes STRICT_RE misses.
ANY_YT_RE = re.compile(r"\S*(?:youtube\.com|youtu\.be)\S*", re.IGNORECASE)

# A permissive ID grab applied to a single URL after percent-decoding.
LOOSE_PATTERNS = [
    re.compile(r"[?&]v=([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
    re.compile(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})"),
    re.compile(r"/watch/([A-Za-z0-9_-]{11})"),
]

# URL shapes that legitimately carry no single video: channels, playlists, socials.
NON_VIDEO_HINTS = (
    "/channel/", "/user/", "/c/", "/@", "/playlist", "list=",
    "/results", "/hashtag/", "/feed/", "/about", "/community",
    "youtube.com/t/", "/premium", "/gaming", "/music",
)


def service():
  token = os.path.join(HERE, "token.json")
  if not os.path.exists(token):
    sys.exit("No token.json. Run build_playlist.py --dry-run first.")
  return build(
      "youtube", "v3",
      credentials=Credentials.from_authorized_user_file(token, SCOPES),
  )


def load_descriptions(youtube):
  """Returns cached descriptions, or fetches and caches them (2 quota units)."""
  if os.path.exists(DESC_CACHE):
    with open(DESC_CACHE, "r", encoding="utf-8") as handle:
      print("Using cached descriptions (0 quota units).")
      return json.load(handle)

  videos = []
  request = youtube.playlistItems().list(
      part="snippet", playlistId=SOURCE_PLAYLIST_ID, maxResults=50
  )
  while request:
    response = request.execute()
    for item in response["items"]:
      snippet = item["snippet"]
      videos.append({
          "title": snippet["title"],
          "description": snippet.get("description", ""),
          "video_id": snippet.get("resourceId", {}).get("videoId"),
      })
    request = youtube.playlistItems().list_next(request, response)

  with open(DESC_CACHE, "w", encoding="utf-8") as handle:
    json.dump(videos, handle, indent=2, ensure_ascii=False)
  print(f"Fetched and cached {len(videos)} descriptions.")
  return videos


def loose_ids(url):
  decoded = urllib.parse.unquote(urllib.parse.unquote(url))
  found = []
  for pattern in LOOSE_PATTERNS:
    found.extend(pattern.findall(decoded))
  return found


def main():
  youtube = service()
  videos = load_descriptions(youtube)
  state = json.load(open(os.path.join(HERE, "state.json"), encoding="utf-8"))

  source_ids = {v["video_id"] for v in videos if v["video_id"]}
  added = set(state["added"])

  strict_hits, loose_hits = [], []
  unparsed = []
  empty_desc = []

  for video in videos:
    desc = video["description"]
    if not desc.strip():
      empty_desc.append(video["title"])
    strict_hits.extend(STRICT_RE.findall(desc))
    for url in ANY_YT_RE.findall(desc):
      ids = loose_ids(url)
      if ids:
        loose_hits.extend(ids)
      elif not any(hint in url.lower() for hint in NON_VIDEO_HINTS):
        unparsed.append((video["title"], url))

  strict_set = set(strict_hits)
  loose_set = set(loose_hits)

  print("\n=== SOURCE ===")
  print(f"Source playlist videos      : {len(videos)}")
  print(f"Descriptions that are empty : {len(empty_desc)}")

  print("\n=== EXTRACTION ===")
  print(f"Unique IDs, builder regex   : {len(strict_set)}")
  print(f"Unique IDs, permissive regex: {len(loose_set)}")

  missed = loose_set - strict_set - source_ids
  print(f"IDs the builder regex MISSED: {len(missed)}")
  for vid in sorted(missed):
    print(f"   MISSED {vid}")

  print("\n=== UNPARSEABLE YOUTUBE URLS (excluding channel/playlist links) ===")
  print(f"Count: {len(unparsed)}")
  for title, url in unparsed[:40]:
    print(f"   [{title[:40]}] {url[:110]}")

  # Everything we believe is in scope.
  universe = (strict_set | loose_set) - source_ids
  print("\n=== COVERAGE vs PLAYLIST ===")
  print(f"Total in-scope unique IDs   : {len(universe)}")
  print(f"Already added               : {len(added)}")

  outstanding = sorted(universe - added)
  print(f"Still to add                : {len(outstanding)}")

  # Confirm the added set is genuinely a subset of what we extracted.
  stray = added - universe
  print(f"Added but not in extraction : {len(stray)}")

  # Which of the outstanding are actually retrievable.
  print("\n=== AVAILABILITY OF OUTSTANDING ===")
  available, unavailable = [], []
  for start in range(0, len(outstanding), 50):
    batch = outstanding[start : start + 50]
    response = (
        youtube.videos().list(part="snippet", id=",".join(batch)).execute()
    )
    returned = {item["id"] for item in response.get("items", [])}
    available.extend(v for v in batch if v in returned)
    unavailable.extend(v for v in batch if v not in returned)

  print(f"Retrievable (will be added) : {len(available)}")
  print(f"Private/deleted (cannot add): {len(unavailable)}")
  for vid in unavailable:
    print(f"   DEAD https://www.youtube.com/watch?v={vid}")

  with open(os.path.join(HERE, "audit_report.json"), "w", encoding="utf-8") as h:
    json.dump({
        "universe": sorted(universe),
        "added": sorted(added),
        "outstanding": outstanding,
        "missed_by_strict_regex": sorted(missed),
        "unparsed_urls": unparsed,
        "dead_ids": unavailable,
        "empty_descriptions": empty_desc,
    }, h, indent=2, ensure_ascii=False)
  print("\nWrote audit_report.json")


if __name__ == "__main__":
  main()
