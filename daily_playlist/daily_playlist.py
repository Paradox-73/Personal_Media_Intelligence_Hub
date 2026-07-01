#!/usr/bin/env python3
"""
Daily auto-updating Spotify playlist builder.

Mixes three buckets into one playlist (default 200 tracks, 33/33/34 split):
  1. Heavy rotation   - your most-played tracks on Last.fm this past week.
  2. Older liked       - Spotify liked songs you haven't scrobbled in a while.
  3. Discovery         - new tracks similar to your taste (Last.fm similar),
                         not already in your liked songs.

Designed to run headless on GitHub Actions on a cron. See README.md.
"""

import os
import sys
import time
import random
import unicodedata

import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import MemoryCacheHandler

# Load the repo-root .env for local runs (GitHub Actions injects env directly).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------------------------------------------------------------------------- #
# Configuration (override any of these with environment variables)
# --------------------------------------------------------------------------- #

SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

LASTFM_ROOT = "http://ws.audioscrobbler.com/2.0/"

PLAYLIST_NAME = os.environ.get("PLAYLIST_NAME", "Daily Mix (auto)")
PLAYLIST_PUBLIC = os.environ.get("PLAYLIST_PUBLIC", "false").lower() == "true"

PLAYLIST_SIZE = int(os.environ.get("PLAYLIST_SIZE", "200"))

# Bucket ratios (must be a 3-tuple; normalized to PLAYLIST_SIZE below).
RATIO_HEAVY = float(os.environ.get("RATIO_HEAVY", "0.33"))
RATIO_OLDER = float(os.environ.get("RATIO_OLDER", "0.33"))
RATIO_DISCOVERY = float(os.environ.get("RATIO_DISCOVERY", "0.34"))

# "Recently heard" window (days) used to decide what counts as an *older* liked
# song. Liked songs scrobbled within this window are excluded from the older
# bucket. Larger = stricter definition of "haven't heard in a while".
RECENT_WINDOW_DAYS = int(os.environ.get("RECENT_WINDOW_DAYS", "180"))
RECENT_MAX_PAGES = int(os.environ.get("RECENT_MAX_PAGES", "25"))  # 200 scrobbles/page

# How many seed tracks to fan out from when finding similar tracks.
DISCOVERY_SEEDS = int(os.environ.get("DISCOVERY_SEEDS", "40"))
SIMILAR_PER_SEED = int(os.environ.get("SIMILAR_PER_SEED", "20"))

RANDOM_SEED = os.environ.get("RANDOM_SEED")  # set for reproducible runs

LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY")
LASTFM_USER = os.environ.get("LASTFM_USER")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log(msg):
    print(msg, flush=True)


def die(msg):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def norm(s):
    """Normalize an artist/track name for cross-service matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    # Drop common noise that differs between Spotify and Last.fm.
    for junk in (" - remastered", " (remastered)", " - remaster", "(remaster)",
                 " - single version", " - radio edit"):
        s = s.replace(junk, "")
    return " ".join(s.split())


def track_key(artist, title):
    return f"{norm(artist)}||{norm(title)}"


# --------------------------------------------------------------------------- #
# Last.fm client
# --------------------------------------------------------------------------- #

def lastfm(method, **params):
    """Call a Last.fm API method with light retry handling."""
    params.update({
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
    })
    for attempt in range(4):
        try:
            r = requests.get(LASTFM_ROOT, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                # Last.fm-level error (e.g. bad params); don't hammer it.
                raise RuntimeError(f"Last.fm {method} error: {data.get('message')}")
            return data
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    return {}


def lastfm_top_tracks(period="7day", limit=100):
    data = lastfm("user.getTopTracks", user=LASTFM_USER, period=period, limit=limit)
    out = []
    for t in data.get("toptracks", {}).get("track", []):
        out.append((t["artist"]["name"], t["name"]))
    return out


def lastfm_loved_tracks(limit=100):
    data = lastfm("user.getLovedTracks", user=LASTFM_USER, limit=limit)
    out = []
    for t in data.get("lovedtracks", {}).get("track", []):
        out.append((t["artist"]["name"], t["name"]))
    return out


def lastfm_recent_keys(days, max_pages):
    """Set of track_keys the user has scrobbled within the last `days`."""
    since = int(time.time()) - days * 86400
    keys = set()
    page = 1
    while page <= max_pages:
        data = lastfm("user.getRecentTracks", user=LASTFM_USER,
                      limit=200, page=page, **{"from": since})
        rt = data.get("recenttracks", {})
        tracks = rt.get("track", [])
        if not tracks:
            break
        for t in tracks:
            # Skip the "now playing" pseudo-track (has no date).
            if t.get("@attr", {}).get("nowplaying"):
                continue
            keys.add(track_key(t["artist"].get("#text", ""), t.get("name", "")))
        attr = rt.get("@attr", {})
        total_pages = int(attr.get("totalPages", "1") or "1")
        if page >= total_pages:
            break
        page += 1
    return keys


def lastfm_similar(artist, title, limit):
    """Return list of (artist, title, match_score) similar to the seed track."""
    data = lastfm("track.getSimilar", artist=artist, track=title,
                  limit=limit, autocorrect=1)
    out = []
    for t in data.get("similartracks", {}).get("track", []):
        try:
            match = float(t.get("match", 0) or 0)
        except (TypeError, ValueError):
            match = 0.0
        out.append((t["artist"]["name"], t["name"], match))
    return out


# --------------------------------------------------------------------------- #
# Spotify client + resolution
# --------------------------------------------------------------------------- #

def get_spotify():
    refresh = os.environ.get("SPOTIFY_REFRESH_TOKEN")
    if not refresh:
        die("SPOTIFY_REFRESH_TOKEN is not set. Run get_refresh_token.py once "
            "locally to obtain it (see README).")
    auth = SpotifyOAuth(
        scope=SCOPE,
        open_browser=False,
        cache_handler=MemoryCacheHandler(),
    )
    token_info = auth.refresh_access_token(refresh)
    return spotipy.Spotify(auth=token_info["access_token"], requests_timeout=30,
                           retries=3)


def get_liked_tracks(sp):
    """All saved tracks -> list of {uri, key, added_at}."""
    out = []
    offset = 0
    while True:
        page = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = page.get("items", [])
        if not items:
            break
        for it in items:
            tr = it.get("track") or {}
            if not tr.get("uri") or tr.get("is_local"):
                continue
            artist = tr["artists"][0]["name"] if tr.get("artists") else ""
            out.append({
                "uri": tr["uri"],
                "key": track_key(artist, tr.get("name", "")),
                "added_at": it.get("added_at", ""),
            })
        offset += len(items)
        if len(items) < 50:
            break
    return out


_resolve_cache = {}


def resolve_to_uri(sp, artist, title):
    """Search Spotify for a track -> uri, or None. Cached per run."""
    key = track_key(artist, title)
    if key in _resolve_cache:
        return _resolve_cache[key]
    uri = None
    try:
        q = f'track:{title} artist:{artist}'
        res = sp.search(q=q, type="track", limit=1)
        items = res.get("tracks", {}).get("items", [])
        if items:
            uri = items[0]["uri"]
        else:
            # Looser fallback query.
            res = sp.search(q=f"{artist} {title}", type="track", limit=1)
            items = res.get("tracks", {}).get("items", [])
            if items:
                uri = items[0]["uri"]
    except spotipy.SpotifyException:
        uri = None
    _resolve_cache[key] = uri
    return uri


# --------------------------------------------------------------------------- #
# Buckets
# --------------------------------------------------------------------------- #

def bucket_heavy_rotation(sp, n, exclude_uris):
    """Most-played tracks this week, resolved to Spotify URIs."""
    seeds = lastfm_top_tracks(period="7day", limit=max(n * 2, 60))
    uris = []
    for artist, title in seeds:
        uri = resolve_to_uri(sp, artist, title)
        if uri and uri not in exclude_uris and uri not in uris:
            uris.append(uri)
        if len(uris) >= n:
            break
    return uris


def bucket_older_liked(liked, recent_keys, n, exclude_uris):
    """Liked songs not scrobbled recently; weighted toward older adds."""
    candidates = [t for t in liked
                  if t["key"] not in recent_keys and t["uri"] not in exclude_uris]
    if not candidates:
        return []
    # Sort oldest-added first, then take a weighted random sample favoring the
    # front of the list so genuinely old favorites surface more often.
    candidates.sort(key=lambda t: t["added_at"])
    weights = [len(candidates) - i for i in range(len(candidates))]
    picked = weighted_sample_without_replacement(candidates, weights,
                                                 min(n, len(candidates)))
    return [t["uri"] for t in picked]


def bucket_discovery(sp, n, exclude_uris, exclude_keys):
    """Tracks similar to loved + recent favorites, not already liked."""
    seeds = lastfm_loved_tracks(limit=DISCOVERY_SEEDS) \
        + lastfm_top_tracks(period="1month", limit=DISCOVERY_SEEDS)
    random.shuffle(seeds)
    seeds = seeds[:DISCOVERY_SEEDS]

    # Aggregate similar tracks by key, summing match scores across seeds.
    scored = {}
    for artist, title in seeds:
        for s_artist, s_title, match in lastfm_similar(artist, title, SIMILAR_PER_SEED):
            k = track_key(s_artist, s_title)
            if k in exclude_keys:
                continue
            if k not in scored:
                scored[k] = {"artist": s_artist, "title": s_title, "score": 0.0}
            scored[k]["score"] += match

    ranked = sorted(scored.values(), key=lambda x: x["score"], reverse=True)

    uris = []
    for cand in ranked:
        uri = resolve_to_uri(sp, cand["artist"], cand["title"])
        if uri and uri not in exclude_uris and uri not in uris:
            uris.append(uri)
        if len(uris) >= n:
            break
    return uris


def weighted_sample_without_replacement(items, weights, k):
    """Weighted random sampling without replacement."""
    items = list(items)
    weights = list(weights)
    chosen = []
    for _ in range(min(k, len(items))):
        total = sum(weights)
        if total <= 0:
            chosen.extend(random.sample(items, min(k - len(chosen), len(items))))
            break
        r = random.uniform(0, total)
        upto = 0
        for i, w in enumerate(weights):
            upto += w
            if upto >= r:
                chosen.append(items.pop(i))
                weights.pop(i)
                break
    return chosen


# --------------------------------------------------------------------------- #
# Playlist management
# --------------------------------------------------------------------------- #

def find_or_create_playlist(sp, name):
    user_id = sp.current_user()["id"]
    offset = 0
    while True:
        page = sp.current_user_playlists(limit=50, offset=offset)
        items = page.get("items", [])
        for pl in items:
            # Only match playlists we own, to avoid grabbing a followed one.
            if pl["name"] == name and pl["owner"]["id"] == user_id:
                return pl["id"]
        if len(items) < 50:
            break
        offset += len(items)
    pl = sp.user_playlist_create(user_id, name, public=PLAYLIST_PUBLIC,
                                 description="Auto-generated daily mix.")
    log(f"Created new playlist '{name}' ({pl['id']}).")
    return pl["id"]


def set_playlist_tracks(sp, playlist_id, uris):
    """Replace all tracks, chunking around Spotify's 100-item limit."""
    first = uris[:100]
    sp.playlist_replace_items(playlist_id, first)
    for i in range(100, len(uris), 100):
        sp.playlist_add_items(playlist_id, uris[i:i + 100])


def update_description(sp, playlist_id, counts):
    from datetime import datetime, timezone
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    desc = (f"Auto daily mix - {today}. "
            f"{counts['heavy']} heavy rotation / {counts['older']} older liked / "
            f"{counts['discovery']} discovery.")
    try:
        sp.playlist_change_details(playlist_id, description=desc)
    except spotipy.SpotifyException:
        pass  # description update is non-critical


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def target_counts():
    total_ratio = RATIO_HEAVY + RATIO_OLDER + RATIO_DISCOVERY
    heavy = round(PLAYLIST_SIZE * RATIO_HEAVY / total_ratio)
    older = round(PLAYLIST_SIZE * RATIO_OLDER / total_ratio)
    discovery = PLAYLIST_SIZE - heavy - older
    return heavy, older, discovery


def main():
    for name, val in [("LASTFM_API_KEY", LASTFM_API_KEY),
                      ("LASTFM_USER", LASTFM_USER)]:
        if not val:
            die(f"{name} is not set.")

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    t_heavy, t_older, t_discovery = target_counts()
    log(f"Targets -> heavy:{t_heavy} older:{t_older} discovery:{t_discovery} "
        f"(total {PLAYLIST_SIZE})")

    sp = get_spotify()
    log("Authenticated with Spotify.")

    liked = get_liked_tracks(sp)
    liked_keys = {t["key"] for t in liked}
    log(f"Fetched {len(liked)} liked tracks.")

    recent_keys = lastfm_recent_keys(RECENT_WINDOW_DAYS, RECENT_MAX_PAGES)
    log(f"Fetched {len(recent_keys)} recently-scrobbled track keys "
        f"(last {RECENT_WINDOW_DAYS} days).")

    chosen = []          # ordered final URIs
    chosen_set = set()   # for dedupe

    def add(uris):
        added = 0
        for u in uris:
            if u not in chosen_set:
                chosen.append(u)
                chosen_set.add(u)
                added += 1
        return added

    # 1. Heavy rotation
    heavy = bucket_heavy_rotation(sp, t_heavy, chosen_set)
    n_heavy = add(heavy)
    log(f"Heavy rotation: {n_heavy} tracks.")

    # 2. Older liked
    older = bucket_older_liked(liked, recent_keys, t_older, chosen_set)
    n_older = add(older)
    log(f"Older liked: {n_older} tracks.")

    # 3. Discovery
    discovery = bucket_discovery(sp, t_discovery, chosen_set, liked_keys)
    n_discovery = add(discovery)
    log(f"Discovery: {n_discovery} tracks.")

    # Fill shortfall so we still hit PLAYLIST_SIZE. Prefer topping up discovery,
    # then any unused older-liked songs.
    if len(chosen) < PLAYLIST_SIZE:
        need = PLAYLIST_SIZE - len(chosen)
        log(f"Short by {need}; topping up.")
        extra_disc = bucket_discovery(sp, need + n_discovery, chosen_set, liked_keys)
        add(extra_disc)
    if len(chosen) < PLAYLIST_SIZE:
        need = PLAYLIST_SIZE - len(chosen)
        leftover = [t["uri"] for t in liked if t["uri"] not in chosen_set]
        random.shuffle(leftover)
        add(leftover[:need])

    if not chosen:
        die("No tracks assembled; aborting so the playlist isn't wiped.")

    # Interleave a little so buckets aren't in three solid blocks.
    random.shuffle(chosen)

    counts = {"heavy": n_heavy, "older": n_older, "discovery": n_discovery}
    playlist_id = find_or_create_playlist(sp, PLAYLIST_NAME)
    set_playlist_tracks(sp, playlist_id, chosen)
    update_description(sp, playlist_id, counts)

    log(f"Done. Playlist '{PLAYLIST_NAME}' now has {len(chosen)} tracks.")


if __name__ == "__main__":
    main()
