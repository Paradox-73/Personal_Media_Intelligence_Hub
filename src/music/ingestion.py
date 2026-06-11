"""
spotify_ingestion.py — Stage 1 of the Music pipeline.

Pulls your full Spotify library and EVERY field Spotify still exposes per track,
then attaches audio features from ReccoBeats (a free, drop-in replacement for
Spotify's deprecated /v1/audio-features endpoint).

Why ReccoBeats: as of Nov 27 2024 Spotify removed Audio Features, Audio Analysis,
Recommendations and Related Artists for all newly created apps. ReccoBeats returns
the same metric set (acousticness, danceability, energy, etc.) keyed off Spotify
track IDs, with no API key required.

Output: data/processed/music_library.csv  (one row per track, fully enriched)

Run:
    python src/music/spotify_ingestion.py
    python src/music/spotify_ingestion.py --limit 500      # cap for a quick test
    python src/music/spotify_ingestion.py --no-playlists   # skip playlist tracks
"""

import argparse
import json
import re
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import requests
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from tqdm import tqdm

from src.music import config

SPOTIFY_ID_RE = re.compile(r"[A-Za-z0-9]{22}")


# --------------------------------------------------------------------------- #
# 1. Auth
# --------------------------------------------------------------------------- #
def get_client() -> spotipy.Spotify:
    config.require(config.SPOTIPY_CLIENT_ID, "SPOTIPY_CLIENT_ID")
    config.require(config.SPOTIPY_CLIENT_SECRET, "SPOTIPY_CLIENT_SECRET")
    auth = SpotifyOAuth(
        client_id=config.SPOTIPY_CLIENT_ID,
        client_secret=config.SPOTIPY_CLIENT_SECRET,
        redirect_uri=config.SPOTIPY_REDIRECT_URI,
        scope=config.SPOTIFY_SCOPES,
        cache_path=str(config.CACHE_DIR / ".spotipy_token"),
    )
    return spotipy.Spotify(auth_manager=auth, requests_timeout=20, retries=5)


# --------------------------------------------------------------------------- #
# 2. Collect the library (saved + top + playlists)
# --------------------------------------------------------------------------- #
def fetch_saved_tracks(sp) -> dict:
    """All 'Liked Songs'. Returns {track_id: raw_track_object}."""
    out, offset = {}, 0
    while True:
        page = sp.current_user_saved_tracks(limit=50, offset=offset)
        items = page.get("items", [])
        if not items:
            break
        for it in items:
            t = it.get("track")
            if t and t.get("id"):
                t["_added_at"] = it.get("added_at")
                t["_source_saved"] = True
                out[t["id"]] = t
        offset += 50
        if offset >= page.get("total", 0):
            break
    return out


def fetch_top_tracks(sp) -> dict:
    """Top tracks across the 3 windows. Records rank so we can build a rating."""
    ranks = {}
    for term in ("short_term", "medium_term", "long_term"):
        offset = 0
        while offset < 200:  # API hard cap is ~99 per term anyway
            page = sp.current_user_top_tracks(limit=50, offset=offset, time_range=term)
            items = page.get("items", [])
            if not items:
                break
            for rank, t in enumerate(items, start=offset):
                if t.get("id"):
                    ranks.setdefault(t["id"], {})[term] = rank
            offset += 50
    return ranks


def fetch_playlist_tracks(sp) -> dict:
    """Every track from playlists you own or follow."""
    out = {}
    me = sp.current_user()["id"]
    offset = 0
    playlists = []
    while True:
        page = sp.current_user_playlists(limit=50, offset=offset)
        items = page.get("items", [])
        if not items:
            break
        playlists.extend(items)
        offset += 50
        if offset >= page.get("total", 0):
            break

    for pl in tqdm(playlists, desc="Playlists"):
        pid, pname = pl["id"], pl["name"]
        off = 0
        while True:
            page = sp.playlist_items(
                pid, limit=100, offset=off,
                additional_types=("track",),
                fields="items(track(id,name)),total",
            )
            items = page.get("items", [])
            if not items:
                break
            for it in items:
                t = it.get("track")
                if t and t.get("id"):
                    out.setdefault(t["id"], {"_playlists": []})["_playlists"].append(pname)
            off += 100
            if off >= page.get("total", 0):
                break
    return out


# --------------------------------------------------------------------------- #
# 3. Hydrate full metadata (tracks -> albums -> artists)
# --------------------------------------------------------------------------- #
def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def hydrate_tracks(sp, track_ids):
    tracks = {}
    for batch in tqdm(list(chunked(track_ids, 50)), desc="Track metadata"):
        for t in sp.tracks(batch).get("tracks", []):
            if t and t.get("id"):
                tracks[t["id"]] = t
    return tracks


def hydrate_artists(sp, artist_ids):
    """Artist objects carry genres, popularity and follower counts."""
    artists = {}
    ids = list(artist_ids)
    for batch in tqdm(list(chunked(ids, 50)), desc="Artist metadata"):
        for a in sp.artists(batch).get("artists", []):
            if a and a.get("id"):
                artists[a["id"]] = a
    return artists


def hydrate_albums(sp, album_ids):
    """Album objects add label, genres and total_tracks."""
    albums = {}
    ids = list(album_ids)
    for batch in tqdm(list(chunked(ids, 20)), desc="Album metadata"):
        for a in sp.albums(batch).get("albums", []):
            if a and a.get("id"):
                albums[a["id"]] = a
    return albums


# --------------------------------------------------------------------------- #
# 4. ReccoBeats audio features (Spotify audio-features replacement)
# --------------------------------------------------------------------------- #
def _extract_spotify_id(recco_obj: dict) -> str | None:
    """ReccoBeats echoes the Spotify URL in an href field; pull the 22-char id."""
    for key in ("href", "spotifyId", "spotify_id", "externalUrls", "url"):
        val = recco_obj.get(key)
        if isinstance(val, str):
            m = SPOTIFY_ID_RE.search(val)
            if m:
                return m.group(0)
        if isinstance(val, dict):
            for v in val.values():
                if isinstance(v, str):
                    m = SPOTIFY_ID_RE.search(v)
                    if m:
                        return m.group(0)
    return None


def fetch_audio_features(track_ids) -> dict:
    """
    Two-step ReccoBeats flow:
      1) GET /v1/track?ids=<spotify_ids>  -> ReccoBeats internal ids (+ spotify href)
      2) GET /v1/track/<recco_id>/audio-features -> the 9 metrics

    Results are cached to data/cached/music/reccobeats.json so re-runs are cheap
    and a flaky response on one track never costs the whole run.
    """
    cache_path = config.CACHE_DIR / "reccobeats.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [tid for tid in track_ids if tid not in cache]
    session = requests.Session()
    session.headers.update({"Accept": "application/json", "User-Agent": config.USER_AGENT})

    def save():
        cache_path.write_text(json.dumps(cache))

    for batch in tqdm(list(chunked(todo, 40)), desc="ReccoBeats features"):
        # Step 1: map Spotify ids -> ReccoBeats ids
        try:
            r = session.get(f"{config.RECCOBEATS_BASE}/track",
                            params={"ids": ",".join(batch)}, timeout=20)
            r.raise_for_status()
            content = r.json().get("content", r.json()) if r.content else []
            if isinstance(content, dict):
                content = content.get("content", [])
        except Exception as e:
            print(f"  ReccoBeats lookup failed for a batch ({e}); skipping.")
            for tid in batch:
                cache[tid] = None
            save()
            continue

        recco_map = {}
        for obj in content:
            sid = _extract_spotify_id(obj)
            if sid and obj.get("id"):
                recco_map[sid] = obj["id"]

        # Step 2: per-track audio features (endpoint takes the ReccoBeats id)
        for tid in batch:
            rid = recco_map.get(tid)
            if not rid:
                cache[tid] = None
                continue
            try:
                fr = session.get(f"{config.RECCOBEATS_BASE}/track/{rid}/audio-features",
                                 timeout=20)
                if fr.status_code == 200:
                    feats = fr.json()
                    cache[tid] = {k: feats.get(k) for k in config.AUDIO_FEATURE_COLS}
                else:
                    cache[tid] = None
            except Exception:
                cache[tid] = None
            time.sleep(0.15)  # be polite to a free service
        save()

    return cache


# --------------------------------------------------------------------------- #
# 5. Implicit rating
# --------------------------------------------------------------------------- #
def implicit_rating(row_meta: dict) -> float:
    """
    Spotify has no star ratings, so we synthesise a 1.0-5.0 affinity score from
    behavioural signals. This is the target the model learns to predict. If you
    keep your own ratings elsewhere, drop a `rating` column into music_library.csv
    after this step and it will be used instead (see feature_engineering.py).

    Signal weights (transparent on purpose, tune to taste):
      saved (Liked Song)        +1.5
      appears in a playlist     +0.4 each, capped at +1.0
      top track (any window)    rank 0 -> +2.0 decaying to ~0 by rank 99
      track popularity (0-100)  scaled to +0.0..+0.5
    """
    score = 2.5  # neutral baseline
    if row_meta.get("_source_saved"):
        score += 1.5
    n_pl = len(row_meta.get("_playlists", []))
    score += min(n_pl * 0.4, 1.0)

    ranks = row_meta.get("_top_ranks", {})
    if ranks:
        best = min(ranks.values())
        score += 2.0 * max(0.0, 1.0 - best / 99.0)

    pop = row_meta.get("popularity") or 0
    score += (pop / 100.0) * 0.5

    return round(max(1.0, min(5.0, score)), 2)


# --------------------------------------------------------------------------- #
# 6. Flatten to a row
# --------------------------------------------------------------------------- #
def build_row(t, artists, albums, audio, top_ranks, saved_meta, playlist_meta):
    tid = t["id"]
    art_objs = [artists.get(a["id"], a) for a in t.get("artists", [])]
    alb = albums.get(t.get("album", {}).get("id"), t.get("album", {}))

    genres = sorted({g for a in art_objs for g in a.get("genres", [])})
    rel_date = alb.get("release_date", "") or ""
    rel_year = int(rel_date[:4]) if rel_date[:4].isdigit() else None

    meta_for_rating = {
        "_source_saved": tid in saved_meta,
        "_playlists": playlist_meta.get(tid, {}).get("_playlists", []),
        "_top_ranks": top_ranks.get(tid, {}),
        "popularity": t.get("popularity"),
    }

    row = {
        "track_id": tid,
        "uri": t.get("uri"),
        "name": t.get("name"),
        "artists": ", ".join(a.get("name", "") for a in art_objs),
        "artist_ids": ", ".join(a.get("id", "") for a in art_objs),
        "primary_artist": art_objs[0].get("name") if art_objs else None,
        "album": alb.get("name"),
        "album_id": alb.get("id"),
        "album_type": alb.get("album_type"),
        "label": alb.get("label"),
        "release_date": rel_date,
        "release_year": rel_year,
        "album_total_tracks": alb.get("total_tracks"),
        "track_number": t.get("track_number"),
        "disc_number": t.get("disc_number"),
        "duration_ms": t.get("duration_ms"),
        "explicit": t.get("explicit"),
        "popularity": t.get("popularity"),
        "isrc": (t.get("external_ids") or {}).get("isrc"),
        "artist_genres": ", ".join(genres),
        "artist_popularity": round(
            sum(a.get("popularity", 0) for a in art_objs) / max(len(art_objs), 1), 1),
        "artist_followers": sum(
            (a.get("followers") or {}).get("total", 0) for a in art_objs),
        "preview_url": t.get("preview_url"),  # often null now (deprecated in multi-get)
        "is_saved": tid in saved_meta,
        "playlists": "; ".join(playlist_meta.get(tid, {}).get("_playlists", [])),
        "top_short": top_ranks.get(tid, {}).get("short_term"),
        "top_medium": top_ranks.get(tid, {}).get("medium_term"),
        "top_long": top_ranks.get(tid, {}).get("long_term"),
        "rating": implicit_rating(meta_for_rating),
    }

    feats = audio.get(tid) or {}
    for col in config.AUDIO_FEATURE_COLS:
        row[col] = feats.get(col)
    row["has_audio_features"] = bool(feats)
    return row


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap number of tracks")
    ap.add_argument("--no-playlists", action="store_true")
    args = ap.parse_args()

    sp = get_client()
    print("Fetching saved tracks...")
    saved = fetch_saved_tracks(sp)
    print(f"  {len(saved)} liked songs")

    print("Fetching top tracks...")
    top_ranks = fetch_top_tracks(sp)

    playlist_meta = {}
    if not args.no_playlists:
        print("Fetching playlist tracks...")
        playlist_meta = fetch_playlist_tracks(sp)
        print(f"  {len(playlist_meta)} tracks across playlists")

    all_ids = set(saved) | set(top_ranks) | set(playlist_meta)
    all_ids = list(all_ids)
    if args.limit:
        all_ids = all_ids[: args.limit]
    print(f"Total unique tracks: {len(all_ids)}")

    tracks = hydrate_tracks(sp, all_ids)
    artist_ids = {a["id"] for t in tracks.values() for a in t.get("artists", []) if a.get("id")}
    album_ids = {t.get("album", {}).get("id") for t in tracks.values() if t.get("album", {}).get("id")}
    artists = hydrate_artists(sp, artist_ids)
    albums = hydrate_albums(sp, album_ids)

    print("Fetching audio features (ReccoBeats)...")
    audio = fetch_audio_features(list(tracks.keys()))

    rows = []
    from src.music.schema import validate_track_data
    
    for t in tracks.values():
        row = build_row(t, artists, albums, audio, top_ranks, saved, playlist_meta)
        try:
            # Validate and coerce types (especially artist_genres)
            validated = validate_track_data(row)
            # Re-flatten the genres to a string for CSV storage, but ensured clean
            validated["artist_genres"] = ", ".join(validated.get("artist_genres", []))
            rows.append(validated)
        except Exception as e:
            print(f"  Warning: track {t.get('id')} failed schema validation ({e}); saving raw.")
            rows.append(row)

    df = pd.DataFrame(rows).sort_values("rating", ascending=False).reset_index(drop=True)

    df.to_csv(config.LIBRARY_CSV, index=False)
    have_af = df["has_audio_features"].sum()
    print(f"\nWrote {len(df)} tracks -> {config.LIBRARY_CSV}")
    print(f"Audio features resolved for {have_af}/{len(df)} tracks "
          f"({have_af / max(len(df),1):.0%}).")


if __name__ == "__main__":
    main()