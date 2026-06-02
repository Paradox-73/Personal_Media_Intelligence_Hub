"""
musicbrainz_enrichment.py — Stage 2 of the Music pipeline.

Adds canonical, open metadata from MusicBrainz: a stable recording MBID, the
canonical artist/release names, and crucially genre + folksonomy TAGS. These
tags backfill the gap left by Spotify killing "Related Artists" and give the
model real genre signal beyond Spotify's sparse artist-level genres.

MusicBrainz is free but strict:
  - max ~1 request/second  (enforced here)
  - a real, contactable User-Agent is REQUIRED  (set MUSICBRAINZ_CONTACT in .env)

Matching strategy, best first:
  1. ISRC lookup (exact)            -> /ws/2/recording?query=isrc:<isrc>
  2. artist + title search          -> /ws/2/recording?query=...
Then a second call hydrates tags/genres for the chosen recording.

Output: data/processed/music_musicbrainz.csv  (track_id + MB fields)

Run:
    python src/music/musicbrainz_enrichment.py
"""

import json
import time

import pandas as pd
import requests
from tqdm import tqdm

import config

RATE_DELAY = 1.05  # seconds between calls; MusicBrainz allows ~1/s


class MusicBrainz:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": config.USER_AGENT, "Accept": "application/json"})
        self._last = 0.0

    def _get(self, path, params):
        wait = RATE_DELAY - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {**params, "fmt": "json"}
        try:
            r = self.s.get(f"{config.MUSICBRAINZ_BASE}/{path}", params=params, timeout=25)
            self._last = time.time()
            if r.status_code == 503:           # rate limited -> back off and retry once
                time.sleep(3)
                r = self.s.get(f"{config.MUSICBRAINZ_BASE}/{path}", params=params, timeout=25)
                self._last = time.time()
            r.raise_for_status()
            return r.json()
        except Exception:
            self._last = time.time()
            return None

    def find_recording(self, isrc, artist, title):
        if isrc:
            data = self._get("recording", {"query": f"isrc:{isrc}", "limit": 1})
            recs = (data or {}).get("recordings", [])
            if recs:
                return recs[0]
        if artist and title:
            q = f'recording:"{title}" AND artist:"{artist}"'
            data = self._get("recording", {"query": q, "limit": 1})
            recs = (data or {}).get("recordings", [])
            if recs:
                return recs[0]
        return None

    def recording_tags(self, mbid):
        """Pull genres + folksonomy tags for a recording (and its artist)."""
        data = self._get(f"recording/{mbid}",
                         {"inc": "genres+tags+artist-credits+releases"})
        if not data:
            return {}
        genres = [g["name"] for g in data.get("genres", [])]
        tags = sorted({t["name"] for t in data.get("tags", [])},
                      key=lambda x: -next((t["count"] for t in data["tags"]
                                           if t["name"] == x), 0))
        artist_credit = data.get("artist-credit", [])
        canonical_artist = artist_credit[0]["name"] if artist_credit else None
        return {
            "mb_recording_id": mbid,
            "mb_canonical_artist": canonical_artist,
            "mb_genres": ", ".join(genres),
            "mb_tags": ", ".join(tags[:15]),
            "mb_length_ms": data.get("length"),
        }


def main():
    if not config.LIBRARY_CSV.exists():
        raise SystemExit("Run spotify_ingestion.py first (music_library.csv missing).")

    df = pd.read_csv(config.LIBRARY_CSV)
    cache_path = config.CACHE_DIR / "musicbrainz.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    mb = MusicBrainz()
    rows = []
    for _, t in tqdm(df.iterrows(), total=len(df), desc="MusicBrainz"):
        tid = t["track_id"]
        if tid in cache:
            rows.append({"track_id": tid, **(cache[tid] or {})})
            continue

        rec = mb.find_recording(
            isrc=None if pd.isna(t.get("isrc")) else t.get("isrc"),
            artist=t.get("primary_artist"),
            title=t.get("name"),
        )
        info = mb.recording_tags(rec["id"]) if rec and rec.get("id") else {}
        cache[tid] = info
        rows.append({"track_id": tid, **info})

        if len(rows) % 25 == 0:                # checkpoint
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    out = pd.DataFrame(rows)
    out.to_csv(config.MUSICBRAINZ_CSV, index=False)
    matched = out["mb_recording_id"].notna().sum() if "mb_recording_id" in out else 0
    print(f"\nWrote {len(out)} rows -> {config.MUSICBRAINZ_CSV}")
    print(f"Matched {matched}/{len(out)} recordings on MusicBrainz.")


if __name__ == "__main__":
    main()