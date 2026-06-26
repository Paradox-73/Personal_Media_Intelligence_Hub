"""
musicbrainz_enrichment.py — Stage 2 of the Music pipeline.

Adds canonical, open metadata from MusicBrainz: a stable recording MBID, the
canonical artist/release names, genre + folksonomy TAGS, and — newer — the
ARTIST'S ORIGIN COUNTRY and the LYRICS LANGUAGE of each track. These tags
backfill the gap left by Spotify killing "Related Artists" and give the model
real genre signal beyond Spotify's sparse artist-level genres; the country and
language power the dashboard world map + language breakdown.

MusicBrainz is free but strict:
  - max ~1 request/second  (enforced here)
  - a real, contactable User-Agent is REQUIRED  (set MUSICBRAINZ_CONTACT in .env)

Matching strategy, best first:
  1. ISRC lookup (exact)            -> /ws/2/recording?query=isrc:<isrc>
  2. artist + title search          -> /ws/2/recording?query=...
Then a second call hydrates tags/genres/language for the chosen recording, and a
third (cached per artist, so it's cheap across an artist's catalogue) hydrates
the artist's origin country.

API fields used (all current on the v2 web service, verified June 2026):
  recording?inc=genres+tags+artist-credits+work-rels
      -> .genres[], .tags[], .artist-credit[0].name / .artist.id, .length,
         and via the "performance" relations: .relations[].work.languages[]
  artist/<mbid>  (no inc needed for core fields)
      -> .country (ISO 3166-1 alpha-2), .area.name, .begin-area.name, .type,
         .life-span.begin

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

# Bump when the set of fields written below changes, so an older cache (missing
# the new artist-country / language columns) is transparently re-enriched.
SCHEMA_VERSION = 2

# ISO 639-3 (and a few 639-1) language codes -> display name. MusicBrainz works
# carry ISO 639-3; "zxx" = no linguistic content (instrumental), "mul" = several.
LANG_NAMES = {
    "eng": "English", "jpn": "Japanese", "spa": "Spanish", "fra": "French",
    "fre": "French", "deu": "German", "ger": "German", "ita": "Italian",
    "por": "Portuguese", "kor": "Korean", "zho": "Chinese", "chi": "Chinese",
    "cmn": "Mandarin Chinese", "yue": "Cantonese", "hin": "Hindi",
    "pan": "Punjabi", "tam": "Tamil", "tel": "Telugu", "mar": "Marathi",
    "ben": "Bengali", "guj": "Gujarati", "kan": "Kannada", "mal": "Malayalam",
    "urd": "Urdu", "rus": "Russian", "ara": "Arabic", "nld": "Dutch",
    "dut": "Dutch", "swe": "Swedish", "nor": "Norwegian", "nob": "Norwegian",
    "dan": "Danish", "fin": "Finnish", "pol": "Polish", "ces": "Czech",
    "cze": "Czech", "tur": "Turkish", "ell": "Greek", "gre": "Greek",
    "heb": "Hebrew", "tha": "Thai", "vie": "Vietnamese", "ind": "Indonesian",
    "msa": "Malay", "may": "Malay", "fas": "Persian", "per": "Persian",
    "ukr": "Ukrainian", "ron": "Romanian", "rum": "Romanian", "hun": "Hungarian",
    "isl": "Icelandic", "ice": "Icelandic", "gle": "Irish", "lat": "Latin",
    "afr": "Afrikaans", "swa": "Swahili", "tgl": "Tagalog", "fil": "Filipino",
    "mul": "Multiple languages", "und": "Undetermined", "zxx": "Instrumental",
}


def _lang_name(code):
    if not code:
        return None
    return LANG_NAMES.get(code.lower(), code.title())


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

    def recording_info(self, mbid):
        """Pull genres, folksonomy tags, the credited artist (+ its MBID) and the
        lyrics language(s) for a recording in a single call."""
        data = self._get(f"recording/{mbid}",
                         {"inc": "genres+tags+artist-credits+work-rels"})
        if not data:
            return {}
        genres = [g["name"] for g in data.get("genres", [])]
        tag_objs = data.get("tags", [])
        tags = sorted({t["name"] for t in tag_objs},
                      key=lambda x: -next((t["count"] for t in tag_objs
                                           if t["name"] == x), 0))
        artist_credit = data.get("artist-credit", [])
        canonical_artist = artist_credit[0]["name"] if artist_credit else None
        artist_mbid = (artist_credit[0].get("artist") or {}).get("id") if artist_credit else None

        # Lyrics language(s) come from the linked Work via "performance" relations.
        langs = []
        for rel in data.get("relations", []):
            work = rel.get("work") or {}
            codes = work.get("languages") or ([work["language"]] if work.get("language") else [])
            for code in codes:
                if code and code not in langs:
                    langs.append(code)
        primary_lang = langs[0] if langs else None

        return {
            "mb_recording_id": mbid,
            "mb_canonical_artist": canonical_artist,
            "mb_artist_mbid": artist_mbid,
            "mb_genres": ", ".join(genres),
            "mb_tags": ", ".join(tags[:15]),
            "mb_length_ms": data.get("length"),
            "mb_language_code": primary_lang,
            "mb_language": _lang_name(primary_lang),
        }

    def artist_info(self, artist_mbid):
        """Origin country / area / type / formation year for an artist MBID.
        Core artist fields need no `inc`, so this is a single cheap lookup."""
        if not artist_mbid:
            return {}
        data = self._get(f"artist/{artist_mbid}", {})
        if not data:
            return {}
        area = data.get("area") or {}
        code = data.get("country")
        if not code:
            iso_codes = area.get("iso-3166-1-codes") or []
            code = iso_codes[0] if iso_codes else None
        return {
            "mb_artist_country_code": code,            # ISO 3166-1 alpha-2 (e.g. "GB")
            "mb_artist_country": area.get("name"),     # human-readable ("United Kingdom")
            "mb_artist_city": (data.get("begin-area") or {}).get("name"),
            "mb_artist_type": data.get("type"),        # "Group" / "Person"
            "mb_artist_begin_year": (data.get("life-span") or {}).get("begin"),
        }


def main():
    if not config.LIBRARY_CSV.exists():
        raise SystemExit("Run spotify_ingestion.py first (music_library.csv missing).")

    df = pd.read_csv(config.LIBRARY_CSV)
    cache_path = config.CACHE_DIR / "musicbrainz.json"
    artist_cache_path = config.CACHE_DIR / "musicbrainz_artists.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    artist_cache = json.loads(artist_cache_path.read_text()) if artist_cache_path.exists() else {}

    def is_current(entry):
        # An entry is fresh if it carries this schema version; misses are stored
        # as {} and are kept (don't re-hammer true non-matches every run).
        return isinstance(entry, dict) and (entry.get("_schema") == SCHEMA_VERSION or entry == {})

    mb = MusicBrainz()
    rows = []
    for _, t in tqdm(df.iterrows(), total=len(df), desc="MusicBrainz"):
        tid = t["track_id"]
        if tid in cache and is_current(cache[tid]):
            info = {k: v for k, v in (cache[tid] or {}).items() if k != "_schema"}
            rows.append({"track_id": tid, **info})
            continue

        # Re-enrich fast: if a prior run already matched this track, reuse that
        # recording MBID and skip the (expensive) search step entirely.
        prev = cache.get(tid) if isinstance(cache.get(tid), dict) else {}
        rid = prev.get("mb_recording_id")
        if not rid:
            rec = mb.find_recording(
                isrc=None if pd.isna(t.get("isrc")) else t.get("isrc"),
                artist=t.get("primary_artist"),
                title=t.get("name"),
            )
            rid = rec["id"] if rec and rec.get("id") else None
        info = mb.recording_info(rid) if rid else {}

        # Enrich with the artist's origin country (cached per artist MBID, so an
        # artist's whole catalogue costs just one extra request).
        amid = info.get("mb_artist_mbid")
        if amid:
            if amid not in artist_cache:
                artist_cache[amid] = mb.artist_info(amid)
                artist_cache_path.write_text(json.dumps(artist_cache))
            info.update(artist_cache[amid] or {})

        cache[tid] = {**info, "_schema": SCHEMA_VERSION}
        rows.append({"track_id": tid, **info})

        if len(rows) % 25 == 0:                # checkpoint
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    artist_cache_path.write_text(json.dumps(artist_cache))
    out = pd.DataFrame(rows)
    out.to_csv(config.MUSICBRAINZ_CSV, index=False)
    matched = out["mb_recording_id"].notna().sum() if "mb_recording_id" in out else 0
    countries = out["mb_artist_country_code"].notna().sum() if "mb_artist_country_code" in out else 0
    langs = out["mb_language_code"].notna().sum() if "mb_language_code" in out else 0
    print(f"\nWrote {len(out)} rows -> {config.MUSICBRAINZ_CSV}")
    print(f"Matched {matched}/{len(out)} recordings on MusicBrainz.")
    print(f"Artist origin country for {countries}/{len(out)} tracks; lyrics language for {langs}/{len(out)}.")


if __name__ == "__main__":
    main()