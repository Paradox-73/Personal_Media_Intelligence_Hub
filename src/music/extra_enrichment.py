"""
extra_enrichment.py — Stage 2b of the Music pipeline (optional, additive).

Pulls metadata that Spotify no longer exposes (it deprecated audio-features,
related-artists, recommendations, … in Nov-2024 / Feb-2026) from free / current
APIs and writes one merge-ready table keyed by track_id.

Sources (all verified current, June 2026):
  * Deezer   (no key)            -> BPM, gain, global rank   [track-level]
  * Discogs  (token, optional)   -> release genres/styles, label, country, year
  * TheAudioDB (key, optional)   -> artist country, genre/style/mood, formed year
  * Wikidata (no key)            -> artist country of origin, genres, label
                                    (joined via the MusicBrainz artist MBID, P434)
  * Last.fm  (key, optional)     -> artist listeners/playcount, crowd tags, similar
                                    artists (Spotify's killed "related"), track plays

Design mirrors the MusicBrainz stage: a polite per-host rate limiter, on-disk
JSON caches (per track and per artist) so the run is fully resumable, and a
schema version so adding fields later transparently re-enriches. Any provider
whose credential is missing is skipped — the rest still run.

Inputs : data/processed/music/music_library.csv      (Spotify)
         data/processed/music/music_musicbrainz.csv   (for mb_artist_mbid)
Output : data/processed/music/music_extra.csv

Run:
    python src/music/extra_enrichment.py
    python src/music/extra_enrichment.py --limit 200 --skip discogs,audiodb
"""

import argparse
import json
import time

import pandas as pd
import requests
from tqdm import tqdm

import config

SCHEMA_VERSION = 2  # bumped: v1 entries predate the Wikidata join (needed mb_artist_mbid)

# Output fields per source — used to build empty rows so the CSV schema is stable
# even when a provider is skipped or finds nothing.
DEEZER_FIELDS = ["deezer_id", "deezer_bpm", "deezer_gain", "deezer_rank"]
DISCOGS_FIELDS = ["discogs_genres", "discogs_styles", "discogs_year",
                  "discogs_label", "discogs_country"]
AUDIODB_FIELDS = ["audiodb_country", "audiodb_genre", "audiodb_style",
                  "audiodb_mood", "audiodb_formed_year"]
WIKIDATA_FIELDS = ["wd_country_of_origin", "wd_genres", "wd_record_label"]
LASTFM_FIELDS = ["lastfm_artist_listeners", "lastfm_artist_playcount", "lastfm_tags",
                 "lastfm_similar", "lastfm_track_playcount", "lastfm_track_listeners"]
ALL_FIELDS = DEEZER_FIELDS + DISCOGS_FIELDS + AUDIODB_FIELDS + WIKIDATA_FIELDS + LASTFM_FIELDS


class RateLimitedClient:
    """Minimal session with a wall-clock min-interval and one 429/503 retry."""

    def __init__(self, min_interval, headers=None):
        self.s = requests.Session()
        if headers:
            self.s.headers.update(headers)
        self.min_interval = min_interval
        self._last = 0.0

    def get_json(self, url, params=None, timeout=25):
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            r = self.s.get(url, params=params, timeout=timeout)
            self._last = time.time()
            if r.status_code in (429, 503):
                time.sleep(5)
                r = self.s.get(url, params=params, timeout=timeout)
                self._last = time.time()
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            self._last = time.time()
            return None


# --- Providers -------------------------------------------------------------

def deezer_track(client, artist, title):
    if not (artist and title):
        return {}
    q = f'artist:"{artist}" track:"{title}"'
    data = client.get_json(f"{config.DEEZER_BASE}/search", {"q": q, "limit": 1})
    res = (data or {}).get("data") or []
    if not res:  # looser query
        data = client.get_json(f"{config.DEEZER_BASE}/search",
                               {"q": f"{artist} {title}", "limit": 1})
        res = (data or {}).get("data") or []
    if not res:
        return {}
    tid = res[0].get("id")
    rank = res[0].get("rank")
    # BPM / gain only exist on the full track resource, not in search results.
    full = client.get_json(f"{config.DEEZER_BASE}/track/{tid}") if tid else None
    bpm = (full or {}).get("bpm")
    return {
        "deezer_id": tid,
        "deezer_bpm": bpm if (bpm and bpm > 0) else None,  # Deezer uses 0 for "unknown"
        "deezer_gain": (full or {}).get("gain"),
        "deezer_rank": rank,
    }


def discogs_release(client, token, artist, title):
    if not (token and artist and title):
        return {}
    data = client.get_json(f"{config.DISCOGS_BASE}/database/search", {
        "artist": artist, "track": title, "type": "release",
        "per_page": 1, "token": token,
    })
    res = (data or {}).get("results") or []
    if not res:
        return {}
    r0 = res[0]
    return {
        "discogs_genres": ", ".join(r0.get("genre") or []),
        "discogs_styles": ", ".join(r0.get("style") or []),
        "discogs_year": r0.get("year"),
        "discogs_label": ", ".join(r0.get("label") or []),
        "discogs_country": r0.get("country"),
    }


def audiodb_artist(client, key, artist):
    if not (key and artist):
        return {}
    data = client.get_json(f"{config.AUDIODB_BASE}/{key}/search.php", {"s": artist})
    arts = (data or {}).get("artists") or []
    if not arts:
        return {}
    a = arts[0]
    return {
        "audiodb_country": a.get("strCountry"),
        "audiodb_genre": a.get("strGenre"),
        "audiodb_style": a.get("strStyle"),
        "audiodb_mood": a.get("strMood"),
        "audiodb_formed_year": a.get("intFormedYear"),
    }


def wikidata_artist(client, mbid):
    """Look an artist up on Wikidata by its MusicBrainz ID (property P434) and
    return country of origin (P495), genres (P136) and record label (P264)."""
    if not mbid:
        return {}
    query = (
        "SELECT ?countryLabel ?genreLabel ?labelLabel WHERE { "
        f'?a wdt:P434 "{mbid}". '
        "OPTIONAL { ?a wdt:P495 ?country. } "
        "OPTIONAL { ?a wdt:P136 ?genre. } "
        "OPTIONAL { ?a wdt:P264 ?label. } "
        'SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } LIMIT 50'
    )
    data = client.get_json(config.WIKIDATA_SPARQL, {"query": query, "format": "json"})
    binds = (((data or {}).get("results") or {}).get("bindings")) or []
    if not binds:
        return {}
    country = None
    genres, labels = [], []
    for b in binds:
        country = country or (b.get("countryLabel") or {}).get("value")
        g = (b.get("genreLabel") or {}).get("value")
        if g and g not in genres:
            genres.append(g)
        lab = (b.get("labelLabel") or {}).get("value")
        if lab and lab not in labels:
            labels.append(lab)
    return {
        "wd_country_of_origin": country,
        "wd_genres": ", ".join(genres) or None,
        "wd_record_label": ", ".join(labels) or None,
    }


def _lastfm(client, key, params):
    return client.get_json(config.LASTFM_BASE, {**params, "api_key": key, "format": "json"})


def lastfm_artist(client, key, artist):
    """Artist-level: global listeners/playcount, crowd tags, similar artists."""
    if not (key and artist):
        return {}
    data = _lastfm(client, key, {"method": "artist.getinfo", "artist": artist, "autocorrect": 1})
    a = (data or {}).get("artist") or {}
    if not a:
        return {}
    stats = a.get("stats") or {}
    tags = [t.get("name") for t in ((a.get("tags") or {}).get("tag") or []) if t.get("name")]
    similar = [s.get("name") for s in ((a.get("similar") or {}).get("artist") or []) if s.get("name")]
    return {
        "lastfm_artist_listeners": stats.get("listeners"),
        "lastfm_artist_playcount": stats.get("playcount"),
        "lastfm_tags": ", ".join(tags[:6]) or None,
        "lastfm_similar": ", ".join(similar[:6]) or None,
    }


def lastfm_track(client, key, artist, title):
    """Track-level: global playcount + listeners for this specific recording."""
    if not (key and artist and title):
        return {}
    data = _lastfm(client, key, {"method": "track.getinfo", "artist": artist,
                                 "track": title, "autocorrect": 1})
    t = (data or {}).get("track") or {}
    if not t:
        return {}
    return {
        "lastfm_track_playcount": t.get("playcount"),
        "lastfm_track_listeners": t.get("listeners"),
    }


# --- Orchestration ---------------------------------------------------------

def primary_artist_of(row):
    pa = row.get("primary_artist")
    if isinstance(pa, str) and pa.strip():
        return pa.strip()
    artists = row.get("artists")
    if isinstance(artists, str) and artists.strip():
        return artists.split(",")[0].strip()  # first credited artist
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="process only the first N tracks")
    ap.add_argument("--skip", default="", help="comma list: deezer,discogs,audiodb,wikidata,lastfm")
    args = ap.parse_args()
    skip = {s.strip().lower() for s in args.skip.split(",") if s.strip()}

    if not config.LIBRARY_CSV.exists():
        raise SystemExit("Run ingestion.py first (music_library.csv missing).")

    df = pd.read_csv(config.LIBRARY_CSV)
    # Join MusicBrainz artist MBID for the Wikidata lookup.
    if config.MUSICBRAINZ_CSV.exists():
        mb = pd.read_csv(config.MUSICBRAINZ_CSV)
        join_cols = [c for c in ("track_id", "mb_artist_mbid", "mb_canonical_artist") if c in mb.columns]
        df = df.merge(mb[join_cols], on="track_id", how="left")
    if args.limit:
        df = df.head(args.limit)

    use_deezer = "deezer" not in skip
    use_discogs = "discogs" not in skip and bool(config.DISCOGS_TOKEN)
    use_audiodb = "audiodb" not in skip and bool(config.AUDIODB_KEY)
    use_wikidata = "wikidata" not in skip
    use_lastfm = "lastfm" not in skip and bool(config.LASTFM_API_KEY)
    if "discogs" not in skip and not config.DISCOGS_TOKEN:
        print("• Discogs skipped: set DISCOGS_TOKEN in .env to enable it.")
    if "lastfm" not in skip and not config.LASTFM_API_KEY:
        print("• Last.fm skipped: set LASTFM_API_KEY in .env to enable it.")
    print(f"• Enabled: deezer={use_deezer} discogs={use_discogs} "
          f"audiodb={use_audiodb} wikidata={use_wikidata} lastfm={use_lastfm}")

    # Per-host clients (Deezer ~50/5s; Discogs 60/min auth; AudioDB + Wikidata polite ~1.5s; Last.fm ~5/s).
    deezer = RateLimitedClient(0.15, {"User-Agent": config.USER_AGENT})
    discogs = RateLimitedClient(1.1, {"User-Agent": config.USER_AGENT})
    audiodb = RateLimitedClient(1.5, {"User-Agent": config.USER_AGENT})
    wikidata = RateLimitedClient(1.5, {"User-Agent": config.USER_AGENT,
                                       "Accept": "application/sparql-results+json"})
    lastfm = RateLimitedClient(0.25, {"User-Agent": config.USER_AGENT})

    cache_path = config.CACHE_DIR / "extra.json"
    artist_audiodb_path = config.CACHE_DIR / "audiodb_artists.json"
    artist_wd_path = config.CACHE_DIR / "wikidata_artists.json"
    artist_lastfm_path = config.CACHE_DIR / "lastfm_artists.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    audiodb_cache = json.loads(artist_audiodb_path.read_text()) if artist_audiodb_path.exists() else {}
    wd_cache = json.loads(artist_wd_path.read_text()) if artist_wd_path.exists() else {}
    lastfm_cache = json.loads(artist_lastfm_path.read_text()) if artist_lastfm_path.exists() else {}

    rows = []
    for _, t in tqdm(df.iterrows(), total=len(df), desc="Extra enrichment"):
        tid = t["track_id"]
        cached = cache.get(tid)
        if isinstance(cached, dict) and cached.get("_schema") == SCHEMA_VERSION:
            rows.append({"track_id": tid, **{k: v for k, v in cached.items() if k != "_schema"}})
            continue

        artist = primary_artist_of(t)
        title = t.get("name")
        info = {f: None for f in ALL_FIELDS}

        if use_deezer:
            info.update(deezer_track(deezer, artist, title))
        if use_discogs:
            info.update(discogs_release(discogs, config.DISCOGS_TOKEN, artist, title))

        # Artist-level lookups are cached by artist so a whole catalogue is one call.
        if use_audiodb and artist:
            akey = artist.lower()
            if akey not in audiodb_cache:
                audiodb_cache[akey] = audiodb_artist(audiodb, config.AUDIODB_KEY, artist)
                artist_audiodb_path.write_text(json.dumps(audiodb_cache))
            info.update({k: v for k, v in (audiodb_cache[akey] or {}).items()})

        mbid = t.get("mb_artist_mbid")
        mbid = mbid if isinstance(mbid, str) and mbid.strip() else None
        if use_wikidata and mbid:
            if mbid not in wd_cache:
                wd_cache[mbid] = wikidata_artist(wikidata, mbid)
                artist_wd_path.write_text(json.dumps(wd_cache))
            info.update({k: v for k, v in (wd_cache[mbid] or {}).items()})

        if use_lastfm and artist:
            akey = artist.lower()
            if akey not in lastfm_cache:
                lastfm_cache[akey] = lastfm_artist(lastfm, config.LASTFM_API_KEY, artist)
                artist_lastfm_path.write_text(json.dumps(lastfm_cache))
            info.update({k: v for k, v in (lastfm_cache[akey] or {}).items()})
            info.update(lastfm_track(lastfm, config.LASTFM_API_KEY, artist, title))

        cache[tid] = {**info, "_schema": SCHEMA_VERSION}
        rows.append({"track_id": tid, **info})

        if len(rows) % 25 == 0:
            cache_path.write_text(json.dumps(cache))

    cache_path.write_text(json.dumps(cache))
    audiodb_cache and artist_audiodb_path.write_text(json.dumps(audiodb_cache))
    wd_cache and artist_wd_path.write_text(json.dumps(wd_cache))
    lastfm_cache and artist_lastfm_path.write_text(json.dumps(lastfm_cache))

    out = pd.DataFrame(rows)
    # Guarantee a stable column set.
    for col in ["track_id"] + ALL_FIELDS:
        if col not in out.columns:
            out[col] = None
    out = out[["track_id"] + ALL_FIELDS]
    out.to_csv(config.EXTRA_CSV, index=False)

    def filled(col):
        return int(out[col].notna().sum()) if col in out else 0
    print(f"\nWrote {len(out)} rows -> {config.EXTRA_CSV}")
    print(f"  Deezer BPM: {filled('deezer_bpm')}  | Discogs styles: {filled('discogs_styles')}")
    print(f"  AudioDB country: {filled('audiodb_country')}  | Wikidata country: {filled('wd_country_of_origin')}")
    print(f"  Last.fm artist listeners: {filled('lastfm_artist_listeners')}  | Last.fm similar: {filled('lastfm_similar')}")


if __name__ == "__main__":
    main()
