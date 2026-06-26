"""
genius_lyrics.py — Stage 3 of the Music pipeline.

Fetches lyrics from Genius for each track and derives features from them:
VADER sentiment (compound/pos/neu/neg) plus simple lexical stats (word count,
unique-word ratio, lines). These mirror the VADER sentiment work already done
on the Movies dashboard.

Notes:
  - The Genius API does NOT return lyrics in its JSON (terms of service). The
    standard approach is to search the API for the song page, then scrape it.
    The `lyricsgenius` library does both; we use it and cache results.
  - Lyrics are copyrighted. This script stores them only in a LOCAL cache
    (data/cached/music/lyrics/) for your own personal analysis and never ships
    raw lyrics into the model-ready CSV — only the derived numeric features and
    a short embedding text. If you'd rather not persist lyrics at all, pass
    --sentiment-only and the raw text is discarded after features are computed.

Output: data/processed/music_lyrics.csv

Run:
    python src/music/genius_lyrics.py
    python src/music/genius_lyrics.py --sentiment-only
"""

import argparse
import hashlib
import re
import shutil
import time

import pandas as pd
import requests
from tqdm import tqdm

import config

try:
    import lyricsgenius
except ImportError as e:
    raise SystemExit("pip install lyricsgenius  (and: pip install vaderSentiment)") from e

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def clean_lyrics(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\[.*?\]", " ", text)                 # [Chorus], [Verse 1]
    text = re.sub(r"\d*Embed\s*$", "", text.strip())      # trailing "…Embed"
    text = re.sub(r"^.*?Lyrics", "", text, count=1, flags=re.S)  # header noise
    return re.sub(r"\s+", " ", text).strip()


_TIMESTAMP_RE = re.compile(r"\(\d{1,3}:\d{2}\)")  # "(4:09)" — tracklist/scrobble dumps


def looks_like_lyrics(text: str) -> bool:
    """Reject obvious non-lyric scrapes (tracklists, articles, wiki/book prose).

    Genius/LRCLIB sometimes return an album tracklist, a "produced by" page, or
    a totally unrelated article. Real song lyrics are short and repetitive; these
    junk pages are long, non-repetitive, and/or full of "(M:SS)" timestamps.

    Counting is whitespace-based (not Latin-only) so non-English lyrics — Hindi,
    Arabic, Korean, etc. — aren't wrongly rejected for having no [A-Za-z] words.
    """
    if not text or not text.strip():
        return False
    if len(_TIMESTAMP_RE.findall(text)) >= 8:      # tracklist / scrobble dump
        return False
    tokens = text.split()
    n = len(tokens)
    if n < 5:
        return False
    if n > 5000:                                    # no real song is this long
        return False
    uniq = len(set(w.lower() for w in tokens)) / n
    if n > 2500 and uniq > 0.5:                     # long, barely-repetitive prose
        return False
    return True


_TITLE_CLEAN = re.compile(r"\(.*?\)|\[.*?\]|feat\.?.*|with .*|-\s*\d{4}.*|remaster.*|remix.*", re.I)


def title_matches(query: str, found: str) -> bool:
    """Loose guard that the Genius hit is actually the requested song.

    Avoids cases where search_song returns a different track (the source of most
    byte-identical-across-songs junk). Passes if either title's significant words
    are a subset of the other's, so subtitles/features don't cause false rejects.
    """
    def toks(s):
        s = _TITLE_CLEAN.sub(" ", str(s or "").lower())
        return {w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 1}
    q, f = toks(query), toks(found)
    if not q or not f:
        return True  # nothing to compare — don't block
    return q <= f or f <= q or len(q & f) / len(q | f) >= 0.5


# --- Lyric cache filenames -------------------------------------------------
# Files are named "Artist - Song [track_id].txt": readable, but the trailing
# [track_id] is the real key. We always look files up by extracting that id
# (never by reconstructing the name), so slight title/artist drift between runs
# can't cause a cache miss or duplicate. extract_track_id also accepts the
# legacy "<track_id>.txt" form so old caches keep working.
_ILLEGAL_FN = re.compile(r'[<>:"/\\|?*\x00-\x1f\[\]]')
_ID_IN_NAME = re.compile(r"\[([A-Za-z0-9]+)\]\.txt$")


def _sanitize_component(s: str, limit: int = 80) -> str:
    s = _ILLEGAL_FN.sub(" ", str(s) if s is not None else "")
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:limit].strip(" .")


def lyric_cache_name(tid: str, name: str, artist: str) -> str:
    artist_s = _sanitize_component(artist)
    name_s = _sanitize_component(name) or str(tid)
    base = f"{artist_s} - {name_s}".strip(" -") or str(tid)
    return f"{base} [{tid}].txt"


def extract_track_id(filename: str):
    m = _ID_IN_NAME.search(filename)
    if m:
        return m.group(1)
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    return stem if re.fullmatch(r"[A-Za-z0-9]+", stem) else None


def index_lyric_cache(lyrics_dir):
    """Map track_id -> Path for every cached lyric file in the dir."""
    idx = {}
    if lyrics_dir.exists():
        for f in lyrics_dir.iterdir():
            if f.suffix.lower() == ".txt":
                tid = extract_track_id(f.name)
                if tid:
                    idx[tid] = f
    return idx


_REMASTER_RE = re.compile(
    r"\b(remaster(ed)?|remix|version|bonus track|live|mono|stereo|single|edit|\d{4}|restored)\b")


def _dup_title_key(filename: str) -> str:
    """Normalized song title — lets us tell genuine same-song dupes from bad scrapes."""
    s = _ID_IN_NAME.sub("", filename)
    if " - " in s:
        s = s.split(" - ", 1)[1]                       # drop "Artist - "
    s = _REMASTER_RE.sub(" ", s.lower())
    s = re.sub(r"\(.*?\)|\[.*?\]|feat\.?.*|with .*", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _is_placeholder_name(filename: str) -> bool:
    """True for "<id> [<id>].txt" files written when the library had no title."""
    m = _ID_IN_NAME.search(filename)
    return bool(m) and filename[:m.start()].strip() == m.group(1)


def revalidate_cache(lyrics_dir):
    """Move existing non-lyric cache files into _rejected/ so they get re-fetched.

    A file is quarantined when its text fails looks_like_lyrics() (tracklist /
    huge / prose), OR when its exact content is shared across more than one
    distinct real song title — the scraper returned the same wrong page for
    several tracks. Genuine same-song duplicates and metadata-less <id>
    placeholder files are preserved. Returns the number of files moved.
    """
    if not lyrics_dir.exists():
        return 0
    files = [f for f in lyrics_dir.iterdir() if f.is_file() and f.suffix.lower() == ".txt"]
    texts, by_hash = {}, {}
    for f in files:
        t = f.read_text(encoding="utf-8", errors="ignore")
        texts[f.name] = t
        h = hashlib.md5(t.strip().encode("utf-8", "ignore")).hexdigest()
        by_hash.setdefault(h, []).append(f.name)

    bad_hashes = {h for h, names in by_hash.items()
                  if len(names) > 1
                  and len({_dup_title_key(n) for n in names if not _is_placeholder_name(n)}) > 1}

    reject_dir = lyrics_dir / "_rejected"
    moved = 0
    for f in files:
        t = texts[f.name]
        h = hashlib.md5(t.strip().encode("utf-8", "ignore")).hexdigest()
        if looks_like_lyrics(t) and h not in bad_hashes:
            continue
        reject_dir.mkdir(exist_ok=True)
        shutil.move(str(f), str(reject_dir / f.name))
        moved += 1
    return moved


def fetch_lrclib(session, name, artist, album=None, duration=None):
    """Free, keyless lyrics fallback (https://lrclib.net). Tries an exact
    signature match first (artist+title+album+duration), then a looser get.
    Returns (text, instrumental_flag)."""
    base = config.LRCLIB_BASE
    attempts = []
    if name and artist:
        p = {"artist_name": artist, "track_name": name}
        if album:
            p = {**p, "album_name": album}
        if duration:
            p = {**p, "duration": int(duration)}
        attempts.append(("get", p))
        attempts.append(("get", {"artist_name": artist, "track_name": name}))  # loose
    for path, params in attempts:
        try:
            r = session.get(f"{base}/{path}", params=params, timeout=15)
            if r.status_code != 200:
                continue
            d = r.json()
            if d.get("instrumental"):
                return "", True
            text = d.get("plainLyrics") or ""
            if text.strip():
                return text, False
        except Exception:
            continue
        finally:
            time.sleep(0.3)  # be polite to a free service
    return "", False


def lyric_features(text: str, vader, source: str = "genius") -> dict:
    if not text:
        return {"lyrics_found": False, "lyric_source": None}
    words = text.split()
    sent = vader.polarity_scores(text)
    return {
        "lyrics_found": True,
        "lyric_source": source,
        "lyric_word_count": len(words),
        "lyric_unique_ratio": round(len(set(w.lower() for w in words)) / max(len(words), 1), 3),
        "lyric_line_count": text.count("\n") + 1,
        "lyric_sentiment": sent["compound"],
        "lyric_pos": sent["pos"],
        "lyric_neu": sent["neu"],
        "lyric_neg": sent["neg"],
        # short, de-duplicated snippet used ONLY to build a semantic embedding
        # in feature_engineering.py (not the full reproduced lyric).
        "lyric_embed_text": " ".join(words[:120]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment-only", action="store_true",
                    help="don't persist raw lyrics, keep only numeric features")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check existing cached files and quarantine non-lyric "
                         "ones into _rejected/ before fetching (so they get re-fetched)")
    args = ap.parse_args()

    if not config.LIBRARY_CSV.exists():
        raise SystemExit("Run spotify_ingestion.py first (music_library.csv missing).")
    config.require(config.GENIUS_ACCESS_TOKEN, "GENIUS_ACCESS_TOKEN")

    df = pd.read_csv(config.LIBRARY_CSV)
    vader = SentimentIntensityAnalyzer()
    lyrics_dir = config.CACHE_DIR / "lyrics"
    lyrics_dir.mkdir(exist_ok=True)
    if args.revalidate:
        moved = revalidate_cache(lyrics_dir)
        print(f"Revalidated cache: quarantined {moved} non-lyric file(s) -> _rejected/")
    cache_index = index_lyric_cache(lyrics_dir)  # track_id -> existing Path

    genius = lyricsgenius.Genius(
        config.GENIUS_ACCESS_TOKEN,
        timeout=15, retries=3, remove_section_headers=True,
        skip_non_songs=True, verbose=False,
    )
    genius.excluded_terms = ["(Remix)", "(Live)"]

    lrclib = requests.Session()
    lrclib.headers.update({"User-Agent": config.USER_AGENT})

    rows = []
    for _, t in tqdm(df.iterrows(), total=len(df), desc="Lyrics (Genius+LRCLIB)"):
        tid = t["track_id"]
        source = "genius"

        # primary_artist may be absent in older library exports; fall back to the
        # first credited name in `artists` so searches still work.
        artist = t.get("primary_artist")
        if not (isinstance(artist, str) and artist.strip()):
            a = t.get("artists")
            artist = str(a).split(",")[0].strip() if isinstance(a, str) and a.strip() else None

        cache_file = cache_index.get(tid)
        if cache_file and cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
            source = "cache"
        else:
            text = ""
            try:
                # Add a sleep to avoid rate limiting during long runs
                time.sleep(2.0)
                song = genius.search_song(t.get("name"), artist)
                if song and song.lyrics:
                    cand = clean_lyrics(song.lyrics)
                    # Only accept if it's the right song AND reads like lyrics —
                    # this is what stops tracklists/articles being cached as lyrics.
                    if title_matches(t.get("name"), getattr(song, "title", "")) \
                            and looks_like_lyrics(cand):
                        text = cand
            except Exception:
                text = ""

            # Fallback: LRCLIB (free, keyless) when Genius came up empty.
            if not text.strip():
                dur = t.get("duration_ms")
                lr_text, _instrumental = fetch_lrclib(
                    lrclib, t.get("name"), artist,
                    album=t.get("album"),
                    duration=(dur / 1000) if pd.notna(dur) else None,
                )
                if lr_text.strip() and looks_like_lyrics(lr_text):
                    text, source = lr_text, "lrclib"

            if text.strip() and not args.sentiment_only:
                cache_file = lyrics_dir / lyric_cache_name(tid, t.get("name"), artist)
                cache_file.write_text(text, encoding="utf-8")
                cache_index[tid] = cache_file

        rows.append({"track_id": tid, **lyric_features(text, vader, source)})

    out = pd.DataFrame(rows)
    out.to_csv(config.LYRICS_CSV, index=False)
    found = out["lyrics_found"].sum()
    by_src = out[out["lyrics_found"]]["lyric_source"].value_counts().to_dict()
    print(f"\nWrote {len(out)} rows -> {config.LYRICS_CSV}")
    print(f"Lyrics found for {found}/{len(out)} tracks ({found / max(len(out),1):.0%}). By source: {by_src}")


if __name__ == "__main__":
    main()
