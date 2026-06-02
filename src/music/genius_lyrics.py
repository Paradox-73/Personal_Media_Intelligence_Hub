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
import re
import time

import pandas as pd
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


def lyric_features(text: str, vader) -> dict:
    if not text:
        return {"lyrics_found": False}
    words = text.split()
    sent = vader.polarity_scores(text)
    return {
        "lyrics_found": True,
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
    args = ap.parse_args()

    if not config.LIBRARY_CSV.exists():
        raise SystemExit("Run spotify_ingestion.py first (music_library.csv missing).")
    config.require(config.GENIUS_ACCESS_TOKEN, "GENIUS_ACCESS_TOKEN")

    df = pd.read_csv(config.LIBRARY_CSV)
    vader = SentimentIntensityAnalyzer()
    lyrics_dir = config.CACHE_DIR / "lyrics"
    lyrics_dir.mkdir(exist_ok=True)

    genius = lyricsgenius.Genius(
        config.GENIUS_ACCESS_TOKEN,
        timeout=15, retries=3, remove_section_headers=True,
        skip_non_songs=True, verbose=False,
    )
    genius.excluded_terms = ["(Remix)", "(Live)"]

    rows = []
    for _, t in tqdm(df.iterrows(), total=len(df), desc="Genius lyrics"):
        tid = t["track_id"]
        cache_file = lyrics_dir / f"{tid}.txt"

        if cache_file.exists():
            text = cache_file.read_text(encoding="utf-8")
        else:
            text = ""
            try:
                # Add a sleep to avoid rate limiting during long runs
                time.sleep(2.0)
                song = genius.search_song(t.get("name"), t.get("primary_artist"))
                if song and song.lyrics:
                    text = clean_lyrics(song.lyrics)
                    if not args.sentiment_only:
                        cache_file.write_text(text, encoding="utf-8")
            except Exception:
                text = ""

        rows.append({"track_id": tid, **lyric_features(text, vader)})

    out = pd.DataFrame(rows)
    out.to_csv(config.LYRICS_CSV, index=False)
    found = out["lyrics_found"].sum()
    print(f"\nWrote {len(out)} rows -> {config.LYRICS_CSV}")
    print(f"Lyrics found for {found}/{len(out)} tracks ({found / max(len(out),1):.0%}).")


if __name__ == "__main__":
    main()
