"""
Sync the enriched movies CSV from a Letterboxd RSS feed (diary/watched entries).

The public RSS feed (https://letterboxd.com/<user>/rss/) carries the member's most
recent activity. We keep ONLY film entries (diary/watched) and ignore list, story
and pure-review items -- film items are the ones that carry a <tmdb:movieId> and
<letterboxd:filmTitle>; list/story items do not.

Matching to the enriched CSV is by TMDB id (the RSS diary <link> is a per-entry URL,
not the boxd.it film shortlink stored in letterboxd_uri, so URIs don't line up).

Behaviour (conservative, mirrors the "don't overwrite" rule):
  * Film in RSS but NOT in the CSV      -> enriched via TMDB/OMDB and appended, with
                                           user_rating / is_liked / rating_date filled.
  * Film already in the CSV             -> only *fills* user_rating / is_liked /
                                           rating_date when the existing cell is blank;
                                           never overwrites an existing value.

Usage:
    python src/movies/letterboxd_rss_sync.py --user Paradox_73 --plan
    python src/movies/letterboxd_rss_sync.py --user Paradox_73 --apply
"""
import argparse
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.ingestion import get_movie_metadata, clean_year, is_effectively_empty

NS = {
    "letterboxd": "https://letterboxd.com",
    "tmdb": "https://themoviedb.org",
    "dc": "http://purl.org/dc/elements/1.1/",
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MediaHub/1.0"


def fetch_rss(user):
    url = f"https://letterboxd.com/{user}/rss/"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    return urllib.request.urlopen(req, timeout=30).read()


def parse_films(xml_bytes):
    """Return film diary entries only (skip lists / stories / non-film items)."""
    root = ET.fromstring(xml_bytes)
    films = []
    for item in root.iter("item"):
        tmdb_el = item.find("tmdb:movieId", NS)
        title_el = item.find("letterboxd:filmTitle", NS)
        # A film entry is defined by having a film title + TMDB movie id. Lists have
        # neither; TV entries would carry tmdb:tvId instead of tmdb:movieId.
        if title_el is None or tmdb_el is None or not (tmdb_el.text or "").strip():
            continue

        def txt(tag):
            el = item.find(tag, NS)
            return el.text.strip() if (el is not None and el.text) else None

        rating = txt("letterboxd:memberRating")
        films.append({
            "tmdb_id": int(tmdb_el.text.strip()),
            "title": (title_el.text or "").strip(),
            "year": clean_year(txt("letterboxd:filmYear")),
            "user_rating": float(rating) if rating else None,
            "is_liked": 1 if (txt("letterboxd:memberLike") or "").lower() == "true" else 0,
            "rating_date": txt("letterboxd:watchedDate"),
            "rewatch": (txt("letterboxd:rewatch") or "").lower() == "true",
            "link": txt("link"),
        })
    # De-dupe by tmdb_id, keeping the most recent (feed is newest-first).
    seen, uniq = set(), []
    for f in films:
        if f["tmdb_id"] in seen:
            continue
        seen.add(f["tmdb_id"])
        uniq.append(f)
    return uniq


def main():
    try:  # emoji/unicode-safe on Windows cp1252 consoles
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="Paradox_73")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    films = parse_films(fetch_rss(args.user))
    print(f"RSS film entries (deduped): {len(films)}")

    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    existing_tmdb = {int(float(x)) for x in df["tmdb_id"].dropna()}
    # title+year fallback index for the ~1 row without a tmdb_id
    ty_index = {}
    for i, r in df.iterrows():
        ty_index[(str(r.get("title")).strip().lower(), clean_year(r.get("year")))] = i
        ty_index[(str(r.get("letterboxd_name")).strip().lower(), clean_year(r.get("year")))] = i

    new_films, fill_rows = [], []
    for f in films:
        idx = None
        if f["tmdb_id"] in existing_tmdb:
            idx = df.index[df["tmdb_id"] == f["tmdb_id"]][0]
        else:
            idx = ty_index.get((f["title"].lower(), f["year"]))
        if idx is None:
            new_films.append(f)
        else:
            fills = {c: f[c] for c in ("user_rating", "is_liked", "rating_date")
                     if is_effectively_empty(df.at[idx, c]) if c in df.columns and not is_effectively_empty(f[c])}
            if fills:
                fill_rows.append((idx, f, fills))

    print(f"\n=== NEW films to add ({len(new_films)}) ===")
    for f in new_films:
        print(f"  + {f['title']} ({f['year']})  tmdb {f['tmdb_id']}  rating={f['user_rating']}")
    print(f"\n=== EXISTING films with blank fields to fill ({len(fill_rows)}) ===")
    for idx, f, fills in fill_rows:
        print(f"  ~ {f['title']} ({f['year']}) -> {fills}")

    if args.plan:
        print("\nPLAN ONLY - nothing changed.")
        return

    # --- apply ---
    for idx, f, fills in fill_rows:
        for c, v in fills.items():
            df.at[idx, c] = v

    added = []
    for f in new_films:
        print(f"[enrich] {f['title']} ({f['year']}) tmdb {f['tmdb_id']}")
        meta = get_movie_metadata(f["title"], f["year"], tmdb_id=f["tmdb_id"])
        meta["letterboxd_name"] = f["title"]
        meta["letterboxd_uri"] = f["link"]
        meta["user_rating"] = f["user_rating"]
        meta["is_liked"] = f["is_liked"]
        meta["rating_date"] = f["rating_date"]
        meta["year"] = meta.get("year") or f["year"]
        added.append(meta)
        time.sleep(0.1)

    if added:
        new_df = pd.DataFrame(added)
        for c in df.columns:
            if c not in new_df.columns:
                new_df[c] = pd.NA
        new_df = new_df[df.columns]
        df = pd.concat([df, new_df], ignore_index=True)

    df.to_csv(config.MOVIES_ENRICHED_DATA_PATH, index=False)
    print(f"\nDONE. filled {len(fill_rows)} existing, added {len(added)} new. "
          f"CSV now {len(df)} rows.")


if __name__ == "__main__":
    main()
