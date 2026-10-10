"""fetch_theatre_release_dates.py — India theatrical release dates for theatre trips.

The Movies dashboard's "Release Gap" tab shows how many days after a film opened
you saw it in a cinema. That needs the release date *in India*, which nothing
else in the project stores: enriched_data.csv and tmdb_cache.json carry TMDB's
single `release_date`, which is the earliest date anywhere — often a festival
or world premiere days or weeks before the Indian opening (Dune: Part Two says
2024-02-27; it opened in India on 2024-03-01).

TMDB's /movie/{id}/release_dates endpoint lists dates per country and per
release type. This script keeps the earliest theatrical (type 3) or limited
(type 2) date for India and for the US; the US date is the fallback when TMDB
has no India entry.

Only films on the theatre list that also have a dated diary entry are fetched,
and films already in the cache are skipped, so re-running after a new cinema
trip costs one or two API calls. Writes data/cache/tmdb_release_dates.json.

    python scripts/fetch_theatre_release_dates.py            # fetch missing
    python scripts/fetch_theatre_release_dates.py --refresh  # refetch all
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
CACHE_PATH = config.CACHE_DIR / "tmdb_release_dates.json"
THEATRE_PATH = config.RAW_DIR / "movies" / "lists" / "theatre-watch.csv"
DIARY_PATH = config.RAW_DIR / "movies" / "diary.csv"
THEATRICAL_TYPES = (2, 3)  # TMDB: 2 = limited theatrical, 3 = theatrical


def read_theatre_list():
    # Letterboxd list exports put list metadata above the film table; find the
    # real header rather than trusting a fixed skiprows.
    with open(THEATRE_PATH, encoding="utf-8") as fh:
        lines = fh.readlines()
    header = next((i for i, ln in enumerate(lines) if ln.startswith("Position,")), 3)
    return pd.read_csv(THEATRE_PATH, skiprows=header)


def cache_key(name, year):
    return f"{name}|{int(year)}"


def find_tmdb_id(name, year, enriched):
    hit = enriched[(enriched["letterboxd_name"] == name) & (enriched["year"] == year)]
    if not hit.empty and pd.notna(hit["tmdb_id"].iloc[0]):
        return int(hit["tmdb_id"].iloc[0])
    r = requests.get("https://api.themoviedb.org/3/search/movie",
                     params={"api_key": TMDB_KEY, "query": name, "year": int(year)}, timeout=20)
    results = r.json().get("results") or []
    return results[0]["id"] if results else None


def earliest_theatrical(payload, country):
    for entry in payload.get("results", []):
        if entry["iso_3166_1"] == country:
            dates = [d["release_date"][:10] for d in entry["release_dates"]
                     if d["type"] in THEATRICAL_TYPES]
            if dates:
                return min(dates)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch films already cached")
    args = ap.parse_args()
    if not TMDB_KEY:
        sys.exit("TMDB_API_KEY missing in .env")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    theatre = read_theatre_list()
    diary = pd.read_csv(DIARY_PATH).dropna(subset=["Watched Date"])
    films = theatre.merge(diary[["Name", "Year"]].drop_duplicates(), on=["Name", "Year"])
    enriched = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH, usecols=["tmdb_id", "letterboxd_name", "year"])

    fetched = 0
    for _, row in films.iterrows():
        key = cache_key(row["Name"], row["Year"])
        if key in cache and not args.refresh:
            continue
        tmdb_id = find_tmdb_id(row["Name"], row["Year"], enriched)
        entry = {"tmdb_id": tmdb_id, "IN": None, "US": None}
        if tmdb_id:
            payload = requests.get(f"https://api.themoviedb.org/3/movie/{tmdb_id}/release_dates",
                                   params={"api_key": TMDB_KEY}, timeout=20).json()
            entry["IN"] = earliest_theatrical(payload, "IN")
            entry["US"] = earliest_theatrical(payload, "US")
        cache[key] = entry
        fetched += 1
        print(f"{key}: IN={entry['IN']} US={entry['US']}")

    CACHE_PATH.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Fetched {fetched}, cache now holds {len(cache)} films -> {CACHE_PATH}")


if __name__ == "__main__":
    main()
