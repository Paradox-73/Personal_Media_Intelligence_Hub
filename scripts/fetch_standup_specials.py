"""fetch_standup_specials.py — TMDB detail for the Letterboxd "Standup Specials" list.

The Movies dashboard has a Standup Specials section. Most of what it shows is
already in enriched_data.csv (your rating, runtime, TMDB/IMDb scores). What the
enriched CSV does not store, and this script adds:

- the comedian as a person (TMDB /person/{id}): birthday, place of birth, photo,
  so the dashboard can group by comedian, by the comedian's country, and by the
  comedian's age when the special came out;
- the special's platform: the network named in its production companies (HBO,
  Netflix, Comedy Central) when there is one, otherwise where it streams now
  (TMDB /movie/{id}/watch/providers, India first, US as fallback). A special's
  home is part of what it is, and a 2008 HBO special now streaming on Prime
  Video is still an HBO special.

The comedian is the first credited cast member; when TMDB has no cast, the
name before the ":" in the Letterboxd title is searched instead. Specials and
people already in the cache are skipped, so a re-run after adding one special
costs two or three API calls.

Writes data/cache/tmdb_standup.json (raw API answers) and
data/processed/movies/standup_specials.csv (one row per list entry).

    python scripts/fetch_standup_specials.py            # fetch missing
    python scripts/fetch_standup_specials.py --refresh  # refetch all
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
API = "https://api.themoviedb.org/3"
LIST_PATH = config.RAW_DIR / "movies" / "lists" / "standup-specials.csv"
DIARY_PATH = config.RAW_DIR / "movies" / "diary.csv"
CACHE_PATH = config.CACHE_DIR / "tmdb_standup.json"
OUT_PATH = config.MOVIES_PROCESSED_DIR / "standup_specials.csv"

# Production-company names that identify the platform when TMDB lists no
# streaming provider (common for YouTube-first Indian specials).
PLATFORM_HINTS = [("netflix", "Netflix"), ("hbo", "HBO"), ("amazon", "Prime Video"),
                  ("prime video", "Prime Video"), ("comedy central", "Comedy Central"),
                  ("youtube", "YouTube"), ("showtime", "Showtime"), ("disney", "Disney+"),
                  ("hulu", "Hulu"), ("peacock", "Peacock"), ("apple", "Apple TV+"),
                  ("800 pound gorilla", "800 Pound Gorilla"), ("dry bar", "Dry Bar")]
# Collapse TMDB provider variants ("Netflix basic with Ads") onto one label.
PROVIDER_ALIASES = {"netflix": "Netflix", "amazon prime video": "Prime Video",
                    "hbo max": "HBO", "max": "HBO", "jiohotstar": "JioHotstar",
                    "apple tv plus": "Apple TV+", "apple tv+": "Apple TV+",
                    "paramount": "Paramount+", "tubi": "Tubi"}
COUNTRY_ALIASES = {"usa": "United States", "us": "United States", "u.s": "United States",
                   "u.s.a.": "United States", "united states of america": "United States",
                   "uk": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom",
                   "wales": "United Kingdom", "northern ireland": "United Kingdom"}
US_STATES = {"alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
             "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
             "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
             "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
             "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
             "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
             "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
             "virginia", "washington", "west virginia", "wisconsin", "wyoming", "dc"}


def tmdb(path, **params):
    r = requests.get(f"{API}{path}", params={"api_key": TMDB_KEY, **params}, timeout=20)
    r.raise_for_status()
    return r.json()


def read_list():
    # Letterboxd list exports put list metadata above the film table.
    with open(LIST_PATH, encoding="utf-8") as fh:
        lines = fh.readlines()
    header = next((i for i, ln in enumerate(lines) if ln.startswith("Position,")), 3)
    return pd.read_csv(LIST_PATH, skiprows=header)


def comedian_from_title(name):
    return name.split(":")[0].strip() if ":" in name else name.strip()


def country_of(place):
    if not isinstance(place, str) or not place.strip():
        return None
    last = place.split(",")[-1].strip().strip(".")
    low = last.lower()
    if low in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[low]
    if low in US_STATES:
        return "United States"
    if "india" in low:
        return "India"
    return last


def platform_of(movie):
    companies = " | ".join(c["name"] for c in movie.get("production_companies", [])).lower()
    for hint, label in PLATFORM_HINTS:
        if hint in companies:
            return label
    prov = (movie.get("watch/providers") or {}).get("results", {})
    for region in ("IN", "US"):
        for kind in ("flatrate", "free", "ads"):
            items = (prov.get(region) or {}).get(kind) or []
            if items:
                name = items[0]["provider_name"]
                key = name.lower()
                for alias, label in PROVIDER_ALIASES.items():
                    if key == alias or key.startswith(alias + " "):
                        return label
                return name
    return "Unknown"


def _origin_country(movie):
    codes = movie.get("origin_country") or [c["iso_3166_1"] for c in movie.get("production_countries", [])]
    names = {"US": "United States", "GB": "United Kingdom", "IN": "India", "CA": "Canada", "AU": "Australia"}
    return names.get(codes[0], codes[0]) if codes else None


def find_tmdb_id(row, enriched):
    hit = enriched[enriched["letterboxd_uri"] == row["URL"]]
    if hit.empty:
        hit = enriched[(enriched["letterboxd_name"] == row["Name"]) & (enriched["year"] == row["Year"])]
    if not hit.empty and pd.notna(hit["tmdb_id"].iloc[0]):
        return int(hit["tmdb_id"].iloc[0])
    res = tmdb("/search/movie", query=row["Name"], year=int(row["Year"])).get("results") or []
    if not res:  # Letterboxd titles are "Comedian: Special"; TMDB often drops the prefix
        tail = row["Name"].split(":", 1)[-1].strip()
        res = tmdb("/search/movie", query=tail, year=int(row["Year"])).get("results") or []
    return res[0]["id"] if res else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="refetch everything already cached")
    args = ap.parse_args()
    if not TMDB_KEY:
        sys.exit("TMDB_API_KEY missing from .env")
    if not LIST_PATH.exists():
        sys.exit(f"List not found: {LIST_PATH}")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    movies, people = cache.setdefault("movies", {}), cache.setdefault("people", {})

    lst = read_list()
    enriched = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    diary = pd.read_csv(DIARY_PATH) if DIARY_PATH.exists() else pd.DataFrame()

    rows, fetched = [], 0
    for _, r in lst.iterrows():
        key = r["URL"]
        entry = movies.get(key)
        if entry is None or args.refresh:
            tid = find_tmdb_id(r, enriched)
            entry = {"tmdb_id": tid}
            if tid:
                entry["movie"] = tmdb(f"/movie/{tid}", append_to_response="credits,watch/providers")
            movies[key] = entry
            fetched += 1
        movie = entry.get("movie") or {}

        # The comedian: first billed cast member, else a person search on the title prefix.
        prefix = comedian_from_title(r["Name"])
        cast = (movie.get("credits") or {}).get("cast") or []
        pid = cast[0]["id"] if cast else None
        if pid is None:
            res = tmdb("/search/person", query=prefix).get("results") or []
            pid = res[0]["id"] if res else None
        person = None
        if pid is not None:
            if str(pid) not in people or args.refresh:
                people[str(pid)] = tmdb(f"/person/{pid}")
                fetched += 1
            person = people[str(pid)]

        e = enriched[enriched["letterboxd_uri"] == key]
        e = e.iloc[0] if not e.empty else None
        watches = diary[diary["Letterboxd URI"].isin([key])] if not diary.empty else pd.DataFrame()
        # Diary entries carry their own per-entry URI, so also match on name + year.
        if watches.empty and not diary.empty:
            watches = diary[(diary["Name"] == r["Name"]) & (diary["Year"] == r["Year"])]
        watched_date = watches["Watched Date"].max() if not watches.empty else None

        release = movie.get("release_date") or None
        birthday = (person or {}).get("birthday")
        age = None
        if release and birthday:
            age = round((pd.Timestamp(release) - pd.Timestamp(birthday)).days / 365.25, 1)

        rows.append({
            "position": r["Position"],
            "name": r["Name"],
            "year": r["Year"],
            "letterboxd_uri": key,
            "comedian": prefix,
            "tmdb_comedian": (person or {}).get("name"),
            "tmdb_id": entry.get("tmdb_id"),
            "user_rating": e["user_rating"] if e is not None else None,
            "is_liked": e["is_liked"] if e is not None else None,
            "imdb_rating": e["imdb_rating"] if e is not None else None,
            "tmdb_vote_average": movie.get("vote_average"),
            "tmdb_vote_count": movie.get("vote_count"),
            "runtime": movie.get("runtime") or None,
            "release_date": release,
            "platform": platform_of(movie) if movie else "Unknown",
            "production": "|".join(c["name"] for c in movie.get("production_companies", [])),
            "poster": movie.get("poster_path"),
            "overview": movie.get("overview"),
            "comedian_birthday": birthday,
            "comedian_birthplace": (person or {}).get("place_of_birth"),
            # Birthplace when TMDB has one, else the special's own origin country.
            "comedian_country": country_of((person or {}).get("place_of_birth"))
                                or _origin_country(movie),
            "comedian_age_at_release": age,
            "comedian_profile": (person or {}).get("profile_path"),
            "watched_date": watched_date,
            "watch_count": len(watches),
        })

    CACHE_PATH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"{len(out)} specials, {fetched} API fetches -> {OUT_PATH}")
    print(f"  rated: {out['user_rating'].notna().sum()} | comedians: {out['comedian'].nunique()} | "
          f"no TMDB match: {out['tmdb_id'].isna().sum()} | unknown platform: {(out['platform'] == 'Unknown').sum()}")


if __name__ == "__main__":
    main()
