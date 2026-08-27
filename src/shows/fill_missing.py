"""
Fill ONLY null (NaN) cells in the enriched TV shows CSV.

Two passes per row that has a fillable null:
  1. Re-fetch TMDB/TVMaze/OMDB via get_tv_show_metadata_ultimate (keyed by tmdb_id).
  2. TheTVDB v4 fallback for the handful of fields TVDB can still supply
     (average runtime, production companies, overview) when they remain null.

Never overwrites an existing non-null value.

Usage:
    python src/shows/fill_missing.py --plan
    python src/shows/fill_missing.py --apply
"""
import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import get_tv_show_metadata_ultimate, is_effectively_empty

load_dotenv()

# CSV column  <-  key from get_tv_show_metadata_ultimate()
FILLABLE = {
    "overview": "overview", "tagline": "tagline", "created_by": "created_by",
    "writer": "writer", "actors": "actors", "genres": "genres",
    "network": "network", "country": "country", "language": "language",
    "production_companies": "production_companies", "homepage": "homepage",
    "runtime": "runtime", "awards": "awards", "age_rating": "age_rating",
    "imdb_rating": "imdb_rating", "imdb_votes": "imdb_votes",
    "vote_average": "vote_average", "vote_count": "vote_count",
    "number_of_episodes": "number_of_episodes", "number_of_seasons": "number_of_seasons",
    "status": "status", "type": "type", "imdb_id": "imdb_id",
}

TVDB_BASE = "https://api4.thetvdb.com/v4"


def tvdb_login():
    key = os.getenv("TVDB_API_KEY")
    if not key:
        return None
    try:
        r = requests.post(f"{TVDB_BASE}/login", json={"apikey": key}, timeout=15)
        return r.json().get("data", {}).get("token")
    except Exception:  # noqa: BLE001
        return None


def tvdb_extended(token, imdb_id=None, name=None):
    """Return a small dict of fillable fields from TheTVDB, or {}."""
    if not token:
        return {}
    h = {"Authorization": f"Bearer {token}"}
    sid = None
    try:
        if imdb_id and not is_effectively_empty(imdb_id):
            r = requests.get(f"{TVDB_BASE}/search/remoteid/{imdb_id}", headers=h, timeout=15)
            for d in r.json().get("data", []):
                if d.get("series"):
                    sid = d["series"].get("id")
                    break
        if sid is None and name:
            r = requests.get(f"{TVDB_BASE}/search", headers=h,
                             params={"query": name, "type": "series"}, timeout=15)
            data = r.json().get("data", [])
            if data:
                sid = str(data[0].get("tvdb_id") or data[0].get("id", "")).lstrip("series-")
        if not sid:
            return {}
        r = requests.get(f"{TVDB_BASE}/series/{sid}/extended", headers=h, timeout=15)
        d = r.json().get("data", {})
        companies = [c.get("name") for c in d.get("companies", []) if c.get("name")]
        return {
            "overview": d.get("overview"),
            "runtime": d.get("averageRuntime"),
            "production_companies": companies or None,
        }
    except Exception:  # noqa: BLE001
        return {}


def clean_val(v):
    if isinstance(v, list):
        return v if v else None
    return v


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    fillable_cols = [c for c in FILLABLE if c in df.columns]
    need = df[fillable_cols].isna().any(axis=1)
    rows = df.index[need].tolist()
    print(f"TV: {len(df)} rows | {len(rows)} rows have >=1 fillable null | "
          f"{int(df[fillable_cols].isna().sum().sum())} null cells")
    print("Per-col nulls:", {c: int(df[c].isna().sum()) for c in fillable_cols if df[c].isna().sum()})

    if args.plan:
        print("\nPLAN ONLY.")
        return

    token = tvdb_login()
    print(f"TVDB token: {'OK' if token else 'UNAVAILABLE'}")

    filled = 0
    for n, idx in enumerate(rows, 1):
        row = df.loc[idx]
        name = row.get("name")
        tmdb_id = row.get("tmdb_id")
        try:
            meta = get_tv_show_metadata_ultimate(
                str(name), row.get("year"), tmdb_id=tmdb_id,
                tvmaze_id=row.get("tvmaze_id"), imdb_id=row.get("imdb_id"))
        except Exception as e:  # noqa: BLE001
            print(f"  [{n}] ERR {name}: {e}")
            meta = {}

        # TVDB fallback only for cols still null after the primary fetch
        still_null = [c for c in fillable_cols
                      if pd.isna(df.at[idx, c]) and is_effectively_empty(meta.get(FILLABLE[c]))]
        if still_null and token:
            tv = tvdb_extended(token, imdb_id=meta.get("imdb_id") or row.get("imdb_id"), name=str(name))
            for c in ("overview", "runtime", "production_companies"):
                if c in still_null and not is_effectively_empty(tv.get(c)):
                    meta[FILLABLE[c]] = tv[c]

        for col, key in FILLABLE.items():
            if col in df.columns and pd.isna(df.at[idx, col]):
                val = clean_val(meta.get(key))
                if not is_effectively_empty(val):
                    df.at[idx, col] = val
                    filled += 1
        time.sleep(0.1)

    df.to_csv(config.TV_SHOWS_ENRICHED_DATA_PATH, index=False)
    print(f"\nDONE. Filled {filled} null cells across {len(rows)} rows.")


if __name__ == "__main__":
    main()
