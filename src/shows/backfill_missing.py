"""
Backfill missing fields in the TV shows enriched dataset.

Goes through every row that still has any empty data field and tries to fill it
using ALL available APIs (TMDB -> TVMaze -> OMDb), reusing the exact aggregation
logic in ingestion.get_tv_show_metadata_ultimate.

Key differences vs the main ingestion run:
  * It only ever FILLS EMPTY CELLS - existing (correct) values are never touched.
  * It robustly resolves a TMDB anchor for rows that have no tmdb_id, by trying
    (1) the existing tmdb_id, (2) TMDB /find via imdb_id, (3) tvmaze -> imdb ->
    /find, and (4) a title search that also tries de-camelCased / cleaned title
    variants (so e.g. "WonderMan" resolves to "Wonder Man").
  * It retries transient fetch failures.

Usage:
    python src/shows/backfill_missing.py            # backfill all rows with gaps
    python src/shows/backfill_missing.py --dry-run  # report only, write nothing
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import (
    TMDB_KEY,
    clean_title,
    clean_year,
    get_tv_show_metadata_ultimate,
    is_effectively_empty,
)

TMDB_BASE = "https://api.themoviedb.org/3"

# Columns that are user-provided or derived - never fetched / never count as gaps.
NON_FETCH_COLS = {"user_rating", "watch_count", "year_watched", "processing_status"}


def _despace_camel(title):
    """'WonderMan' -> 'Wonder Man', 'X-MenTAS' left mostly alone."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(title))


def tmdb_find_by_imdb(imdb_id):
    if not imdb_id or is_effectively_empty(imdb_id):
        return None
    try:
        r = requests.get(
            f"{TMDB_BASE}/find/{imdb_id}",
            params={"api_key": TMDB_KEY, "external_source": "imdb_id"},
            timeout=8,
        )
        if r.status_code == 200:
            res = r.json().get("tv_results", [])
            if res:
                return res[0]["id"]
    except Exception:
        pass
    return None


def tvmaze_to_imdb(tvmaze_id):
    if not tvmaze_id or is_effectively_empty(tvmaze_id):
        return None
    try:
        r = requests.get(
            f"https://api.tvmaze.com/shows/{int(float(tvmaze_id))}", timeout=8
        )
        if r.status_code == 200:
            return (r.json().get("externals") or {}).get("imdb")
    except Exception:
        pass
    return None


def tvmaze_average_runtime(tvmaze_id):
    """TVMaze exposes both `runtime` (often null) and `averageRuntime` - the main
    ingestion only reads `runtime`, so this recovers the common averageRuntime case."""
    if not tvmaze_id or is_effectively_empty(tvmaze_id):
        return None
    try:
        r = requests.get(
            f"https://api.tvmaze.com/shows/{int(float(tvmaze_id))}", timeout=8
        )
        if r.status_code == 200:
            j = r.json()
            return j.get("runtime") or j.get("averageRuntime")
    except Exception:
        pass
    return None


def tmdb_search_id(title, year):
    """Title search across a few title variants; prefer a year match, else popularity."""
    variants = []
    for v in (str(title), _despace_camel(title), clean_title(title), clean_title(_despace_camel(title))):
        v = v.strip()
        if v and v not in variants:
            variants.append(v)
    best_any = None
    for v in variants:
        try:
            r = requests.get(
                f"{TMDB_BASE}/search/tv",
                params={"api_key": TMDB_KEY, "query": v},
                timeout=8,
            )
            if r.status_code != 200:
                continue
            results = r.json().get("results", [])
            if not results:
                continue
            if best_any is None:
                best_any = results[0]["id"]
            if year:
                for res in results:
                    fad = res.get("first_air_date") or ""
                    if not fad:
                        continue
                    try:
                        ry = int(fad.split("-")[0])
                    except Exception:
                        continue
                    if abs(ry - year) <= 1:
                        return res["id"]
        except Exception:
            continue
    return best_any


def resolve_tmdb_id(row, year):
    """Best-effort TMDB id for a row, trying every anchor in priority order."""
    if not is_effectively_empty(row.get("tmdb_id")):
        return int(float(row["tmdb_id"]))
    tid = tmdb_find_by_imdb(row.get("imdb_id"))
    if tid:
        return tid
    imdb_from_tvmaze = tvmaze_to_imdb(row.get("tvmaze_id"))
    if imdb_from_tvmaze:
        tid = tmdb_find_by_imdb(imdb_from_tvmaze)
        if tid:
            return tid
    return tmdb_search_id(row.get("name"), year)


def fetch_with_retry(name, year, tmdb_id, tvmaze_id, imdb_id, attempts=3):
    last = {}
    for i in range(attempts):
        data = get_tv_show_metadata_ultimate(
            name, year, tmdb_id=tmdb_id, tvmaze_id=tvmaze_id, imdb_id=imdb_id
        )
        last = data or {}
        # consider it good enough if we anchored to TMDB or got an imdb rating
        if not is_effectively_empty(last.get("tmdb_id")) or not is_effectively_empty(
            last.get("imdb_rating")
        ):
            return last
        time.sleep(1.5 * (i + 1))
    return last


def run_backfill(dry_run=False):
    path = config.TV_SHOWS_ENRICHED_DATA_PATH
    df = pd.read_csv(path)
    fill_cols = [c for c in df.columns if c not in NON_FETCH_COLS]

    total_filled = 0
    rows_touched = 0
    for idx, row in df.iterrows():
        d = row.to_dict()
        missing = [c for c in fill_cols if is_effectively_empty(d.get(c))]
        if not missing:
            continue

        name = str(d.get("name") or "").strip()
        year = clean_year(d.get("year")) or clean_year(d.get("year_watched"))
        tmdb_id = resolve_tmdb_id(d, year)
        tvmaze_id = None if is_effectively_empty(d.get("tvmaze_id")) else d.get("tvmaze_id")
        imdb_id = None if is_effectively_empty(d.get("imdb_id")) else d.get("imdb_id")

        print(f"[{idx}] {name!r}: {len(missing)} gaps -> anchor tmdb={tmdb_id}")
        fetched = fetch_with_retry(name, year, tmdb_id, tvmaze_id, imdb_id)

        # Runtime fallback: TVMaze averageRuntime (main fetch only uses `runtime`).
        if "runtime" in missing and is_effectively_empty(fetched.get("runtime")):
            avg = tvmaze_average_runtime(fetched.get("tvmaze_id") or tvmaze_id)
            if not is_effectively_empty(avg):
                fetched["runtime"] = avg

        filled_here = []
        for c in missing:
            v = fetched.get(c)
            if not is_effectively_empty(v):
                df.at[idx, c] = v
                filled_here.append(c)
        if filled_here:
            rows_touched += 1
            total_filled += len(filled_here)
            if not is_effectively_empty(df.at[idx, "tmdb_id"]) or not is_effectively_empty(
                df.at[idx, "imdb_id"]
            ):
                df.at[idx, "processing_status"] = "success"
            print(f"        filled {len(filled_here)}: {', '.join(filled_here)}")
        else:
            print("        (nothing available to fill)")

    print(
        f"\nSummary: filled {total_filled} cells across {rows_touched} rows."
    )
    if dry_run:
        print("DRY RUN - no file written.")
        return
    df = df.replace({r"\x00": ""}, regex=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_backfill(dry_run=args.dry_run)
