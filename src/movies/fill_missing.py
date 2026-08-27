"""
Fill ONLY null (NaN) cells in the enriched movies CSV by re-fetching TMDB/OMDB
metadata. Never overwrites an existing non-null value; only rows that have at
least one fillable null are re-fetched.

Usage:
    python src/movies/fill_missing.py --plan     # count rows/cells only
    python src/movies/fill_missing.py --apply
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.ingestion import get_movie_metadata, clean_year, is_effectively_empty

# CSV column  <-  key returned by get_movie_metadata()
FILLABLE = {
    "title": "title", "overview": "overview", "plot": "plot", "tagline": "tagline",
    "director": "director", "writer": "writer", "actors": "actors", "genre": "genre",
    "imdb_rating": "imdb_rating", "imdb_votes": "imdb_votes", "metascore": "metascore",
    "rotten_tomatoes_rating": "rotten_tomatoes_rating", "vote_average": "vote_average",
    "vote_count": "vote_count", "rated": "rated", "released": "released",
    "runtime": "runtime", "language": "language", "country": "country",
    "awards": "awards", "box_office": "box_office", "production": "production",
    "poster": "poster", "imdb_id": "imdb_id",
}
# Columns present in the CSV that this fetch canNOT source (reported, left as-is):
#   website, status, processing_status, type, popularity, backdrop_path


def clean_meta_value(v):
    if isinstance(v, list):
        return ", ".join(str(x) for x in v) if v else None
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
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    fillable_cols = [c for c in FILLABLE if c in df.columns]

    need = df[fillable_cols].isna().any(axis=1)
    rows_to_fetch = df.index[need].tolist()
    if args.limit:
        rows_to_fetch = rows_to_fetch[: args.limit]

    total_null_cells = int(df[fillable_cols].isna().sum().sum())
    print(f"Movies: {len(df)} rows | {len(rows_to_fetch)} rows have >=1 fillable null "
          f"| {total_null_cells} null cells across {len(fillable_cols)} fillable cols")
    unfillable = [c for c in ("website", "status", "processing_status", "type",
                              "popularity", "backdrop_path") if c in df.columns]
    print("Not API-sourceable here (left untouched):",
          {c: int(df[c].isna().sum()) for c in unfillable})

    if args.plan:
        print("\nPLAN ONLY.")
        return

    filled_cells = 0
    for n, idx in enumerate(rows_to_fetch, 1):
        row = df.loc[idx]
        title = row.get("title") if not is_effectively_empty(row.get("title")) else row.get("letterboxd_name")
        year = clean_year(row.get("year"))
        tmdb_id = row.get("tmdb_id")
        imdb_id = row.get("imdb_id")
        try:
            meta = get_movie_metadata(str(title), year, tmdb_id=tmdb_id, imdb_id=imdb_id)
        except Exception as e:  # noqa: BLE001
            print(f"  [{n}/{len(rows_to_fetch)}] ERR {title}: {e}")
            continue
        row_filled = 0
        for col, key in FILLABLE.items():
            if col not in df.columns:
                continue
            if pd.isna(df.at[idx, col]):  # only true nulls
                val = clean_meta_value(meta.get(key))
                if not is_effectively_empty(val):
                    df.at[idx, col] = val
                    row_filled += 1
        filled_cells += row_filled
        if n % 25 == 0:
            df.to_csv(config.MOVIES_ENRICHED_DATA_PATH, index=False)
            print(f"  ...{n}/{len(rows_to_fetch)} processed, {filled_cells} cells filled (checkpoint saved)")
        time.sleep(0.08)

    df.to_csv(config.MOVIES_ENRICHED_DATA_PATH, index=False)
    print(f"\nDONE. Filled {filled_cells} null cells across {len(rows_to_fetch)} rows.")


if __name__ == "__main__":
    main()
