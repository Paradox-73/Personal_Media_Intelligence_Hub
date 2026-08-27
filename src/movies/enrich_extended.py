"""Fill every gap the APIs we already pay for can fill, then add the fields the
pipeline never collected.

FILLS (null cells only, never overwrites):
  OMDb  -> metascore, rotten_tomatoes_rating, box_office, awards, rated,
           imdb_rating, imdb_votes, runtime, country, language
  TMDB  -> runtime, tagline, overview, poster, popularity, vote_average/count

ADDS (new columns):
  keywords            TMDB /movie/{id}/keywords -- the real "aboutness" signal,
                      as discrete labels instead of mean-pooled plot embeddings
  collection_name     TMDB belongs_to_collection -- franchise membership
  revenue, budget     TMDB financials (fills box office where OMDb has no value)
  original_language   TMDB

Usage:
    python -m src.movies.enrich_extended --plan
    python -m src.movies.enrich_extended --apply [--limit N]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.ingestion import (TMDB_KEY, OMDB_KEY, fetch_omdb_direct,
                                  is_effectively_empty, clean_int, clean_float)

CACHE = config.CACHE_DIR / "tmdb_extended.json"
TIMEOUT = 12

# CSV column -> how to pull it out of an OMDb response
OMDB_FIELDS = {
    "metascore": lambda r: r.get("Metascore"),
    "rotten_tomatoes_rating": lambda r: next(
        (x["Value"] for x in r.get("Ratings", []) if x["Source"] == "Rotten Tomatoes"), None),
    "box_office": lambda r: r.get("BoxOffice"),
    "awards": lambda r: r.get("Awards"),
    "rated": lambda r: r.get("Rated"),
    "imdb_rating": lambda r: r.get("imdbRating"),
    "imdb_votes": lambda r: r.get("imdbVotes"),
    "runtime": lambda r: r.get("Runtime"),
    "country": lambda r: r.get("Country"),
    "language": lambda r: r.get("Language"),
    "plot": lambda r: r.get("Plot"),
}
TMDB_FILL = {
    "runtime": "runtime", "tagline": "tagline", "overview": "overview",
    "popularity": "popularity", "vote_average": "vote_average",
    "vote_count": "vote_count", "poster": "poster_path",
}
NEW_COLS = ["keywords", "collection_name", "revenue", "budget", "original_language",
            "composer", "cinematographer", "editor", "producer"]

# TMDB crew job -> CSV column. These live in the /credits payload the pipeline
# already fetched and discarded, keeping only Director and the Writing dept.
CREW_JOBS = {
    "composer": ("Original Music Composer", "Music"),
    "cinematographer": ("Director of Photography",),
    "editor": ("Editor",),
    "producer": ("Producer",),
}


def _load_cache():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(c):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c), encoding="utf-8")


def tmdb_extended(tmdb_id, cache):
    """details + keywords + crew in one cached record.

    Cached records written before a field was added lack its key, so a plain
    `k in cache` hit would silently skip the new fetch forever. Re-fetch whenever
    the cached record predates the current field set.
    """
    k = str(int(float(tmdb_id)))
    cached = cache.get(k)
    if cached is not None and ("_error" in cached or all(c in cached for c in CREW_JOBS)):
        return cached
    rec = {}
    try:
        d = requests.get(f"https://api.themoviedb.org/3/movie/{k}",
                         params={"api_key": TMDB_KEY}, timeout=TIMEOUT).json()
        if "id" in d:
            coll = d.get("belongs_to_collection") or {}
            rec = {
                "runtime": d.get("runtime"), "tagline": d.get("tagline"),
                "overview": d.get("overview"), "popularity": d.get("popularity"),
                "vote_average": d.get("vote_average"), "vote_count": d.get("vote_count"),
                "poster_path": d.get("poster_path"), "revenue": d.get("revenue"),
                "budget": d.get("budget"),
                "original_language": d.get("original_language"),
                "collection_name": coll.get("name") if isinstance(coll, dict) else None,
            }
        kw = requests.get(f"https://api.themoviedb.org/3/movie/{k}/keywords",
                          params={"api_key": TMDB_KEY}, timeout=TIMEOUT).json()
        rec["keywords"] = [w["name"] for w in kw.get("keywords", [])]

        cr = requests.get(f"https://api.themoviedb.org/3/movie/{k}/credits",
                          params={"api_key": TMDB_KEY}, timeout=TIMEOUT).json()
        crew = cr.get("crew", [])
        for col, jobs in CREW_JOBS.items():
            names = []
            for c in crew:
                if c.get("job") in jobs and c.get("name") not in names:
                    names.append(c["name"])
            rec[col] = names[:3] if col == "producer" else names
    except Exception as e:  # noqa: BLE001
        rec = {"_error": str(e)}
    cache[k] = rec
    return rec


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    path = config.MOVIES_ENRICHED_DATA_PATH
    df = pd.read_csv(path)
    for c in NEW_COLS:
        if c not in df.columns:
            # object dtype: these hold comma-joined names/keywords. Creating them
            # as pd.NA makes a float64 column, and writing a string into it warns
            # (and will raise in a future pandas).
            df[c] = pd.Series([pd.NA] * len(df), dtype="object")

    def missing(col):
        return df[col].apply(is_effectively_empty) if col in df.columns else pd.Series(True, index=df.index)

    gaps = {c: int(missing(c).sum()) for c in list(OMDB_FIELDS) + NEW_COLS}
    print(f"{len(df)} movies. Fillable gaps:")
    for c, n in sorted(gaps.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"   {c:26} {n:5} missing")

    need = df.index[
        pd.concat([missing(c) for c in list(OMDB_FIELDS) + NEW_COLS], axis=1).any(axis=1)]
    print(f"\n{len(need)} rows need at least one fetch.")
    if args.plan:
        print("PLAN ONLY.")
        return

    if args.limit:
        need = need[: args.limit]
    cache = _load_cache()
    filled = {c: 0 for c in list(OMDB_FIELDS) + NEW_COLS}

    for n, idx in enumerate(need, 1):
        row = df.loc[idx]

        if not is_effectively_empty(row.get("tmdb_id")):
            rec = tmdb_extended(row["tmdb_id"], cache)
            for col, key in TMDB_FILL.items():
                if col in df.columns and is_effectively_empty(df.at[idx, col]):
                    v = rec.get(key)
                    if not is_effectively_empty(v):
                        df.at[idx, col] = v
                        filled[col] = filled.get(col, 0) + 1
            for col, key in (("revenue", "revenue"), ("budget", "budget"),
                             ("collection_name", "collection_name"),
                             ("original_language", "original_language")):
                v = rec.get(key)
                if is_effectively_empty(df.at[idx, col]) and not is_effectively_empty(v):
                    df.at[idx, col] = v
                    filled[col] += 1
            kws = rec.get("keywords")
            if kws and is_effectively_empty(df.at[idx, "keywords"]):
                df.at[idx, "keywords"] = ", ".join(kws)
                filled["keywords"] += 1
            for col in CREW_JOBS:
                names = rec.get(col)
                if names and is_effectively_empty(df.at[idx, col]):
                    df.at[idx, col] = ", ".join(names)
                    filled[col] += 1

        if any(is_effectively_empty(df.at[idx, c]) for c in OMDB_FIELDS if c in df.columns):
            res = fetch_omdb_direct(imdb_id=row.get("imdb_id"),
                                    title=row.get("title"), year=row.get("year"))
            if res:
                for col, get in OMDB_FIELDS.items():
                    if col in df.columns and is_effectively_empty(df.at[idx, col]):
                        v = get(res)
                        if not is_effectively_empty(v):
                            df.at[idx, col] = v
                            filled[col] += 1

        if n % 50 == 0:
            df.to_csv(path, index=False)
            _save_cache(cache)
            print(f"   ...{n}/{len(need)} rows processed")
        time.sleep(0.05)

    df.to_csv(path, index=False)
    _save_cache(cache)
    print("\nFILLED:")
    for c, n in sorted(filled.items(), key=lambda kv: -kv[1]):
        if n:
            print(f"   {c:26} +{n}")
    kw_cov = (~df["keywords"].apply(is_effectively_empty)).mean()
    print(f"\nkeywords coverage now {kw_cov:.1%}; "
          f"collection {(~df['collection_name'].apply(is_effectively_empty)).mean():.1%}")


if __name__ == "__main__":
    main()
