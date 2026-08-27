"""Fetch the TMDB item-item neighbour graph for movies and TV.

Why this exists
---------------
Every feature in the rating models describes the *item* (crowd score, runtime,
genre, who made it). Nothing describes how an item sits relative to the rest of
the library. TMDB already publishes two neighbour lists per title:

    /movie/{id}/similar          - keyword/genre overlap, editorial-ish
    /movie/{id}/recommendations  - behavioural, from TMDB user co-rating

The second is the interesting one: it is a collaborative signal, the only one
available anywhere in this project. Turning it into "my mean rating over the
neighbours I have already rated" gives the model a genuine item-item feature
instead of another restatement of how good the crowd thinks the film is.

This module only FETCHES and caches the neighbour ids. Turning them into
leakage-safe features happens inside feature engineering, because the neighbour
means must be rebuilt per CV fold from training-fold ratings only.

Cache: data/cache/tmdb_neighbors.json  {"movie:1234": {"similar": [...],
                                        "recommendations": [...]}}
Only ids missing from the cache are fetched, so re-runs are nearly free.

Run:  python -m src.common.tmdb_neighbors            # movies + tv
      python -m src.common.tmdb_neighbors --kind tv
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
CACHE = config.CACHE_DIR / "tmdb_neighbors.json"
BASE = "https://api.themoviedb.org/3"
ENDPOINTS = ("similar", "recommendations")


def _load():
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}
    return {}


def _save(cache):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")


def _get(kind, tmdb_id, endpoint, session, retries=3):
    """One page (20 neighbours) is plenty -- TMDB orders them by relevance."""
    url = f"{BASE}/{kind}/{int(tmdb_id)}/{endpoint}"
    for attempt in range(retries):
        try:
            r = session.get(url, params={"api_key": TMDB_KEY, "page": 1}, timeout=15)
            if r.status_code == 429:  # rate limited -- honour the header
                time.sleep(float(r.headers.get("Retry-After", 2)) + 0.5)
                continue
            if r.status_code == 404:
                return []
            r.raise_for_status()
            return [int(x["id"]) for x in r.json().get("results", []) if x.get("id")]
        except requests.RequestException:
            if attempt == retries - 1:
                return None  # distinguish "failed" from "genuinely empty"
            time.sleep(1.5 * (attempt + 1))
    return None


def fetch_neighbors(kind, tmdb_ids, cache=None, save_every=50):
    """Populate the cache for `tmdb_ids`. Returns the cache dict."""
    if not TMDB_KEY:
        print("❌ TMDB_API_KEY missing -- cannot fetch neighbours.")
        return cache or {}
    cache = _load() if cache is None else cache
    ids = [int(i) for i in pd.Series(tmdb_ids).dropna().unique()]
    todo = [i for i in ids if f"{kind}:{i}" not in cache]
    print(f"   [{kind}] {len(ids)} ids, {len(todo)} to fetch "
          f"({len(ids) - len(todo)} cached)")
    session = requests.Session()
    failed = 0
    for n, tid in enumerate(todo, 1):
        rec = {}
        for ep in ENDPOINTS:
            got = _get(kind, tid, ep, session)
            if got is None:
                failed += 1
                got = []
            rec[ep] = got
        cache[f"{kind}:{tid}"] = rec
        if n % save_every == 0:
            _save(cache)
            print(f"      {n}/{len(todo)} fetched", flush=True)
        time.sleep(0.05)  # ~20 req/s, well inside TMDB's limit
    _save(cache)
    if failed:
        print(f"   ⚠️ {failed} endpoint call(s) failed and were stored as empty.")
    return cache


def _report(kind, cache, ids):
    """Coverage summary -- how many titles actually got neighbours."""
    have = [cache.get(f"{kind}:{int(i)}", {}) for i in ids if pd.notna(i)]
    n = len(have)
    if not n:
        return
    for ep in ENDPOINTS:
        nz = sum(1 for h in have if h.get(ep))
        sz = sum(len(h.get(ep, [])) for h in have) / max(nz, 1)
        print(f"   [{kind}] {ep:16} non-empty {nz}/{n} ({nz / n * 100:.0f}%), "
              f"mean {sz:.1f} neighbours")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["movie", "tv", "both"], default="both")
    args = ap.parse_args()

    cache = _load()
    if args.kind in ("movie", "both"):
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        ids = pd.to_numeric(df["tmdb_id"], errors="coerce").dropna()
        cache = fetch_neighbors("movie", ids, cache)
        _report("movie", cache, ids)
    if args.kind in ("tv", "both"):
        df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        ids = pd.to_numeric(df["tmdb_id"], errors="coerce").dropna()
        cache = fetch_neighbors("tv", ids, cache)
        _report("tv", cache, ids)
    # Plain ASCII: this module is run directly as well as through run_pipeline.py
    # (which sets PYTHONUTF8=1), and a bare console on cp1252 raised
    # UnicodeEncodeError on the emoji *after* the cache had already been written.
    print(f"[ok] neighbour cache -> {CACHE}")


if __name__ == "__main__":
    main()
