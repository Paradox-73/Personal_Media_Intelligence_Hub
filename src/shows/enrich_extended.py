"""Shows counterpart of src/movies/enrich_extended.py.

FILLS (null cells only): tagline, created_by, writer, awards, age_rating,
imdb_rating/votes, overview, runtime, country, language -- from TMDB + OMDb.

ADDS: keywords (TMDB /tv/{id}/keywords), networks, in_production, last_air_date,
episode_run_time, origin_country.

Usage:
    python -m src.shows.enrich_extended --plan
    python -m src.shows.enrich_extended --apply
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
from src.shows.ingestion import is_effectively_empty

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")
CACHE = config.CACHE_DIR / "tmdb_tv_extended.json"
TIMEOUT = 12

OMDB_FIELDS = {
    "awards": lambda r: r.get("Awards"),
    "age_rating": lambda r: r.get("Rated"),
    "imdb_rating": lambda r: r.get("imdbRating"),
    "imdb_votes": lambda r: r.get("imdbVotes"),
    "country": lambda r: r.get("Country"),
    "language": lambda r: r.get("Language"),
    "writer": lambda r: r.get("Writer"),
    "actors": lambda r: r.get("Actors"),
}
TMDB_FILL = {
    "tagline": "tagline", "overview": "overview", "runtime": "episode_run_time",
    "last_air_date": "last_air_date", "created_by": "created_by",
    "network": "network",
}
NEW_COLS = ["keywords", "in_production", "origin_country",
            # TVMaze
            "tvmaze_weight", "tvmaze_rating", "tvmaze_status", "web_channel",
            # TVDB v4
            "tvdb_id", "tvdb_score", "tvdb_tags", "tvdb_status",
            "tvdb_avg_runtime", "content_rating"]

TVDB_BASE = "https://api4.thetvdb.com/v4"
_TVDB_TOKEN = {"v": None}


def tvdb_token():
    """TVDB v4 exchanges the API key for a short-lived bearer token."""
    if _TVDB_TOKEN["v"]:
        return _TVDB_TOKEN["v"]
    key = os.getenv("TVDB_API_KEY")
    if not key:
        return None
    try:
        r = requests.post(f"{TVDB_BASE}/login", json={"apikey": key}, timeout=TIMEOUT)
        _TVDB_TOKEN["v"] = r.json().get("data", {}).get("token")
    except Exception:  # noqa: BLE001
        _TVDB_TOKEN["v"] = None
    return _TVDB_TOKEN["v"]


def tvdb_series(imdb_id, name, year, cache):
    """Resolve by IMDb remote id, fall back to a name search, then /extended.

    TVDB `tags` are thematic ("Workplace", "Boss/Employee") rather than the
    coarse genres TMDB returns -- closer to nanogenres than anything else we have.
    """
    ck = str(imdb_id or f"{name}|{year}")
    if ck in cache and "_error" not in cache[ck]:  # retry past failures
        return cache[ck]
    tok = tvdb_token()
    if not tok:
        return {}
    h = {"Authorization": f"Bearer {tok}"}
    rec, sid = {}, None
    try:
        if imdb_id and not is_effectively_empty(imdb_id):
            r = requests.get(f"{TVDB_BASE}/search/remoteid/{imdb_id}", headers=h,
                             timeout=TIMEOUT).json()
            for item in (r.get("data") or []):
                ser = item.get("series") or {}
                if ser.get("id"):
                    sid = ser["id"]
                    break
        if not sid and name:
            r = requests.get(f"{TVDB_BASE}/search", headers=h,
                             params={"query": str(name), "type": "series"},
                             timeout=TIMEOUT).json()
            hits = r.get("data") or []
            if hits:
                sid = hits[0].get("tvdb_id") or hits[0].get("id")
        if sid:
            e = requests.get(f"{TVDB_BASE}/series/{str(sid).lstrip('series-')}/extended",
                             headers=h, timeout=TIMEOUT).json().get("data") or {}
            ratings = {c.get("country"): c.get("name") for c in (e.get("contentRatings") or [])}
            rec = {
                "tvdb_id": e.get("id"),
                "tvdb_score": e.get("score"),
                "tvdb_status": (e.get("status") or {}).get("name"),
                "tvdb_avg_runtime": e.get("averageRuntime"),
                "tvdb_tags": ", ".join(t.get("name") for t in (e.get("tags") or [])
                                       if t.get("name")),
                "content_rating": ratings.get("usa") or next(iter(ratings.values()), None),
            }
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    cache[ck] = rec
    return rec


def tvmaze_show(imdb_id, name, cache):
    """TVMaze `weight` is its own popularity score (0-100) -- a crowd-attention
    signal none of the other APIs give us."""
    ck = str(imdb_id or name)
    if ck in cache and "_error" not in cache[ck]:  # retry past failures
        return cache[ck]
    rec = {}
    try:
        d = None
        if imdb_id and not is_effectively_empty(imdb_id):
            r = requests.get("https://api.tvmaze.com/lookup/shows",
                             params={"imdb": imdb_id}, timeout=TIMEOUT)
            if r.status_code == 200:
                d = r.json()
        if not d and name:
            r = requests.get("https://api.tvmaze.com/singlesearch/shows",
                             params={"q": str(name)}, timeout=TIMEOUT)
            if r.status_code == 200:
                d = r.json()
        if d:
            rec = {
                "tvmaze_weight": d.get("weight"),
                "tvmaze_rating": (d.get("rating") or {}).get("average"),
                "tvmaze_status": d.get("status"),
                "web_channel": ((d.get("webChannel") or {}) or {}).get("name"),
            }
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    cache[ck] = rec
    return rec


def _load_named(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _load():
    return _load_named(CACHE)


def tmdb_tv(tmdb_id, cache):
    k = str(int(float(tmdb_id)))
    if k in cache and "_error" not in cache[k]:  # retry past failures
        return cache[k]
    rec = {}
    try:
        d = requests.get(f"https://api.themoviedb.org/3/tv/{k}",
                         params={"api_key": TMDB_KEY}, timeout=TIMEOUT).json()
        if "id" in d:
            ert = d.get("episode_run_time") or []
            rec = {
                "tagline": d.get("tagline"), "overview": d.get("overview"),
                "episode_run_time": ert[0] if ert else None,
                "last_air_date": d.get("last_air_date"),
                "in_production": d.get("in_production"),
                "origin_country": ", ".join(d.get("origin_country") or []),
                "created_by": ", ".join(c["name"] for c in (d.get("created_by") or [])),
                "network": ", ".join(n["name"] for n in (d.get("networks") or [])),
            }
        kw = requests.get(f"https://api.themoviedb.org/3/tv/{k}/keywords",
                          params={"api_key": TMDB_KEY}, timeout=TIMEOUT).json()
        rec["keywords"] = [w["name"] for w in (kw.get("results") or kw.get("keywords") or [])]
    except Exception as e:  # noqa: BLE001
        return {"_error": str(e)}
    cache[k] = rec
    return rec


def omdb_tv(imdb_id, title, year):
    try:
        if imdb_id and not is_effectively_empty(imdb_id):
            r = requests.get("http://www.omdbapi.com/",
                             params={"i": imdb_id, "apikey": OMDB_KEY}, timeout=TIMEOUT).json()
            if r.get("Response") == "True":
                return r
        if title:
            r = requests.get("http://www.omdbapi.com/",
                             params={"t": str(title), "type": "series", "apikey": OMDB_KEY},
                             timeout=TIMEOUT).json()
            if r.get("Response") == "True":
                return r
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    path = config.TV_SHOWS_ENRICHED_DATA_PATH
    df = pd.read_csv(path)
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    def missing(col):
        return df[col].apply(is_effectively_empty) if col in df.columns \
            else pd.Series(True, index=df.index)

    targets = list(OMDB_FIELDS) + list(TMDB_FILL) + NEW_COLS
    print(f"{len(df)} shows. Fillable gaps:")
    for c in sorted(set(targets)):
        n = int(missing(c).sum())
        if n:
            print(f"   {c:24} {n:4} missing")
    need = df.index[pd.concat([missing(c) for c in set(targets)], axis=1).any(axis=1)]
    print(f"\n{len(need)} rows need at least one fetch.")
    if args.plan:
        print("PLAN ONLY.")
        return

    cache = _load()
    tvdb_cache = _load_named(config.CACHE_DIR / "tvdb_series.json")
    maze_cache = _load_named(config.CACHE_DIR / "tvmaze_shows.json")
    filled = {}
    for n, idx in enumerate(need, 1):
        row = df.loc[idx]
        if not is_effectively_empty(row.get("tmdb_id")):
            rec = tmdb_tv(row["tmdb_id"], cache)
            for col, key in TMDB_FILL.items():
                if col in df.columns and is_effectively_empty(df.at[idx, col]):
                    v = rec.get(key)
                    if not is_effectively_empty(v):
                        df.at[idx, col] = v
                        filled[col] = filled.get(col, 0) + 1
            for col in ("in_production", "origin_country"):
                if is_effectively_empty(df.at[idx, col]) and not is_effectively_empty(rec.get(col)):
                    df.at[idx, col] = rec.get(col)
                    filled[col] = filled.get(col, 0) + 1
            kws = rec.get("keywords")
            if kws and is_effectively_empty(df.at[idx, "keywords"]):
                df.at[idx, "keywords"] = ", ".join(kws)
                filled["keywords"] = filled.get("keywords", 0) + 1

        tvdb_cols = ["tvdb_id", "tvdb_score", "tvdb_tags", "tvdb_status",
                     "tvdb_avg_runtime", "content_rating"]
        if any(is_effectively_empty(df.at[idx, c]) for c in tvdb_cols):
            rec = tvdb_series(row.get("imdb_id"), row.get("name"), row.get("year"), tvdb_cache)
            for c in tvdb_cols:
                if is_effectively_empty(df.at[idx, c]) and not is_effectively_empty(rec.get(c)):
                    df.at[idx, c] = rec[c]
                    filled[c] = filled.get(c, 0) + 1

        maze_cols = ["tvmaze_weight", "tvmaze_rating", "tvmaze_status", "web_channel"]
        if any(is_effectively_empty(df.at[idx, c]) for c in maze_cols):
            rec = tvmaze_show(row.get("imdb_id"), row.get("name"), maze_cache)
            for c in maze_cols:
                if is_effectively_empty(df.at[idx, c]) and not is_effectively_empty(rec.get(c)):
                    df.at[idx, c] = rec[c]
                    filled[c] = filled.get(c, 0) + 1

        if any(is_effectively_empty(df.at[idx, c]) for c in OMDB_FIELDS if c in df.columns):
            res = omdb_tv(row.get("imdb_id"), row.get("name"), row.get("year"))
            if res:
                for col, get in OMDB_FIELDS.items():
                    if col in df.columns and is_effectively_empty(df.at[idx, col]):
                        v = get(res)
                        if not is_effectively_empty(v):
                            df.at[idx, col] = v
                            filled[col] = filled.get(col, 0) + 1
        if n % 25 == 0:
            df.to_csv(path, index=False)
            CACHE.write_text(json.dumps(cache), encoding="utf-8")
            (config.CACHE_DIR / "tvdb_series.json").write_text(json.dumps(tvdb_cache), encoding="utf-8")
            (config.CACHE_DIR / "tvmaze_shows.json").write_text(json.dumps(maze_cache), encoding="utf-8")
            print(f"   ...{n}/{len(need)}")
        time.sleep(0.05)

    df.to_csv(path, index=False)
    CACHE.write_text(json.dumps(cache), encoding="utf-8")
    print("\nFILLED:")
    for c, v in sorted(filled.items(), key=lambda kv: -kv[1]):
        print(f"   {c:24} +{v}")
    print(f"\nkeywords coverage now "
          f"{(~df['keywords'].apply(is_effectively_empty)).mean():.1%}")


if __name__ == "__main__":
    main()
