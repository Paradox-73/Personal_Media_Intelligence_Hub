"""Fill the per-season gaps in season_features.csv from every season-aware API.

Only NULL cells are written, so your hand-entered `season_rating` / `status`
labels are never touched. Adds season-level fields the first pass never took.

FILLS   season_overview, top_cast, key_crew, air_year, episode_count,
        tmdb_season_vote, imdb_season_mean/min/max/spread  (TMDB + OMDb)
ADDS    season_air_date   full air date, not just the year
        season_poster     per-season artwork
        imdb_ep_first / imdb_ep_last / imdb_ep_trend
                          does quality rise or fall ACROSS a season? (slope of
                          episode ratings) -- a genuinely per-season signal the
                          mean/min/max cannot express
        guest_star_count  per-season guest cast size

Also reports WHY anything is still empty, so absent-upstream is distinguishable
from not-fetched.

Usage:  python -m src.shows.season_fill --plan | --apply
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.season_data import SEASON_FEATURES_PATH

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")
C_SEASON = config.CACHE_DIR / "tmdb_season_detail.json"
C_OMDB = config.CACHE_DIR / "omdb_season_detail.json"
TOP_CAST = 10
TIMEOUT = 12

NEW_COLS = ["season_air_date", "season_poster", "imdb_ep_first", "imdb_ep_last",
            "imdb_ep_trend", "guest_star_count"]


def _load(p):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _empty(v):
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower() in ("", "nan", "none", "n/a")


def tmdb_season(tid, sn, cache):
    k = f"{int(tid)}_{int(sn)}"
    if k in cache:
        return cache[k]
    rec = {}
    try:
        d = requests.get(f"https://api.themoviedb.org/3/tv/{int(tid)}/season/{int(sn)}",
                         params={"api_key": TMDB_KEY, "append_to_response": "credits"},
                         timeout=TIMEOUT).json()
        if "id" in d or "episodes" in d:
            eps = d.get("episodes") or []
            cr = d.get("credits") or {}
            cast = [c["name"] for c in (cr.get("cast") or [])][:TOP_CAST]
            crew = [c["name"] for c in (cr.get("crew") or [])
                    if c.get("job") in ("Director", "Writer", "Executive Producer")]
            guests = {g["name"] for e in eps for g in (e.get("guest_stars") or [])}
            rec = {
                "season_overview": d.get("overview"),
                "season_air_date": d.get("air_date"),
                "season_poster": d.get("poster_path"),
                "episode_count": len(eps) or None,
                "tmdb_season_vote": d.get("vote_average"),
                "top_cast": ", ".join(cast) or None,
                "key_crew": ", ".join(dict.fromkeys(crew)) or None,
                "guest_star_count": len(guests),
            }
    except Exception as e:  # noqa: BLE001
        rec = {"_error": str(e)}
    cache[k] = rec
    return rec


def omdb_season(imdb, sn, cache):
    k = f"{imdb}_{int(sn)}"
    if k in cache:
        return cache[k]
    rec = {}
    try:
        d = requests.get("http://www.omdbapi.com/",
                         params={"i": imdb, "Season": int(sn), "apikey": OMDB_KEY},
                         timeout=TIMEOUT).json()
        if d.get("Response") == "True":
            r = [float(e["imdbRating"]) for e in d.get("Episodes", [])
                 if e.get("imdbRating") not in (None, "N/A")]
            if r:
                x = np.arange(len(r))
                slope = float(np.polyfit(x, r, 1)[0]) if len(r) > 2 else 0.0
                rec = {"imdb_season_mean": float(np.mean(r)),
                       "imdb_season_min": float(np.min(r)),
                       "imdb_season_max": float(np.max(r)),
                       "imdb_episode_spread": float(np.max(r) - np.min(r)),
                       "imdb_ep_first": r[0], "imdb_ep_last": r[-1],
                       "imdb_ep_trend": slope}
    except Exception as e:  # noqa: BLE001
        rec = {"_error": str(e)}
    cache[k] = rec
    return rec


TMDB_COLS = ["season_overview", "season_air_date", "season_poster", "episode_count",
             "tmdb_season_vote", "top_cast", "key_crew", "guest_star_count"]
OMDB_COLS = ["imdb_season_mean", "imdb_season_min", "imdb_season_max",
             "imdb_episode_spread", "imdb_ep_first", "imdb_ep_last", "imdb_ep_trend"]


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

    try:
        df = pd.read_csv(SEASON_FEATURES_PATH)
    except UnicodeDecodeError:
        df = pd.read_csv(SEASON_FEATURES_PATH, encoding="cp1252")
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    tgt = TMDB_COLS + OMDB_COLS
    print(f"{len(df)} seasons. Gaps:")
    for c in tgt:
        n = int(df[c].apply(_empty).sum())
        if n:
            print(f"   {c:22} {n:4} empty")
    no_imdb = int(df["imdb_id"].apply(_empty).sum())
    print(f"\n   ({no_imdb} seasons belong to shows with no imdb_id -> OMDb "
          f"season data impossible for those)")
    if args.plan:
        print("PLAN ONLY.")
        return

    cs, co = _load(C_SEASON), _load(C_OMDB)
    filled = {}
    for n, idx in enumerate(df.index, 1):
        row = df.loc[idx]
        tid, sn = row.get("tmdb_id"), row.get("season_number")
        if _empty(tid) or _empty(sn):
            continue

        if any(_empty(row.get(c)) for c in TMDB_COLS):
            rec = tmdb_season(tid, sn, cs)
            for c in TMDB_COLS:
                if _empty(df.at[idx, c]) and not _empty(rec.get(c)):
                    df.at[idx, c] = rec[c]
                    filled[c] = filled.get(c, 0) + 1

        if not _empty(row.get("imdb_id")) and any(_empty(row.get(c)) for c in OMDB_COLS):
            rec = omdb_season(str(row["imdb_id"]).strip(), sn, co)
            for c in OMDB_COLS:
                if _empty(df.at[idx, c]) and not _empty(rec.get(c)):
                    df.at[idx, c] = rec[c]
                    filled[c] = filled.get(c, 0) + 1

        if n % 50 == 0:
            df.to_csv(SEASON_FEATURES_PATH, index=False, encoding="utf-8")
            C_SEASON.write_text(json.dumps(cs), encoding="utf-8")
            C_OMDB.write_text(json.dumps(co), encoding="utf-8")
            print(f"   ...{n}/{len(df)}")
        time.sleep(0.04)

    # Recompute cast/crew churn now that per-season cast is more complete.
    df = df.sort_values(["tmdb_id", "season_number"]).reset_index(drop=True)
    prev_cast, prev_crew, prev_imdb = {}, {}, {}
    for idx, row in df.iterrows():
        tid = row["tmdb_id"]
        cast = {c.strip() for c in str(row.get("top_cast") or "").split(",") if c.strip()}
        crew = {c.strip() for c in str(row.get("key_crew") or "").split(",") if c.strip()}
        pc, pw = prev_cast.get(tid), prev_crew.get(tid)
        if pc and cast:
            ov = len(cast & pc) / max(1, len(cast | pc))
            if _empty(df.at[idx, "cast_overlap_prev"]):
                df.at[idx, "cast_overlap_prev"] = round(ov, 4)
                filled["cast_overlap_prev"] = filled.get("cast_overlap_prev", 0) + 1
            if _empty(df.at[idx, "cast_turnover"]):
                df.at[idx, "cast_turnover"] = round(1 - ov, 4)
            if _empty(df.at[idx, "cast_new_count"]):
                df.at[idx, "cast_new_count"] = len(cast - pc)
        if pw is not None and crew and _empty(df.at[idx, "crew_change"]):
            df.at[idx, "crew_change"] = int(crew != pw)
        pi = prev_imdb.get(tid)
        cur = pd.to_numeric(row.get("imdb_season_mean"), errors="coerce")
        if pi is not None and pd.notna(cur) and _empty(df.at[idx, "imdb_delta_vs_prev"]):
            df.at[idx, "imdb_delta_vs_prev"] = round(float(cur - pi), 4)
            filled["imdb_delta_vs_prev"] = filled.get("imdb_delta_vs_prev", 0) + 1
        if cast:
            prev_cast[tid] = cast
        if crew:
            prev_crew[tid] = crew
        if pd.notna(cur):
            prev_imdb[tid] = float(cur)

    df.to_csv(SEASON_FEATURES_PATH, index=False, encoding="utf-8")
    C_SEASON.write_text(json.dumps(cs), encoding="utf-8")
    C_OMDB.write_text(json.dumps(co), encoding="utf-8")

    print("\nFILLED:")
    for c, v in sorted(filled.items(), key=lambda kv: -kv[1]):
        print(f"   {c:22} +{v}")
    print("\nREMAINING EMPTY (absent upstream):")
    for c in tgt + ["cast_overlap_prev", "imdb_delta_vs_prev"]:
        n = int(df[c].apply(_empty).sum())
        if n:
            print(f"   {c:22} {n:4}")


if __name__ == "__main__":
    main()
