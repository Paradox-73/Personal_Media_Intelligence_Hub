"""
Season-level ingestion — multi-API, whole-feature-space collection scaffold.

71% of rated shows are multi-season and ratings swing season-to-season, so the single
show rating is a noisy average. This builds the per-season FEATURE matrix from every
season-aware API we have, and emits a fill-in sheet with a blank `season_rating` you log.

Per season it pulls / derives:
  TMDB  /tv/{id}                       -> season air year, episode count, overview,
                                          per-season vote_average, poster
  TMDB  /tv/{id}/season/{n}/credits    -> per-season top cast + crew (director/writer)
  OMDb  ?i={imdb}&Season={n}           -> per-season IMDb: episode-rating mean/min/max/spread
  derived (cast CHANGES season->season):
        cast_overlap_prev  (Jaccard of top cast vs previous season)
        cast_turnover      (1 - overlap; the "new cast" signal you mentioned)
        cast_new_count, crew_change (director/writer set changed vs prev season)
        rating_delta_vs_prev (per-season IMDb change — did quality move?)

Everything is cached (resumable); re-runs preserve any `season_rating` you've filled in.
Output: data/processed/shows/season_features.csv  (+ per-API JSON caches)
Run:    python -m src.shows.season_ingestion
"""
import os
import sys
import json
import time
from pathlib import Path

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

load_dotenv(dotenv_path=str(config.BASE_DIR / ".env"))
TMDB_KEY = os.getenv("TMDB_API_KEY")
OMDB_KEY = os.getenv("OMDB_API_KEY")
CDIR = getattr(config, "CACHE_DIR", config.TV_SHOWS_PROCESSED_DIR)
C_SHOW = CDIR / "tmdb_seasons.json"
C_CRED = CDIR / "tmdb_season_credits.json"
C_OMDB = CDIR / "omdb_seasons.json"
OUT = config.TV_SHOWS_PROCESSED_DIR / "season_features.csv"
TOP_CAST = 10


def _load(p):
    return json.loads(p.read_text()) if p.exists() else {}


def _save(p, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


def _get(url, params, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"   ⚠️ {url.split('/3/')[-1] if '/3/' in url else url}: {e}")
        return None


def fetch_show(tmdb_id, cache):
    k = str(int(tmdb_id))
    if k not in cache:
        data = _get(f"https://api.themoviedb.org/3/tv/{k}", {"api_key": TMDB_KEY}) or {}
        cache[k] = [{"season_number": s.get("season_number"), "name": s.get("name"),
                     "overview": s.get("overview"), "air_date": s.get("air_date"),
                     "episode_count": s.get("episode_count"), "vote_average": s.get("vote_average")}
                    for s in data.get("seasons", [])]
    return cache[k]


def fetch_credits(tmdb_id, season_n, cache):
    k = f"{int(tmdb_id)}_{season_n}"
    if k not in cache:
        d = _get(f"https://api.themoviedb.org/3/tv/{int(tmdb_id)}/season/{season_n}/credits",
                 {"api_key": TMDB_KEY}) or {}
        cast = [c.get("name") for c in (d.get("cast") or [])[:TOP_CAST] if c.get("name")]
        crew = [c.get("name") for c in (d.get("crew") or [])
                if c.get("job") in ("Director", "Writer", "Executive Producer") and c.get("name")]
        cache[k] = {"cast": cast, "crew": sorted(set(crew))[:10]}
    return cache[k]


def fetch_omdb_season(imdb_id, season_n, cache):
    if not (OMDB_KEY and isinstance(imdb_id, str) and imdb_id.startswith("tt")):
        return {}
    k = f"{imdb_id}_{season_n}"
    if k not in cache:
        d = _get("http://www.omdbapi.com/", {"apikey": OMDB_KEY, "i": imdb_id, "Season": season_n})
        eps = (d or {}).get("Episodes", []) if (d or {}).get("Response") == "True" else []
        rs = [float(e["imdbRating"]) for e in eps
              if e.get("imdbRating") not in (None, "N/A") and _isnum(e.get("imdbRating"))]
        cache[k] = {"imdb_mean": round(float(np.mean(rs)), 2) if rs else None,
                    "imdb_min": min(rs) if rs else None, "imdb_max": max(rs) if rs else None,
                    "imdb_n": len(rs)}
    return cache[k]


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return None
    return round(len(a & b) / max(len(a | b), 1), 3)


def main():
    if not TMDB_KEY:
        print("❌ TMDB_API_KEY missing in .env"); return
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    df = df[df["user_rating"].notna() & df["tmdb_id"].notna()].copy()
    print(f"📺 Multi-API season ingestion for {len(df)} rated shows "
          f"(TMDB seasons+credits, OMDb IMDb, cast-change derivation)...")

    c_show, c_cred, c_omdb = _load(C_SHOW), _load(C_CRED), _load(C_OMDB)
    # Preserve anything you've hand-entered: both the numeric season_rating AND the
    # status code (see legend below). These are your labels; never clobber them on re-run.
    prior, prior_status = {}, {}
    if OUT.exists():
        old = pd.read_csv(OUT)
        for _, r in old.iterrows():
            k = (int(r["tmdb_id"]), int(r["season_number"]))
            if "season_rating" in old.columns and pd.notna(r.get("season_rating")) and str(r.get("season_rating")).strip():
                prior[k] = r["season_rating"]
            if "status" in old.columns and pd.notna(r.get("status")) and str(r.get("status")).strip():
                prior_status[k] = str(r.get("status")).strip()

    rows = []
    for i, (_, show) in enumerate(df.iterrows(), 1):
        tid = int(show["tmdb_id"])
        imdb_id = show.get("imdb_id") if isinstance(show.get("imdb_id"), str) else None
        seasons = sorted([s for s in fetch_show(tid, c_show)
                          if s.get("season_number") and s["season_number"] >= 1],
                         key=lambda s: s["season_number"])
        prev_cast, prev_crew, prev_imdb = None, None, None
        for s in seasons:
            sn = s["season_number"]
            cred = fetch_credits(tid, sn, c_cred)
            omdb = fetch_omdb_season(imdb_id, sn, c_omdb)
            air = str(s.get("air_date") or "")
            year = int(air[:4]) if air[:4].isdigit() else None
            overlap = _jaccard(cred["cast"], prev_cast) if prev_cast is not None else None
            crew_change = (None if prev_crew is None
                           else int(set(cred["crew"]) != set(prev_crew)))
            imdb_mean = omdb.get("imdb_mean")
            delta = (round(imdb_mean - prev_imdb, 2)
                     if (imdb_mean is not None and prev_imdb is not None) else None)
            rows.append({
                "tmdb_id": tid, "imdb_id": imdb_id or "",
                "show_name": show.get("name"), "show_user_rating": show.get("user_rating"),
                "season_number": sn, "season_name": s.get("name"), "air_year": year,
                "episode_count": s.get("episode_count"),
                "tmdb_season_vote": s.get("vote_average"),
                "imdb_season_mean": imdb_mean, "imdb_season_min": omdb.get("imdb_min"),
                "imdb_season_max": omdb.get("imdb_max"),
                "imdb_episode_spread": (round(omdb["imdb_max"] - omdb["imdb_min"], 2)
                                        if omdb.get("imdb_max") is not None else None),
                "imdb_delta_vs_prev": delta,
                "cast_overlap_prev": overlap,
                "cast_turnover": (round(1 - overlap, 3) if overlap is not None else None),
                "cast_new_count": (None if prev_cast is None
                                   else len(set(cred["cast"]) - set(prev_cast))),
                "crew_change": crew_change,
                "top_cast": ", ".join(cred["cast"]),
                "key_crew": ", ".join(cred["crew"]),
                "season_overview": (s.get("overview") or "")[:500],
                # ---- YOUR LABELS (fill these) ----
                # status: watched | dropped | skipped | backlog   (legend printed on run)
                "status": prior_status.get((tid, sn), ""),
                # season_rating: 0.5–5.0, ONLY for status=watched (leave blank otherwise)
                "season_rating": prior.get((tid, sn), ""),
            })
            prev_cast, prev_crew, prev_imdb = cred["cast"], cred["crew"], imdb_mean
        if i % 20 == 0:
            _save(C_SHOW, c_show); _save(C_CRED, c_cred); _save(C_OMDB, c_omdb)
            print(f"   {i}/{len(df)} shows...")
        time.sleep(0.03)

    _save(C_SHOW, c_show); _save(C_CRED, c_cred); _save(C_OMDB, c_omdb)
    out = pd.DataFrame(rows).sort_values(["show_name", "season_number"]).reset_index(drop=True)

    # Auto-flag ANTHOLOGY shows so you don't mislabel their unwatched seasons as "dropped".
    # Signal: seasons that keep swapping the whole cast (mean cast_turnover high across the
    # show). True Detective = 1.0 every season. These seasons' "didn't watch" is a viewing
    # CHOICE (different story), not a taste signal — suggest status=skipped, not dropped.
    turn = out.groupby("tmdb_id")["cast_turnover"].mean()
    anthology_ids = set(turn[turn >= 0.7].index)
    out["is_anthology"] = out["tmdb_id"].isin(anthology_ids).astype(int)
    # Gentle prefill: on anthology shows, blank-status seasons default to a 'skipped?' hint
    # (you confirm/override). Watched/rated seasons and any you've labelled are left alone.
    mask = (out["is_anthology"] == 1) & (out["status"].astype(str).str.strip() == "")
    out.loc[mask, "status"] = "skipped?"

    # order columns so your two label columns sit at the end, next to the anthology hint
    label_cols = ["is_anthology", "status", "season_rating"]
    out = out[[c for c in out.columns if c not in label_cols] + label_cols]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    tmdb_ok = out["tmdb_season_vote"].fillna(0).gt(0).sum()
    imdb_ok = out["imdb_season_mean"].notna().sum()
    cast_ok = (out["top_cast"].astype(str).str.len() > 0).sum()
    turn_ok = out["cast_turnover"].notna().sum()
    print(f"\n✅ Wrote {len(out)} season rows ({out['tmdb_id'].nunique()} shows) -> {OUT}")
    print(f"   TMDB per-season vote:   {tmdb_ok}/{len(out)}")
    print(f"   IMDb per-season rating: {imdb_ok}/{len(out)}")
    print(f"   per-season cast:        {cast_ok}/{len(out)}   cast-change rows: {turn_ok}")
    print(f"   anthology shows auto-flagged: {len(anthology_ids)} (seasons hinted status='skipped?')")
    print("\n👉 Label two columns per season:")
    print("   status:")
    print("     watched  -> you saw it; also put a 0.5–5.0 in season_rating")
    print("     dropped  -> you quit from LACK OF INTEREST -> negative/censored signal")
    print("                 (leave season_rating blank; training treats it as 'rating <= drop threshold')")
    print("     skipped  -> didn't watch by CHOICE (anthology/different story) -> EXCLUDED, not negative")
    print("     backlog  -> haven't gotten to it -> missing, ignored")
    print("   season_rating: 0.5–5.0, ONLY when status=watched.")
    print("   (anthology seasons are pre-hinted 'skipped?' — change to 'skipped' to confirm, or")
    print("    'watched'+rating / 'dropped' if that's actually what happened.)")
    print("\n   Re-run any time — caches make it fast; your status + season_rating are preserved.")


if __name__ == "__main__":
    main()
