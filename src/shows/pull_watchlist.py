"""
Pull the Serializd **watchlist** (want-to-watch shows) and persist it to
data/raw/shows/watchlist.csv — the TV parallel of the Letterboxd movie
watchlist.csv. The shows active-learning queue uses this as its candidate pool.

Usage:
    python src/shows/pull_watchlist.py
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.serializd_sync import make_client

load_dotenv()

WATCHLIST_PATH = config.TV_SHOWS_RAW_DIR / "watchlist.csv"


def fetch_watchlist_rows():
    token = os.getenv("SERIALIZD_TOKEN")
    username = os.getenv("SERIALIZD_USERNAME", "Paradox73")
    if not token:
        raise SystemExit("ERROR: SERIALIZD_TOKEN missing in .env")
    client = make_client(token)
    rows, page, total = [], 1, 1
    while page <= total:
        r = client.session.get(
            f"user/{username}/watchlistpage_v2/{page}?sort_by=date_added_desc")
        if r.status_code != 200 or "json" not in (r.headers.get("content-type") or ""):
            raise SystemExit(f"Failed to read watchlist page {page} (status {r.status_code})")
        d = r.json()
        total = d.get("totalPages", 1) or 1
        for it in d.get("items", []):
            rows.append({
                "showId": it.get("showId"),
                "showName": it.get("showName"),
                "dateAdded": it.get("dateAdded"),
                "numSeasons": it.get("numSeasons"),
                "numEpisodes": it.get("numEpisodes"),
                "premiereDate": it.get("premiereDate"),
            })
        page += 1
        time.sleep(0.3)
    return rows


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    rows = fetch_watchlist_rows()
    df = pd.DataFrame(rows).dropna(subset=["showId"])
    df["showId"] = df["showId"].astype(int)
    df = df.drop_duplicates(subset=["showId"]).reset_index(drop=True)
    WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(WATCHLIST_PATH, index=False)
    print(f"Pulled {len(df)} watchlist shows -> {WATCHLIST_PATH}")
    print(df[["showId", "showName"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
