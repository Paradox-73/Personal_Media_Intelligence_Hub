"""
Two-way sync between the enriched TV shows CSV and a Serializd.com account.

Source of truth = the CSV. After a sync:
  * Shows in the CSV but NOT logged on Serializd  -> logged as watched (all seasons)
  * Shows logged on Serializd but NOT in the CSV   -> unlogged (removed)
The CSV is never modified.

Auth: Serializd uses a session cookie named `tvproject_credentials`. Google-login
accounts have no password, so we authenticate with that cookie value, provided via
the SERIALIZD_TOKEN environment variable (.env). Username is auto-detected from the
token, or set SERIALIZD_USERNAME to override.

Usage:
    python src/shows/serializd_sync.py --plan     # show diff only, change nothing
    python src/shows/serializd_sync.py --apply    # add missing + remove extras
    python src/shows/serializd_sync.py --apply --add-only     # only add, never remove
    python src/shows/serializd_sync.py --apply --remove-only  # only remove
"""
import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import is_effectively_empty

from serializd import SerializdClient

load_dotenv()

WATCHED_SORT = "date_added_desc"
# serializd.com sits behind a Vercel "Security Checkpoint" (JS bot challenge) that
# blocks non-browser HTTP clients. The Render backend serves the same API without it.
BACKEND_BASE = "https://serializd.onrender.com/api/"
BACKEND_HOST = "serializd.onrender.com"
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
)


def make_client(token):
    """SerializdClient pointed at the Render backend with the auth cookie loaded."""
    client = SerializdClient()
    client.session.base_url = httpx.URL(BACKEND_BASE)
    client.session.headers.update({"User-Agent": BROWSER_UA, "Accept": "application/json"})
    client.session.cookies.set("tvproject_credentials", token, domain=BACKEND_HOST)
    return client


def get_watched_shows(client, username):
    """Return {tmdb_show_id: show_name} for every show logged as watched."""
    out = {}
    page = 1
    total_pages = 1
    while page <= total_pages:
        path = f"user/{username}/watchedpage_v2/{page}?sort_by={WATCHED_SORT}"
        resp = None
        for attempt in range(4):
            resp = client.session.get(path)
            if resp.status_code == 200 and "json" in (resp.headers.get("content-type") or ""):
                break
            time.sleep(2 * (attempt + 1))  # back off on 429 / transient
        if resp is None or resp.status_code != 200:
            raise RuntimeError(
                f"Failed to read watched page {page} (status "
                f"{getattr(resp, 'status_code', '?')}). Token expired or rate-limited."
            )
        data = resp.json()
        total_pages = data.get("totalPages", 1) or 1
        for item in data.get("items", []):
            sid = item.get("showId")
            if sid is not None:
                out[int(sid)] = item.get("showName")
        page += 1
        time.sleep(0.5)
    return out


def get_csv_shows():
    """Return {tmdb_show_id: show_name} for every CSV row that has a tmdb_id."""
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    out = {}
    for _, row in df.iterrows():
        if is_effectively_empty(row.get("tmdb_id")):
            continue
        out[int(float(row["tmdb_id"]))] = row.get("name")
    return out


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="show diff only")
    mode.add_argument("--apply", action="store_true", help="execute the sync")
    parser.add_argument("--add-only", action="store_true", help="never remove")
    parser.add_argument("--remove-only", action="store_true", help="never add")
    args = parser.parse_args()

    token = os.getenv("SERIALIZD_TOKEN")
    if not token:
        print("ERROR: SERIALIZD_TOKEN missing in .env (the tvproject_credentials cookie value).")
        sys.exit(1)

    username = os.getenv("SERIALIZD_USERNAME", "Paradox73")
    client = make_client(token)
    print(f"Backend: {BACKEND_BASE} | user: {username}")

    csv_shows = get_csv_shows()
    serializd_shows = get_watched_shows(client, username)
    print(f"CSV shows (with tmdb_id): {len(csv_shows)}")
    print(f"Serializd watched shows:  {len(serializd_shows)}")

    to_add = {sid: nm for sid, nm in csv_shows.items() if sid not in serializd_shows}
    to_remove = {sid: nm for sid, nm in serializd_shows.items() if sid not in csv_shows}

    print(f"\n=== TO ADD to Serializd ({len(to_add)}) ===")
    for sid, nm in sorted(to_add.items(), key=lambda x: str(x[1])):
        print(f"  + {nm}  (tmdb {sid})")
    print(f"\n=== TO REMOVE from Serializd ({len(to_remove)}) ===")
    for sid, nm in sorted(to_remove.items(), key=lambda x: str(x[1])):
        print(f"  - {nm}  (tmdb {sid})")

    if args.plan:
        print("\nPLAN ONLY - nothing changed.")
        return

    # --- apply ---
    if not args.remove_only:
        print(f"\nAdding {len(to_add)} shows...")
        for sid, nm in to_add.items():
            try:
                ok = client.log_show(sid)
                print(f"  {'OK ' if ok else 'FAIL'} +{nm} (tmdb {sid})")
            except Exception as e:
                print(f"  ERR +{nm} (tmdb {sid}): {e}")
            time.sleep(0.4)

    if not args.add_only:
        print(f"\nRemoving {len(to_remove)} shows...")
        for sid, nm in to_remove.items():
            try:
                ok = client.unlog_show(sid)
                print(f"  {'OK ' if ok else 'FAIL'} -{nm} (tmdb {sid})")
            except Exception as e:
                print(f"  ERR -{nm} (tmdb {sid}): {e}")
            time.sleep(0.4)

    print("\nDONE.")


if __name__ == "__main__":
    main()
