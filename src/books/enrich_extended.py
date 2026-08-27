"""Pull the Hardcover tag categories the pipeline was discarding.

`cached_tags` comes back with FOUR categories and ingestion only ever read
`Genre` (`_genres_from_cached_tags`). The other three are pure taste signal and
cost nothing extra -- they are already in the library payload:

    Mood             reflective, dark, tense, funny, emotional, challenging
    Tag              Plot driven, Strong Character Development, Loveable Characters
    Content Warning  violence, death, animal cruelty, gaslighting

`moods` in particular is the closest thing Books has to the keyword/tag channel
every other domain now contributes to the shared latent space.

Matching is by title (the enriched CSV keeps no Hardcover book id), normalised
case/punctuation, so re-runs are idempotent.

Usage:  python -m src.books.enrich_extended --plan | --apply
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.books.ingestion import _gql, ME_QUERY

NEW_COLS = ["moods", "hc_tags", "content_warnings"]
CATEGORIES = {"moods": "Mood", "hc_tags": "Tag", "content_warnings": "Content Warning"}

LIB_TAGS_QUERY = """
query($uid: Int!, $limit: Int!, $offset: Int!) {
  user_books(where: {user_id: {_eq: $uid}}, limit: $limit, offset: $offset) {
    book { title cached_tags }
  }
}
"""


def _norm(s):
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def _names(tags, category, limit=10):
    if not isinstance(tags, dict):
        return []
    out = []
    for t in (tags.get(category) or [])[:limit]:
        name = t.get("tag") if isinstance(t, dict) else t
        if name and name not in out:
            out.append(str(name))
    return out


def fetch_library_tags():
    me = _gql(ME_QUERY) or {}
    m = me.get("me")
    uid = (m[0] if isinstance(m, list) else m or {}).get("id")
    if not uid:
        sys.exit("❌ Could not resolve Hardcover user (check HARDCOVER_API_KEY).")
    print(f"📚 Hardcover user id {uid}")

    by_title, offset, page = {}, 0, 100
    while True:
        d = _gql(LIB_TAGS_QUERY, {"uid": int(uid), "limit": page, "offset": offset})
        rows = (d or {}).get("user_books") or []
        if not rows:
            break
        for ub in rows:
            b = ub.get("book") or {}
            t = b.get("cached_tags")
            if b.get("title") and isinstance(t, dict):
                by_title[_norm(b["title"])] = t
        offset += page
        if len(rows) < page:
            break
    print(f"   pulled tags for {len(by_title)} library books")
    return by_title


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

    path = config.BOOKS_ENRICHED_DATA_PATH
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")
    for c in NEW_COLS:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df), dtype="object")

    print(f"{len(df)} books. Missing: "
          + ", ".join(f"{c} {int(df[c].isna().sum())}" for c in NEW_COLS))
    if args.plan:
        print("PLAN ONLY.")
        return

    tags_by_title = fetch_library_tags()
    filled = {c: 0 for c in NEW_COLS}
    unmatched = []
    for idx, row in df.iterrows():
        t = tags_by_title.get(_norm(row.get("title")))
        if not t:
            unmatched.append(row.get("title"))
            continue
        for col, cat in CATEGORIES.items():
            if pd.isna(df.at[idx, col]):
                vals = _names(t, cat)
                if vals:
                    df.at[idx, col] = ", ".join(vals)
                    filled[col] += 1

    df.to_csv(path, index=False)
    print("\nFILLED:")
    for c, n in filled.items():
        print(f"   {c:18} +{n}  ({df[c].notna().mean():.1%} coverage)")
    if unmatched:
        print(f"\n{len(unmatched)} books had no Hardcover tag match: "
              f"{unmatched[:6]}{'...' if len(unmatched) > 6 else ''}")


if __name__ == "__main__":
    main()
