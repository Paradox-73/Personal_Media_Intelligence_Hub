"""
merge_youtube_takeouts.py — fold every YouTube Takeout export into one history file.

WHY THIS EXISTS
    Google's watch-history export is a ROLLING WINDOW, not a full archive. Each new
    Takeout covers only a recent slice, so replacing `watch-history.json` with the
    latest export silently deletes everything that fell off the back. Older exports
    are therefore not redundant -- they are the only copy of the early history.

    This was previously done by hand, which is where the stray `.bak_*.json` files
    came from. This script makes it repeatable: drop a new Takeout anywhere under
    data/raw/youtube/ and run it.

WHAT IT DOES
    Finds every watch-history.json under data/raw/youtube/ (including nested Takeout
    folders), unions the rows, de-duplicates on (titleUrl, time), sorts newest first
    and writes data/raw/youtube/watch-history.json. The previous file is backed up
    with its own last-modified date in the name.

Run:  python -m scripts.merge_youtube_takeouts
      python -m scripts.merge_youtube_takeouts --dry-run
"""
import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

RAW_DIR = Path("data/raw/youtube")
TARGET = RAW_DIR / "watch-history.json"


def _key(row):
    """Identity of one watch event. titleUrl is stable; title is not (videos get
    renamed), so title is only a fallback for rows with no URL (deleted videos)."""
    return (row.get("titleUrl") or row.get("title", ""), row.get("time", ""))


def _sources():
    """Every watch-history.json under the raw dir, target file included."""
    seen, out = set(), []
    for p in sorted(RAW_DIR.rglob("watch-history*.json")):
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report the merge, write nothing")
    args = ap.parse_args()

    srcs = _sources()
    if not srcs:
        print(f"No watch-history*.json found under {RAW_DIR}")
        return 1

    merged, per_file = {}, []
    for p in srcs:
        try:
            rows = json.load(open(p, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - a corrupt export must not lose the rest
            print(f"  !! skipped {p} ({e})")
            continue
        before = len(merged)
        for r in rows:
            # keep=first: sources are visited in a stable order, and duplicate rows
            # across exports are byte-identical, so which copy wins does not matter.
            merged.setdefault(_key(r), r)
        ts = [r.get("time", "") for r in rows if r.get("time")]
        per_file.append((p, len(rows), len(merged) - before,
                         (min(ts)[:10], max(ts)[:10]) if ts else ("-", "-")))

    print(f"Merging {len(per_file)} export(s):")
    for p, n, added, (lo, hi) in per_file:
        rel = p.relative_to(RAW_DIR)
        print(f"  {str(rel)[:58]:58} {n:>7} rows  {lo}..{hi}  +{added} new")

    out = sorted(merged.values(), key=lambda r: r.get("time", ""), reverse=True)
    ts = [r.get("time", "") for r in out if r.get("time")]
    print(f"\nTotal unique watch events: {len(out)}  ({min(ts)[:10]} .. {max(ts)[:10]})")

    # Gaps are worth knowing about: they are months where no export was ever taken,
    # and no amount of re-merging will recover them.
    days = sorted({r.get("time", "")[:10] for r in out if r.get("time")})
    gaps = []
    for a, b in zip(days, days[1:]):
        d = (datetime.fromisoformat(b) - datetime.fromisoformat(a)).days
        if d > 30:
            gaps.append((a, b, d))
    if gaps:
        print("Gaps over 30 days (no export covers these):")
        for a, b, d in gaps:
            print(f"   {a} -> {b}  ({d} days missing)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if TARGET.exists():
        stamp = datetime.fromtimestamp(TARGET.stat().st_mtime).strftime("%Y%m%d")
        bak = TARGET.with_name(f"watch-history.bak_{stamp}.json")
        shutil.copy2(TARGET, bak)
        print(f"\nBacked up previous -> {bak.name}")

    with open(TARGET, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"Wrote {TARGET} ({len(out)} events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
