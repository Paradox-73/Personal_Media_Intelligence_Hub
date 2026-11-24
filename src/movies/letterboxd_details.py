"""
Merge the Letterboxd film-page export and network export into the movie data.

Inputs (pulled in the browser from letterboxd.com, saved to data/raw/movies/):
  letterboxd_my_films_details.xlsx  every film on Paradox_73's Films page
  letterboxd_network.xlsx           following/followers and every film they logged

What it writes:
  enriched_data.csv
    * Film not in the CSV -> fetched from TMDB/OMDb via get_movie_metadata and
      appended (same path as letterboxd_rss_sync.py). rating_date stays blank: the
      workbook has no watch dates; the next export/RSS sync fills it.
    * Film already in the CSV -> only ADDS: blank cells are filled, and the
      name-list columns (genre/director/actors/production) gain names they lack.
      Nothing existing is overwritten or removed. A set user_rating is never changed
      here (the export sync owns it); mismatches are only reported, blanks filled.
    * lb_* columns -> the Letterboxd crowd snapshot for every matched film.
      A newer non-blank value replaces an older one; a blank never erases one.
  letterboxd_network_members.csv, letterboxd_network_films.csv (processed/movies)

Usage:
    python -m src.movies.letterboxd_details --plan
    python -m src.movies.letterboxd_details --apply
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.parsing import canon_genre, parse_list
from src.movies.ingestion import get_movie_metadata, clean_year, is_effectively_empty

RAW = config.DATA_DIR / "raw" / "movies"
FILMS_XLSX = RAW / "letterboxd_my_films_details.xlsx"
NETWORK_XLSX = RAW / "letterboxd_network.xlsx"
NET_MEMBERS = config.MOVIES_PROCESSED_DIR / "letterboxd_network_members.csv"
NET_FILMS = config.MOVIES_PROCESSED_DIR / "letterboxd_network_films.csv"

# workbook column -> lb_* column (Letterboxd snapshot, refreshed each run)
LB_COLS = {
    "Letterboxd average rating": "lb_avg_rating",
    "Ratings": "lb_rating_count",
    "Reviews": "lb_review_count",
    "Fans (rounded)": "lb_fans",
    "Watched by (members)": "lb_watched_by",
    "Appears in lists": "lb_list_count",
    "Liked by (members)": "lb_like_count",
    "Themes": "lb_themes",
    "Letterboxd link": "lb_film_url",
}
# Network columns are deliberately NOT copied into enriched_data.csv: that file
# feeds the model, which must stay single-user (no other people's ratings).
# Network data lives only in the letterboxd_network_*.csv files for the dashboard.
# workbook column -> existing column, filled only where the CSV cell is blank
FILL_COLS = {
    "Runtime (mins)": "runtime", "Tagline": "tagline", "Synopsis": "overview",
    "IMDb ID": "imdb_id", "Countries": "country", "Primary language": "language",
}
# "Collection" is deliberately not filled: Letterboxd spells it "Spider Man Mcu
# Collection", TMDB (what the Oracle serves) "Spider-Man (MCU) Collection", and the
# model encodes collection_name.
# list column -> where its names come from in the workbook's one-name-per-row
# sheets (the Films sheet joins names with commas, which splits "Tyler, The Creator")
UNION_COLS = ["genre", "director", "actors", "production"]
STARS = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
# share of Letterboxd ratings at each star value (0..1), lb_share_05 = half a star
SHARE_COLS = [f"lb_share_{int(v * 10):02d}" for v in STARS]
# per-film derived columns copied like LB_COLS (computed in load_films)
DERIVED_COLS = ["lb_rating_std", "lb_top500_rank", "lb_similar_urls", "lb_related_urls"] + SHARE_COLS


def load_films():
    films = pd.read_excel(FILMS_XLSX, sheet_name="Films")
    curve = pd.read_excel(FILMS_XLSX, sheet_name="Rating curve")
    # Both sheets list the same films in the same order; verify, then take the
    # star-share columns positionally to get the spread of the crowd's ratings.
    assert len(films) == len(curve) and (films["Film"].values == curve["Film"].values).all(), \
        "Films and Rating curve sheets are not row-aligned"
    shares = curve[[c for c in curve.columns if c.endswith("share (%)")]].to_numpy(float) / 100
    for j, c in enumerate(SHARE_COLS):
        films[c] = shares[:, j]
    mean = shares @ STARS
    films["lb_rating_std"] = np.sqrt(np.clip(shares @ np.square(STARS) - mean ** 2, 0, None))
    films.loc[curve["Total ratings"].fillna(0).to_numpy() == 0, "lb_rating_std"] = np.nan
    films["lb_top500_rank"] = pd.to_numeric(
        films["Top list rank"].astype("string").str.extract(r"#(\d+)")[0], errors="coerce")
    films["tmdb_id_int"] = pd.to_numeric(
        films["TMDB ID"].astype("string").str.extract(r"(\d+)")[0], errors="coerce")
    # Themes arrive "; "-joined because theme names contain commas. Keep that.

    key = lambda d: d["Film"].astype(str) + "|" + d["Year"].astype(str)
    gt = pd.read_excel(FILMS_XLSX, sheet_name="Genres and themes")
    crew = pd.read_excel(FILMS_XLSX, sheet_name="Crew")
    cast = pd.read_excel(FILMS_XLSX, sheet_name="Cast")
    det = pd.read_excel(FILMS_XLSX, sheet_name="Details")
    sources = {
        "genre": gt[gt["Type"] == "Genre"].assign(n=lambda d: d["Value"]),
        "director": crew[crew["Role"] == "Director"].assign(n=lambda d: d["Name"]),
        "actors": cast[cast["Billing order"] <= 10].sort_values("Billing order")
                  .assign(n=lambda d: d["Actor"]),
        "production": det[det["Field"] == "Studio"].assign(n=lambda d: d["Value"]),
    }
    # Letterboxd's "similar films" (ordered) and same-collection films, as "; "-joined
    # film links so they join to lb_film_url.
    sim = pd.read_excel(FILMS_XLSX, sheet_name="Similar films").sort_values("Order")
    rel = pd.read_excel(FILMS_XLSX, sheet_name="Related films")
    sources_urls = {"lb_similar_urls": sim, "lb_related_urls": rel}
    fk = key(films)
    for col, d in sources_urls.items():
        urls = d.groupby(key(d).values)["Letterboxd link"].apply(lambda s: "; ".join(s.dropna()))
        films[col] = [urls.get(k, np.nan) or np.nan for k in fk]
    for col, d in sources.items():
        names = d.groupby(key(d).values)["n"].apply(lambda s: [str(x).strip() for x in s if pd.notna(x)])
        films[f"names_{col}"] = [names.get(k, []) for k in fk]
    return films


def union_cell(old, new_names, col):
    """old cell + the names it lacks, written back as a list repr (None = no change)."""
    have = parse_list(old)
    key = canon_genre if col == "genre" else (lambda s: s.strip().lower())
    seen = {key(x) for x in have}
    added = [n for n in dict.fromkeys(new_names) if n and key(n) not in seen]
    return (str(have + added) if added else None), len(added)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    films = load_films()
    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    n_before = len(df)
    for c in list(LB_COLS.values()) + DERIVED_COLS:
        if c not in df.columns:
            df[c] = pd.Series([pd.NA] * len(df), dtype="object")

    by_tmdb = {int(float(t)): i for i, t in df["tmdb_id"].items() if pd.notna(t)}
    ty_index = {}
    for i, r in df.iterrows():
        for nm in (r.get("letterboxd_name"), r.get("title")):
            if pd.notna(nm):
                ty_index.setdefault((str(nm).strip().lower(), clean_year(r.get("year"))), i)

    new_films, rating_diffs = [], []
    filled = {c: 0 for c in FILL_COLS.values()}
    unioned = {c: 0 for c in UNION_COLS}
    matched = 0
    for _, f in films.iterrows():
        idx = by_tmdb.get(int(f["tmdb_id_int"])) if pd.notna(f["tmdb_id_int"]) else None
        if idx is None:
            idx = ty_index.get((str(f["Film"]).strip().lower(), clean_year(f["Year"])))
        if idx is None:
            new_films.append(f)
            continue
        matched += 1
        for src, dst in LB_COLS.items():
            if not is_effectively_empty(f[src]):
                df.at[idx, dst] = f[src]
        for c in DERIVED_COLS:
            if not is_effectively_empty(f[c]):
                df.at[idx, c] = f[c]
        for src, dst in FILL_COLS.items():
            if dst in df.columns and is_effectively_empty(df.at[idx, dst]) \
                    and not is_effectively_empty(f[src]):
                df.at[idx, dst] = f[src]
                filled[dst] += 1
        for col in UNION_COLS:
            cell, _ = union_cell(df.at[idx, col], f[f"names_{col}"], col)
            if cell is not None:
                df.at[idx, col] = cell
                unioned[col] += 1
        mine = df.at[idx, "user_rating"]
        if pd.notna(mine) and pd.notna(f["My rating (out of 5)"]) \
                and float(mine) != float(f["My rating (out of 5)"]):
            rating_diffs.append((f["Film"], mine, f["My rating (out of 5)"]))
        elif pd.isna(mine) and pd.notna(f["My rating (out of 5)"]):
            df.at[idx, "user_rating"] = f["My rating (out of 5)"]

    print(f"Workbook films: {len(films)} | matched to CSV: {matched} | new: {len(new_films)}")
    print(f"Blank cells filled: {filled}")
    print(f"List cells that gained names: {unioned}")
    print(f"Ratings that differ from the CSV (left unchanged, export sync owns them): {len(rating_diffs)}")
    for t, a, b in rating_diffs[:15]:
        print(f"   {t}: csv {a} vs letterboxd {b}")
    print(f"\n=== NEW films to fetch from TMDB/OMDb ({len(new_films)}) ===")
    for f in new_films:
        print(f"  + {f['Film']} ({f['Year']})  tmdb {f['tmdb_id_int']}  rating={f['My rating (out of 5)']}")

    members = pd.read_excel(NETWORK_XLSX, sheet_name="Members")
    members = members[members["Username"].notna()]  # drop the "Total" summary row
    net_films = pd.read_excel(NETWORK_XLSX, sheet_name="Films")
    print(f"\nNetwork: {len(members)} members, {len(net_films):,} film rows")

    if args.plan:
        print("\nPLAN ONLY - nothing changed.")
        return

    added = []
    for f in new_films:
        print(f"[enrich] {f['Film']} ({f['Year']})")
        tmdb_id = f["tmdb_id_int"] if pd.notna(f["tmdb_id_int"]) else None
        meta = get_movie_metadata(str(f["Film"]), clean_year(f["Year"]), tmdb_id=tmdb_id)
        meta["letterboxd_name"] = f["Film"]
        meta["letterboxd_uri"] = f["Letterboxd link"]
        meta["user_rating"] = f["My rating (out of 5)"]
        meta["is_liked"] = 1 if f["I liked it"] == "Yes" else 0
        meta["year"] = meta.get("year") or clean_year(f["Year"])
        for src, dst in LB_COLS.items():
            meta[dst] = f[src]
        for c in DERIVED_COLS:
            meta[c] = f[c]
        for src, dst in FILL_COLS.items():
            if is_effectively_empty(meta.get(dst)) and not is_effectively_empty(f[src]):
                meta[dst] = f[src]
        for col in UNION_COLS:
            cell, _ = union_cell(meta.get(col), f[f"names_{col}"], col)
            if cell is not None:
                meta[col] = cell
        added.append(meta)
        time.sleep(0.1)

    if added:
        new_df = pd.DataFrame(added)
        for c in df.columns:
            if c not in new_df.columns:
                new_df[c] = pd.NA
        df = pd.concat([df, new_df[df.columns]], ignore_index=True)

    assert len(df) == n_before + len(added), "row count changed unexpectedly"
    df.to_csv(config.MOVIES_ENRICHED_DATA_PATH, index=False)
    members.to_csv(NET_MEMBERS, index=False)
    net_films.to_csv(NET_FILMS, index=False)
    print(f"\nDONE. updated {matched}, added {len(added)}; CSV {n_before} -> {len(df)} rows.")
    if added:
        print("Next: python -m src.movies.enrich_extended --apply  (keywords/crew for the new rows)")


if __name__ == "__main__":
    main()
