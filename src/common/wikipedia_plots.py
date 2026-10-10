"""wikipedia_plots.py — pull real beat-by-beat plot text for the library.

WHY
    Every description the pipeline currently embeds is a BLURB: a TMDB overview
    (median 1 sentence for TV, 2 for film), a RAWG marketing paragraph, a
    publisher's jacket copy. None of them has a beginning/middle/end, which is why
    splitting descriptions into positional thirds and measuring narrative arc found
    nothing -- there was no arc in the text to find.

    A Wikipedia `== Plot ==` section is the opposite: an ordered, spoiler-complete
    retelling, typically 400-800 words. It is the only free, non-generated source
    of ordered plot text at library scale.

WHAT THIS DOES NOT DO
    It does not touch the feature pipeline. Plot text changes the length and the
    REGISTER of a domain's text, and register asymmetry across domains is exactly
    the confound `src/common/text_embedding.py` was written to remove -- give films
    encyclopedic prose while games keep marketing copy and cross-domain cosine
    starts measuring prose style. So this module's job is to fetch and REPORT
    coverage. Decide from the coverage table whether the text is even usable before
    wiring it into anything.

HOW (public MediaWiki Action API, no key, no HTML scraping)
    1. title -> QID via `build_entity_links.resolve_qid`, which disambiguates on a
       type hint. This step is not optional: a bare Wikipedia title search cannot
       tell the 1965 novel `Dune` from the 2021 film, and the library holds both.
    2. QID -> English article title via `wbgetentities&props=sitelinks` (50 QIDs
       per request).
    3. article -> plaintext via `prop=extracts&explaintext`. ONE title per request:
       `exlimit` is capped at 1 unless `exintro` is set, and `exintro` returns only
       the lead, which is the one part of the article that is NOT the plot.
    4. Regex the plot heading and take everything up to the next `==` heading.

RATE LIMITS -- the part that bit
    A first pass at ~8 req/s earned HTTP 429s on 417 of 461 requests, and because
    the cache helper persisted error blobs, those failures became permanent: the
    coverage report then read them as "no article exists" and reported 0% for books
    and 4.5% for films. Both numbers were fiction.

    So: 1 req/s by default (Wikimedia's guidance for anonymous clients), exponential
    backoff on 429/5xx, and failures are NEVER written to the cache -- a re-run
    retries them. Expect roughly `n_items` seconds for a cold run.

USAGE
    python -m src.common.wikipedia_plots                    # whole library
    python -m src.common.wikipedia_plots --domains tv book  # some domains
    python -m src.common.wikipedia_plots --limit 40         # a taste, per domain
    python -m src.common.wikipedia_plots --report-only      # re-report from cache

    Writes data/processed/wikipedia_plots.csv and prints the coverage table.
    Resumable: rows already in the CSV are skipped unless --refresh.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.parsing import clean_year
from src.linking.build_entity_links import DOMAINS, UA, resolve_qid

CACHE_DIR = config.CACHE_DIR / "wikipedia"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# config exposes PROCESSED_DIR (there is no PROCESSED_DATA_DIR), so the old
# hasattr fallback always fired and wrote to a RELATIVE path — fine from the repo
# root, wrong from anywhere else. Resolve it absolutely.
OUT_PATH = config.PROCESSED_DIR / "wikipedia_plots.csv"

DEFAULT_SLEEP = 1.0

# Ordered by preference: a "Plot" section beats a "Premise" one, which is often a
# single sentence of setup rather than the story.
PLOT_HEADING = re.compile(
    r"^==\s*(Plot|Plot summary|Synopsis|Plot synopsis|Story|Storyline|Premise)\s*==\s*$",
    re.IGNORECASE | re.MULTILINE)
ANY_HEADING = re.compile(r"^==[^=].*==\s*$", re.MULTILINE)


def _cache_path(url):
    """Hash the whole URL.

    The sibling helper in build_entity_links used to truncate the quoted URL to 180
    chars, which is unique for single-QID URLs but collides for batched ones -- all
    50-id requests mapped to one file and every batch after the first read the
    first batch's answer.
    """
    return CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")


def wget(url, retries=5, sleep=DEFAULT_SLEEP):
    """Cached GET. Retries 429/5xx with exponential backoff; never caches a failure."""
    key = _cache_path(url)
    if key.exists():
        try:
            cached = json.loads(key.read_text(encoding="utf-8"))
            if "__error__" not in cached:
                return cached
        except Exception:
            pass

    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=30).read())
            key.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(sleep)
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return {"__error__": repr(e)}
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            return {"__error__": repr(e)}
    return {"__error__": "retries exhausted"}


def _sitelink_batch(chunk, sleep):
    url = ("https://www.wikidata.org/w/api.php?action=wbgetentities&ids="
           + "|".join(chunk) + "&props=sitelinks&sitefilter=enwiki&format=json")
    res = wget(url, sleep=sleep)
    got, answered = {}, set()
    for qid, ent in (res.get("entities") or {}).items():
        answered.add(qid)
        link = (ent.get("sitelinks") or {}).get("enwiki")
        if link:
            got[qid] = link["title"]
    return got, answered


def enwiki_titles(qids, sleep=DEFAULT_SLEEP):
    """{QID: English Wikipedia article title} — 50 QIDs per request.

    A batch that fails (or comes back partial) is RETRIED in smaller pieces rather
    than skipped. The first version just did `for qid, ent in res.get("entities")`,
    so a single failed 50-id request silently dropped 50 items and they showed up
    in the report as "no Wikipedia article" -- that cost 137 films, including
    Reservoir Dogs, The Lion King and The Usual Suspects, every one of which
    resolves fine on its own. Missing-because-the-request-failed and
    missing-because-it-does-not-exist must not look the same.
    """
    out = {}
    qids = [q for q in qids if q]
    missing = []
    for i in range(0, len(qids), 50):
        chunk = qids[i:i + 50]
        got, answered = _sitelink_batch(chunk, sleep)
        out.update(got)
        missing += [q for q in chunk if q not in answered]

    # Second pass for anything the API never answered for, in batches of 5.
    if missing:
        print(f"    retrying {len(missing)} QID(s) the batch step did not answer")
        still = []
        for i in range(0, len(missing), 5):
            chunk = missing[i:i + 5]
            got, answered = _sitelink_batch(chunk, sleep)
            out.update(got)
            still += [q for q in chunk if q not in answered]
        if still:
            print(f"    {len(still)} QID(s) still unanswered after retry")
    return out


def article_text(title, sleep=DEFAULT_SLEEP):
    """Full article plaintext, or None. One title per request (see module docstring)."""
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1"
           "&format=json&redirects=1&titles=" + urllib.parse.quote(title, safe=""))
    res = wget(url, sleep=sleep)
    for page in ((res.get("query") or {}).get("pages") or {}).values():
        if page.get("extract"):
            return page["extract"]
    return None


def plot_section(text):
    """The plot section body, or None. Stops at the next top-level heading.

    NOTE the output column is `plot_text`, not `plot`: `df.plot` is pandas' plotting
    accessor, so a column named `plot` silently returns a PlotAccessor on attribute
    access instead of the data.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    m = PLOT_HEADING.search(text)
    if not m:
        return None
    rest = text[m.end():]
    nxt = ANY_HEADING.search(rest)
    body = (rest[:nxt.start()] if nxt else rest).strip()
    return body or None


# Where each domain keeps its release year. The year is not cosmetic: it is what
# separates `Dune` the 1965 novel from `Dune` the 2021 film, and the library holds
# both.
YEAR_COLS = {
    "movie": ("year", "released"),
    "tv": ("year", "first_air_date"),
    "game": ("released",),
    "book": ("publishedDate", "pub_year"),
}
_YEAR_IN = re.compile(r"(1[89]\d{2}|20\d{2})")


def _year_of(row, media):
    for col in YEAR_COLS.get(media, ()):
        v = row.get(col)
        if v is None or (isinstance(v, float) and v != v):
            continue
        m = _YEAR_IN.search(str(v))
        if m:
            return int(m.group(1))
    return None


def library_items(domains=None, limit=None):
    """One row per library item, in the unified build order, with a type hint + year."""
    rows = []
    for media, (path, title_col, _creators, hint) in DOMAINS.items():
        if domains and media not in domains:
            continue
        p = Path(path)
        if not p.exists():
            print(f"  skip {media}: {path} missing")
            continue
        enr = pd.read_csv(p, low_memory=False)
        # movies/tv drop unrated rows in the unified build; games/books keep all
        for rating_col in ("user_rating", "my_rating"):
            if rating_col in enr.columns:
                enr = enr[enr[rating_col].notna()]
                break
        enr = enr.reset_index(drop=True)
        if limit:
            enr = enr.head(limit)
        for i, row in enr.iterrows():
            rows.append({"media_type": media, "row": int(i),
                         "title": str(row.get(title_col, "")), "hint": hint,
                         "year": _year_of(row, media)})
    return pd.DataFrame(rows)


def build(domains=None, limit=None, sleep=DEFAULT_SLEEP, refresh=False):
    items = library_items(domains, limit)
    if items.empty:
        print("no items")
        return items
    print(f"  items: {items.media_type.value_counts().to_dict()}")

    if OUT_PATH.exists() and not refresh:
        prev = pd.read_csv(OUT_PATH)
        done = set(zip(prev.media_type, prev.row))
        todo = items[[t not in done for t in zip(items.media_type, items.row)]]
        print(f"  resuming: {len(items) - len(todo)} already fetched, {len(todo)} to go")
    else:
        prev, todo = pd.DataFrame(), items

    if not todo.empty:
        print("  resolving QIDs...")
        todo = todo.copy()
        todo["qid"] = [resolve_qid(t, h, y)
                       for t, h, y in zip(todo.title, todo.hint, todo.year)]
        print(f"    resolved {todo.qid.notna().mean():.1%}")

        print("  resolving English Wikipedia articles...")
        tmap = enwiki_titles(todo.qid.dropna().unique().tolist(), sleep=sleep)
        todo["wiki_title"] = todo.qid.map(tmap)
        print(f"    have an article: {todo.wiki_title.notna().mean():.1%}")

        print(f"  fetching articles (~{sleep:.1f}s each, "
              f"{todo.wiki_title.notna().sum()} to fetch)...")
        plots = []
        for n, wt in enumerate(todo.wiki_title, 1):
            plots.append(plot_section(article_text(wt, sleep=sleep))
                         if isinstance(wt, str) else None)
            if n % 50 == 0:
                print(f"    {n}/{len(todo)}")
        todo["plot_text"] = plots
        todo["plot_words"] = [len(p.split()) if isinstance(p, str) else 0 for p in plots]
        out = pd.concat([prev, todo.drop(columns=["hint"])], ignore_index=True) \
            if len(prev) else todo.drop(columns=["hint"])
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUT_PATH, index=False)
        print(f"  wrote {OUT_PATH} ({len(out)} rows)")
    else:
        out = prev
    return out


def report(df):
    if df is None or df.empty:
        print("nothing to report")
        return
    has_plot = df["plot_text"].apply(lambda p: isinstance(p, str) and bool(p.strip()))
    agg = df.assign(_p=has_plot).groupby("media_type").agg(
        n=("title", "size"),
        has_article=("wiki_title", lambda s: s.notna().mean()),
        has_plot=("_p", "mean"),
        median_words=("plot_words", lambda s: s[s > 0].median() if (s > 0).any() else 0),
    )
    print("\n" + "=" * 66)
    print("WIKIPEDIA PLOT COVERAGE")
    print("=" * 66)
    print(agg.round(3).to_string())
    print("\nA domain below ~70% has_plot cannot be used as a text source on its own:")
    print("partial coverage means the covered items get encyclopedic prose and the")
    print("rest keep a blurb, which is a register split INSIDE one domain.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--domains", nargs="*", default=None,
                    help="movie tv game book (default: all)")
    ap.add_argument("--limit", type=int, default=None, help="first N per domain")
    ap.add_argument("--sleep", type=float, default=DEFAULT_SLEEP,
                    help="seconds between requests (default 1.0; lower earns 429s)")
    ap.add_argument("--refresh", action="store_true", help="ignore existing CSV")
    ap.add_argument("--report-only", action="store_true", help="re-report from the CSV")
    a = ap.parse_args()

    if a.report_only:
        report(pd.read_csv(OUT_PATH) if OUT_PATH.exists() else None)
        return
    report(build(a.domains, a.limit, a.sleep, a.refresh))


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Feature-pipeline hook
# --------------------------------------------------------------------------- #
_PLOT_CACHE = None


def plot_lookup(refresh=False):
    """{(media_type, lowercased title): [(year, plot_text), ...]} for every item that has one.

    Keyed on TITLE + YEAR, not the `row` column: the CSV was fetched when books held
    100 rows and TV 173, and both have grown since, so a positional join would
    silently misalign the newest items onto someone else's plot. The year keeps
    remakes apart (Ghostbusters 1984/2016 carry different plots).
    """
    global _PLOT_CACHE
    if _PLOT_CACHE is not None and not refresh:
        return _PLOT_CACHE
    out = {}
    if OUT_PATH.exists():
        w = pd.read_csv(OUT_PATH)
        for mt, title, yr, txt in zip(w["media_type"], w["title"], w["year"],
                                      w.get("plot_text", [])):
            if isinstance(txt, str) and txt.strip():
                out.setdefault((str(mt), str(title).strip().lower()), []).append(
                    (clean_year(yr), txt.strip()))
    _PLOT_CACHE = out
    return out


def plot_for(media_type, title, fallback="", year=None):
    """Wikipedia plot for this item, else `fallback`.

    Coverage is uneven by construction -- movies 93.5%, TV 65.3%, games 54.2%,
    books 49.0% -- because a Diary of a Wimpy Kid volume has no standalone article
    and FIFA 09 has no plot to summarise. So a domain is a MIX of encyclopedic plot
    prose and marketing blurb, which is a register split inside one domain. That is
    a real confound and it is why this is opt-in per pipeline rather than silently
    applied; the honest read of any gain is "on this mixture", and the covered-only
    subset is the like-for-like comparison.
    """
    if not title:
        return fallback
    # WIKI_PLOTS_OFF=tv,game turns the Wikipedia plot off for those media types
    # (falls back to the blurb), for like-for-like ablations of the text change.
    if str(media_type) in {m.strip() for m in os.environ.get("WIKI_PLOTS_OFF", "").split(",") if m.strip()}:
        return fallback
    cands = plot_lookup().get((str(media_type), str(title).strip().lower()), [])
    y = clean_year(year)
    exact = [txt for yr, txt in cands if y is not None and yr == y]
    if exact:
        return exact[0]
    # Without a year match, only an unambiguous title may borrow a plot: a remake
    # or a new same-titled item must not inherit another item's plot.
    if len(cands) == 1 and (y is None or cands[0][0] is None):
        return cands[0][1]
    return fallback
