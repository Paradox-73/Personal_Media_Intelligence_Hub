"""
Phase 3 — Cross-domain entity bridges (Task 3.1).

Builds data/processed/entity_links.csv linking library items ACROSS domains via
Wikidata relations:
  * adaptation     — P144 "based on" / P4969 "derivative work" connecting two
                     library items in different domains (e.g. a film based on a book).
  * shared_creator — same person QID via P50 (author) / P57 (director) /
                     P86 (composer) across domains.
  * same_franchise — same P179 "part of the series" QID across domains.

Item -> source_id mapping is POSITIONAL: each domain's enriched_data.csv is in the
same row order as the unified slice (verified by count), so row i maps to the unified
source_id for that domain. Columns: source_id_a, source_id_b, link_type, confidence,
plus human-readable title_a/title_b/media_a/media_b for the manual verification pass.

Network: uses the public Wikidata REST/Action API (no key). Be polite: small sleep,
on-disk cache so re-runs are cheap and deterministic.
"""
import sys
import json
import re
import time
import hashlib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

UA = {"User-Agent": "PersonalMediaIntelligenceHub/1.0 (research; contact: local)"}
# Wikimedia asks anonymous API clients to stay near 1 req/s; 0.15s earned a
# burst of HTTP 429s that the old cache then made permanent.
RATE_LIMIT_SLEEP = 1.0
CACHE_DIR = config.CACHE_DIR / "wikidata"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# enriched file, title column, creator columns, wikidata type hint
DOMAINS = {
    "movie": (config.MOVIES_ENRICHED_DATA_PATH, "title", ["director"], "film"),
    "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "name", [], "television series"),
    "game": (config.GAMES_ENRICHED_DATA_PATH, "name", [], "video game"),
    "book": (config.BOOKS_ENRICHED_DATA_PATH, "title", ["authors"], "book"),
}
CREATOR_PROPS = ["P50", "P57", "P86", "P58", "P170"]   # author/director/composer/writer/creator
ADAPT_PROPS = ["P144", "P4969"]                         # based on / derivative work
SERIES_PROP = "P179"


def _get(url, retries=4):
    """Cached GET. Retries transient failures and NEVER caches a failure.

    The previous version wrote `{"__error__": ...}` into the cache on any exception
    and returned the cached copy forever after. A single burst of HTTP 429s from
    Wikidata therefore became permanent: those items could never resolve a QID on
    any later run, however long you waited, and the only symptom was a quietly
    smaller entity_links.csv. 15 entries in the on-disk cache were stuck this way.

    Errors are now returned but not persisted, so a re-run retries them, and 429/5xx
    back off instead of burning the attempt.
    """
    # New keys are a hash of the full URL. The old scheme truncated the quoted URL
    # to 180 chars, which is unique for the single-QID URLs used here but silently
    # collides for any batched (multi-id) URL. Legacy keys are still READ so the
    # ~2.7k already-cached responses are not re-fetched.
    key = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")
    legacy = CACHE_DIR / (urllib.parse.quote(url, safe="")[:180] + ".json")
    for cand in (key, legacy):
        if not cand.exists():
            continue
        try:
            cached = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Tolerate (and heal) error blobs written by the older implementation.
        if "__error__" not in cached:
            return cached

    delay = 1.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = json.loads(urllib.request.urlopen(req, timeout=20).read())
            key.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(RATE_LIMIT_SLEEP)
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


# Words that identify each media type in a Wikidata description, plus the types a
# match must NOT be. "Batman" the film and "Batman" the Turkish province both match
# a bare title search; only the description separates them.
TYPE_WORDS = {
    "film": ("film", "movie"),
    "television series": ("television series", "tv series", "television programme",
                          "miniseries", "anime television series", "web series",
                          "television show", "sitcom"),
    "video game": ("video game", "videogame"),
    "book": ("book", "novel", "comic", "manga", "graphic novel", "literary work",
             "short story", "novella", "written work"),
}
_ALL_TYPE_WORDS = {w: k for k, ws in TYPE_WORDS.items() for w in ws}
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")

# Phrases that contain a wanted type word but describe something that is NOT the
# work itself. Positional matching alone cannot catch these, because the type
# word really is first: "film song written by..." starts with "film", and
# "CGI-animated film series" starts with "film" too. They cost us Ghostbusters
# (the SONG) and Toy Story (the FRANCHISE, whose article has no plot section).
_REJECT_PHRASES = {
    "film": ("film series", "series of films", "franchise", "film score",
             "film song", "song", "album", "soundtrack", "theme song",
             "musical", "cancelled film", "unproduced"),
    "television series": ("franchise", "song", "album", "soundtrack",
                          "episode of", "season of"),
    "video game": ("video game series", "game series", "franchise", "song",
                   "album", "soundtrack"),
    "book": ("book series", "series of novels", "franchise", "song", "album",
             "publisher", "magazine"),
}


def _first_index(desc, words):
    """Earliest position of any of `words` in `desc`, or None."""
    hits = [desc.index(w) for w in words if w in desc]
    return min(hits) if hits else None


def _score_candidate(cand, type_hint, title, year):
    """How well does this Wikidata hit match the thing we are looking for?

    Returns None to REJECT. Rejecting is the point: the old resolver fell back to
    `cands[0]` whenever nothing matched the type hint, which silently returned
    Batman Province (Turkey) for the film "Batman", the Katy Perry album for
    "Smile", and a 1986 video game for "Top Gun". A wrong QID is worse than no
    QID -- it pollutes entity_links.csv with confident nonsense and sends the plot
    fetcher to an article that has no plot.

    Matching is POSITIONAL, because a Wikidata description leads with what the
    thing IS and may mention other media later: "action video game developed by
    Ocean Software based on the 1989 film" is a game, not a film, and a bare
    substring test for "film" accepts it. Whichever type word appears first wins.
    """
    desc = (cand.get("description") or "").lower()
    label = (cand.get("label") or "").strip().lower()
    if not desc:
        return None
    want = TYPE_WORDS.get(type_hint, (type_hint,))
    competing = [w for w, k in _ALL_TYPE_WORDS.items() if k != type_hint]

    # Reject collections and derivatives outright, before positional scoring: they
    # lead with the right type word but are not the work ("film series", "film song").
    for bad in _REJECT_PHRASES.get(type_hint, ()):
        if bad in desc:
            return None

    want_at = _first_index(desc, want)
    comp_at = _first_index(desc, competing)
    if want_at is None:
        return None                      # never claims to be the right kind of thing
    if comp_at is not None and comp_at < want_at:
        return None                      # leads with a different medium

    score = 4
    if year:
        years = _YEAR_RE.findall(desc)
        if years:
            score += 3 if str(int(year)) in years else -2
    if label == str(title).strip().lower():
        score += 1
    # Earlier is better: "1989 film by Tim Burton" beats a description that only
    # mentions films halfway through.
    if want_at <= 12:
        score += 1
    return score if score > 0 else None


def _search(title, limit, cached_only=False):
    """One wbsearchentities call (cache-first). `cached_only` never hits the network."""
    q = urllib.parse.quote(title)
    url = (f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}"
           f"&language=en&format=json&limit={limit}")
    if cached_only:
        key = CACHE_DIR / (hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json")
        legacy = CACHE_DIR / (urllib.parse.quote(url, safe="")[:180] + ".json")
        for cand in (key, legacy):
            if cand.exists():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if "__error__" not in data:
                    return data.get("search", [])
        return None                      # not cached -- caller decides whether to fetch
    return _get(url).get("search", [])


def _best(cands, type_hint, title, year):
    best, best_score = None, 0
    for c in cands or []:
        sc = _score_candidate(c, type_hint, title, year)
        if sc is not None and sc > best_score:
            best, best_score = c["id"], sc
    return best


def resolve_qid(title, type_hint, year=None, limit=20):
    """Title -> Wikidata QID, or None when nothing is a confident match.

    TWO STAGE, to keep a full-library re-resolve affordable. ~1,400 uncached
    searches at the polite 1 req/s cost about 2.5 hours, and this repo already has
    1,383 `limit=5` results on disk from earlier runs. So: re-score the CACHED
    5-result search first, and only pay for a `limit=20` fetch when those five hold
    no confident match.

    The wide search is still needed for the hard cases -- for "Batman" the top five
    hits are a Turkish province, a Turkish city and three video games, so the film
    is not among them however they are scored -- but those are the minority.
    """
    if not isinstance(title, str) or not title.strip():
        return None

    cached = _search(title, 5, cached_only=True)
    if cached is not None:
        hit = _best(cached, type_hint, title, year)
        if hit:
            return hit

    return _best(_search(title, limit), type_hint, title, year)


def get_claims(qid):
    if not qid:
        return {}
    url = (f"https://www.wikidata.org/w/api.php?action=wbgetentities&ids={qid}"
           f"&props=claims&format=json")
    res = _get(url)
    ent = res.get("entities", {}).get(qid, {})
    return ent.get("claims", {})


def _claim_qids(claims, prop):
    out = []
    for c in claims.get(prop, []):
        try:
            dv = c["mainsnak"]["datavalue"]["value"]
            if isinstance(dv, dict) and "id" in dv:
                out.append(dv["id"])
        except (KeyError, TypeError):
            continue
    return out


def load_library():
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    items = []
    for media, (path, title_col, creators, hint) in DOMAINS.items():
        if not Path(path).exists():
            print(f"  skip {media}: {path} missing")
            continue
        enr = pd.read_csv(path)
        uni_slice = uni[uni.media_type == media].reset_index(drop=True)
        if len(enr) != len(uni_slice):
            print(f"  ⚠️ {media}: enriched rows {len(enr)} != unified slice {len(uni_slice)} -- positional map skipped")
            continue
        for i, row in enr.reset_index(drop=True).iterrows():
            items.append({
                "source_id": int(uni_slice.loc[i, "source_id"]),
                "media_type": media,
                "title": str(row.get(title_col, "")),
                "type_hint": hint,
            })
    return pd.DataFrame(items)


def build(limit=None, movie_cap=None):
    lib = load_library()
    if limit:
        lib = lib.groupby("media_type").head(limit).reset_index(drop=True)
    if movie_cap is not None:
        # Pilot: resolve the small domains fully, cap movies (the bulk) for tractable
        # wall-clock. Cross-domain adaptations concentrate in books/games/TV anyway.
        lib = pd.concat([lib[lib.media_type != "movie"],
                         lib[lib.media_type == "movie"].head(movie_cap)]).reset_index(drop=True)
    print(f"Resolving {len(lib)} items to Wikidata QIDs "
          f"(movies capped at {movie_cap})..." if movie_cap else
          f"Resolving {len(lib)} items to Wikidata QIDs...")

    qid_of, claims_of = {}, {}
    for _, it in lib.iterrows():
        qid = resolve_qid(it["title"], it["type_hint"])
        qid_of[it["source_id"]] = qid
        claims_of[it["source_id"]] = get_claims(qid) if qid else {}

    # Reverse index: QID -> library source_ids (for adaptation detection).
    sid_by_qid = {}
    for sid, q in qid_of.items():
        if q:
            sid_by_qid.setdefault(q, []).append(sid)

    by_sid = {r["source_id"]: r for _, r in lib.iterrows()}
    links = []

    def add(a, b, ltype, conf):
        if a == b:
            return
        ra, rb = by_sid[a], by_sid[b]
        if ra["media_type"] == rb["media_type"] and ltype != "same_franchise":
            return  # bridges are cross-domain
        lo, hi = sorted([a, b])
        links.append({"source_id_a": lo, "source_id_b": hi, "link_type": ltype,
                      "confidence": conf, "title_a": by_sid[lo]["title"],
                      "media_a": by_sid[lo]["media_type"], "title_b": by_sid[hi]["title"],
                      "media_b": by_sid[hi]["media_type"]})

    # adaptation: A based-on QID that is another library item
    for sid, claims in claims_of.items():
        for prop in ADAPT_PROPS:
            for tgt_qid in _claim_qids(claims, prop):
                for other_sid in sid_by_qid.get(tgt_qid, []):
                    add(sid, other_sid, "adaptation", 0.95)

    # shared_creator: same creator QID across domains
    creators_by_sid = {sid: set(q for p in CREATOR_PROPS for q in _claim_qids(c, p))
                       for sid, c in claims_of.items()}
    sids = list(creators_by_sid)
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = sids[i], sids[j]
            shared = creators_by_sid[a] & creators_by_sid[b]
            if shared and by_sid[a]["media_type"] != by_sid[b]["media_type"]:
                add(a, b, "shared_creator", 0.8)

    # same_franchise: same series QID (any domain pair)
    series_by_sid = {sid: set(_claim_qids(c, SERIES_PROP)) for sid, c in claims_of.items()}
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = sids[i], sids[j]
            if series_by_sid[a] & series_by_sid[b]:
                add(a, b, "same_franchise", 0.85)

    out = pd.DataFrame(links).drop_duplicates(subset=["source_id_a", "source_id_b", "link_type"])
    out_path = config.PROCESSED_DIR / "entity_links.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ {len(out)} links -> {out_path}")
    if len(out):
        print(out["link_type"].value_counts().to_string())
        print("\nFor manual verification (Task 3.1): eyeball the 'adaptation' rows below.")
        print(out[out.link_type == "adaptation"][["title_a", "media_a", "title_b", "media_b"]].to_string(index=False))
    return out


if __name__ == "__main__":
    # Default: pilot (movies capped at 300; small domains full). `full` resolves all.
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        build()
    else:
        build(movie_cap=300)
