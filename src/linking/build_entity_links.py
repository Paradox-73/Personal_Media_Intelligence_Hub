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
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

UA = {"User-Agent": "PersonalMediaIntelligenceHub/1.0 (research; contact: local)"}
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


def _get(url):
    key = CACHE_DIR / (urllib.parse.quote(url, safe="") [:180] + ".json")
    if key.exists():
        return json.loads(key.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request(url, headers=UA)
        data = json.loads(urllib.request.urlopen(req, timeout=15).read())
    except Exception as e:
        data = {"__error__": repr(e)}
    key.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.15)
    return data


def resolve_qid(title, type_hint):
    if not isinstance(title, str) or not title.strip():
        return None
    q = urllib.parse.quote(title)
    url = (f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={q}"
           f"&language=en&format=json&limit=5")
    res = _get(url)
    cands = res.get("search", [])
    if not cands:
        return None
    # Prefer a candidate whose description matches the media type hint.
    for c in cands:
        desc = (c.get("description") or "").lower()
        if type_hint.split()[0] in desc or type_hint in desc:
            return c["id"]
    return cands[0]["id"]


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
