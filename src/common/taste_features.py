"""Shared taste-signal features for the per-domain rating models.

Background
----------
An audit on 2026-08-15 found the movie matrix was doing nothing personal: the 12
crowd/metadata columns alone scored MAE 0.4807 / R2 0.5592 while the full 85
columns scored 0.4821 / 0.5479. The 73 "personal" columns (two target encodings,
40 PCA text dims, 21 genre one-hots) added nothing and fed the overfit gap.

Two causes, both fixed here.

1. **The crew was never featurised.** `enrich_extended.py` fetches composer,
   cinematographer, editor and producer out of the TMDB /credits payload, and
   movies never even encoded `writer`. Leave-one-out entity means against the
   user's ratings: writer 0.563, director 0.460, producer 0.424, cinematographer
   0.418, composer 0.390, editor 0.385, production company 0.352 (country 0.017 --
   excluded, it carries nothing). Enrichment writes columns into enriched_data.csv
   and feature engineering has a hardcoded feature list; nothing connected them.

2. **The encodings that did exist were smoothed flat.** The old default was m=10
   on a library where the median entity has 1-2 credits, so
   (n*mean + m*global) / (n + m) sat on top of the global mean (observed std
   0.05-0.13). Measured sweep, movies, 10-fold, inner-OOF:
   m=0 0.4769 | m=1 0.4732 | **m=2 0.4694** | m=3 0.4765 | m=5 0.4713 |
   m=10 0.4802 | m=20 0.4769. m=10 is indistinguishable from having no feature.

Plus a genuinely new signal: TMDB's item-item neighbour graph
(`src/common/tmdb_neighbors.py`). "My mean rating over the neighbours I have
already rated" is the only collaborative feature available in this project --
everything else restates how good the crowd thinks the item is.

Leakage
-------
`oof_entity_encode` and `oof_neighbor_encode` both build the training column with
an inner KFold so no row's own rating reaches its own feature. That inner OOF is
load-bearing: a control that dropped it scored MAE 0.6243 against 0.5075 for
simply deleting the columns. Each function also returns the full-library map for
inference, so the serve path applies the identical keys -- the train/serve skew
that has bitten this repo twice.
"""
import ast
import re
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

DEFAULT_M = 2
DEFAULT_SPLITS = 5
DEFAULT_SEED = 42


_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def entity_key(name):
    """Canonical dictionary key for one entity name.

    The same person is spelled differently by different sources, and these are dict
    lookups -- any difference silently splits one creator into two and the link
    never fires. Real cases in this library:

        TMDB / TVMaze : "Bryan Lee O'Malley"          (straight apostrophe)
        Hardcover CSV : "Bryan Lee O�Malley"       (apostrophe lost to a bad
                                                       encoding round-trip)
        elsewhere     : "Alejandro Gonzalez Inarritu" vs "Alejandro Gonzalez Inarritu"
                                                       (diacritics present/absent)

    Rather than guess what a corrupted byte used to be, strip to lowercase
    alphanumerics and single spaces after removing diacritics. All three spellings
    above collapse to one key.
    """
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(_NON_ALNUM.sub("", s.lower()).split())


def _keys(vals):
    """entity_key() over a sequence, dropping anything that normalises to empty."""
    return [k for k in (entity_key(v) for v in vals) if k]


def parse_people(val, limit=None):
    """Parse a multi-value cell. The CSVs mix "['A', 'B']" and "A, B" for the same
    column (director: 967 bracket / 95 comma; composer: 24 / 975), so both shapes
    must produce identical keys or the map lookups silently miss."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, (list, tuple)):
        out = _keys(val)
        return out[:limit] if limit else out
    s = str(val).strip()
    if not s or s.upper() in ("N/A", "NAN", "NONE", "[]"):
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            out = _keys(ast.literal_eval(s))
            return out[:limit] if limit else out
        except (ValueError, SyntaxError):
            pass
    out = _keys(s.split(","))
    return out[:limit] if limit else out


# --- entity target encoding ------------------------------------------------------

def _fit_entity_map(lists, y, idx, m, gmean):
    """Unrated rows (NaN target) still need a feature value, but must never
    contribute to the map -- Games keeps 'Incomplete' rows with a NaN rating, and
    adding one would turn the whole entity mean into NaN."""
    tot, cnt = defaultdict(float), defaultdict(int)
    for i in idx:
        if np.isnan(y[i]):
            continue
        for e in lists[i]:
            tot[e] += y[i]
            cnt[e] += 1
    mapping = {e: (tot[e] + m * gmean) / (cnt[e] + m) for e in tot}
    return mapping, dict(cnt)


def apply_entity_map(cell, mapping, counts, gmean, limit=None):
    """Encode ONE cell against a fitted map. Used by the inference path."""
    items = parse_people(cell, limit)
    vals = [mapping[e] for e in items if e in mapping]
    if not vals:
        return float(gmean), 0.0
    sup = float(np.mean([counts.get(e, 0) for e in items])) if counts else 0.0
    return float(np.mean(vals)), sup


def oof_entity_encode(cells, y, m=DEFAULT_M, limit=None,
                      n_splits=DEFAULT_SPLITS, seed=DEFAULT_SEED):
    """Return (encoded, support, artifact) for a multi-value column.

    `encoded` is built with an inner KFold so a row never sees its own rating.
    `artifact` = {"map", "counts", "mean", "m", "limit"} -- the full-library map
    for inference.
    """
    y = np.asarray(y, dtype=float)
    lists = [parse_people(v, limit) for v in cells]
    n = len(lists)
    gmean = float(np.nanmean(y))
    enc = np.full(n, gmean)
    sup = np.zeros(n)

    splitter = KFold(n_splits=min(n_splits, max(2, n)), shuffle=True, random_state=seed)
    for tr, va in splitter.split(np.arange(n)):
        mapping, counts = _fit_entity_map(lists, y, tr, m, gmean)
        for i in va:
            enc[i], sup[i] = apply_entity_map(cells[i] if not isinstance(cells, pd.Series)
                                              else cells.iloc[i],
                                              mapping, counts, gmean, limit)
    full_map, full_counts = _fit_entity_map(lists, y, np.arange(n), m, gmean)
    return enc, sup, {"map": full_map, "counts": full_counts, "mean": gmean,
                      "m": m, "limit": limit}


# --- TMDB neighbour graph --------------------------------------------------------

def _fit_neighbor_index(item_ids, y, idx):
    """id -> rating, restricted to rows in `idx` (the training side)."""
    out = {}
    for i in idx:
        tid = item_ids[i]
        if tid is None or (isinstance(tid, float) and np.isnan(tid)) or np.isnan(y[i]):
            continue
        out[int(tid)] = y[i]
    return out


def apply_neighbor_map(neighbor_ids, rated_index, gmean, self_id=None):
    """Mean of my ratings over the neighbours I have already rated."""
    if not neighbor_ids:
        return float(gmean), 0.0
    vals = [rated_index[int(t)] for t in neighbor_ids
            if int(t) in rated_index and (self_id is None or int(t) != int(self_id))]
    if not vals:
        return float(gmean), 0.0
    return float(np.mean(vals)), float(len(vals))


def oof_neighbor_encode(item_ids, neighbor_lists, y,
                        n_splits=DEFAULT_SPLITS, seed=DEFAULT_SEED):
    """Return (encoded, support, artifact) for one neighbour endpoint.

    Inner KFold again: within a fold the neighbour ratings come only from the
    training rows, so an item cannot be scored using a neighbour whose rating is
    itself held out with it.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    gmean = float(np.nanmean(y))
    enc = np.full(n, gmean)
    sup = np.zeros(n)

    splitter = KFold(n_splits=min(n_splits, max(2, n)), shuffle=True, random_state=seed)
    for tr, va in splitter.split(np.arange(n)):
        index = _fit_neighbor_index(item_ids, y, tr)
        for i in va:
            enc[i], sup[i] = apply_neighbor_map(neighbor_lists[i], index, gmean,
                                                self_id=item_ids[i])
    full_index = _fit_neighbor_index(item_ids, y, np.arange(n))
    return enc, sup, {"index": full_index, "mean": gmean}


def load_neighbor_lists(cache, kind, tmdb_ids, endpoint):
    """Pull one endpoint's neighbour list per row out of the fetch cache."""
    out = []
    for t in tmdb_ids:
        if t is None or (isinstance(t, float) and np.isnan(t)):
            out.append([])
            continue
        rec = cache.get(f"{kind}:{int(t)}") or {}
        out.append(list(rec.get(endpoint) or []))
    return out


# --- cheap numerics that were fetched and never used -----------------------------

def movie_numeric_extras(df):
    """budget/revenue/ROI/vote_count/release-month/age-at-rating.

    All present in enriched_data.csv at 90-100% coverage and none of them reached
    the model. Worth -0.008 MAE on top of the crew block.
    """
    out = pd.DataFrame(index=df.index)
    rev = pd.to_numeric(df.get("revenue"), errors="coerce")
    bud = pd.to_numeric(df.get("budget"), errors="coerce")
    out["budget_log"] = np.log1p(bud.clip(lower=0)).fillna(0)
    out["revenue_log"] = np.log1p(rev.clip(lower=0)).fillna(0)
    # NOTE: a revenue/budget ROI column was tried and REMOVED. Even clipped to
    # [0, 50] it stays heavy-tailed with a -1 sentinel for the 10% of films with no
    # budget, and it cost more than it paid on the registry folds with the deployed
    # movie ensemble: all-new 0.5108 MAE / R2 0.4820 vs all-new-minus-roi 0.5094 /
    # 0.4873. budget_log and revenue_log carry the same information more stably.
    out["vote_count_log"] = np.log1p(
        pd.to_numeric(df.get("vote_count"), errors="coerce")).fillna(0)
    rel = pd.to_datetime(df.get("released"), errors="coerce", format="mixed")
    out["rel_month"] = rel.dt.month.fillna(0)
    rat = pd.to_datetime(df.get("rating_date"), errors="coerce", format="mixed")
    out["age_at_rating"] = ((rat - rel).dt.days / 365.25).clip(-1, 100).fillna(-1)
    return out


def show_numeric_extras(df):
    """TV's missing crowd block.

    Shows carry only imdb_rating (rho 0.551) and vote_average -- no metascore, no
    Rotten Tomatoes, no critic_avg_5 -- which is the likeliest reason TV has the
    worst val R2 (0.26) and the widest train/val gap (+0.62) in the project.
    tvdb_score, tvmaze_rating and tvmaze_weight are already in the enriched CSV at
    95-100% coverage and were never read.
    """
    out = pd.DataFrame(index=df.index)
    for src, dst in (("tvdb_score", "tvdb_score"),
                     ("tvmaze_rating", "tvmaze_rating"),
                     ("tvmaze_weight", "tvmaze_weight"),
                     ("tvdb_avg_runtime", "tvdb_avg_runtime")):
        v = pd.to_numeric(df.get(src), errors="coerce")
        out[dst] = v.fillna(v.median() if v.notna().any() else 0)
    # A single consensus column, mirroring movies' critic_avg_5. Each source is
    # rescaled to the 0-5 star grid before averaging so they are commensurable.
    parts = []
    imdb = pd.to_numeric(df.get("imdb_rating"), errors="coerce") / 2.0
    tmdb = pd.to_numeric(df.get("vote_average"), errors="coerce") / 2.0
    maze = pd.to_numeric(df.get("tvmaze_rating"), errors="coerce") / 2.0
    for p in (imdb, tmdb, maze):
        if p is not None and p.notna().any():
            parts.append(p)
    if parts:
        crit = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
        out["critic_avg_5"] = crit.fillna(crit.median())
    first = pd.to_datetime(df.get("first_air_date"), errors="coerce", format="mixed")
    out["air_month"] = first.dt.month.fillna(0)
    ep = pd.to_numeric(df.get("number_of_episodes"), errors="coerce")
    se = pd.to_numeric(df.get("number_of_seasons"), errors="coerce")
    out["eps_per_season"] = (ep / se.replace(0, np.nan)).clip(0, 100).fillna(0)
    return out
