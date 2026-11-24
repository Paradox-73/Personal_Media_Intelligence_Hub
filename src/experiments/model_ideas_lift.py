"""
Which of the user's model ideas lift the n=1 movie rating model, and survive serving?

Same harness as letterboxd_lift.py (XGBoost, repeated 5x5 CV, skill vs predict-the-mean
and Spearman of out-of-fold predictions, paired per-repeat deltas against the deployed
FULL feature set). Every candidate is built the way process_features would build it:
the out-of-fold encodings use src.common.taste_features so a film's own rating never
reaches its own feature, and the crowd numerics come from movie_crowd_extras.

Ideas:
  1 lb_avg_rating + log(rating count)                 (serve: Letterboxd film page)
  2 theme target-encoding / multi-hot                 (serve: theme names on that page)
  3 my mean rating over Letterboxd's similar films    (serve: BLOCKED, Cloudflare)
    vs the TMDB neighbour means already in FULL      (serve: tmdb_neighbors cache)
  4 my mean rating over the film's TMDB collection    (serve: collection_name)
  5 lb_avg_rating - imdb_rating/2                     (serve: both available)
  6 rating curve (std + star shares)                  (serve: only via a local cache,
    so also scored with the curve masked at test time, the realistic serve case)
  7 first release type / premiere-to-wide days        (serve: TMDB /release_dates)
Output: reports/model_ideas_lift.json (+ console table)
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common import taste_features as tf
from src.experiments.text_vs_crowd import _load, _xgb, _skill
from src.experiments.letterboxd_lift import N_REPEATS, _summary, _paired
from src.movies.feature_engineering import TE_PRIOR, movie_crowd_extras
from src.movies.ingestion import _cached, TMDB_KEY
from src.movies.letterboxd_details import SHARE_COLS

CURVE = ["lb_rating_std"] + SHARE_COLS
MIN_THEME_FILMS = 20
# Ideas 2 and 7 measured ~0 on top of the Letterboxd block, so their serve-side code was
# removed; the experiment keeps its own copies so the table can be reproduced.
RELEASE_CACHE = config.CACHE_DIR / "tmdb_release_types.json"


def lb_themes_list(cell):
    """'; '-joined because the theme names contain commas."""
    return [t.strip() for t in str(cell).split(";") if t.strip()] if isinstance(cell, str) else []


def release_types(tmdb_id):
    """[[type, 'YYYY-MM-DD'], ...] date-sorted over every country, from TMDB
    /release_dates (1 premiere, 2 limited, 3 theatrical, 4 digital, 5 physical, 6 TV)."""
    import requests

    def fetch(key):
        try:
            j = requests.get(f"https://api.themoviedb.org/3/movie/{key}/release_dates",
                             params={"api_key": TMDB_KEY}, timeout=8).json()
        except (requests.RequestException, ValueError):
            return None
        if "results" not in j:
            return None
        return sorted([d["type"], d["release_date"][:10]]
                      for c in j["results"] for d in c["release_dates"] if d.get("release_date"))
    return _cached(RELEASE_CACHE, tmdb_id, fetch)


def release_type_cols(rts):
    """first_release_type and days from the first date to the first wide theatrical one."""
    first, days = [], []
    for rt in rts:
        wide = next((d for t, d in rt if t == 3), None) if rt else None
        first.append(rt[0][0] if rt else np.nan)
        days.append((pd.Timestamp(wide) - pd.Timestamp(rt[0][1])).days if wide else np.nan)
    return first, days


def _rated():
    """Training matrix + the enriched rows it was built from (positional join, guarded)."""
    X, y = _load("movie")
    e = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
    rated = e[e["user_rating"].notna()].reset_index(drop=True)
    if len(rated) < len(X):
        sys.exit(f"ABORT: {len(rated)} rated enriched rows < {len(X)} training rows")
    rated = rated.iloc[:len(X)].copy()
    ur = pd.to_numeric(rated["user_rating"], errors="coerce").values
    bad = np.flatnonzero(~np.isclose(ur, y))
    if len(bad):
        sys.exit(f"ABORT: positional join broken at {len(bad)} rows (first {bad[:5].tolist()})")
    return X, y, rated


def _candidates(y, rated):
    """Every candidate column, built as process_features would build it."""
    C = movie_crowd_extras(rated)
    C["first_release_type"], C["premiere_to_wide_days"] = release_type_cols(
        [release_types(t) for t in rated["tmdb_id"]])
    C[CURVE] = rated[CURVE].apply(pd.to_numeric, errors="coerce")

    themes = rated["lb_themes"].apply(lb_themes_list).tolist()
    enc, sup, _ = tf.oof_entity_encode(themes, y, m=TE_PRIOR)
    C["lb_themes_encoded"], C["lb_themes_n"] = enc, sup
    counts = pd.Series([t for ts in themes for t in ts]).value_counts()
    hot = counts[counts >= MIN_THEME_FILMS].index
    for t in hot:
        C[f"theme_{tf.entity_key(t)}"] = [float(t in ts) for ts in themes]

    coll = [[c] if isinstance(c, str) and c.strip() else [] for c in rated["collection_name"]]
    enc, sup, _ = tf.oof_entity_encode(coll, y, m=TE_PRIOR)
    C["collection_name_encoded"], C["collection_name_n"] = enc, sup

    url_to_id = {u: int(t) for u, t in zip(rated["lb_film_url"], rated["tmdb_id"])
                 if isinstance(u, str) and pd.notna(t)}
    ids = pd.to_numeric(rated["tmdb_id"], errors="coerce").tolist()
    sim = [[url_to_id[u] for u in str(s).split("; ") if u in url_to_id] if isinstance(s, str) else []
           for s in rated["lb_similar_urls"]]
    enc, sup, _ = tf.oof_neighbor_encode(ids, sim, y)
    C["lbsim_mean"], C["lbsim_n"] = enc, sup
    return C, [f"theme_{tf.entity_key(t)}" for t in hot]


def _per_repeat(X, y, cols, mask_test=(), mask_train_frac=0.0):
    """OOF predictions per repeat. mask_test: columns set to NaN on the test side (the
    serve case for data only a local cache holds); mask_train_frac: share of training
    rows given the same NaN so the model learns to live without them."""
    rkf = RepeatedKFold(n_splits=5, n_repeats=N_REPEATS, random_state=0)
    P = np.zeros((N_REPEATS, len(y)))
    rng = np.random.default_rng(0)
    for i, (tr, te) in enumerate(rkf.split(X)):
        Xtr, Xte = X.iloc[tr][cols].copy(), X.iloc[te][cols].copy()
        if mask_test:
            Xte[list(mask_test)] = np.nan
            if mask_train_frac:
                Xtr.loc[rng.random(len(tr)) < mask_train_frac, list(mask_test)] = np.nan
        m = _xgb(); m.fit(Xtr, y[tr])
        P[i // 5, te] = m.predict(Xte)
    return P


def main():
    X, y, rated = _rated()
    C, hot_cols = _candidates(y, rated)
    # training_features.csv now carries the kept features; baseline = the set before them
    X = pd.concat([X.drop(columns=[c for c in X.columns if c.startswith(('lb_', 'collection_name_'))] + [c for c in C.columns if c in X.columns]), C], axis=1)
    full = [c for c in X.columns if c not in C.columns]
    nb = [c for c in full if c.startswith("nb_")]
    i1 = ["lb_avg_rating", "lb_count_log"]
    i7 = ["first_release_type", "premiere_to_wide_days"]
    configs = {
        "full": (full, {}),
        "1_lb_avg+count": (full + i1, {}),
        "2a_themes_te": (full + ["lb_themes_encoded", "lb_themes_n"], {}),
        "2b_themes_hot": (full + hot_cols, {}),
        "3a_full-nb_tmdb": ([c for c in full if c not in nb], {}),
        "3b_lbsim": (full + ["lbsim_mean", "lbsim_n"], {}),
        "3c_lbsim_instead_of_nb": ([c for c in full if c not in nb] + ["lbsim_mean", "lbsim_n"], {}),
        "4_collection": (full + ["collection_name_encoded", "collection_name_n"], {}),
        "5_lb_gap": (full + ["lb_gap"], {}),
        "5b_lb_avg+count+gap": (full + i1 + ["lb_gap"], {}),
        "6a_curve": (full + i1 + CURVE, {}),
        "6b_curve_masked_at_serve": (full + i1 + CURVE, {"mask_test": CURVE}),
        "6c_curve_half_masked_train": (full + i1 + CURVE, {"mask_test": CURVE, "mask_train_frac": 0.5}),
        "7_release_type": (full + i7, {}),
    }
    res = {k: _summary(y, _per_repeat(X, y, cols, **kw)) for k, (cols, kw) in configs.items()}
    paired = {k: {m: _paired(res["full"], res[k], f"per_repeat_{m}") for m in ("skill", "spearman")}
              for k in configs if k != "full"}

    keep = [k for k in configs if k != "full" and paired[k]["skill"]["improved"] >= 3
            and k not in ("3a_full-nb_tmdb", "3b_lbsim", "3c_lbsim_instead_of_nb",
                          "6a_curve", "6b_curve_masked_at_serve", "6c_curve_half_masked_train")]
    combined = list(dict.fromkeys(full + [c for k in keep for c in configs[k][0]]))
    res["combined"] = _summary(y, _per_repeat(X, y, combined))
    paired["combined"] = {m: _paired(res["full"], res["combined"], f"per_repeat_{m}")
                          for m in ("skill", "spearman")}
    configs["combined"] = (combined, {})

    print(f"n={len(y)}  lb_avg present {int(X['lb_avg_rating'].notna().sum())}, "
          f"release_types present {int(X['first_release_type'].notna().sum())}, "
          f"themes present {int((X['lb_themes_n'] > 0).sum())}, hot themes {len(hot_cols)}, "
          f"lbsim>0 {int((X['lbsim_n'] > 0).sum())}, collection>0 {int((X['collection_name_n'] > 0).sum())}")
    print(f"{'config':30} {'#feat':>5} {'skill':>7} {'spearman':>8} | paired skill (mean ± sd, improved)   paired spearman")
    for k, (cols, _) in configs.items():
        r = res[k]
        line = f"{k:30} {len(cols):>5} {r['skill']:>7.4f} {r['spearman']:>8.4f}"
        if k in paired:
            s, p = paired[k]["skill"], paired[k]["spearman"]
            line += (f" | {s['mean']:+.4f} ± {s['sd']:.4f} ({s['improved']}/{s['of']})"
                     f"   {p['mean']:+.4f} ± {p['sd']:.4f} ({p['improved']}/{p['of']})")
        print(line)
    print("kept for combined:", keep)
    out = {"n": len(y), "configs": {k: v[0] for k, v in configs.items()}, "results": res,
           "paired": paired, "kept": keep}
    Path("reports/model_ideas_lift.json").write_text(json.dumps(out, indent=2))
    print("Wrote reports/model_ideas_lift.json")


if __name__ == "__main__":
    main()
