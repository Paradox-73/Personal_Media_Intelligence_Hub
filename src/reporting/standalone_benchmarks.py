"""
Standalone per-domain benchmarks on the FROZEN registry folds.

Phase 0 bug fix: the previous writer published the *unified* model's per-domain
OOF slice as if it were the standalone domain models (benchmarks == slices, with
N inflated 5x by counting OOF rows instead of items). This module regenerates
genuine standalone-model OOF on the same 50 registry folds, deduplicated to one
row per item, so the benchmark table reports the local models -- not the unified
slice.

Design decisions (documented so the docs can cite them):
  * Each domain runs its OWN deployed regressor architecture:
      - movie : XGBoost (reg:absoluteerror, balanced sample weights)   -> "XGBoost"
      - tv    : Simplex-weighted stack of XGB + CatBoost + SVR          -> "Simplex-Stack"
      - game  : local SVR(rbf), NO unified prior (distillation DROPPED) -> "Local SVR"
      - book  : local SVR(rbf), NO unified prior (distillation DROPPED) -> "Local SVR"
  * Standalone feature CSVs carry no source_id, so each row is mapped to its
    registry global_id positionally against the unified domain slice (the row
    order is identical and every standalone item is rated; validated by an exact
    target-value alignment check, same pattern as distillation_ablation.py).
  * Each item lands in 5 test folds (5 CV repeats). Predictions are averaged
    across repeats per item (mean of raw predictions), THEN rounded to the
    nearest 0.5 once, THEN scored. N is the unique-item count.
  * Target-derived columns (entity target encodings, TMDB neighbour means) are
    refit inside every fold from the training rows (`reencode_fold`). The CSV
    versions come from one inner KFold over the whole library, so a training
    row's encoding carried test ratings into every registry fold.
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import catboost as ctb
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import Parallel, delayed

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common import taste_features as tf
from src.common.ensemble import model_threads
from src.common.parsing import canon_genres, people_key

DOMAIN_PATHS = {
    "movie": config.TRAINING_DATA_PATH,
    "tv": config.TV_SHOWS_TRAINING_DATA_PATH,
    "game": config.GAMES_TRAINING_DATA_PATH,
    "book": config.BOOKS_TRAINING_DATA_PATH,
}
DOMAIN_MODEL_LABEL = {
    "movie": "XGBoost",
    "tv": "Simplex-Stack",
    "game": "Local SVR",
    "book": "Local SVR",
}
DROP_COLS = ["target_reg", "target_class", "target_ordinal", "source_id", "global_id"]


def _round_half(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def skill(y_true, y_pred, baseline):
    """1 - MAE(model) / MAE(constant baseline). Callers pass y_pred already rounded if they want it."""
    base = mean_absolute_error(y_true, np.full_like(y_true, baseline, dtype=float))
    if base == 0:
        return 0.0
    return 1.0 - mean_absolute_error(y_true, y_pred) / base


def load_registry():
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        return json.load(f)


def registry_folds(df, registry, fold_ids=range(50)):
    """Yield (fold_id, test_mask) for every registry fold with at least one test row in df."""
    for fold_id in fold_ids:
        test_mask = df["global_id"].apply(lambda g: fold_id in registry.get(g, []))
        if test_mask.any():
            yield fold_id, test_mask


# Each fold trains a few small models (~1k rows) that cannot keep 12 threads busy, so
# whole folds run side by side instead. FOLD_JOBS=1 restores the old serial loop.
FOLD_JOBS = int(os.getenv("FOLD_JOBS", max(1, (os.cpu_count() or 2) // 2)))


def _in_worker(fn, args, calibrate):
    # Workers outlive set_calibration() calls in the parent, so the flag travels with the task.
    if calibrate is not None:
        os.environ["CALIBRATE"] = calibrate
    return fn(*args)


def map_folds(fn, tasks, n_jobs=None):
    """[fn(*t) for t in tasks], run across FOLD_JOBS processes; results keep task order.

    joblib caps each worker's OpenMP threads (XGBoost) at cpu_count // n_jobs; CatBoost
    ignores that cap, so callers pass thread_count=model_threads() to it.
    """
    n_jobs = n_jobs or FOLD_JOBS
    if n_jobs == 1:
        return [fn(*t) for t in tasks]
    cal = os.environ.get("CALIBRATE")
    return Parallel(n_jobs=n_jobs)(delayed(_in_worker)(fn, t, cal) for t in tasks)


def set_calibration(domain):
    """Per-domain rank calibration, as the benchmark runs it: off for CALIBRATE_SKIP (default tv)."""
    import os
    os.environ["CALIBRATE"] = "0" if domain in os.getenv("CALIBRATE_SKIP", "tv").split(",") else "1"


def _metrics(y_true, y_pred_raw):
    yp = _round_half(np.asarray(y_pred_raw))
    yt = np.asarray(y_true)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "R2": float(r2_score(yt, yp)),
        "Acc": float((np.abs(yt - yp) <= 0.5).mean() * 100),
    }


# ---- per-domain model builders -------------------------------------------------

# depth 8 / 600 trees, not the old depth 6 / 200. Movies measured ~95% bias-dominated
# (reports/bias_variance.json), which is exactly the regime where more capacity helps.
# Chosen by NESTED CV -- setting picked on an inner 3-fold of each training fold and
# scored on the untouched outer fold, so this is not winner's curse:
#     frozen depth6/n300   MAE 0.4746  R2 0.5628
#     nested-tuned         MAE 0.4655  R2 0.5788   (modal choice depth8/n600, 15/30)
# See reports/capacity_tuning.json.
XGB_DEPTH, XGB_TREES = 8, 600


def _fit_predict_xgb(X_tr, y_tr, X_te):
    w = _balanced_weights(y_tr)
    m = xgb.XGBRegressor(
        n_estimators=XGB_TREES, learning_rate=0.03, max_depth=XGB_DEPTH,
        subsample=0.8, colsample_bytree=0.8,
        objective="reg:absoluteerror", random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=w)
    return _maybe_calibrate(m.predict(X_tr), y_tr, m.predict(X_te))


# Must track src/{books,games}/model_trainer.py SVR_C -- this module exists to
# benchmark the DEPLOYED local model, so a divergence here would republish numbers
# for a model that is not shipped (the exact failure the movie RidgeCV-vs-mean
# mismatch caused). Nested-CV tuned; see reports/capacity_tuning.json.
SVR_C = 3.0


def _fit_predict_svr(X_tr, y_tr, X_te):
    m = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=SVR_C, gamma="scale", epsilon=0.1)),
    ])
    m.fit(X_tr, y_tr, svr__sample_weight=_balanced_weights(y_tr))
    return _maybe_calibrate(m.predict(X_tr), y_tr, m.predict(X_te))


def _fit_predict_simplex_stack(X_tr, y_tr, X_te):
    """XGB + CatBoost + SVR base models, simplex-weighted by inner 5-fold OOF MAE."""
    base = {
        "xgb": lambda: xgb.XGBRegressor(n_estimators=150, learning_rate=0.03,
                                        max_depth=5, random_state=42),
        "cat": lambda: ctb.CatBoostRegressor(iterations=200, learning_rate=0.05, thread_count=model_threads(),
                                             depth=5, loss_function="MAE",
                                             verbose=0, random_seed=42),
        "svr": lambda: Pipeline([("scaler", StandardScaler()),
                                 ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1))]),
    }
    # Inner OOF to fit simplex weights
    inner = KFold(n_splits=3, shuffle=True, random_state=42)
    oof = {k: np.zeros(len(y_tr)) for k in base}
    y_arr = np.asarray(y_tr)
    X_arr = X_tr.reset_index(drop=True)
    for tr_idx, va_idx in inner.split(X_arr):
        for k, mk in base.items():
            mdl = mk()
            mdl.fit(X_arr.iloc[tr_idx], y_arr[tr_idx])
            oof[k][va_idx] = mdl.predict(X_arr.iloc[va_idx])
    Z = np.column_stack([oof[k] for k in base])

    def obj(w):
        return mean_absolute_error(y_arr, Z @ w)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    bounds = [(0, 1)] * len(base)
    w0 = np.ones(len(base)) / len(base)
    weights = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons).x

    # Refit base models on full train, blend train+test predictions
    w_full = _balanced_weights(y_tr)
    test_preds, train_preds = [], []
    for k, mk in base.items():
        mdl = mk()
        if k == "svr":
            mdl.fit(X_tr, y_tr, svr__sample_weight=w_full)
        else:
            mdl.fit(X_tr, y_tr, sample_weight=w_full)
        test_preds.append(mdl.predict(X_te))
        train_preds.append(mdl.predict(X_tr))
    blend_te = np.column_stack(test_preds) @ weights
    blend_tr = np.column_stack(train_preds) @ weights
    return _maybe_calibrate(blend_tr, y_tr, blend_te)


DOMAIN_FIT = {
    "movie": _fit_predict_xgb,
    "tv": _fit_predict_simplex_stack,
    "game": _fit_predict_svr,
    "book": _fit_predict_svr,
}


def _balanced_weights(y):
    import os
    from sklearn.utils.class_weight import compute_sample_weight
    classes = (np.asarray(y) * 2).round() / 2
    w = compute_sample_weight(class_weight="balanced", y=classes)
    # EDGE_BOOST (env, default 1.0 = OFF). Superseded by quantile calibration below; kept
    # as a toggle for the edge-weight ablation only.
    eb = float(os.getenv("EDGE_BOOST", "1.0"))
    edge = (classes <= 1.5) | (classes >= 4.5)
    return w * np.where(edge, eb, 1.0)


def _maybe_calibrate(tr_pred, y_tr, te_pred):
    """Leakage-safe rank/quantile calibration: map each test prediction's RANK (vs the
    train predictions) onto the empirical train-rating distribution, so the output
    distribution matches the real rating curve and the model can predict the full range
    (incl. 5.0). Ranking-preserving, so it only helps where the model ranks well. The map
    is fit on the TRAIN fold only -> no leakage. Toggle with CALIBRATE=0 for the baseline."""
    import os
    if os.getenv("CALIBRATE", "1") != "1":
        return te_pred
    sp = np.sort(np.asarray(tr_pred))
    if len(sp) < 2:
        return te_pred
    q = np.interp(np.asarray(te_pred), sp, np.linspace(0.0, 1.0, len(sp)))
    return np.quantile(np.asarray(y_tr), q)


# ---- driver --------------------------------------------------------------------

def _load_domain(domain, uni_df, registry):
    """Load a standalone domain frame and attach registry global_ids positionally."""
    loc = pd.read_csv(DOMAIN_PATHS[domain])
    uni_slice = uni_df[uni_df["media_type"] == domain].reset_index(drop=True)
    if len(uni_slice) != len(loc):
        raise ValueError(
            f"{domain}: standalone rows={len(loc)} != unified slice={len(uni_slice)}. "
            "Positional source_id mapping invalid -- re-run feature engineering."
        )
    loc = loc.reset_index(drop=True)
    loc["source_id"] = uni_slice["source_id"].values
    # Validate positional alignment with a NaN-SAFE check. The standalone Games
    # file keeps "Incomplete"-status rows with NaN target_reg (no rating); those
    # must not silently pass a plain abs-diff (NaN > tol is False). Require that
    # every RATED standalone row matches the unified slice target at its position.
    a = uni_slice["target_reg"].values.astype(float)
    b = loc["target_reg"].values.astype(float)
    rated = ~np.isnan(b)
    if not np.allclose(a[rated], b[rated], atol=1e-6):
        raise ValueError(f"{domain}: rated target alignment failed (positional map invalid).")
    n_unrated = int((~rated).sum())
    if n_unrated:
        print(f"      ({domain}: {n_unrated} unrated/Incomplete rows excluded -- no target)")
    loc["global_id"] = domain + "_" + loc["source_id"].astype(str)
    loc = loc[loc["target_reg"].notna()].copy()
    return loc


# ---- fold-safe target encodings ------------------------------------------------

RAW_PATHS = {
    "movie": (config.MOVIES_ENRICHED_DATA_PATH, "user_rating"),
    "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "user_rating"),
    "game": (config.GAMES_ENRICHED_DATA_PATH, "my_rating"),
    "book": (config.BOOKS_ENRICHED_DATA_PATH, "my_rating"),
}


def _encode_spec(domain):
    """(prior m, [(feature stem, raw column, names kept)]) for every entity encoding
    the domain's feature_engineering writes. Imported lazily: those modules load torch."""
    if domain == "movie":
        from src.movies.feature_engineering import CREW_ENCODE, TE_PRIOR
        return TE_PRIOR, [(c, c, lim) for c, lim in CREW_ENCODE] + [("dir_genre", "dir_genre_key", None)]
    if domain == "tv":
        from src.shows.feature_engineering import TE_PRIOR
        return TE_PRIOR, [("creator", "created_by", None), ("actors", "actors", None),
                          ("writer", "writer", None)]
    if domain == "game":
        from src.games.feature_engineering import GAME_ENCODE, TE_PRIOR
        return TE_PRIOR, [(c, c, lim) for c, lim in GAME_ENCODE]
    from src.books.feature_engineering import BOOK_ENCODE, TE_PRIOR
    return TE_PRIOR, [(c, c, lim) for c, lim in BOOK_ENCODE]


def load_domain_raw(domain, loc):
    """The enriched rows behind `loc`, in its row order. The feature builders keep the
    enriched order and drop only unrated rows, so the same filter lines them up;
    checked on count and target rather than assumed."""
    path, target = RAW_PATHS[domain]
    try:
        raw = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        raw = pd.read_csv(path, low_memory=False, encoding="latin1")
    y = pd.to_numeric(raw[target], errors="coerce")
    raw = raw[y.notna()].reset_index(drop=True)
    if len(raw) != len(loc) or not np.allclose(y.dropna().to_numpy(),
                                               loc["target_reg"].to_numpy(dtype=float), atol=1e-6):
        raise ValueError(f"{domain}: enriched rows do not line up with the feature rows")
    if domain == "movie":
        # The composite key feature_engineering encodes, pre-wrapped as a one-element
        # list so a two-director cell is not split on its comma.
        raw["dir_genre_key"] = [[people_key(d) + "_" + (canon_genres(g) or ["Unknown"])[0]]
                                for d, g in zip(raw["director"], raw["genre"])]
    return raw


def _tv_encode(cells_tr, y_tr, cells_te, m):
    """TV keeps its own encoder in shows/feature_engineering (raw names, unseen entity
    = global mean) rather than taste_features; refitting with taste_features moved
    16-51 of 175 rows by up to 0.23 stars, so the benchmark would have scored a
    feature the deployed model never sees."""
    from src.shows.feature_engineering import global_te_mapping, item_mean, parse_people, target_encode_list
    frame = pd.DataFrame({"L": [parse_people(c) for c in cells_tr], "y": y_tr})
    enc = target_encode_list(frame, "L", target="y", m=m).to_numpy()
    mapping, g = global_te_mapping(frame, "L", target="y", m=m)
    return enc, [item_mean(parse_people(c), mapping, g) for c in cells_te]


def reencode_fold(domain, loc, raw, test_mask, feat_cols):
    """(X_train, X_test) for one fold with every target-derived column refit on the
    training rows only: training rows get an inner-OOF encoding over the training
    fold, test rows the training-fold map. Same shape as
    unified_repeated_cv._reencode_entities for the unified matrix."""
    test_mask = np.asarray(test_mask, dtype=bool)
    tr, te = np.flatnonzero(~test_mask), np.flatnonzero(test_mask)
    X = loc[feat_cols].reset_index(drop=True).copy()
    y = loc["target_reg"].to_numpy(dtype=float)
    m, spec = _encode_spec(domain)
    done = set()
    for stem, raw_col, limit in spec:
        if f"{stem}_encoded" not in X.columns:
            continue
        cells = raw[raw_col].tolist()
        if domain == "tv":
            assert f"{stem}_n" not in X.columns, "TV's encoder writes no support column"
            enc, te_enc = _tv_encode([cells[i] for i in tr], y[tr], [cells[i] for i in te], m)
        else:
            enc, sup, art = tf.oof_entity_encode([cells[i] for i in tr], y[tr], m=m, limit=limit)
            pairs = [tf.apply_entity_map(cells[i], art["map"], art["counts"], art["mean"], limit) for i in te]
            te_enc = [e for e, _ in pairs]
            if f"{stem}_n" in X.columns:
                X.loc[tr, f"{stem}_n"], X.loc[te, f"{stem}_n"] = sup, [n for _, n in pairs]
        X.loc[tr, f"{stem}_encoded"], X.loc[te, f"{stem}_encoded"] = enc, te_enc
        done.add(f"{stem}_encoded")
    if "nb_similar_mean" in X.columns:
        from src.movies.feature_engineering import tf_neighbors_cache
        cache = tf_neighbors_cache()
        ids = pd.to_numeric(raw["tmdb_id"], errors="coerce").tolist()
        for ep in ("similar", "recommendations"):
            lists = tf.load_neighbor_lists(cache, domain, ids, ep)
            enc, sup, art = tf.oof_neighbor_encode([ids[i] for i in tr], [lists[i] for i in tr], y[tr])
            pairs = [tf.apply_neighbor_map(lists[i], art["index"], art["mean"], self_id=ids[i]) for i in te]
            X.loc[tr, f"nb_{ep}_mean"], X.loc[te, f"nb_{ep}_mean"] = enc, [e for e, _ in pairs]
            X.loc[tr, f"nb_{ep}_n"], X.loc[te, f"nb_{ep}_n"] = sup, [n for _, n in pairs]
    left = [c for c in X.columns if c.endswith("_encoded") and c not in done]
    assert not left, f"{domain}: target-derived columns left as built on the whole library: {left}"
    return X.iloc[tr], X.iloc[te]


def benchmarks_from_oof(oof_df):
    """Recompute benchmark rows from an existing per-item standalone OOF frame."""
    bms = []
    for domain in ["movie", "tv", "game", "book"]:
        d = oof_df[oof_df["media_type"] == domain]
        if d.empty:
            continue
        n = d["source_id"].nunique()
        bms.append({"Domain": domain, "N": int(n),
                    "Model": DOMAIN_MODEL_LABEL[domain],
                    **_metrics(d["target_reg"], d["pred"])})
    return bms


def generate_standalone_oof(uni_df, registry, oof_out_path="reports/standalone_oof.csv",
                            use_cache=False):
    if use_cache and Path(oof_out_path).exists():
        print(f"   [standalone] reusing cached per-item OOF <- {oof_out_path}")
        oof_df = pd.read_csv(oof_out_path)
        return benchmarks_from_oof(oof_df), oof_df
    return _generate_standalone_oof(uni_df, registry, oof_out_path)


def _generate_standalone_oof(uni_df, registry, oof_out_path="reports/standalone_oof.csv"):
    """Run all four standalone models on registry folds; return (benchmarks, oof_df)."""
    benchmarks = []
    all_oof = []
    for domain in ["movie", "tv", "game", "book"]:
        set_calibration(domain)
        print(f"   [standalone] {domain} ...", flush=True)
        loc = _load_domain(domain, uni_df, registry)
        raw = load_domain_raw(domain, loc)
        feat_cols = [c for c in loc.columns if c not in DROP_COLS]
        fit_fn = DOMAIN_FIT[domain]

        def one_fold(test_mask):
            train_mask = ~test_mask
            X_tr, X_te = reencode_fold(domain, loc, raw, test_mask, feat_cols)
            y_tr = loc.loc[train_mask, "target_reg"]
            preds = fit_fn(X_tr, y_tr, X_te)
            sub = loc.loc[test_mask, ["source_id", "target_reg"]].copy()
            sub["pred"] = np.asarray(preds)
            sub["media_type"] = domain
            return sub

        # (source_id, target_reg, raw_pred) per (item, repeat)
        rows = map_folds(one_fold, [(m,) for _, m in registry_folds(loc, registry)])

        oof = pd.concat(rows, ignore_index=True)
        # Per-item dedup: mean of raw predictions across the 5 CV repeats.
        per_item = oof.groupby("source_id").agg(
            target_reg=("target_reg", "first"),
            pred=("pred", "mean"),
        ).reset_index()
        per_item["media_type"] = domain

        n_items = per_item["source_id"].nunique()
        assert len(per_item) == n_items, f"{domain}: dedup left duplicate items"
        met = _metrics(per_item["target_reg"], per_item["pred"])
        benchmarks.append({"Domain": domain, "N": int(n_items),
                           "Model": DOMAIN_MODEL_LABEL[domain], **met})
        all_oof.append(per_item)
        print(f"      N={n_items}  MAE={met['MAE']:.4f}  R2={met['R2']:.4f}  Acc={met['Acc']:.1f}%")

    oof_df = pd.concat(all_oof, ignore_index=True)
    Path(oof_out_path).parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_csv(oof_out_path, index=False)
    print(f"   [standalone] per-item OOF saved -> {oof_out_path}")
    return benchmarks, oof_df


if __name__ == "__main__":
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    reg = load_registry()
    bms, _ = generate_standalone_oof(uni, reg)
    print(json.dumps(bms, indent=2))
