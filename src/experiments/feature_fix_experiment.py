"""
Feature-fix validation — the three levers from err.txt, measured on the FROZEN folds.

err.txt says: engineer the dropped features, then "prove any improvement with the
paired Wilcoxon on frozen folds before believing it." This script does exactly that
as a CONTROLLED before/after: same model, same 50 registry folds, only the feature
set changes. It does NOT overwrite the committed matrices, the frozen registry, the
deployed models, or the docs — persisting is a separate, deliberate step.

Three levers:
  (1) MOVIES — add the leakage-safe Director/Actors/Dir×Genre target encodings that
      feature_engineering.py computes but the persisted matrix dropped. (model: XGB)
  (2) TV     — add the dropped relational/structural signal: showrunner (created_by),
      actors, writer, network target-encodings (out-of-fold, HARD m=20 smoothing),
      plus episode/season/runtime/imdb_rating numerics and status one-hot.
      (model: the deployed TV simplex-stack)
  (3) BOOKS  — restore the three dead UNIFIED channels (critic_avg_5←averageRating,
      runtime←pageCount, gen_*←categories) and re-measure the pooled book SLICE.
      (model: pooled shared-space XGB, matching the unified design)

All target encodings are computed PER FOLD on the training rows only (registry-fold
out-of-fold), so the comparison is leakage-safe.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.reporting.standalone_benchmarks import (
    _load_domain, _fit_predict_xgb, _fit_predict_simplex_stack, DROP_COLS, _round_half,
)

SEED = 42


def _load_registry():
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni["global_id"] = uni["media_type"] + "_" + uni["source_id"].astype(str)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        reg = json.load(f)
    return uni, reg


def _parse_people(v):
    if pd.isna(v) or str(v).strip() == "":
        return []
    s = str(v).strip()
    if s.startswith("["):
        try:
            import ast
            return [str(x).strip() for x in ast.literal_eval(s)]
        except Exception:
            pass
    return [x.strip() for x in s.split(",") if x.strip()]


def _te_map(train_df, col, target="target_reg", m=10):
    """Bayesian-smoothed mean-rating map from a TRAIN slice (multi-value safe)."""
    gmean = train_df[target].mean()
    tmp = pd.DataFrame({"items": train_df[col].apply(_parse_people).values,
                        "t": train_df[target].values})
    ft = tmp.explode("items").dropna(subset=["items"])
    stats = ft.groupby("items")["t"].agg(["count", "mean"])
    mp = ((stats["count"] * stats["mean"] + m * gmean) / (stats["count"] + m)).to_dict()
    return mp, gmean


def _te_apply(series, mp, gmean):
    return series.apply(lambda v: np.mean([mp.get(i, gmean) for i in _parse_people(v)])
                        if _parse_people(v) else gmean)


def _metrics(yt, yp_raw):
    yp = _round_half(np.asarray(yp_raw))
    yt = np.asarray(yt)
    base = mean_absolute_error(yt, np.full_like(yt, yt.mean(), dtype=float))
    return dict(MAE=float(mean_absolute_error(yt, yp)), R2=float(r2_score(yt, yp)),
                Acc=float((np.abs(yt - yp) <= 0.5).mean() * 100),
                skill=float(1 - mean_absolute_error(yt, yp) / base))


def _wilcoxon(yt, old_raw, new_raw):
    ea = np.abs(np.asarray(yt) - _round_half(np.asarray(old_raw)))
    eb = np.abs(np.asarray(yt) - _round_half(np.asarray(new_raw)))
    if np.allclose(ea, eb):
        return float("nan")
    return float(wilcoxon(ea, eb).pvalue)


# --------------------------------------------------------------------- MOVIES / TV
def run_local_fix(domain, enrich_path, te_specs, extra_numeric, status_col, fit_fn, m):
    uni, reg = _load_registry()
    loc = _load_domain(domain, uni, reg).reset_index(drop=True)
    enr = pd.read_csv(enrich_path)
    rating_col = "user_rating" if "user_rating" in enr.columns else "my_rating"
    enr = enr[enr[rating_col].notna()].reset_index(drop=True)
    assert len(enr) == len(loc), f"{domain}: enriched {len(enr)} != loc {len(loc)}"
    for _, src in te_specs:
        loc[src] = enr[src] if src in enr.columns else ""
    for c in extra_numeric:
        loc[f"__num_{c}"] = pd.to_numeric(enr[c], errors="coerce") if c in enr.columns else np.nan
    if status_col and status_col in enr.columns:
        loc["__status"] = enr[status_col].fillna("Unknown").astype(str).str.strip().replace({"": "Unknown"})
    base_feats = [c for c in loc.columns if c not in DROP_COLS
                  and not c.startswith("__") and c not in [s for _, s in te_specs]]
    # fill extra numerics with median
    for c in extra_numeric:
        col = f"__num_{c}"
        med = loc[col].median()
        loc[col] = loc[col].fillna(med if pd.notna(med) else 0)
    status_dummies = (pd.get_dummies(loc["__status"], prefix="status")
                      if status_col and "__status" in loc else pd.DataFrame(index=loc.index))

    rows_old, rows_new = [], []
    for fold_id in range(50):
        tmask = loc["global_id"].apply(lambda g: fold_id in reg.get(g, []))
        if not tmask.any():
            continue
        tr, te = loc[~tmask], loc[tmask]
        # OLD feature matrix
        Xtr_o, Xte_o = tr[base_feats], te[base_feats]
        # NEW = base + extra numerics + status + target encodings.
        # CRITICAL: train-row encodings are computed with NESTED out-of-fold CV so a
        # row never sees its own label (matches feature_engineering.py); test rows use
        # the full-train map. Without nesting the TE leaks the label and overfits.
        from sklearn.model_selection import KFold
        tr_reset = tr.reset_index(drop=True)
        inner = KFold(n_splits=5, shuffle=True, random_state=SEED)
        te_new_tr, te_new_te = {}, {}
        for name, src in te_specs:
            mp, gm = _te_map(tr, src, m=m)
            te_new_te[name] = _te_apply(te[src], mp, gm).values
            arr = np.full(len(tr_reset), gm, dtype=float)
            for itr, iva in inner.split(tr_reset):
                mpi, gmi = _te_map(tr_reset.iloc[itr], src, m=m)
                arr[iva] = _te_apply(tr_reset.iloc[iva][src], mpi, gmi).values
            te_new_tr[name] = arr
        add_tr = pd.DataFrame(te_new_tr, index=tr.index)
        add_te = pd.DataFrame(te_new_te, index=te.index)
        num_cols = [f"__num_{c}" for c in extra_numeric]
        Xtr_n = pd.concat([tr[base_feats + num_cols], add_tr,
                           status_dummies.loc[tr.index]], axis=1)
        Xte_n = pd.concat([te[base_feats + num_cols], add_te,
                           status_dummies.loc[te.index]], axis=1)
        Xte_n = Xte_n.reindex(columns=Xtr_n.columns, fill_value=0)
        y_tr = tr["target_reg"]
        p_old = fit_fn(Xtr_o, y_tr, Xte_o)
        p_new = fit_fn(Xtr_n, y_tr, Xte_n)
        for sid, t, po, pn in zip(te["source_id"], te["target_reg"], p_old, p_new):
            rows_old.append((sid, t, po)); rows_new.append((sid, t, pn))

    def _dedup(rows):
        d = pd.DataFrame(rows, columns=["sid", "t", "p"])
        return d.groupby("sid").agg(t=("t", "first"), p=("p", "mean")).reset_index()
    do, dn = _dedup(rows_old), _dedup(rows_new)
    mo, mn = _metrics(do["t"], do["p"]), _metrics(dn["t"], dn["p"])
    p = _wilcoxon(dn["t"], do["p"], dn["p"])
    return mo, mn, p, len(do), [n for n, _ in te_specs], num_cols, list(status_dummies.columns)


# --------------------------------------------------------------------------- BOOKS
def run_books_fix():
    uni, reg = _load_registry()
    rated = uni[uni.media_type.isin(["movie", "tv", "game", "book"])].copy()
    book_enr = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    book_enr = book_enr[book_enr["my_rating"].notna()].reset_index(drop=True)
    book_mask = rated.media_type == "book"
    n_book = int(book_mask.sum())
    assert len(book_enr) == n_book, f"book enriched {len(book_enr)} != unified book {n_book}"

    # Sanity check: CAN the restored critic channel help? Only if the crowd's
    # averageRating actually tracks THIS user's ratings.
    from scipy.stats import pearsonr, spearmanr
    user = pd.to_numeric(book_enr["my_rating"], errors="coerce")
    crowd = pd.to_numeric(book_enr["averageRating"], errors="coerce")
    pages = pd.to_numeric(book_enr["pageCount"], errors="coerce")
    m2 = user.notna() & crowd.notna() & (crowd != 0)
    pr = pearsonr(user[m2], crowd[m2])[0] if m2.sum() > 3 else float("nan")
    sr = spearmanr(user[m2], crowd[m2]).correlation if m2.sum() > 3 else float("nan")
    mp = user.notna() & pages.notna() & (pages != 0)
    pr_pg = pearsonr(user[mp], pages[mp])[0] if mp.sum() > 3 else float("nan")
    print(f"\n  [books sanity] corr(my_rating, averageRating): Pearson {pr:+.3f}, Spearman {sr:+.3f}  (n={int(m2.sum())})")
    print(f"  [books sanity] corr(my_rating, pageCount):     Pearson {pr_pg:+.3f}  (n={int(mp.sum())})")
    print(f"  -> a restored channel can only help if these are meaningfully non-zero.")

    feat = [c for c in rated.columns if (c.startswith(("pca_", "gen_", "g_", "lang_", "rated_"))
            or c in {"year", "popularity", "critic_avg_5", "runtime", "box_office_log",
                     "imdb_rating", "imdb_votes", "vote_average"})]
    rated_new = rated.copy()
    # restore the three dead book channels from the (current) book enriched data
    avg = pd.to_numeric(book_enr["averageRating"], errors="coerce").fillna(0).values
    pages = pd.to_numeric(book_enr["pageCount"], errors="coerce").fillna(0).values
    rated_new.loc[book_mask, "critic_avg_5"] = avg            # averageRating already on a 0-5 scale
    if "runtime" in rated_new:
        rated_new.loc[book_mask, "runtime"] = pages
    # categories -> existing gen_ columns
    gen_cols = [c for c in rated.columns if c.startswith("gen_")]
    for i, cats in enumerate(book_enr["categories"].fillna("")):
        toks = {t.strip() for t in str(cats).split(",") if t.strip()}
        ridx = rated_new.index[book_mask][i]
        for c in gen_cols:
            name = c[4:]
            if name in toks:
                rated_new.at[ridx, c] = 1

    # identity-aware feature set (matches the PRODUCTION unified model, which keeps
    # is_*/has_* masks; the domain-BLIND set matches the transfer grid).
    id_cols = [c for c in rated.columns if c.startswith("is_") or c.startswith("has_")]
    feat_id = feat + id_cols

    def pooled_oof(frame, fcols):
        rows = []
        for fold_id in range(50):
            tmask = frame["global_id"].apply(lambda g: fold_id in reg.get(g, []))
            test_books = tmask & (frame.media_type == "book")
            if not test_books.any():
                continue
            tr = frame[~tmask]
            te = frame[test_books]
            m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                                 subsample=0.8, colsample_bytree=0.8,
                                 objective="reg:absoluteerror", random_state=SEED)
            m.fit(tr[fcols], tr["target_reg"])
            pr = m.predict(te[fcols])
            for sid, t, p in zip(te["source_id"], te["target_reg"], pr):
                rows.append((sid, t, p))
        d = pd.DataFrame(rows, columns=["sid", "t", "p"])
        return d.groupby("sid").agg(t=("t", "first"), p=("p", "mean")).reset_index()

    res = {}
    for tag, fcols in [("domain-blind (transfer-style)", feat), ("identity-aware (production-style)", feat_id)]:
        do, dn = pooled_oof(rated, fcols), pooled_oof(rated_new, fcols)
        mo, mn = _metrics(do["t"], do["p"]), _metrics(dn["t"], dn["p"])
        pv = _wilcoxon(dn["t"], do["p"], dn["p"])
        res[tag] = (mo, mn, pv, len(do))
    return res


def _print(title, mo, mn, p, n, added=""):
    print(f"\n=== {title}  (N={n}) ===")
    if added:
        print(f"  added: {added}")
    print(f"  {'':6} {'MAE':>8} {'R2':>8} {'Acc':>7} {'skill':>8}")
    print(f"  {'OLD':6} {mo['MAE']:>8.4f} {mo['R2']:>8.4f} {mo['Acc']:>6.1f}% {mo['skill']:>8.4f}")
    print(f"  {'NEW':6} {mn['MAE']:>8.4f} {mn['R2']:>8.4f} {mn['Acc']:>6.1f}% {mn['skill']:>8.4f}")
    print(f"  Δ      {mo['MAE']-mn['MAE']:>+8.4f} {mn['R2']-mo['R2']:>+8.4f} "
          f"{mn['Acc']-mo['Acc']:>+6.1f}% {mn['skill']-mo['skill']:>+8.4f}   (ΔMAE>0 = better)")
    sig = "n/a" if p != p else f"{p:.4f}{' *' if p < 0.05 else ''}"
    print(f"  paired Wilcoxon p = {sig}")
    return dict(domain=title, old=mo, new=mn, wilcoxon_p=(None if p != p else p), N=n, added=added)


def main():
    out = {}
    print("#" * 70)
    print("# FEATURE-FIX VALIDATION — before/after on the 50 frozen folds")
    print("#" * 70)

    # (1) MOVIES — target encodings
    mo, mn, p, n, te_names, _, _ = run_local_fix(
        "movie", config.MOVIES_ENRICHED_DATA_PATH,
        te_specs=[("director_te", "director"), ("actors_te", "actors")],
        extra_numeric=[], status_col=None, fit_fn=_fit_predict_xgb, m=10)
    out["movies"] = _print("MOVIES + target encodings (director, actors)", mo, mn, p, n,
                           added="director_te, actors_te (OOF, m=10)")

    # (2) TV — dropped relational + structural signal
    mo, mn, p, n, te_names, num_cols, stat_cols = run_local_fix(
        "tv", config.TV_SHOWS_ENRICHED_DATA_PATH,
        te_specs=[("creator_te", "created_by"), ("actors_te", "actors"),
                  ("writer_te", "writer"), ("network_te", "network")],
        extra_numeric=["imdb_rating", "number_of_episodes", "runtime"],
        status_col="status", fit_fn=_fit_predict_simplex_stack, m=20)
    out["tv"] = _print("TV + showrunner/cast/writer/network TE + structural", mo, mn, p, n,
                       added="creator_te, actors_te, writer_te, network_te (OOF, m=20); "
                             "+imdb_rating, episodes, runtime; +status one-hot")

    # (3) BOOKS — restore the three dead unified channels, two model variants
    books_res = run_books_fix()
    out["books"] = {}
    for tag, (mo, mn, p, n) in books_res.items():
        out["books"][tag] = _print(f"BOOKS pooled slice [{tag}] + restored channels", mo, mn, p, n,
                                    added="critic_avg_5←averageRating, runtime←pageCount, gen_*←categories")

    Path("reports/feature_fix_experiment.json").write_text(json.dumps(out, indent=2, default=str))
    print("\nWrote reports/feature_fix_experiment.json")


if __name__ == "__main__":
    main()
