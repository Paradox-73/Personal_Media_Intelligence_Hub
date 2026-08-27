"""data_sanity_check.py — is the data on disk actually what we think it is?

WHY THIS EXISTS
    Every bug this project has lost time to was silent. A column parses to NaN and
    gets median-filled, so 70% of it becomes one constant. A dtype guard stops firing
    and "92%" survives as a string. A double-encoded title never resolves. None of it
    moves an offline metric, because the metric is computed on the same corrupted
    frame.

    The specific worry this answers: enrichment is NULL-CELLS-ONLY
    (`if is_effectively_empty(df.at[idx, col])` in src/*/enrich_extended.py), so a
    cell that was written with a WRONG non-empty value is never re-fetched. If bad
    values were ever persisted into enriched_data.csv, they are frozen there.

    Median imputation happens in feature_engineering, in memory, on the way to
    training_features.csv — it is NOT written back to enriched_data.csv. This script
    checks that separation actually holds, per domain, per column.

WHAT IT CHECKS
    1. Row counts agree: enriched (rated) -> training_features -> unified slice.
    2. Persisted imputation: a numeric column in ENRICHED data where one value
       repeats implausibly often (the signature of a median written back).
    3. Parse health: values that survive as strings and coerce to NaN — the
       "92%" / "7,13,701" / "87 min" family.
    4. Constant / near-constant features in training_features (post-imputation this
       is expected for cross-domain columns, so it reports rather than fails).
    5. Mojibake (double-encoded UTF-8) in any text column.
    6. Rating sanity: user ratings on the 0.5-5.0 half-star grid, no zeros, no dupes.

USAGE
    python -m scripts.data_sanity_check              # all domains
    python -m scripts.data_sanity_check movies games # some
    python -m scripts.data_sanity_check --verbose    # per-column detail
Exit code 1 if any FAIL-level check trips, so it can gate the pipeline.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

MOJIBAKE = re.compile(r"Ã.|â€|Â.|ÃƒÂ")
# Strings that look numeric to a human but not to pd.to_numeric.
DIRTY_NUMERIC = re.compile(r"^\s*[\d.,]+\s*(%|min|mins|minutes|hrs|h)?\s*$", re.I)

DOMAINS = {
    "movies": dict(enriched=config.MOVIES_ENRICHED_DATA_PATH,
                   training=config.TRAINING_DATA_PATH,
                   rating="user_rating", title="title", media="movie"),
    "shows":  dict(enriched=config.TV_SHOWS_ENRICHED_DATA_PATH,
                   training=config.TV_SHOWS_PROCESSED_DIR / "training_features.csv",
                   rating="user_rating", title="name", media="tv"),
    "games":  dict(enriched=config.GAMES_ENRICHED_DATA_PATH,
                   training=config.GAMES_PROCESSED_DIR / "training_features.csv",
                   rating="my_rating", title="name", media="game"),
    "books":  dict(enriched=config.BOOKS_ENRICHED_DATA_PATH,
                   training=config.BOOKS_PROCESSED_DIR / "training_features.csv",
                   rating="my_rating", title="title", media="book"),
}

FAILS, WARNS = [], []


def fail(dom, msg):
    FAILS.append(f"[{dom}] {msg}")
    print(f"   FAIL  {msg}")


def warn(dom, msg):
    WARNS.append(f"[{dom}] {msg}")
    print(f"   warn  {msg}")


def ok(msg):
    print(f"   ok    {msg}")


def _read(p):
    try:
        return pd.read_csv(p, low_memory=False)
    except UnicodeDecodeError:
        # NOTE: reading as latin1 is what MANUFACTURES mojibake when the frame is
        # later written back as utf-8. Report it rather than doing it silently.
        print(f"   warn  {p.name} is not valid UTF-8; reading latin1 (this is how "
              f"double-encoding starts)")
        return pd.read_csv(p, encoding="latin1", low_memory=False)


def check_domain(name, spec, verbose=False):
    print(f"\n=== {name} ===")
    ep, tp = Path(spec["enriched"]), Path(spec["training"])
    if not ep.exists():
        fail(name, f"enriched file missing: {ep}")
        return None
    enr = _read(ep)
    rating_col = spec["rating"]

    # ---------- 1. row counts ----------
    rated = enr[pd.to_numeric(enr.get(rating_col), errors="coerce").notna()] \
        if rating_col in enr.columns else enr
    print(f"   enriched rows={len(enr)}  rated={len(rated)}")
    if tp.exists():
        tr = _read(tp)
        # Games deliberately KEEP unrated ("Incomplete") rows so they can still be
        # scored; movies/TV drop them. Compare on the right basis per domain.
        expected = len(enr) if name == "games" else len(rated)
        if len(tr) != expected:
            fail(name, f"training_features has {len(tr)} rows but expects {expected} "
                       f"(enriched={len(enr)}, rated={len(rated)}) — STALE, rebuild")
        else:
            ok(f"training_features row count matches ({len(tr)})")
    else:
        warn(name, f"no training_features at {tp}")
        tr = None

    # ---------- 2. persisted imputation in ENRICHED data ----------
    # A median written back would show as one value repeating across many rows.
    suspects = []
    for c in enr.columns:
        s = pd.to_numeric(enr[c], errors="coerce")
        nn = s.notna().sum()
        if nn < 20 or s.nunique() < 2:
            continue
        top_frac = s.value_counts(normalize=True).iloc[0]
        top_val = s.value_counts().index[0]
        # A genuinely repeated value (runtime 30, rating 4.0) is normal for small
        # integer/graded columns. Flag only continuous-looking columns.
        if top_frac > 0.30 and s.nunique() > 20:
            suspects.append((c, top_val, top_frac, nn))
    if suspects:
        for c, v, f, nn in suspects:
            fail(name, f"ENRICHED '{c}': value {v} on {f:.0%} of {nn} non-null rows "
                       f"— looks like an imputed constant written back to source")
    else:
        ok("no persisted imputation detected in enriched data")

    # ---------- 3. parse health ----------
    # Sentinel values that legitimately do not parse. Games mark a not-finished
    # title 'I' (Incomplete) in my_rating; feature_engineering reads that as unrated
    # on purpose ("Rated: 66, Incomplete: 6"), so it is not a parse failure.
    SENTINELS = {"i", "incomplete", "n/a", "tbd", "-"}
    dirty = []
    for c in enr.columns:
        col = enr[c]
        if pd.api.types.is_numeric_dtype(col):
            continue
        s = col.dropna().astype(str)
        if s.empty:
            continue
        looks_numeric = s.str.match(DIRTY_NUMERIC).mean()
        if looks_numeric < 0.6:
            continue
        # Judge AFTER the same cleaning the feature pipeline applies. The raw form
        # ("92%", "7,13,701", "87 min") is fine as long as the cleaner handles it;
        # what matters is whether the value reaches the model as a number.
        cleaned = (col.astype('string')
                   .str.replace(',', '', regex=False)
                   .str.replace('%', '', regex=False)
                   .str.replace(r'\s*(min|mins|minutes|hrs|h)\.?$', '', regex=True)
                   .str.strip())
        parses = pd.to_numeric(cleaned, errors='coerce').notna().sum()
        if parses < len(s) * 0.95:
            bad = s[pd.to_numeric(cleaned, errors='coerce').isna().reindex(s.index, fill_value=True)]
            bad = bad[~bad.str.strip().str.lower().isin(SENTINELS)]
            parses = len(s) - len(bad)
            if parses >= len(s) * 0.95:
                continue
            dirty.append((c, len(s), parses, bad.iloc[0] if len(bad) else '?'))
    if dirty:
        for c, tot, parses, sample in dirty:
            fail(name, f"ENRICHED '{c}': {tot - parses} of {tot} values do not coerce "
                       f"(e.g. {sample!r}) — will become NaN then median-filled")
    else:
        ok("all numeric-looking text columns coerce cleanly")

    # ---------- 4. constant features in TRAINING data ----------
    if tr is not None:
        flat = []
        for c in tr.columns:
            s = pd.to_numeric(tr[c], errors="coerce")
            if s.notna().sum() < 20 or s.nunique() <= 1:
                continue
            top = s.value_counts(normalize=True).iloc[0]
            if top > 0.50 and s.nunique() > 10:
                flat.append((c, top, s.value_counts().index[0]))
        if flat:
            for c, top, v in sorted(flat, key=lambda x: -x[1])[:8]:
                warn(name, f"training '{c}': {top:.0%} of rows == {v} (imputed?)")
        else:
            ok("no continuous training feature collapsed onto one value")

    # ---------- 5. mojibake ----------
    moji = {}
    for c in enr.columns:
        s = enr[c].dropna().astype(str)
        if s.empty:
            continue
        n = int(s.str.contains(MOJIBAKE, regex=True, na=False).sum())
        if n:
            moji[c] = n
    if moji:
        fail(name, f"mojibake (double-encoded UTF-8) in {moji}")
    else:
        ok("no mojibake in text columns")

    # ---------- 6. rating sanity ----------
    if rating_col in enr.columns:
        r = pd.to_numeric(enr[rating_col], errors="coerce").dropna()
        bad_grid = r[(r * 2) % 1 != 0]
        if len(bad_grid):
            warn(name, f"{len(bad_grid)} rating(s) off the half-star grid: "
                       f"{sorted(bad_grid.unique())[:5]}")
        oob = r[(r < 0.5) | (r > 5.0)]
        if len(oob):
            fail(name, f"{len(oob)} rating(s) outside 0.5-5.0: {sorted(oob.unique())[:5]}")
        else:
            ok(f"all {len(r)} ratings on the 0.5-5.0 grid")
    tcol = spec["title"]
    if tcol in enr.columns:
        d = enr[tcol].astype(str).str.strip().str.lower()
        dups = d[d.duplicated()].unique()
        if len(dups):
            warn(name, f"{len(dups)} duplicate title(s): {list(dups)[:4]}")
    return enr


def check_unified(verbose=False):
    print("\n=== unified ===")
    up = Path(config.UNIFIED_TRAINING_DATA_PATH)
    if not up.exists():
        warn("unified", "no unified training_features")
        return
    u = _read(up)
    print(f"   unified rows={len(u)}")
    if "target_reg" in u.columns:
        z = int((pd.to_numeric(u.target_reg, errors="coerce") <= 0).sum())
        if z:
            # The value is still written (fillna(0) in the builder), but every
            # consumer now filters it: create_frozen_folds, the trainer and the
            # evaluator all require target_reg > 0. Kept visible because the day one
            # of those filters is dropped, these rows silently become 0-star labels.
            warn("unified", f"{z} row(s) carry target_reg == 0 (unrated). All three "
                            f"consumers filter them; this is a standing hazard, not "
                            f"a live bug.")
        else:
            ok("no zero/negative targets")
    for name, spec in DOMAINS.items():
        ep = Path(spec["enriched"])
        if not ep.exists():
            continue
        enr = _read(ep)
        rc = spec["rating"]
        # build_universal_dataset drops unrated rows for movies/TV only; games and
        # books are concatenated whole. Compare against what the builder actually takes.
        if name == "games" or rc not in enr.columns:
            n_rated = len(enr)
        else:
            n_rated = int(pd.to_numeric(enr[rc], errors="coerce").notna().sum())
        n_uni = int((u.media_type == spec["media"]).sum())
        if n_uni != n_rated:
            fail("unified", f"{spec['media']}: unified slice {n_uni} != rated enriched "
                            f"{n_rated} — the positional source_id map is INVALID")
        else:
            ok(f"{spec['media']}: unified slice matches rated enriched ({n_uni})")
    emb = up.with_name("text_embeddings.npy")
    if emb.exists():
        E = np.load(emb)
        if len(E) != len(u):
            fail("unified", f"text_embeddings.npy has {len(E)} rows vs {len(u)} features")
        else:
            ok(f"text_embeddings.npy aligned ({E.shape})")
    state = Path(config.UNIFIED_PREPROCESSOR_STATE)
    if state.exists():
        import joblib
        try:
            from src.unified_model.unified_feature_engineering import MusicProfile  # noqa: F401
            st = joblib.load(state)
            if st.get("aligner") is None:
                fail("unified", "preprocessor_state has NO 'aligner' — a feature rebuild "
                                "stripped it; inference will silently skip alignment "
                                "until advanced_unified_model_trainer runs again")
            else:
                ok("preprocessor_state carries the DomainAligner")
        except Exception as e:
            warn("unified", f"could not inspect preprocessor_state: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("domains", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    picked = {k: v for k, v in DOMAINS.items() if not a.domains or k in a.domains}
    for name, spec in picked.items():
        check_domain(name, spec, a.verbose)
    if not a.domains:
        check_unified(a.verbose)

    print("\n" + "=" * 62)
    if FAILS:
        print(f"{len(FAILS)} FAIL, {len(WARNS)} warn")
        for f in FAILS:
            print(f"  FAIL  {f}")
        sys.exit(1)
    print(f"All checks passed ({len(WARNS)} warning(s)).")
    for w in WARNS:
        print(f"  warn  {w}")


if __name__ == "__main__":
    main()
