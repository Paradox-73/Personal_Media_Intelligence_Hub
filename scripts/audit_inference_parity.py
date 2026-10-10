"""
audit_inference_parity.py — does the Oracle feed the model what the trainer fed it?

Every domain has TWO code paths that must produce the same feature vector:

  * the BATCH builder   (`*/feature_engineering.py::process_features`) -> training_features.csv
  * the SINGLE-ITEM path (`transform_single_*`, `transform_single_media`) -> Oracle inference

They are written separately, and nothing has been comparing them. When they drift,
the model is served a vector it never saw in training and the prediction is wrong in
a way no metric catches — the offline scores stay good because they are computed on
the batch frame.

This script pushes the REAL library rows through the single-item path and diffs the
result against the batch frame those same rows produced, column by column.

USAGE
  python -m scripts.audit_inference_parity              # all domains
  python -m scripts.audit_inference_parity books        # one domain
  python -m scripts.audit_inference_parity --live       # also hit the APIs (slow)

Exit code is 1 if any BLOCKING skew is found, so this can gate the pipeline.
"""
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.unified_model.unified_utils import text_feature_cols

warnings.filterwarnings("ignore")

# A column is flagged when the single-item path and the batch builder disagree by
# more than this, relative to the column's own spread in training.
REL_TOL = 0.10
# Columns whose absolute spread is below this are compared on absolute difference only.
ABS_FLOOR = 1e-6

# Target encodings are DELIBERATELY different between the two paths: the batch
# builder writes out-of-fold values (to keep the target out of the features) while
# inference applies the full-data map. A gap here is the design working, not a bug,
# so these are reported separately instead of counting as skew.
EXPECTED_SUFFIXES = ("_encoded", "_n")
EXPECTED_PREFIXES = ("nb_",)   # OOF neighbour encodings, same reasoning


def _expected(col):
    return col.endswith(EXPECTED_SUFFIXES) or col.startswith(EXPECTED_PREFIXES)


# Sentence-transformer output is not bitwise identical between a batched encode and
# a single-item encode (padding/attention differ), so the low-variance tail PCA
# components always wobble a little. Only flag a text component when it moves a
# meaningful fraction of its own spread — a real template mismatch shifts it far more.
PCA_REL_TOL = 0.35


def _read(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kw)


# --------------------------------------------------------------------------- #
# Domain adapters. Each returns (enriched_rows_as_dicts, batch_frame, transform_fn).
# The enriched rows are filtered EXACTLY as the batch builder filters them, so row i
# of the dict list corresponds to row i of the batch frame.
# --------------------------------------------------------------------------- #

def _books():
    from src.books.feature_engineering import transform_single_book
    df = _read(config.BOOKS_ENRICHED_DATA_PATH).dropna(subset=["my_rating"]).reset_index(drop=True)
    batch = _read(config.BOOKS_TRAINING_DATA_PATH)
    state = joblib.load(config.BOOKS_PREPROCESSOR_STATE)
    return df, batch, (lambda raw: transform_single_book(raw, state)), state


def _games():
    from src.games.feature_engineering import transform_single_game
    df = _read(config.GAMES_ENRICHED_DATA_PATH).reset_index(drop=True)
    batch = _read(config.GAMES_TRAINING_DATA_PATH)
    state = joblib.load(config.GAMES_MODEL_PREPROCESSOR_STATE)
    return df, batch, (lambda raw: transform_single_game(raw, state)), state


def _movies():
    from src.movies.feature_engineering import transform_single_movie
    df = _read(config.MOVIES_ENRICHED_DATA_PATH).dropna(subset=["user_rating"]).reset_index(drop=True)
    batch = _read(config.TRAINING_DATA_PATH)
    state = joblib.load(config.PREPROCESSOR_STATE)
    return df, batch, (lambda raw: transform_single_movie(raw, state)), state


def _shows():
    from src.shows.feature_engineering import transform_single_show
    df = _read(config.TV_SHOWS_ENRICHED_DATA_PATH).dropna(subset=["user_rating"]).reset_index(drop=True)
    batch = _read(config.TV_SHOWS_TRAINING_DATA_PATH)
    state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    return df, batch, (lambda raw: transform_single_show(raw, state)), state


def _unified(media_type):
    """The unified backbone, exercised through the same entry point the Oracles use."""
    from src.unified_model.unified_oracle import _load, _to_backbone
    from src.unified_model.unified_feature_engineering import transform_single_media
    c = _load()
    state = c["state"]
    batch_all = _read(config.UNIFIED_TRAINING_DATA_PATH)
    batch = batch_all[batch_all["media_type"] == media_type].reset_index(drop=True)

    # The unified pca_* columns are stored UNALIGNED. build_universal_dataset writes
    # raw PCA on purpose (aligning there would leak test-fold statistics), and the
    # per-domain DomainAligner is applied inside the trainer per fold and again by
    # transform_single_media at inference. So the single-item path legitimately
    # returns aligned coordinates while the CSV holds unaligned ones, and diffing
    # them raw reports every non-reference domain as skewed -- a constant per-domain
    # centroid offset (books were flagged at a flat 0.171 on pca_1 for EVERY row,
    # games at 0.179 on pca_0, movies at 0 because movie is the reference domain).
    # Align the batch side the same way the trainer does so this compares like with
    # like; otherwise the check cries wolf on unified and a real text-template skew
    # would be invisible in the noise.
    aligner = state.get("aligner")
    if aligner is not None:
        # `text_feature_cols` accepts emb_* AND pca_*. Hardcoding "pca_" here meant that
        # when the matrix moved to the full 384-d embedding this selector matched
        # nothing, the batch side was left unaligned against an aligned serve side, and
        # the audit reported all 384 emb_* columns as skewed in every unified domain --
        # the precise false alarm the comment above was written to prevent.
        pca_cols = text_feature_cols(batch)
        if pca_cols:
            # replace_text_block handles both states: a plain aligner keeps emb_*,
            # AlignedTextPCA turns them into pca_0..9 exactly as the trainer does.
            from src.unified_model.unified_utils import replace_text_block
            batch = replace_text_block(batch, pca_cols, aligner,
                                       np.array([media_type] * len(batch)))

    # Filter exactly as build_universal_dataset does: it drops unrated movies/TV but
    # keeps every game and book (unrated ones with target 0). Filtering games/books
    # here too shifted rows, so game A's features were diffed against game B's.
    src_map = {
        "book": (config.BOOKS_ENRICHED_DATA_PATH, None),
        "game": (config.GAMES_ENRICHED_DATA_PATH, None),
        "movie": (config.MOVIES_ENRICHED_DATA_PATH, "user_rating"),
        "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "user_rating"),
    }
    path, rating_col = src_map[media_type]
    df = _read(path)
    if rating_col:
        df = df.dropna(subset=[rating_col])
    df = df.reset_index(drop=True)
    # Row i of df is the item with source_id batch.source_id[i]; a length mismatch
    # means the enriched CSV changed since the features were built.
    if len(df) != len(batch):
        raise RuntimeError(f"unified:{media_type}: {len(df)} enriched rows vs {len(batch)} "
                           f"feature rows -- features are stale, rebuild them")

    def fn(raw):
        return transform_single_media(_to_backbone(raw, media_type), state,
                                      media_type=media_type)

    return df, batch, fn, state


DOMAINS = {
    "books": _books,
    "games": _games,
    "movies": _movies,
    "shows": _shows,
    "unified:book": lambda: _unified("book"),
    "unified:game": lambda: _unified("game"),
    "unified:movie": lambda: _unified("movie"),
    "unified:tv": lambda: _unified("tv"),
}


def audit(name, n_sample=12, verbose=True):
    """Transform n_sample real library rows and diff against the batch frame."""
    try:
        df, batch, fn, state = DOMAINS[name]()
    except Exception as e:  # noqa: BLE001 -- a broken adapter must fail the audit, not pass it
        print(f"  ❌ {name}: could not load: {e!r}")
        return [("__load__", float("inf"), 0.0, repr(e))]

    n = min(n_sample, len(df), len(batch))
    if n == 0:
        print(f"  ⚠️  {name}: no rows to compare")
        return []
    # Spread across the frame rather than taking the head, so a skew that only
    # affects later (newer) rows is not missed.
    idx = np.linspace(0, min(len(df), len(batch)) - 1, n).astype(int)

    rows = []
    for i in idx:
        try:
            rows.append(fn(df.iloc[i].to_dict()))
        except Exception as e:
            print(f"  ❌ {name}: transform failed on row {i}: {e!r}")
            return [("__transform__", float("inf"), 0.0, repr(e))]
    infer = pd.concat(rows, ignore_index=True)

    feat_cols = [c for c in infer.columns if c in batch.columns]
    b = batch.iloc[idx][feat_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    a = infer[feat_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    spread = batch[feat_cols].apply(pd.to_numeric, errors="coerce").std().fillna(0)
    diff = (a - b).abs().mean()

    findings = []
    for c in feat_cols:
        s, d = float(spread.get(c, 0)), float(diff.get(c, 0))
        if not np.isfinite(d):
            findings.append((c, d, s, "NaN produced at inference"))
            continue
        if s <= ABS_FLOOR:
            # Constant in training. Any deviation means the Oracle sends a value
            # the model has literally never seen for this domain.
            if d > ABS_FLOOR:
                findings.append((c, d, s, f"constant {b[c].iloc[0]:.4g} in training, "
                                          f"inference sends {a[c].iloc[0]:.4g}"))
        else:
            # The looser text tolerance must follow the text block whatever it is
            # called. Keying it on "pca_" alone meant that after the move to the full
            # 384-d embedding every emb_* column was judged against REL_TOL (0.10), the
            # tolerance meant for engineered features, rather than the 0.35 that exists
            # precisely because batched and single-item sentence-transformer encodes are
            # not bitwise identical. That flagged ~375 columns per domain as skew.
            tol = PCA_REL_TOL if (c.startswith("pca_") or c.startswith("emb_")) else REL_TOL
            if d > tol * s:
                findings.append((c, d, s, f"mean|Δ|={d:.4g} vs train σ={s:.4g}"))

    findings.sort(key=lambda r: -(r[1] / max(r[2], ABS_FLOOR)))
    real = [f for f in findings if not _expected(f[0])]
    by_design = [f for f in findings if _expected(f[0])]
    missing = [c for c in batch.columns
               if c not in infer.columns and not c.startswith("target")
               and c not in ("source_id", "media_type", "rating_date", "status")]

    if verbose:
        head = f"  {name}: {len(feat_cols)} shared cols, {n} rows sampled"
        if not real and not missing:
            print(f"✅{head} — parity OK"
                  + (f"  ({len(by_design)} OOF-encoded col(s) differ by design)" if by_design else ""))
        else:
            print(f"❌{head} — {len(real)} skewed column(s)")
            for c, d, s, why in real[:15]:
                print(f"     • {c:28} {why}")
            if len(real) > 15:
                print(f"     … and {len(real) - 15} more")
            if by_design:
                print(f"     (plus {len(by_design)} OOF target-encoding col(s) — expected)")
        if missing:
            print(f"     ⚠️  {len(missing)} batch column(s) the single-item path never emits: "
                  f"{missing[:8]}{' …' if len(missing) > 8 else ''}")

    # A batch column the single-item path never emits is served as 0 or not at all.
    return real + [(c, float("inf"), 0.0, "never emitted by the single-item path")
                   for c in missing]


def main(argv):
    live = "--live" in argv
    picked = [a for a in argv if not a.startswith("-")]
    names = picked or list(DOMAINS)
    bad = {}
    print("=" * 74)
    print("INFERENCE PARITY AUDIT — single-item Oracle path vs batch training frame")
    print("=" * 74)
    for nm in names:
        if nm not in DOMAINS:
            print(f"  ⚠️  unknown domain '{nm}' (choose from {list(DOMAINS)})")
            continue
        f = audit(nm)
        if f:
            bad[nm] = f
    print("=" * 74)
    if bad:
        print(f"❌ {len(bad)} domain(s) with feature skew: {list(bad)}")
        print("   The Oracle is serving vectors the model was never trained on.")
    else:
        print("✅ All audited domains agree between inference and training.")
    if live:
        print("\n(--live API-completeness check: see scripts/audit_api_coverage.py)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
