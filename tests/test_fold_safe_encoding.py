"""Benchmark folds must not let a test row's rating reach a training row's features.

The training CSVs' `*_encoded` columns come from one inner KFold over the whole
library. Inside a registry fold that means a training row's encoding already holds
test rows' ratings. `reencode_fold` refits them from the training rows only.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import taste_features as tf
from src.reporting.standalone_benchmarks import reencode_fold

N, TEST = 40, np.arange(0, 40, 4)
SRC = Path(__file__).resolve().parent.parent / "src"


def test_every_fold_scorer_refits_target_encodings():
    """Structural guard: a module under src/ that splits rows for scoring must refit the
    target encodings inside the split, or score only the shared space, which carries
    none. A new scorer that feeds CSV columns straight into registry folds fails here."""
    # Trainers also hold out rows (train_test_split) but must train on the whole-library
    # columns for serving, so only the experiments dir is held to the ad-hoc-split rule.
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        splits = (re.search(r"\bregistry_folds\(", text) and path.parent.name in ("experiments", "reporting", "unified_model")) \
            or (re.search(r"\btrain_test_split\(", text) and path.parent.name == "experiments")
        if not splits:
            continue
        refits = "reencode_fold" in text or "_reencode_entities" in text
        shared_space_only = "select_features(" in text or "def shared_space_columns" in text
        if not (refits or shared_space_only):
            offenders.append(path.relative_to(SRC.parent).as_posix())
    assert not offenders, (
        "these scorers feed whole-library target encodings into their folds; call "
        "standalone_benchmarks.reencode_fold (local matrix) or unified_repeated_cv._reencode_entities "
        f"(unified matrix) inside each fold: {offenders}")


def test_shared_space_keeps_no_target_encoding():
    """The allowance above rests on this: transfer_study's shared space has no encoded column."""
    from src.experiments.transfer_study import shared_space_columns
    cols = ["year", "critic_avg_5", "creator_encoded", "creator_n", "studio_encoded", "gen_Drama", "emb_0"]
    assert shared_space_columns(cols) == ["year", "critic_avg_5", "gen_Drama", "emb_0"]


def _library(seed=0):
    rng = np.random.default_rng(seed)
    authors = [f"a{i}" for i in range(6)]
    raw = pd.DataFrame({"authors": [", ".join(rng.choice(authors, 2, replace=False)) for _ in range(N)]})
    y = rng.choice([2.0, 3.0, 3.5, 4.0, 5.0], N)
    return raw, y


def _features(raw, y):
    """What books/feature_engineering.py writes: one OOF encoding over every row."""
    enc, sup, _ = tf.oof_entity_encode(raw["authors"].tolist(), y, m=2, limit=2)
    return pd.DataFrame({"year": np.arange(N), "authors_encoded": enc, "authors_n": sup, "target_reg": y})


def _perturb_test_targets(y):
    y2 = y.copy()
    y2[TEST] = np.where(y2[TEST] < 3.0, 5.0, 0.5)
    return y2


def test_csv_columns_leak_test_targets_into_training_rows():
    raw, y = _library()
    train = np.setdiff1d(np.arange(N), TEST)
    before = _features(raw, y).loc[train, "authors_encoded"]
    after = _features(raw, _perturb_test_targets(y)).loc[train, "authors_encoded"]
    assert not np.allclose(before, after), "the whole-library encoding must move when test targets move"


def test_reencode_fold_training_features_ignore_test_targets():
    raw, y = _library()
    mask = np.zeros(N, bool)
    mask[TEST] = True
    feat = ["year", "authors_encoded", "authors_n"]
    loc = _features(raw, y)
    loc2 = _features(raw, _perturb_test_targets(y))

    X_tr, X_te = reencode_fold("book", loc, raw, mask, feat)
    X_tr2, X_te2 = reencode_fold("book", loc2, raw, mask, feat)

    pd.testing.assert_frame_equal(X_tr, X_tr2)
    pd.testing.assert_frame_equal(X_te, X_te2)
    assert len(X_tr) == N - len(TEST) and len(X_te) == len(TEST)
    assert not np.allclose(X_tr["authors_encoded"], loc.loc[~mask, "authors_encoded"]), \
        "training encodings were rebuilt, not copied from the CSV"
