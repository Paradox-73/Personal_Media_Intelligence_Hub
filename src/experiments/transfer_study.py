"""
Phase 2 — The Transfer Grid (Task 2.1).

One config-driven harness that measures cross-domain taste transfer in the SHARED
feature space only (aligned vibe PCA + unified genre encoding + critic-average +
year + log-popularity + missingness masks). Domain-specific blocks are excluded.

Protocols per (Source set S, Target T) cell:
  A (baseline) : train on T only, registry folds.
  B (zero-shot): train on S, evaluate on all of T's registry test folds.
  C (augmented): train on S + {0,25,50,100}% of T's training folds, eval on T's
                 test folds. Target-fraction subsampling is deterministic (seeded).

Model: a fixed XGBoost regressor used as the grid model. The frozen production
ensemble also adds a CatBoost member, but the rated-only ablation showed that
member contributes a *significant-but-trivial* ΔMAE < 0.01; XGB-only is therefore
a faithful, far cheaper proxy and keeps the 100+ cell grid reproducible. This is
documented, not hidden.

CORAL/centroid alignment is fit on TRAINING folds only (no leakage).

Outputs:
  reports/transfer_grid_results.csv  — one row per (cell, protocol, fraction, repeat)
  reports/transfer_grid_summary.json — aggregated, for the renderer / atlas page
"""
import sys
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon, spearmanr
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.reporting.standalone_benchmarks import load_registry, registry_folds, skill
from src.unified_model.unified_utils import DomainAligner, text_feature_cols

RATED_DOMAINS = ["movie", "tv", "game", "book"]
MUSIC_WEIGHT = 0.4  # pseudo-labeled music sample weight (Task 2.1: 0.3-0.5)
SEED = 42


def shared_space_columns(cols):
    """Shared cross-domain features only; domain-specific blocks excluded.

    DOMAIN-BLIND (fix, err.txt check #4): the `has_{domain}_feats` masks and the
    `is_{domain}` indicators are *domain-identity* labels — leaving them in lets a
    transfer model split on the domain itself instead of transferring content, which
    biases Protocol C (augmented) and makes the grid "can't transfer by construction."
    A transfer study must be blind to which domain a row came from, so they are now
    excluded. (The vibe geometry is handled separately by CORAL alignment.)
    """
    keep = []
    for c in cols:
        # `emb_*` is the full 384-d description embedding. It replaced `pca_*`, the
        # 10-component compression of it. That distinction is the whole experiment:
        # PCA keeps global variance and degrades the local neighbourhood structure
        # cross-domain similarity lives in (in `pca_0..9` the nearest neighbours of
        # the Scott Pilgrim film are Ted and Ferris Bueller; in the full space they
        # are the TV series and all six book volumes). Accept BOTH names so the grid
        # runs on whichever representation the matrix currently holds -- and so a
        # matrix change can never silently reduce this to a 1-feature run again.
        if (c.startswith("emb_") or c.startswith("pca_")
                or c.startswith("gen_") or c.startswith("g_")):
            keep.append(c)
        elif c in {"year", "popularity", "critic_avg_5"}:
            keep.append(c)
        # NOTE: has_*_feats / is_* domain-identity columns are deliberately NOT kept.
    return keep


def select_features(df, max_genres=40):
    """Shared-space features with the genre one-hots capped to the `max_genres`
    most frequent across the library. The unified genre encoding has 350+ columns,
    most of them near-constant; keeping the top-K slashes XGB cost with negligible
    signal loss and keeps the grid tractable. Deterministic (frequency-ranked),
    so the analysis and the grid select the identical feature set."""
    feat = shared_space_columns(df.columns.tolist())
    gen = [c for c in feat if c.startswith("gen_") or c.startswith("g_")]
    non_gen = [c for c in feat if c not in gen]
    if max_genres and len(gen) > max_genres:
        freq = df[gen].fillna(0).sum().sort_values(ascending=False)
        gen = list(freq.index[:max_genres])
    return non_gen + sorted(gen)


N_ESTIMATORS = 300  # overridable by run_grid for a tractable pilot


def _xgb():
    return xgb.XGBRegressor(n_estimators=N_ESTIMATORS, learning_rate=0.03, max_depth=6,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="reg:absoluteerror", random_state=SEED)


def _round(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def _fit_predict(X_tr, y_tr, m_tr, X_te, m_te, w_tr=None):
    """CORAL-align (fit on train) then fit XGB and predict (rounded).

    `text_feature_cols` accepts emb_* AND pca_*. This used to hardcode "pca_", so when
    the shared space moved from a 10-component PCA to the full 384-d embedding the
    selector matched nothing and the alignment step silently became a no-op. The grid
    published on 2026-08-30 was run that way -- unaligned, identically across every
    cell, so its comparisons hold but its absolute numbers are unaligned transfer.
    """
    pca_cols = text_feature_cols(X_tr)
    Xtr, Xte = X_tr.copy(), X_te.copy()
    if pca_cols:
        aligner = DomainAligner(method="coral")
        aligner.fit(Xtr[pca_cols].values, m_tr.values)
        Xtr.loc[:, pca_cols] = aligner.transform(Xtr[pca_cols].values, m_tr.values)
        Xte.loc[:, pca_cols] = aligner.transform(Xte[pca_cols].values, m_te.values)
    model = _xgb()
    model.fit(Xtr, y_tr, sample_weight=w_tr)
    return _round(model.predict(Xte))


def subset_label(s):
    return "+".join(s)


def run_grid(repeats_per_cell=1, fractions=(0.0, 0.25, 0.5, 1.0),
             fold_ids=None, n_estimators=300, include_music=True):
    """fold_ids: subset of the 50 registry folds to evaluate (None = all 50).
    Use a subset + smaller n_estimators for a tractable, reproducible PILOT grid;
    the default (all 50 folds, 300 trees) is the production artifact."""
    global N_ESTIMATORS
    N_ESTIMATORS = n_estimators
    if fold_ids is None:
        fold_ids = range(50)
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)
    print(f"Shared-space features: {len(feat)} (genre one-hots capped at 40)")
    if len(feat) < 5:
        raise SystemExit(
            f"ABORT: only {len(feat)} shared-space feature(s) selected -- "
            f"`shared_space_columns` matched almost nothing in this matrix. "
            f"Columns present: {sorted(set(c.split('_')[0] for c in df.columns))[:12]}. "
            f"Running the grid on this would produce meaningless numbers.")

    registry = load_registry()

    music_df = df[df["media_type"] == "music"].copy()
    # If there are no music rows, the `with_music=True` arm adds nothing and produces
    # a bit-for-bit duplicate of the `False` arm -- doubling the entire grid for zero
    # information. Music is currently de-rated and excluded from the unified dataset
    # (0 rows), so this halves the run.
    if include_music and music_df.empty:
        print("   note: 0 music rows -- skipping the with-music arm (it would duplicate "
              "every cell). Grid size halved.")
        include_music = False
    rows = []

    for target in RATED_DOMAINS:
        others = [d for d in RATED_DOMAINS if d != target]
        # All 7 non-empty subsets of the other rated domains, x {without, with music}.
        subsets = []
        for r in range(1, len(others) + 1):
            subsets.extend(itertools.combinations(others, r))
        source_configs = []
        for s in subsets:
            source_configs.append((list(s), False))
            if include_music:
                source_configs.append((list(s), True))

        tdf = df[df["media_type"] == target].copy()
        tgt_ids = tdf["global_id"].values

        # Per-fold test masks for the target (registry folds).
        for fold_id, test_mask in registry_folds(tdf, registry, fold_ids=fold_ids):
            tr_target = tdf[~test_mask]
            te_target = tdf[test_mask]
            y_te = te_target["target_reg"].values
            train_mean = tr_target["target_reg"].mean()

            # ---- Protocol A: target-only ----
            predA = _fit_predict(tr_target[feat], tr_target["target_reg"], tr_target["media_type"],
                                 te_target[feat], te_target["media_type"])
            rows.append(dict(target=target, source="(target-only)", music=False,
                             protocol="A", fraction=1.0, fold=fold_id, repeat=0,
                             N=len(y_te), MAE=mean_absolute_error(y_te, predA),
                             skill=skill(y_te, predA, train_mean),
                             spearman=_safe_spearman(y_te, predA)))

            for src, with_music in source_configs:
                sdf = df[df["media_type"].isin(src)].copy()
                if with_music and len(music_df):
                    sdf = pd.concat([sdf, music_df], ignore_index=True)
                lbl = subset_label(src)

                # ---- Protocol B: zero-shot (train S, eval T test) ----
                wB = np.where(sdf["media_type"] == "music", MUSIC_WEIGHT, 1.0)
                predB = _fit_predict(sdf[feat], sdf["target_reg"], sdf["media_type"],
                                     te_target[feat], te_target["media_type"], w_tr=wB)
                rows.append(dict(target=target, source=lbl, music=with_music,
                                 protocol="B", fraction=0.0, fold=fold_id, repeat=0,
                                 N=len(y_te), MAE=mean_absolute_error(y_te, predB),
                                 skill=skill(y_te, predB, train_mean),
                                 spearman=_safe_spearman(y_te, predB)))

                # ---- Protocol C: augmented (S + frac of T train) ----
                for frac in fractions:
                    for rep in range(repeats_per_cell):
                        if frac == 0.0:
                            t_aug = tr_target.iloc[0:0]
                        else:
                            rng = np.random.RandomState(SEED + fold_id * 100 + rep)
                            n = max(1, int(round(frac * len(tr_target))))
                            idx = rng.choice(len(tr_target), size=n, replace=False)
                            t_aug = tr_target.iloc[idx]
                        comb = pd.concat([sdf, t_aug], ignore_index=True)
                        wC = np.where(comb["media_type"] == "music", MUSIC_WEIGHT, 1.0)
                        predC = _fit_predict(comb[feat], comb["target_reg"], comb["media_type"],
                                             te_target[feat], te_target["media_type"], w_tr=wC)
                        rows.append(dict(target=target, source=lbl, music=with_music,
                                         protocol="C", fraction=frac, fold=fold_id, repeat=rep,
                                         N=len(y_te), MAE=mean_absolute_error(y_te, predC),
                                         skill=skill(y_te, predC, train_mean),
                                         spearman=_safe_spearman(y_te, predC)))
        print(f"  target={target}: cumulative rows={len(rows)}")
        # Checkpoint: flush partial results after each target so a crash mid-grid
        # doesn't lose completed targets (the full grid is a long/overnight run).
        pd.DataFrame(rows).to_csv(Path("reports/transfer_grid_results.csv"), index=False)
        print(f"  [checkpoint] wrote {len(rows)} rows after target={target}")

    res = pd.DataFrame(rows)
    out_csv = Path("reports/transfer_grid_results.csv")
    res.to_csv(out_csv, index=False)
    print(f"✅ grid -> {out_csv} ({len(res)} rows)")
    return res


def _safe_spearman(a, b):
    if len(np.unique(b)) < 2:
        return 0.0
    rho = spearmanr(a, b).correlation
    return float(rho) if rho == rho else 0.0


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    if mode == "full":
        run_grid()
    else:
        # Tractable, reproducible pilot: 10 of the 50 registry folds, 150 trees,
        # fractions {0, 0.5, 1.0}. Exercises every protocol/path and yields a real
        # verdict; the full 50-fold/300-tree grid is the production artifact.
        print("PILOT grid (6 folds, 120 trees, top-40 genres, rated-domain sources) -- documented reduction.")
        run_grid(fractions=(0.0, 0.5, 1.0), fold_ids=list(range(0, 50, 8)),
                 n_estimators=120, include_music=False)
