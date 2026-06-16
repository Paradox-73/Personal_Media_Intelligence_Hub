import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import catboost as ctb

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_utils import DomainAligner, compute_temporal_weights
from src.reporting.metrics_writer import write_latest_metrics
from src.reporting.standalone_benchmarks import generate_standalone_oof
from src.unified_model.unified_repeated_cv import run_unified_ablation_study

LAMBDA_DECAY = 0.000429


def get_metrics(y_true, y_pred):
    """Round to nearest 0.5, then score (project-wide convention)."""
    y_pred_r = np.round(np.clip(y_pred, 0.5, 5.0) * 2) / 2
    mae = mean_absolute_error(y_true, y_pred_r)
    r2 = r2_score(y_true, y_pred_r)
    acc = (np.abs(np.asarray(y_true) - y_pred_r) <= 0.5).mean() * 100
    return {"MAE": float(mae), "R2": float(r2), "Acc": float(acc)}


def _fit_mean_ensemble(X_train, y_train, X_test, w_train):
    m1 = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)
    m2 = ctb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6,
                               loss_function='MAE', verbose=0, random_seed=42)
    m1.fit(X_train, y_train, sample_weight=w_train)
    m2.fit(X_train, y_train, sample_weight=w_train)
    return (m1.predict(X_test) + m2.predict(X_test)) / 2


def unified_oof_on_registry(df, registry):
    """
    Unified Mean-Ensemble OOF on the 50 frozen registry folds.
    Returns a per-(item, repeat) frame: media_type, source_id, target_reg, pred(raw).
    Music is never in the registry, so this covers the 1,264 rated items only.
    """
    pca_cols = [c for c in df.columns if c.startswith('pca_')]
    X_cols = [c for c in df.columns if c not in
              ['target_reg', 'target_ordinal', 'target_class', 'source_id',
               'media_type', 'rating_date', 'global_id']]
    results = []
    for fold_id in range(50):
        test_mask = df['global_id'].apply(lambda g: fold_id in registry.get(g, []))
        if not test_mask.any():
            continue
        train_mask = ~test_mask

        X_train = df.loc[train_mask, X_cols].copy()
        X_test = df.loc[test_mask, X_cols].copy()
        y_train = df.loc[train_mask, 'target_reg']
        m_train = df.loc[train_mask, 'media_type']
        m_test = df.loc[test_mask, 'media_type']
        d_train = pd.to_datetime(df.loc[train_mask, 'rating_date'])

        aligner = DomainAligner(method='centroid')
        aligner.fit(X_train[pca_cols].values, m_train)
        X_train.loc[:, pca_cols] = aligner.transform(X_train[pca_cols].values, m_train)
        X_test.loc[:, pca_cols] = aligner.transform(X_test[pca_cols].values, m_test)

        w_train = compute_temporal_weights(d_train, lambda_decay=LAMBDA_DECAY)
        preds = _fit_mean_ensemble(X_train, y_train, X_test, w_train)

        res = df.loc[test_mask, ['media_type', 'source_id', 'target_reg']].copy()
        res['pred'] = preds
        results.append(res)
        if (fold_id + 1) % 10 == 0:
            print(f"      unified registry fold {fold_id + 1}/50")
    return pd.concat(results, ignore_index=True)


def dedup_per_item(oof):
    """Average raw predictions across the 5 CV repeats -> one row per item."""
    return (oof.groupby(['media_type', 'source_id'], as_index=False)
               .agg(target_reg=('target_reg', 'first'), pred=('pred', 'mean')))


def full_pool_with_music(df_full, registry):
    """
    Secondary, footnoted row: train on the FULL pool (incl. 3,688 music PU
    pseudo-labels) and evaluate INCLUDING music. Music has no frozen registry,
    so this row uses a separate RepeatedKFold(5x1) over the full pool and is NOT
    a frozen-fold / actual-taste metric -- it exists only to show transparently
    what the pooled headline (~0.50) actually measures.

    Uses an XGBoost proxy (not the full XGB+CatBoost ensemble): the rated-only
    ablation showed the CatBoost member contributes a significant-but-trivial
    delta-MAE < 0.01, and this is a secondary/footnoted row, so the proxy is
    faithful and keeps the 4,952-item pass tractable.
    """
    pca_cols = [c for c in df_full.columns if c.startswith('pca_')]
    X_cols = [c for c in df_full.columns if c not in
              ['target_reg', 'target_ordinal', 'target_class', 'source_id',
               'media_type', 'rating_date', 'global_id']]
    df = df_full.reset_index(drop=True)
    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
    preds_accum = np.zeros(len(df))
    counts = np.zeros(len(df))
    for tr_idx, te_idx in rkf.split(df):
        X_train = df.loc[tr_idx, X_cols].copy()
        X_test = df.loc[te_idx, X_cols].copy()
        y_train = df.loc[tr_idx, 'target_reg']
        m_train = df.loc[tr_idx, 'media_type']
        m_test = df.loc[te_idx, 'media_type']
        d_train = pd.to_datetime(df.loc[tr_idx, 'rating_date'])
        aligner = DomainAligner(method='centroid')
        aligner.fit(X_train[pca_cols].values, m_train)
        X_train.loc[:, pca_cols] = aligner.transform(X_train[pca_cols].values, m_train)
        X_test.loc[:, pca_cols] = aligner.transform(X_test[pca_cols].values, m_test)
        w_train = compute_temporal_weights(d_train, lambda_decay=LAMBDA_DECAY)
        model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                                 random_state=42)
        model.fit(X_train, y_train, sample_weight=w_train)
        preds_accum[te_idx] += model.predict(X_test)
        counts[te_idx] += 1
    per_item = preds_accum / np.maximum(counts, 1)
    return get_metrics(df['target_reg'], per_item), len(df)


def load_unified_params():
    """Report the params actually deployed in the evaluated ensemble.

    The frozen-fold Mean Ensemble uses XGB(MAE)+CatBoost(MAE) with temporal
    weighting -- it does NOT use the AsymmetricEdgePenalty objective, so its
    alpha_hi/alpha_lo are not applied here. If the advanced trainer has persisted
    tuned values (models/unified/best_params.json) we surface them as 'tuned
    (production asymmetric objective)'; otherwise alphas are reported null with a
    note rather than as misleading hardcoded defaults.
    """
    params = {"lambda": LAMBDA_DECAY,
              "lambda_note": "temporal decay applied in the evaluated Mean Ensemble (half-life ~1,615 days)",
              "alpha_hi": None, "alpha_lo": None,
              "alpha_note": ("AsymmetricEdgePenalty alphas are tuned by the advanced trainer "
                             "but NOT applied in the frozen-fold Mean Ensemble evaluation")}
    bp = config.UNIFIED_MODEL_DIR / "best_params.json"
    if bp.exists():
        try:
            saved = json.loads(bp.read_text())
            params["alpha_hi"] = saved.get("alpha_hi")
            params["alpha_lo"] = saved.get("alpha_lo")
            params["alpha_note"] = "tuned via Optuna in the advanced (production) asymmetric objective"
            if "lambda_decay" in saved:
                params["lambda"] = saved["lambda_decay"]
        except Exception:
            pass
    return params


def run_master_evaluation():
    print("🚀 Running Master Evaluation Pipeline (Phase 0 corrected)...")

    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df['global_id'] = df['media_type'] + "_" + df['source_id'].astype(str)
    df_rated = df[df['media_type'] != 'music'].copy()

    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)

    # 1) Unified rated OOF on registry folds -> headline + per-domain slices.
    #    Reuse the cached per-(item,repeat) OOF if present (it is deterministic and
    #    expensive: 50 folds x XGB+CatBoost). Delete reports/oof_predictions.csv to
    #    force a fresh recompute.
    oof_cache = Path("reports/oof_predictions.csv")
    if oof_cache.exists():
        uni_oof = pd.read_csv(oof_cache)
        if {'media_type', 'source_id', 'target_reg', 'pred'}.issubset(uni_oof.columns) \
                and 'music' not in set(uni_oof['media_type']):
            print("📊 Reusing cached unified registry OOF (reports/oof_predictions.csv)")
        else:
            print("📊 Unified (rated) OOF on registry folds (cache stale)...")
            uni_oof = unified_oof_on_registry(df_rated, registry)
            uni_oof.to_csv(oof_cache, index=False)
    else:
        print("📊 Unified (rated) OOF on registry folds...")
        uni_oof = unified_oof_on_registry(df_rated, registry)
        uni_oof.to_csv(oof_cache, index=False)
    uni_item = dedup_per_item(uni_oof)
    met_rated = get_metrics(uni_item['target_reg'], uni_item['pred'])
    n_rated = uni_item[['media_type', 'source_id']].drop_duplicates().shape[0]

    slices = []
    for domain in ['movie', 'tv', 'game', 'book']:
        d = uni_item[uni_item['media_type'] == domain]
        if d.empty:
            continue
        slices.append({"Domain": domain, "N": int(len(d)),
                       **get_metrics(d['target_reg'], d['pred'])})

    # 2) Unified full pool (incl. music PU pseudo-labels) -- secondary, footnoted
    print("📊 Unified (full pool incl. music pseudo-labels) -- secondary CV...")
    met_full, n_full = full_pool_with_music(df, registry)

    # 3) Standalone per-domain benchmarks on registry folds (genuine local models)
    print("📊 Standalone per-domain benchmarks (registry folds)...")
    standalone_bms, _ = generate_standalone_oof(df, registry, use_cache=True)
    standalone_by_domain = {b['Domain']: b for b in standalone_bms}

    # 4) Assemble benchmarks (standalone locals + dual unified headline)
    benchmarks = [
        standalone_by_domain['movie'],
        {"Domain": "unified_rated", "N": int(n_rated), "Model": "Mean Ensemble", **met_rated},
        {"Domain": "unified_full", "N": int(n_full),
         "Model": "Mean Ensemble (+music pool)", **met_full},
        standalone_by_domain['tv'],
        standalone_by_domain['game'],
        standalone_by_domain['book'],
    ]

    # ---- Phase 0 acceptance assertions -------------------------------------
    # (a) Standalone benchmarks must NOT be the unified slice (the original bug).
    slice_by_domain = {s['Domain']: s for s in slices}
    for dom in ['movie', 'tv', 'game', 'book']:
        b = standalone_by_domain[dom]
        s = slice_by_domain.get(dom, {})
        assert not (abs(b['MAE'] - s.get('MAE', -1)) < 1e-9 and
                    abs(b['R2'] - s.get('R2', -1)) < 1e-9), \
            f"benchmarks == slices for {dom} -- standalone/slice not separated!"
    # (b) N must be unique item counts, never OOF rows (no 5x inflation).
    #   Unified slices cover all rated registry items: movie 980, tv 159, game 62, book 63.
    #   Standalone Games covers only 55 -- 7 "Incomplete"-status games have NaN target_reg
    #   in the standalone pipeline and cannot be scored locally (provenance, documented).
    slice_counts = {'movie': 980, 'tv': 159, 'game': 62, 'book': 63}
    standalone_counts = {'movie': 980, 'tv': 159, 'game': 55, 'book': 63}
    for dom in ['movie', 'tv', 'game', 'book']:
        assert slice_by_domain[dom]['N'] == slice_counts[dom], \
            f"{dom} slice N={slice_by_domain[dom]['N']} != unique items {slice_counts[dom]}"
        assert standalone_by_domain[dom]['N'] == standalone_counts[dom], \
            f"{dom} standalone N={standalone_by_domain[dom]['N']} != {standalone_counts[dom]}"
    print("✅ Assertions passed: benchmarks != slices; N == unique items (no 5x inflation).")

    # 5) Ablation (rated-only registry folds)
    ablation_rows = run_unified_ablation_study()

    # 6) Per-domain DROP verdict for distillation prior (load if present)
    distill = []
    dpath = Path("reports/distillation_ablation_results.json")
    if dpath.exists():
        distill = json.loads(dpath.read_text())

    final_metrics = {
        "benchmarks": benchmarks,
        "slices": slices,
        "ablation": ablation_rows,
        "distillation": distill,
        "params": load_unified_params(),
        "notes": {
            "dedup": "per-item: raw predictions averaged across 5 CV repeats, then rounded to 0.5",
            "standalone_models": {"movie": "XGBoost", "tv": "Simplex-Stack",
                                  "game": "Local SVR (no prior)", "book": "Local SVR (no prior)"},
            "unified_full": ("trained on rated + 3,688 music PU pseudo-labels and evaluated "
                             "INCLUDING music via RepeatedKFold(5x1), XGB proxy; not a frozen-fold / "
                             "actual-taste metric -- secondary line only"),
        },
    }

    write_latest_metrics(final_metrics)
    print("🏁 Master evaluation complete.")


if __name__ == "__main__":
    run_master_evaluation()
