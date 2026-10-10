import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as ctb
import joblib
import sys
import json
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import wilcoxon

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective
from src.unified_model.unified_utils import (AlignedTextPCA, compute_temporal_weights,
                                             replace_text_block, text_feature_cols)
from src.reporting.metrics_writer import write_latest_metrics

def run_unified_ablation_study():
    print("🧪 Starting Unified Model Ablation Study (RATED-ONLY, Frozen Folds)...")

    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        print("❌ Error: Unified training data not found.")
        return

    df_full = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # Task 0.4: RATED-ONLY
    # Music has target_reg but it's pseudo-labels. Rated items are movie, tv, game, book.
    # In unified_feature_engineering, media_type is set.
    #
    # `target_reg > 0` is the second half of the filter and is NOT optional. Unrated
    # items are stored as target_reg == 0.0 -- a fillna in the builder, not NaN -- so
    # dropping music alone left six "Incomplete" games training and scoring against an
    # impossible label (0.0 is off the bottom of the 0.5-5.0 half-star grid). That is
    # the same defect already fixed in the fold registry, the trainer and the
    # comprehensive evaluator; this module was missed, so it ran on N=1413 while every
    # other artifact ran on N=1407, and its numbers reach latest_metrics.json.
    df = df_full[(df_full['media_type'] != 'music') & (df_full['target_reg'] > 0)].copy()
    _n_dropped = (df_full['media_type'] != 'music').sum() - len(df)
    print(f"📊 Filtered to RATED-ONLY items: N={len(df)}"
          + (f" ({_n_dropped} unrated/0-target row(s) excluded)" if _n_dropped else ""))

    df['rating_date'] = pd.to_datetime(df['rating_date'])
    pca_cols = text_feature_cols(df)
    
    X_full = df.drop(columns=['target_reg', 'target_ordinal', 'source_id', 'media_type', 'rating_date', 'global_id'], errors='ignore')
    y = df['target_reg']
    media_types = df['media_type']
    dates = df['rating_date']

    registry_path = config.UNIFIED_MODEL_DIR / "fold_registry.json"
    if not registry_path.exists():
        print("❌ Error: Fold registry missing. Run create_frozen_folds.py first.")
        return
        
    with open(registry_path, "r") as f:
        registry = json.load(f)
    
    df['global_id'] = df['media_type'] + "_" + df['source_id'].astype(str)
    
    # Protocol evaluation
    fold_preds = {}

    def evaluate_protocol(name, mode='base'):
        print(f"\n🚀 Evaluating Protocol: {name}")
        
        all_y_true = []
        all_y_pred = []
        all_ids = []
        
        # We need to iterate over the 50 folds (10-fold x 5-repeat)
        for fold_id in range(50):
            # Items in test set for this fold
            test_mask = df['global_id'].apply(lambda x: fold_id in registry.get(x, []))
            train_mask = ~test_mask
            
            if not test_mask.any(): continue
            
            X_train, X_test = X_full[train_mask].copy(), X_full[test_mask].copy()
            y_train, y_test = y[train_mask], y[test_mask]
            m_train, m_test = media_types[train_mask], media_types[test_mask]
            d_train = dates[train_mask]
            
            block = AlignedTextPCA().fit(X_train[pca_cols].values, m_train)
            X_train = replace_text_block(X_train, pca_cols, block, m_train)
            X_test = replace_text_block(X_test, pca_cols, block, m_test)

            w_train = compute_temporal_weights(d_train, lambda_decay=0.000429)
            
            if mode == 'base':
                model = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)
                model.fit(X_train, y_train, sample_weight=w_train)
                preds = model.predict(X_test)
            elif mode == 'mean_ensemble':
                m1 = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)
                m2 = ctb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function='MAE', verbose=0, random_seed=42)
                m1.fit(X_train, y_train, sample_weight=w_train)
                m2.fit(X_train, y_train, sample_weight=w_train)
                preds = (m1.predict(X_test) + m2.predict(X_test)) / 2
            
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(preds.tolist())
            all_ids.extend(df.loc[test_mask, 'global_id'].tolist())
            
            if (fold_id + 1) % 10 == 0:
                print(f"   Processed {fold_id + 1} folds...")
            
        # Each item is tested in 5 of the 50 folds: score one averaged prediction per item,
        # otherwise every item counts 5 times and the paired test's p-value is too small.
        per_item = pd.DataFrame({'id': all_ids, 't': all_y_true, 'p': all_y_pred}).groupby('id', sort=True).mean()
        y_true_arr = per_item['t'].to_numpy()
        y_pred_arr = per_item['p'].to_numpy()
        y_pred_rounded = np.round(np.clip(y_pred_arr, 0.5, 5.0) * 2) / 2
        
        mae = mean_absolute_error(y_true_arr, y_pred_rounded)
        r2 = r2_score(y_true_arr, y_pred_rounded)
        
        fold_preds[name] = y_pred_arr
        return {"MAE": mae, "R2": r2, "Y_TRUE": y_true_arr}

    # Run active protocols
    base_results = evaluate_protocol("Base", 'base')
    ens_results = evaluate_protocol("MeanEnsemble", 'mean_ensemble')
    
    # Calculate significance
    y_true = base_results['Y_TRUE']
    _rh = lambda a: np.round(np.clip(a, 0.5, 5.0) * 2) / 2
    base_err = np.abs(y_true - _rh(fold_preds["Base"]))
    ens_err = np.abs(y_true - _rh(fold_preds["MeanEnsemble"]))
    _, p_val = wilcoxon(base_err, ens_err)
    
    # Effect size: paired Cohen's d on the per-item absolute-error differences.
    diff = base_err - ens_err
    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0.0
    mae_delta = base_results['MAE'] - ens_results['MAE']
    # ~1.5k paired items make microscopic effects "significant"; flag triviality.
    triviality = "trivial" if abs(mae_delta) < 0.02 else "material"
    print(f"   Ensemble vs Base: dMAE={mae_delta:+.4f}  Cohen's d={cohens_d:.4f}  "
          f"p={p_val:.4g}  -> significant-but-{triviality}")

    # Task 0.4: 4-row table. Rows 3-4 are QUARANTINED legacy paths kept for the
    # record. Their numbers are the historical 5-fold ad-hoc run (NOT the frozen
    # registry), so they are directionally indicative only and not strictly
    # comparable to rows 1-2. Historical values restored from the original log:
    #   Ridge Stack:     MAE 0.5609, R^2 0.3837
    #   +Residual Heads: MAE 0.6088, R^2 0.2688
    ablation_rows = [
        {"Protocol": "Base XGB (Aligned)", "MAE": base_results['MAE'], "R2": base_results['R2'],
         "EffectSize": "ref", "p": "ref"},
        {"Protocol": "Mean Ensemble (XGB+Cat)", "MAE": ens_results['MAE'], "R2": ens_results['R2'],
         "EffectSize": f"d={cohens_d:.3f} ({triviality})", "p": f"{p_val:.4f}"},
        {"Protocol": "Ridge Stack (removed — legacy 5-fold protocol)", "MAE": 0.5609, "R2": 0.3837,
         "EffectSize": "n/a (legacy)", "p": "n/a (legacy)"},
        {"Protocol": "+Residual Heads (removed — legacy 5-fold protocol)", "MAE": 0.6088, "R2": 0.2688,
         "EffectSize": "n/a (legacy)", "p": "n/a (legacy)"},
    ]
    
    # Update latest_metrics.json (partial update, will be combined by comprehensive_evaluator)
    # Actually, let's just write the ablation report and return the data
    report_path = Path("reports/UNIFIED_ABLATION_REPORT.md")
    res_df = pd.DataFrame(ablation_rows)
    with open(report_path, "w", encoding='utf-8') as f:
        f.write("# Unified Model Ablation & Decomposition Report (RATED-ONLY)\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n*Results generated on " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*\n")
    
    print(f"✅ Ablation report saved to {report_path}")
    return ablation_rows

if __name__ == "__main__":
    run_unified_ablation_study()
