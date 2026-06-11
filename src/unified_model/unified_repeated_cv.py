import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as ctb
import joblib
import sys
from pathlib import Path
from sklearn.model_selection import RepeatedKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, wilcoxon

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective
from src.unified_model.unified_utils import DomainAligner, compute_temporal_weights
from src.unified_model.advanced_unified_model_trainer import DomainResidualCorrector

def run_unified_ablation_study():
    print("🧪 Starting Unified Model Ablation Study (Repeated CV)...")

    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        print("❌ Error: Unified training data not found.")
        return

    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    
    pca_cols = [c for c in df.columns if c.startswith('pca_')]
    mask_cols = [c for c in df.columns if c.startswith('has_') and c.endswith('_feats')]
    
    X_full = df.drop(columns=['target_reg', 'target_ordinal', 'source_id', 'media_type', 'rating_date'], errors='ignore')
    y = df['target_reg']
    media_types = df['media_type']
    dates = df['rating_date']

    rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42) 
    
    # Track fold-level predictions for significance testing
    fold_preds = {}

    def evaluate_protocol(name, use_decay=False, use_masks=True, mode='base', use_alignment=True):
        print(f"\n🚀 Evaluating Protocol: {name}")
        X = X_full.copy()
        if not use_masks:
            X = X.drop(columns=mask_cols)
        
        all_mae = []
        all_r2 = []
        all_y_true = []
        all_y_pred = []
        
        fold_idx = 1
        for train_idx, test_idx in rkf.split(X):
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            m_train, m_test = media_types.iloc[train_idx], media_types.iloc[test_idx]
            d_train = dates.iloc[train_idx]
            
            if use_alignment:
                aligner = DomainAligner(method='centroid')
                aligner.fit(X_train[pca_cols].values, m_train)
                X_train.loc[:, pca_cols] = aligner.transform(X_train[pca_cols].values, m_train)
                X_test.loc[:, pca_cols] = aligner.transform(X_test[pca_cols].values, m_test)

            w_train = compute_temporal_weights(d_train, lambda_decay=0.0001) if use_decay else None
            
            # Base models with tuned params
            xgb_reg = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)
            cat_reg = ctb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function='MAE', verbose=0, random_seed=42)
            svr_reg = Pipeline([('scaler', StandardScaler()), ('svr', SVR(C=1.0))])

            if mode == 'base':
                if w_train is not None:
                    xgb_reg.fit(X_train, y_train, sample_weight=w_train)
                else:
                    xgb_reg.fit(X_train, y_train)
                preds = xgb_reg.predict(X_test)
            elif mode == 'mean_stack':
                xgb_reg.fit(X_train, y_train, sample_weight=w_train)
                cat_reg.fit(X_train, y_train, sample_weight=w_train)
                preds = (xgb_reg.predict(X_test) + cat_reg.predict(X_test)) / 2
            elif mode == 'ridge_stack':
                base_estimators = [('xgb', xgb_reg), ('catboost', cat_reg), ('svr', svr_reg)]
                model = StackingRegressor(estimators=base_estimators, final_estimator=Ridge(alpha=10.0), cv=5)
                # StackingRegressor doesn't take sample_weight in fit directly for base estimators easily in this version
                # But we'll fit it normally.
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            elif mode == 'full_residual':
                base_estimators = [('xgb', xgb_reg), ('catboost', cat_reg), ('svr', svr_reg)]
                stacking_base = StackingRegressor(estimators=base_estimators, final_estimator=Ridge(alpha=10.0), cv=5)
                model = DomainResidualCorrector(stacking_base)
                model.fit(X_train, y_train, m_train)
                preds = model.predict(X_test, m_test)
            
            y_pred_rounded = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2
            all_mae.append(mean_absolute_error(y_test, y_pred_rounded))
            all_r2.append(r2_score(y_test, y_pred_rounded))
            
            all_y_true.extend(y_test.tolist())
            all_y_pred.extend(preds.tolist())
            
            print(f"   Fold {fold_idx}/5: MAE={all_mae[-1]:.4f}, R2={all_r2[-1]:.4f}")
            fold_idx += 1
            
        fold_preds[name] = np.array(all_y_pred)
        return {
            "MAE": np.mean(all_mae),
            "R2": np.mean(all_r2),
            "Y_TRUE": np.array(all_y_true)
        }

    results = []
    # Decomposition path
    results.append({"Protocol": "1. Base (XGB, Aligned)", **evaluate_protocol("Base", True, True, 'base', True)})
    results.append({"Protocol": "2. Mean Ensemble (XGB+Cat)", **evaluate_protocol("MeanStack", True, True, 'mean_stack', True)})
    results.append({"Protocol": "3. Ridge Stacking (XGB+Cat+SVR)", **evaluate_protocol("RidgeStack", True, True, 'ridge_stack', True)})
    results.append({"Protocol": "4. Full Model (+Residual Heads)", **evaluate_protocol("FullModel", True, True, 'full_residual', True)})
    
    res_df = pd.DataFrame(results).drop(columns=['Y_TRUE'])
    
    # Significance Testing
    y_true = results[0]['Y_TRUE']
    base_err = np.abs(y_true - fold_preds["Base"])
    
    p_values = {}
    for name in ["MeanStack", "RidgeStack", "FullModel"]:
        err = np.abs(y_true - fold_preds[name])
        _, p = wilcoxon(base_err, err)
        p_values[name] = p

    print("\n" + "="*80)
    print("📊 UNIFIED MODEL ABLATION & DECOMPOSITION")
    print("="*80)
    print(res_df.to_string(index=False))
    print("-" * 80)
    for name, p in p_values.items():
        sig = " (Significant)" if p < 0.05 else " (Not Significant)"
        print(f"   {name} vs Base P-Value: {p:.4f}{sig}")
    print("="*80)
    
    # Save results to a report file
    report_path = Path("reports/UNIFIED_ABLATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Unified Model Ablation & Decomposition Report\n\n")
        f.write("## 1. Component Attribution\n\n")
        f.write(res_df.to_markdown(index=False))
        f.write("\n\n## 2. Statistical Significance (Wilcoxon signed-rank on MAE)\n\n")
        for name, p in p_values.items():
            f.write(f"- **{name} vs Base:** p={p:.4f} " + ("✅ Significant" if p < 0.05 else "❌ Not Significant") + "\n")
        
        f.write("\n## 3. Findings\n\n")
        # Check for pathology
        base_r2 = results[0]['R2']
        stack_r2 = results[2]['R2']
        if stack_r2 < base_r2:
            f.write(f"⚠️ **Pathology Confirmed:** The Ridge Stacker (R² {stack_r2:.4f}) underperforms the XGB base (R² {base_r2:.4f}). This indicates the meta-learner is being dragged down by weaker base models or over-fitting on the limited OOF pool.\n")
        else:
            f.write(f"✅ **Stacking Gain:** The Ridge Stacker successfully lifted R² by {stack_r2 - base_r2:.4f}.\n")
            
        res_r2 = results[3]['R2']
        f.write(f"- **Residual Heads Lift:** {res_r2 - stack_r2:+.4f} R² gain from domain-specific residual correction.\n")

        f.write("\n\n*Results generated on " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "*\n")
    
    print(f"✅ Ablation report saved to {report_path}")

if __name__ == "__main__":
    run_unified_ablation_study()
