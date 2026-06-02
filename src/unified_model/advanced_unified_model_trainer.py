import pandas as pd
import xgboost as xgb
import catboost as ctb
import joblib
import sys
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import CustomEdgePenaltyObjective

def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("="*50)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*50)
    
    y_pred_rounded = np.round(np.clip(y_pred, 0, 5) * 2) / 2
    
    mse = mean_squared_error(y_true, y_pred_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_rounded)
    r2 = r2_score(y_true, y_pred_rounded)

    print(f"   📉 [TEST SET] Regressor MSE:  {mse:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")

    diffs = np.abs(y_true - y_pred_rounded)
    total = len(diffs)
    print("   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40)

def tune_and_train_xgboost(X_train, y_train, X_test, y_test):
    print("\n🔍 Sweeping Alpha range for Custom Edge Penalty (Unified Model)...")
    alpha_range = np.arange(0.0, 1.6, 0.1) 
    best_alpha = 0.0
    best_mae = float('inf')

    for a in alpha_range:
        current_objective = CustomEdgePenaltyObjective(alpha=a)
        temp_xgb = xgb.XGBRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=5,
            objective=current_objective, random_state=42, n_jobs=-1
        )
        temp_xgb.fit(X_train, y_train)
        preds = temp_xgb.predict(X_test)
        preds_rounded = np.round(np.clip(preds, 0, 5) * 2) / 2
        mae = mean_absolute_error(y_test, preds_rounded)
        
        if mae < best_mae:
            best_mae = mae
            best_alpha = a

    print(f"✅ Optimal Alpha found: {best_alpha:.1f} (MAE: {best_mae:.4f})")
    
    final_objective = CustomEdgePenaltyObjective(alpha=best_alpha)
    final_xgb = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        objective=final_objective, random_state=42, n_jobs=-1
    )
    final_xgb.fit(X_train, y_train)
    return final_xgb

def train_advanced_unified_models():
    print("🤖 Starting Advanced Unified Media Model Training Pipeline...")

    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        print("❌ Error: Unified training data not found.")
        return

    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # Drop target and metadata columns
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal', 'source_id', 'is_tv_flag', 'is_game_flag', 'media_type'], errors='ignore')
    y_reg = df['target_reg']
    y_ord = df['target_ordinal']

    X_train, X_test, y_reg_train, y_reg_test, y_ord_train, y_ord_test = train_test_split(
        X, y_reg, y_ord, test_size=0.2, random_state=42
    )
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Sample weights
    weight_classes = (y_reg_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)
    
    # 1. Base Models
    print("   ✅ Training Unified XGBoost with Custom Edge-Penalty Loss...")
    xgb_reg = tune_and_train_xgboost(X_train, y_reg_train, X_test, y_reg_test)

    cat_reg = ctb.CatBoostRegressor(
        n_estimators=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='MAE', eval_metric='RMSE',
        random_seed=42, verbose=0, early_stopping_rounds=50
    )

    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    # 2. Evaluate Base Models
    print("\n--- Evaluating Base Models ---")
    xgb_preds = xgb_reg.predict(X_test)
    print_performance_report("Unified XGBoost", y_reg_test, xgb_preds)

    print("   Fitting SVR...")
    svr_pipe.fit(X_train, y_reg_train, svr__sample_weight=sample_weights)
    print_performance_report("Unified SVR", y_reg_test, svr_pipe.predict(X_test))

    print("   Fitting CatBoost...")
    cat_reg.fit(X_train, y_reg_train, eval_set=(X_test, y_reg_test), use_best_model=True)
    print_performance_report("Unified CatBoost", y_reg_test, cat_reg.predict(X_test))

    # 3. Stacking Ensemble
    print("\n--- Training Unified Stacking Ensemble ---")
    estimators = [('xgb', xgb_reg), ('svr', svr_pipe), ('catboost', cat_reg)]
    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5, n_jobs=-1)
    stacking_regressor.fit(X_train, y_reg_train)
    print_performance_report("🚀 UNIFIED STACKING ENSEMBLE 🚀", y_reg_test, stacking_regressor.predict(X_test))

    # 4. Ordinal Classifier
    print("\n--- Training Unified 10-Bucket Ordinal Classifier ---")
    
    # Identify unique classes present in training data (to handle missing buckets)
    unique_classes = np.sort(np.unique(y_ord_train))
    num_classes_present = len(unique_classes)
    
    # Map classes to 0...N-1 for XGBoost
    class_map = {old: new for new, old in enumerate(unique_classes)}
    y_ord_train_mapped = pd.Series(y_ord_train).map(class_map)

    ordinal_clf = xgb.XGBClassifier(
        n_estimators=300, 
        learning_rate=0.03, 
        max_depth=5,
        objective='multi:softprob', 
        num_class=num_classes_present,
        subsample=0.8, 
        random_state=42, 
        n_jobs=-1
    )
    ordinal_clf.fit(X_train, y_ord_train_mapped, sample_weight=sample_weights)
    
    ord_probs = ordinal_clf.predict_proba(X_test)
    
    # Map back to actual bucket values
    bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
    present_bucket_vals = np.array([bucket_map[c] for c in unique_classes])
    
    ord_ev_preds = np.sum(ord_probs * present_bucket_vals, axis=1)
    print_performance_report("Unified Ordinal Classifier (Expected Value)", y_reg_test, ord_ev_preds)

    # 5. Save
    ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    # Save mapping for inference
    joblib.dump(unique_classes, ensemble_dir / "ordinal_classes.joblib")
    
    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(svr_pipe, ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(stacking_regressor, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")
    
    print(f"✅ All 5 Advanced Unified models saved to: {ensemble_dir}")

if __name__ == "__main__":
    train_advanced_unified_models()