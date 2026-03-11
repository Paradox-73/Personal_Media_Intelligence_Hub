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
# Add project root to path                                                                                                                    │    
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
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

def train_universal_models():
    print("🤖 Starting Unified Media Model Training Pipeline...")

    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # Drop target and metadata columns
    X = df.drop(columns=['target_reg', 'source_id'])
    y = df['target_reg']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Sample weights
    weight_classes = (y_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)
    
    # --- Model Definitions ---
    xgb_reg = xgb.XGBRegressor(
        n_estimators=300, # Increased for larger dataset
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=42,
        n_jobs=-1
    )

    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    cat_reg = ctb.CatBoostRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function='MAE',
        eval_metric='RMSE',
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50
    )

    # --- Training ---
    print("\n   Fitting XGBoost...")
    xgb_reg.fit(X_train, y_train, sample_weight=sample_weights)
    print_performance_report("Unified XGBoost", y_test, xgb_reg.predict(X_test))

    print("   Fitting SVR...")
    svr_pipe.fit(X_train, y_train)
    print_performance_report("Unified SVR", y_test, svr_pipe.predict(X_test))

    print("   Fitting CatBoost...")
    cat_reg.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
    print_performance_report("Unified CatBoost", y_test, cat_reg.predict(X_test))

    # --- Stacking ---
    print("\n   Fitting Unified Stacking Regressor...")
    estimators = [('xgb', xgb_reg), ('svr', svr_pipe), ('catboost', cat_reg)]
    stacking_regressor = StackingRegressor(estimators=estimators, final_estimator=RidgeCV(), cv=5, n_jobs=-1)
    stacking_regressor.fit(X_train, y_train)
    print_performance_report("🚀 UNIFIED STACKING ENSEMBLE 🚀", y_test, stacking_regressor.predict(X_test))

    # --- Save ---
    unified_model_dir = config.UNIFIED_ENSEMBLE_DIR
    unified_model_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(xgb_reg, unified_model_dir / "xgb_base.joblib")
    joblib.dump(svr_pipe, unified_model_dir / "svr_base.joblib")
    joblib.dump(cat_reg, unified_model_dir / "catboost_base.joblib")
    joblib.dump(stacking_regressor, unified_model_dir / "stacking_ensemble.joblib")
    print(f"\n✅ All Unified models saved to {unified_model_dir}")

if __name__ == "__main__":
    train_universal_models()