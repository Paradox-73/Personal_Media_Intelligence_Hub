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
from sklearn.linear_model import RidgeCv
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add Project Root to Path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("" + "="*50)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*50)
    
    # Clip and round predictions to valid rating values (0.5 increments)
    y_pred_rounded = np.round(np.clip(y_pred, 0, 5) * 2) / 2
    
    mse = mean_squared_error(y_true, y_pred_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_rounded)
    r2 = r2_score(y_true, y_pred_rounded)

    print(f"   📉 [TEST SET] Regressor MSE:  {mse:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")

    # Detailed Difference Report
    diffs = np.abs(y_true - y_pred_rounded)
    total = len(diffs)
    print("" + "   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40)
    print("="*50)


def train_advanced_models():
    print("🤖 Starting Advanced Movie Model Training Pipeline (No MLP)...")

    if not config.TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found at:", config.TRAINING_DATA_PATH)
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    X = df.drop(columns=['target_reg', 'target_class'])
    y_reg = df['target_reg']

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Calculate sample weights to FORCE models to care about 0.5s and 5.0s
    weight_classes = (y_reg_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)
    
    # 2. Define Base Models
    print("   Defining base models...")
    
    xgb_reg = xgb.XGBRegressor(
        n_estimators=200,
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

    # 3. Train and Evaluate Each Base Model Individually
    print("--- Training and Evaluating Base Models ---")
    
    print("   Fitting XGBoost...")
    xgb_reg.fit(X_train, y_reg_train, sample_weight=sample_weights)
    xgb_preds = xgb_reg.predict(X_test)
    print_performance_report("XGBoost Regressor", y_reg_test, xgb_preds)

    print("   Fitting SVR...")
    # Double underscore passes the weights directly to the SVR step in the pipeline
    svr_pipe.fit(X_train, y_reg_train)
    svr_preds = svr_pipe.predict(X_test)
    print_performance_report("Support Vector Regressor (SVR)", y_reg_test, svr_preds)

    print("   Fitting CatBoost...")
    cat_reg.fit(X_train, y_reg_train, eval_set=(X_test, y_reg_test), use_best_model=True)
    cat_preds = cat_reg.predict(X_test)
    print_performance_report("CatBoost Regressor", y_reg_test, cat_preds)

    # 4. Train Stacking Ensemble
    print("--- Training Stacking Ensemble ---")
    
    estimators = [
        ('xgb', xgb_reg),
        ('svr', svr_pipe),
        ('catboost', cat_reg)
    ]

    # Using RidgeCv removes the L2 shrinkage, allowing for more extreme predictions
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCv(), 
        cv=5,
        n_jobs=-1
    )

    print("   Fitting Stacking Regressor... (This may take a while)")
    stacking_regressor.fit(X_train, y_reg_train)
    
    stack_preds = stacking_regressor.predict(X_test)
    print_performance_report("🚀 STACKING ENSEMBLE 🚀", y_reg_test, stack_preds)

    # 5. Save All Trained Models
    print("\n--- Saving All Trained Models ---")
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    print(f"   Saving models to: {ensemble_dir}")

    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(svr_pipe, ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(stacking_regressor, ensemble_dir / "stacking_ensemble_regressor.joblib")
    
    print(f"\n✅ All 4 models (3 Base + 1 Ensemble) saved successfully.")

if __name__ == "__main__":
    train_advanced_models()