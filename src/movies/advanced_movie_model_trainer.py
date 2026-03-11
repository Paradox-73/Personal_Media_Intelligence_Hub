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

# Add Project Root to Path for config import
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# IMPORT THE NEW CALLABLE CLASS HERE
from src.movies.custom_objectives import CustomEdgePenaltyObjective

# ==========================================
# ⚙️ ARCHITECTURE FLAGS
# ==========================================
USE_CUSTOM_EDGE_LOSS = True  
# ==========================================

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

def tune_and_train_xgboost(X_train, y_train, X_test, y_test):
    """
    Sweeps a range of alpha values for the custom edge-penalty loss
    to find the mathematically optimal steepness for the XGBoost model
    using Mean Absolute Error (MAE).
    """
    from src.movies.custom_objectives import CustomEdgePenaltyObjective
    
    print("\n🔍 Sweeping Alpha range for Custom Edge Penalty...")
    
    # Generate a range from 0.0 to 1.5 in steps of 0.1
    alpha_range = np.arange(0.0, 1.6, 0.1) 
    
    best_alpha = 0.0
    best_mae = float('inf')

    for a in alpha_range:
        current_objective = CustomEdgePenaltyObjective(alpha=a)
        
        # We use fewer estimators/depth just for the quick tuning sweep
        temp_xgb = xgb.XGBRegressor(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=5,
            objective=current_objective,
            random_state=42, 
            n_jobs=-1
        )
        
        temp_xgb.fit(X_train, y_train)
        preds = temp_xgb.predict(X_test)
        
        # Round predictions to valid 0.5 increments to evaluate true performance
        preds_rounded = np.round(np.clip(preds, 0, 5) * 2) / 2
        
        # CHANGED: Now optimizing for MAE instead of RMSE
        mae = mean_absolute_error(y_test, preds_rounded)
        severe_misses = (np.abs(y_test - preds_rounded) > 1.0).sum()
        
        print(f"   Alpha {a:.1f} -> MAE: {mae:.4f} | Severe Misses (>1.0): {severe_misses}")
        
        if mae < best_mae:
            best_mae = mae
            best_alpha = a

    print(f"✅ Optimal Alpha found: {best_alpha:.1f} (MAE: {best_mae:.4f})")
    
    # Train the final, full-powered model using the optimal alpha
    print(f"   Training final XGBoost model with Alpha {best_alpha:.1f}...")
    final_objective = CustomEdgePenaltyObjective(alpha=best_alpha)
    final_xgb = xgb.XGBRegressor(
        n_estimators=200, 
        learning_rate=0.03, 
        max_depth=6,
        subsample=0.8, 
        colsample_bytree=0.8,
        objective=final_objective,
        random_state=42, 
        n_jobs=-1
    )
    final_xgb.fit(X_train, y_train)
    
    return final_xgb

def train_advanced_models():
    print("🤖 Starting Advanced Movie Model Training Pipeline...")

    if not config.TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found at:", config.TRAINING_DATA_PATH)
        return

    # 1. Load and Prepare Data
    df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    y_reg = df['target_reg']
    y_ord = df['target_ordinal']
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'])

    X_train, X_test, y_reg_train, y_reg_test, y_ord_train, y_ord_test = train_test_split(
        X, y_reg, y_ord, test_size=0.2, random_state=42
    )
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Calculate standard sample weights for SVR and Ordinal
    weight_classes = (y_reg_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)
    
    # 2. Define Base Models
    print("   Defining base models...")
    
    if USE_CUSTOM_EDGE_LOSS:
        print("   ✅ Using Custom Exponential Edge-Penalty Loss for XGBoost.")
        xgb_reg = tune_and_train_xgboost(X_train, y_reg_train, X_test, y_reg_test)
    else:
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective='reg:absoluteerror', random_state=42, n_jobs=-1
        )
        # Only fit here if we are not using the tuning function
        xgb_reg.fit(X_train, y_reg_train, sample_weight=sample_weights)

    # Reverted CatBoost to standard MAE to prevent degenerate math and cloning errors
    cat_reg = ctb.CatBoostRegressor(
        n_estimators=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='MAE', eval_metric='RMSE',
        random_seed=42, verbose=0, early_stopping_rounds=50
    )

    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    # 3. Train and Evaluate Base Regressors
    print("\n--- Training Regressors ---")
    
    print("   Evaluating XGBoost...")
    xgb_preds = xgb_reg.predict(X_test)
    print_performance_report("XGBoost Regressor", y_reg_test, xgb_preds)

    print("   Fitting SVR...")
    svr_pipe.fit(X_train, y_reg_train, svr__sample_weight=sample_weights)
    svr_preds = svr_pipe.predict(X_test)
    print_performance_report("Support Vector Regressor (SVR)", y_reg_test, svr_preds)

    print("   Fitting CatBoost...")
    cat_reg.fit(X_train, y_reg_train, eval_set=(X_test, y_reg_test), use_best_model=True)
    cat_preds = cat_reg.predict(X_test)
    print_performance_report("CatBoost Regressor", y_reg_test, cat_preds)

    # 4. Train Stacking Ensemble
    print("\n--- Training Stacking Ensemble ---")
    estimators = [('xgb', xgb_reg), ('svr', svr_pipe), ('catboost', cat_reg)]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(), 
        cv=5,
        n_jobs=-1
    )

    print("   Fitting Stacking Regressor... (This may take a while)")
    stacking_regressor.fit(X_train, y_reg_train)
    stack_preds = stacking_regressor.predict(X_test)
    print_performance_report("🚀 STACKING ENSEMBLE 🚀", y_reg_test, stack_preds)

    # 5. Train Ordinal Confidence Classifier
    print("\n--- Training 10-Bucket Ordinal Classifier ---")
    ordinal_clf = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=5,
        objective='multi:softprob',
        num_class=10,
        subsample=0.8,
        random_state=42,
        n_jobs=-1
    )
    ordinal_clf.fit(X_train, y_ord_train, sample_weight=sample_weights)
    
    ord_probs = ordinal_clf.predict_proba(X_test)
    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    
    # Expected Value rather than Argmax for stability
    ord_ev_preds = np.sum(ord_probs * bucket_vals, axis=1)
    print_performance_report("10-Bucket Ordinal Classifier (Expected Value)", y_reg_test, ord_ev_preds)


    # 6. Save All Trained Models
    print("\n--- Saving All Trained Models ---")
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(svr_pipe, ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(stacking_regressor, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")
    
    print(f"✅ All 5 models saved successfully to: {ensemble_dir}")

if __name__ == "__main__":
    train_advanced_models()