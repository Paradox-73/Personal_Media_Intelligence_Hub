import pandas as pd
import numpy as np
import xgboost as xgb
import catboost as ctb
import joblib
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import RidgeCV, LassoCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective

from sklearn.base import BaseEstimator, RegressorMixin

class SimplexWeightedAverager(BaseEstimator, RegressorMixin):
    """
    Constrained non-negative weighted average (simplex-constrained).
    Weights are derived from OOF performance or a simple optimization.
    At small N, this is more robust than a learned RidgeCV.
    """
    _estimator_type = "regressor"
    
    def __init__(self):
        self.weights_ = None
        self.n_features_in_ = None

    def fit(self, X, y):
        from scipy.optimize import minimize
        self.n_features_in_ = X.shape[1]
        
        def objective(w):
            preds = X @ w
            return mean_absolute_error(y, preds)
        
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(self.n_features_in_)]
        initial_w = np.ones(self.n_features_in_) / self.n_features_in_
        
        res = minimize(objective, initial_w, method='SLSQP', bounds=bounds, constraints=cons)
        self.weights_ = res.x
        return self

    def predict(self, X):
        return X @ self.weights_

    def get_params(self, deep=True):
        return {}

def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("="*55)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*55)
    
    # Ensure predictions are within bounds and rounded to nearest 0.5
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
    print("   " + "-"*45)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("="*55 + "\n")

def evaluate_robustly(model, X, y):
    """10-fold x 5-repeat cross-validation for statistically meaningful metrics on small N."""
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    scores = cross_val_score(model, X, y, cv=rkf, scoring='r2', n_jobs=-1)
    print(f"   📈 [Repeated CV] R²: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    return np.mean(scores), np.std(scores)

def tune_and_train_xgboost(X_train, y_train, X_test, y_test):
    print("\n🔍 Sweeping Alphas for Asymmetric Edge Penalty (TV Shows)...")
    import optuna
    
    def objective(trial):
        a_hi = trial.suggest_float("alpha_hi", 0.01, 0.4)
        a_lo = trial.suggest_float("alpha_lo", 0.01, 0.4)
        model = xgb.XGBRegressor(
            objective=AsymmetricEdgePenaltyObjective(alpha_hi=a_hi, alpha_lo=a_lo),
            n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42
        )
        rkf = RepeatedKFold(n_splits=5, n_repeats=1, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=rkf, scoring='neg_mean_absolute_error')
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    final_obj = AsymmetricEdgePenaltyObjective(
        alpha_hi=study.best_params['alpha_hi'], 
        alpha_lo=study.best_params['alpha_lo']
    )
    final_xgb = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=5,
        objective=final_obj, random_state=42, n_jobs=-1
    )
    final_xgb.fit(X_train, y_train)
    return final_xgb

def train_shows_model():
    print("🤖 Starting Consolidated TV Show Model Training Pipeline...")

    if not config.TV_SHOWS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.TV_SHOWS_TRAINING_DATA_PATH)
    
    # Drop target columns to isolate features
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'])
    y_reg = df['target_reg']

    # 80/20 Split for initial assessment, but evaluate_robustly uses whole X
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print(f"   Dataset Size: {len(X)} | Training CV: 10-fold x 5-repeat")

    # --- 1. Base Models ---
    print("\n🔍 Tuning XGBoost...")
    xgb_reg = tune_and_train_xgboost(X_train, y_reg_train, X_test, y_reg_test)
    evaluate_robustly(xgb_reg, X, y_reg)

    print("🐱 Training CatBoost...")
    cat_reg = ctb.CatBoostRegressor(
        n_estimators=500, learning_rate=0.03, depth=5,
        loss_function='MAE', verbose=0, random_seed=42
    )
    evaluate_robustly(cat_reg, X, y_reg)

    print("⚖️ Training SVR...")
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])
    evaluate_robustly(svr_pipe, X, y_reg)

    # --- 2. Stacking Ensemble with Positive Ridge Meta-Learner ---
    print("\n🚀 Training Positive-Weighted Stacking Ensemble...")
    estimators = [('xgb', xgb_reg), ('svr', svr_pipe), ('catboost', cat_reg)]

    # Using Ridge with positive=True acts as a constrained non-negative weighted average
    # which is very robust for small N stacking.
    stacking_reg = StackingRegressor(
        estimators=estimators, 
        final_estimator=Ridge(positive=True), 
        cv=5, 
        n_jobs=-1
    )

    stacking_reg.fit(X_train, y_reg_train)
    print_performance_report("SIMPLEX STACKING", y_reg_test, stacking_reg.predict(X_test))
    evaluate_robustly(stacking_reg, X, y_reg)

    # --- 4. Ordinal Classifier (Bucket Probability -> EV) ---
    print("\n🔢 Training 10-Bucket Ordinal Classifier...")
    # Load all targets for this part
    y_ord = df['target_ordinal']
    y_class = df['target_class']
    _, _, y_ord_train, y_ord_test = train_test_split(X, y_ord, test_size=0.2, random_state=42)
    
    unique_classes = np.sort(np.unique(y_ord_train))
    class_map = {old: new for new, old in enumerate(unique_classes)}
    y_ord_train_mapped = pd.Series(y_ord_train).map(class_map)
    
    # Weights for ordinal
    weight_classes = (y_reg_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)

    ordinal_clf = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        objective='multi:softprob', num_class=len(unique_classes),
        subsample=0.8, random_state=42, n_jobs=-1
    )
    ordinal_clf.fit(X_train, y_ord_train_mapped, sample_weight=sample_weights)

    # --- 5. Save Artifacts ---
    ensemble_dir = config.TV_SHOWS_MODEL_DIR / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    
    # Baseline
    gbr_baseline = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.03, subsample=0.7, random_state=42
    )
    gbr_baseline.fit(X_train, y_reg_train)
    joblib.dump(gbr_baseline, config.TV_SHOWS_MODEL_REGRESSOR)
    
    # evaluate_robustly only cross-validates (fits internal clones); fit the base
    # estimators themselves before persisting so they are usable standalone.
    svr_pipe.fit(X_train, y_reg_train)
    cat_reg.fit(X_train, y_reg_train)

    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(svr_pipe, ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(stacking_reg, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")
    joblib.dump(unique_classes, ensemble_dir / "ordinal_classes.joblib")
    
    print(f"✅ Models saved to: {ensemble_dir}")
    
    print(f"✅ All models saved to: {config.TV_SHOWS_MODEL_DIR}")

if __name__ == "__main__":
    train_shows_model()
