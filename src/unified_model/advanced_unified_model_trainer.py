import pandas as pd
import xgboost as xgb
import catboost as ctb
import joblib
import sys
import numpy as np
import optuna
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.optimize import minimize

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective
from src.unified_model.unified_utils import DomainAligner, compute_temporal_weights

class SimplexWeightedAverager:
    """
    Constrained non-negative weighted averager (Simplex).
    Finds weights that sum to 1 and are all >= 0.
    """
    def __init__(self):
        self.weights = None
        self.n_models = 0

    def fit(self, X, y):
        self.n_models = X.shape[1]
        def loss(w):
            preds = X @ w
            return mean_squared_error(y, preds)

        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(self.n_models)]
        res = minimize(loss, x0=np.ones(self.n_models)/self.n_models, bounds=bounds, constraints=cons)
        self.weights = res.x
        return self

    def predict(self, X):
        return X @ self.weights

def train_advanced_unified_models():
    print("🤖 Starting Advanced Unified Media Model Training Pipeline (v4)...")

    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        print("❌ Error: Unified training data not found.")
        return

    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df['rating_date'] = pd.to_datetime(df['rating_date'])
    pca_cols = [c for c in df.columns if c.startswith('pca_')]
    
    X = df.drop(columns=['target_reg', 'target_ordinal', 'source_id', 'media_type', 'rating_date'], errors='ignore')
    y_reg = df['target_reg']
    media_types = df['media_type']
    dates = df['rating_date']

    # 1. Tune Hyperparameters
    print("🚀 Tuning Temporal Decay (λ) and Asymmetric Penalties...")
    def objective(trial):
        l_decay = trial.suggest_float("lambda_decay", 1e-5, 0.01, log=True)
        alpha_hi = trial.suggest_float("alpha_hi", 0.05, 0.3)
        alpha_lo = trial.suggest_float("alpha_lo", 0.05, 0.3)
        xt, xv, yt, yv, mt, mv, dt, dv = train_test_split(X, y_reg, media_types, dates, test_size=0.2, random_state=42)
        
        aligner = DomainAligner(method='centroid')
        aligner.fit(xt[pca_cols].values, mt)
        xt_p = xt.copy(); xt_p.loc[:, pca_cols] = aligner.transform(xt[pca_cols].values, mt)
        xv_p = xv.copy(); xv_p.loc[:, pca_cols] = aligner.transform(xv[pca_cols].values, mv)
        
        w = compute_temporal_weights(dt, lambda_decay=l_decay)
        model = xgb.XGBRegressor(objective=AsymmetricEdgePenaltyObjective(alpha_hi=alpha_hi, alpha_lo=alpha_lo), n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(xt_p, yt, sample_weight=w)
        return mean_absolute_error(yv, model.predict(xv_p))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    best_lambda = study.best_params['lambda_decay']

    # 2. Final Training
    weights = compute_temporal_weights(dates, lambda_decay=best_lambda)
    X_train, X_test, y_train, y_test, w_train, w_test, m_train, m_test = train_test_split(X, y_reg, weights, media_types, test_size=0.2, random_state=42)

    aligner = DomainAligner(method='centroid')
    aligner.fit(X_train[pca_cols].values, m_train)
    X_train_a = X_train.copy(); X_train_a.loc[:, pca_cols] = aligner.transform(X_train[pca_cols].values, m_train)
    X_test_a = X_test.copy(); X_test_a.loc[:, pca_cols] = aligner.transform(X_test[pca_cols].values, m_test)

    # Base Models
    xgb_reg = xgb.XGBRegressor(objective=AsymmetricEdgePenaltyObjective(alpha_hi=study.best_params['alpha_hi'], alpha_lo=study.best_params['alpha_lo']), n_estimators=300, learning_rate=0.03, max_depth=6, random_state=42)
    xgb_reg.fit(X_train_a, y_train, sample_weight=w_train)

    cat_reg = ctb.CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, loss_function='MAE', verbose=0, random_seed=42)
    cat_reg.fit(X_train_a, y_train, sample_weight=w_train)

    # 3. Simplex Ensemble
    print("   Training Simplex Ensemble (XGB + CatBoost)...")
    oof_xgb = cross_val_predict(xgb_reg, X_train_a, y_train, cv=5)
    oof_cat = cross_val_predict(cat_reg, X_train_a, y_train, cv=5)
    X_oof = np.column_stack([oof_xgb, oof_cat])
    
    ensemble = SimplexWeightedAverager()
    ensemble.fit(X_oof, y_train)
    print(f"   Ensemble Weights: XGB={ensemble.weights[0]:.3f}, Cat={ensemble.weights[1]:.3f}")

    # 4. Evaluation
    def get_met(yt, yp):
        yp_r = np.round(np.clip(yp, 0.5, 5.0) * 2) / 2
        return {"MAE": mean_absolute_error(yt, yp_r), "R2": r2_score(yt, yp_r)}

    xgb_preds = xgb_reg.predict(X_test_a)
    cat_preds = cat_reg.predict(X_test_a)
    final_preds = ensemble.predict(np.column_stack([xgb_preds, cat_preds]))

    print("\n" + "="*60)
    print("📊 UNIFIED PERFORMANCE REPORT (v4)")
    print("="*60)
    print(f"   XGB Base:      {get_met(y_test, xgb_preds)}")
    print(f"   CatBoost Base: {get_met(y_test, cat_preds)}")
    print(f"   FULL ENSEMBLE: {get_met(y_test, final_preds)}")
    print("-" * 60)
    for d in m_test.unique():
        mask = (m_test == d)
        if mask.any():
            m = get_met(y_test[mask], final_preds[mask])
            print(f"   [{d.upper():<7}] (N={mask.sum():>3}) MAE: {m['MAE']:.4f} | R²: {m['R2']:.4f}")
    print("="*60)

    # 5. Save
    ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
    joblib.dump(ensemble, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump({'best_lambda': best_lambda, 'aligner': aligner, 'pca_cols': pca_cols, 'weights': ensemble.weights}, ensemble_dir / "model_metadata.joblib")
    
    # Update state for inference
    state = joblib.load(config.UNIFIED_PREPROCESSOR_STATE)
    state['aligner'] = aligner
    joblib.dump(state, config.UNIFIED_PREPROCESSOR_STATE)
    
    print(f"✅ Models saved to: {ensemble_dir}")

if __name__ == "__main__":
    train_advanced_unified_models()
