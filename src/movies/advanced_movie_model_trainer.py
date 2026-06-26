import pandas as pd
import xgboost as xgb
import catboost as ctb
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
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
from src.movies.custom_objectives import AsymmetricEdgePenaltyObjective

# ==========================================
# ⚙️ ARCHITECTURE FLAGS
# ==========================================
USE_CUSTOM_EDGE_LOSS = True  
# ==========================================

from sklearn.base import BaseEstimator, RegressorMixin

class OrdinalExpectedValueRegressor(RegressorMixin, BaseEstimator):
    """Wraps the Ordinal Classifier to act as a regressor for stacking.

    Mixin must precede BaseEstimator so RegressorMixin.__sklearn_tags__ sets
    estimator_type='regressor' (sklearn >=1.6 tag-based is_regressor); the
    reverse order shadows it and StackingRegressor rejects this as 'not a regressor'.
    """
    _estimator_type = "regressor"
    
    def __init__(self, clf, unique_classes):
        self.clf = clf
        self.unique_classes = unique_classes
        self.bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
        self.present_bucket_vals = np.array([self.bucket_map[c] for c in unique_classes])

    def fit(self, X, y, sample_weight=None):
        # CV-safe: StackingRegressor clones this and refits clones on CV folds, so a
        # no-op fit leaves the inner classifier unfitted -> NotFittedError. Fit a fresh
        # clone here. Stacking passes the continuous regression target, so map ratings
        # -> ordinal bucket indices (0.5->0 ... 5.0->9), remapped contiguously per fold.
        from sklearn.base import clone
        raw = np.clip(np.rint(np.asarray(y, dtype=float) * 2).astype(int) - 1, 0, 9)
        classes = np.sort(np.unique(raw))
        cmap = {int(c): i for i, c in enumerate(classes)}
        y_mapped = np.array([cmap[int(c)] for c in raw])
        clf = clone(self.clf)
        if hasattr(clf, "set_params"):
            try:
                clf.set_params(num_class=len(classes))
            except Exception:
                pass
        try:
            clf.fit(X, y_mapped, sample_weight=sample_weight)
        except TypeError:
            clf.fit(X, y_mapped)
        self.clf_ = clf
        self.bucket_vals_ = np.array([self.bucket_map[int(c)] for c in classes])
        return self

    def predict(self, X):
        # Use the fold-fitted clone if present; else fall back to a pre-fitted clf
        # (standalone use of the original wrapper).
        clf = getattr(self, "clf_", None)
        if clf is None:
            return np.sum(self.clf.predict_proba(X) * self.present_bucket_vals, axis=1)
        return np.sum(clf.predict_proba(X) * self.bucket_vals_, axis=1)
    
    def get_params(self, deep=True):
        return {"clf": self.clf, "unique_classes": self.unique_classes}

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
    rho, _ = spearmanr(y_true, y_pred)

    print(f"   📉 [TEST SET] Regressor MSE:      {mse:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE:     {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:      {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:       {r2:.4f}")
    print(f"   🔝 [TEST SET] Regressor Spearman: {rho:.4f}")

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

def tune_with_optuna(X_train, y_train):
    """
    Jointly tunes asymmetric alpha_hi/alpha_lo and tree params using Optuna.
    Uses 5x2 RepeatedKFold for robust validation as recommended.
    """
    import optuna
    from sklearn.model_selection import RepeatedKFold

    def objective(trial):
        a_hi = trial.suggest_float("alpha_hi", 0.01, 0.5, log=True)
        a_lo = trial.suggest_float("alpha_lo", 0.01, 0.5, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        
        rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)
        maes = []
        
        for t_idx, v_idx in rkf.split(X_train):
            xt, xv = X_train.iloc[t_idx], X_train.iloc[v_idx]
            yt, yv = y_train.iloc[t_idx], y_train.iloc[v_idx]
            
            model = xgb.XGBRegressor(
                objective=AsymmetricEdgePenaltyObjective(alpha_hi=a_hi, alpha_lo=a_lo),
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                reg_lambda=reg_lambda,
                n_estimators=150,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
            model.fit(xt, yt)
            preds = model.predict(xv)
            preds_rounded = np.round(np.clip(preds, 0, 5) * 2) / 2
            maes.append(mean_absolute_error(yv, preds_rounded))
            
        return np.mean(maes)

    print("\n🚀 Starting Optuna Hyperparameter Sweep...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)
    
    print(f"✅ Best Trial: {study.best_params}")
    
    best_obj = AsymmetricEdgePenaltyObjective(
        alpha_hi=study.best_params['alpha_hi'],
        alpha_lo=study.best_params['alpha_lo']
    )
    final_xgb = xgb.XGBRegressor(
        objective=best_obj,
        max_depth=study.best_params['max_depth'],
        min_child_weight=study.best_params['min_child_weight'],
        reg_lambda=study.best_params['reg_lambda'],
        n_estimators=300,
        learning_rate=0.03,
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
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'], errors='ignore')

    X_train, X_test, y_reg_train, y_reg_test, y_ord_train, y_ord_test = train_test_split(
        X, y_reg, y_ord, test_size=0.2, random_state=42
    )
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # Calculate standard sample weights for SVR and Ordinal
    weight_classes = (y_reg_train * 2).round() / 2
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)
    # Edge boost: heavily up-weight extreme-rating items (<=1.5 / >=4.5) on TOP of the
    # balanced weights, so missing a 5/4.5 or a 1/0.5 is penalised hard. Without this the
    # MAE-optimal move on a 3.0-3.5-clustered distribution is to hug the mode (the model
    # otherwise never predicts >=4.5); this deliberately trades a little overall MAE for
    # genuine calibration at the extremes the user cares about.
    EDGE_WEIGHT_BOOST = 5.0
    _edge = (weight_classes <= 1.5) | (weight_classes >= 4.5)
    sample_weights = sample_weights * np.where(_edge, EDGE_WEIGHT_BOOST, 1.0)

    # 2. Define and Train Ordinal Classifier FIRST (to fuse it)
    print("\n--- Training 10-Bucket Ordinal Classifier ---")
    unique_classes = np.sort(np.unique(y_ord_train))
    num_classes_present = len(unique_classes)
    class_map = {old: new for new, old in enumerate(unique_classes)}
    y_ord_train_mapped = pd.Series(y_ord_train).map(class_map)

    ordinal_clf = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.03, max_depth=5,
        objective='multi:softprob', num_class=num_classes_present,
        subsample=0.8, random_state=42, n_jobs=-1
    )
    ordinal_clf.fit(X_train, y_ord_train_mapped, sample_weight=sample_weights)
    
    ord_wrapper = OrdinalExpectedValueRegressor(ordinal_clf, unique_classes)

    # 3. Define Base Regressors
    print("   Defining base models...")

    if USE_CUSTOM_EDGE_LOSS:
        xgb_reg = tune_with_optuna(X_train, y_reg_train)
    else:
        xgb_reg = xgb.XGBRegressor(
            n_estimators=200, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            objective='reg:absoluteerror', random_state=42, n_jobs=-1
        )
        xgb_reg.fit(X_train, y_reg_train, sample_weight=sample_weights)

    cat_reg = ctb.CatBoostRegressor(
        n_estimators=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='MAE', eval_metric='RMSE',
        random_seed=42, verbose=0, early_stopping_rounds=50
    )

    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
    ])

    # 4. Train and Evaluate Base Regressors
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

    # 5. Train Stacking Ensemble (FUSED with Ordinal)
    print("\n--- Training Stacking Ensemble ---")
    # Fusing the ordinal classifier expected value into the stack
    estimators = [
        ('xgb', xgb_reg), 
        ('svr', svr_pipe), 
        ('catboost', cat_reg),
        ('ordinal_ev', ord_wrapper)
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(), 
        cv=5,
        n_jobs=-1
    )

    print("   Fitting Stacking Regressor (Fused)...")
    stacking_regressor.fit(X_train, y_reg_train)
    stack_preds = stacking_regressor.predict(X_test)
    print_performance_report("🚀 FUSED STACKING ENSEMBLE 🚀", y_reg_test, stack_preds)

    # Ordinal expected-value predictions (used in the diagnostic plots below).
    # Previously referenced but never computed -> NameError that aborted the run.
    ord_ev_preds = ord_wrapper.predict(X_test)

    # --- 6. Visualizations (Advanced Ensemble) ---
    print("\n--- Generating Ensemble Visualizations ---")
    results_dir = config.BASE_DIR / "results" / "movies" / "advanced"
    results_dir.mkdir(parents=True, exist_ok=True)

    def round_h(x): return np.round(np.clip(x, 0, 5) * 2) / 2

    # Prep data for plotting
    plot_df = pd.DataFrame({
        'Actual': y_reg_test,
        'Stacking': round_h(stack_preds),
        'XGB': round_h(xgb_preds),
        'CatBoost': round_h(cat_preds),
        'SVR': round_h(svr_preds),
        'Ordinal_EV': round_h(ord_ev_preds)
    })

    # KDE Plot
    plt.figure(figsize=(14, 7))
    sns.kdeplot(plot_df['Actual'], label='Actual', color='black', fill=True, alpha=0.1, linewidth=3)
    sns.kdeplot(plot_df['Stacking'], label='Stacking Ensemble', color='red', linewidth=2)
    sns.kdeplot(plot_df['Ordinal_EV'], label='Ordinal EV', color='blue', linestyle='--')
    plt.title("Advanced Ensemble: Rating Distribution (KDE)")
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.legend()
    plt.savefig(results_dir / "advanced_kde.png")
    plt.close()

    # Histogram Grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    cols = ['Actual', 'Stacking', 'Ordinal_EV', 'XGB', 'CatBoost', 'SVR']
    colors = ['gray', 'red', 'blue', 'orange', 'green', 'purple']

    for i, col in enumerate(cols):
        sns.histplot(plot_df[col], bins=np.arange(0.25, 5.75, 0.5), ax=axes[i], color=colors[i])
        axes[i].set_title(f"{col} Distribution")
        axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))

    plt.suptitle("Advanced Ensemble: Component Histograms")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(results_dir / "advanced_histograms.png")
    plt.close()

    # 7. Save All Trained Models
    print("\n--- Saving All Trained Models ---")
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)

    # Save the class mapping alongside the model
    joblib.dump(unique_classes, ensemble_dir / "ordinal_classes.joblib")

    joblib.dump(xgb_reg, ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(svr_pipe, ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(cat_reg, ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(stacking_regressor, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")

    print(f"✅ All 5 models and ensemble graphs saved to {results_dir}")

if __name__ == "__main__":
    train_advanced_models()