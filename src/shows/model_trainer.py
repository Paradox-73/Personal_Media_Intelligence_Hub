import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.ensemble import mean_ensemble


def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("="*55)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*55)
    
    # Ensure predictions are within bounds and rounded to nearest 0.5
    y_pred_rounded = np.round(np.clip(y_pred, 0.5, 5) * 2) / 2
    
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


def train_shows_model():
    """Train the deployed TV model: an equal-weight mean of XGB, CatBoost and SVR.

    Replaces an Optuna-tuned edge-penalty XGB + positive-Ridge stack that scored below
    the plain mean on the 50 registry folds (skill 0.229 vs 0.255) and was fit on only
    80% of the ~175 rated shows; see src/common/ensemble.py.
    """
    print("🤖 Starting TV Show Model Training (mean ensemble)...")
    if not config.TV_SHOWS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.TV_SHOWS_TRAINING_DATA_PATH)
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'])
    y_reg, y_ord = df['target_reg'], df['target_ordinal']

    # Quick 80/20 read-out only; the saved models are refit on every rated show.
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    print_performance_report("Mean Ensemble (80/20 hold-out)",
                             y_test, mean_ensemble().fit(X_train, y_train).predict(X_test))

    print(f"   Refitting on all {len(X)} rated shows...")
    ensemble = mean_ensemble().fit(X, y_reg)
    gbr_baseline = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.03,
                                             subsample=0.7, random_state=42).fit(X, y_reg)

    # Rating-bucket classifier: the Oracle's probability breakdown, not part of the rating.
    unique_classes = np.sort(np.unique(y_ord))
    ordinal_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                    objective='multi:softprob', num_class=len(unique_classes),
                                    subsample=0.8, random_state=42, n_jobs=-1)
    ordinal_clf.fit(X, pd.Series(y_ord).map({c: i for i, c in enumerate(unique_classes)}))

    ensemble_dir = config.TV_SHOWS_MODEL_DIR / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    # File names are what predict_ratings.py and the Oracle page load.
    joblib.dump(gbr_baseline, config.TV_SHOWS_MODEL_REGRESSOR)
    joblib.dump(ensemble, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["xgb"], ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["svr"], ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["cat"], ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")
    joblib.dump(unique_classes, ensemble_dir / "ordinal_classes.joblib")
    print(f"✅ Models saved to: {ensemble_dir}")


if __name__ == "__main__":
    train_shows_model()
