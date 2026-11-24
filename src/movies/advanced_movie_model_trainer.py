import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.ensemble import mean_ensemble


def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("" + "="*50)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*50)

    # Clip and round predictions to valid rating values (0.5 increments). The floor
    # is 0.5, not 0: Letterboxd has no 0-star rating, so clipping to 0 let a low
    # prediction round to an impossible 0.0 and inflated the error against it.
    y_pred_rounded = np.round(np.clip(y_pred, 0.5, 5.0) * 2) / 2

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


def train_advanced_models():
    """Train the deployed movie model: an equal-weight mean of XGB, CatBoost and SVR.

    The Optuna-tuned edge-penalty stack (ordinal member, crowd anchor, 5-fold inner CV)
    this replaced scored below the plain mean of its own members on the 50 registry
    folds (skill 0.429 vs 0.458); see src/common/ensemble.py.
    """
    print("🤖 Starting Movie Model Training (mean ensemble)...")
    if not config.TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found at:", config.TRAINING_DATA_PATH)
        return

    df = pd.read_csv(config.TRAINING_DATA_PATH)
    y_reg, y_ord = df['target_reg'], df['target_ordinal']
    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'], errors='ignore')

    # Quick 80/20 read-out only; the 50-fold benchmark is the real evaluation, and the
    # saved models are refit on every rating below.
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    holdout = mean_ensemble().fit(X_train, y_train)
    print_performance_report("Mean Ensemble (80/20 hold-out)", y_test, holdout.predict(X_test))
    test_preds = {"Ensemble": holdout.predict(X_test),
                  **{k: m.predict(X_test) for k, m in holdout.named_estimators_.items()}}

    print(f"   Refitting on all {len(X)} rated films...")
    ensemble = mean_ensemble().fit(X, y_reg)

    # Rating-bucket classifier: shown on the Oracle as a probability breakdown only;
    # it is not part of the predicted rating.
    unique_classes = np.sort(np.unique(y_ord))
    ordinal_clf = xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                    objective='multi:softprob', num_class=len(unique_classes),
                                    subsample=0.8, random_state=42, n_jobs=-1)
    ordinal_clf.fit(X, pd.Series(y_ord).map({c: i for i, c in enumerate(unique_classes)}))

    results_dir = config.BASE_DIR / "results" / "movies" / "advanced"
    results_dir.mkdir(parents=True, exist_ok=True)
    round_h = lambda x: np.round(np.clip(x, 0.5, 5) * 2) / 2
    plot_df = pd.DataFrame({'Actual': y_test, **{k: round_h(v) for k, v in test_preds.items()}})

    plt.figure(figsize=(14, 7))
    sns.kdeplot(plot_df['Actual'], label='Actual', color='black', fill=True, alpha=0.1, linewidth=3)
    sns.kdeplot(plot_df['Ensemble'], label='Mean Ensemble', color='red', linewidth=2)
    plt.title("Movie Model: Rating Distribution (KDE, 80/20 hold-out)")
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.legend()
    plt.savefig(results_dir / "advanced_kde.png")
    plt.close()

    cols = list(plot_df.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4), sharey=True)
    for ax, col in zip(axes, cols):
        sns.histplot(plot_df[col], bins=np.arange(0.25, 5.75, 0.5), ax=ax)
        ax.set_title(f"{col} Distribution")
        ax.set_xticks(np.arange(0.5, 5.5, 1.0))
    plt.tight_layout()
    plt.savefig(results_dir / "advanced_histograms.png")
    plt.close()

    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    # File names are what predict_ratings.py and the Oracle page load.
    joblib.dump(ensemble, ensemble_dir / "stacking_ensemble_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["xgb"], ensemble_dir / "xgb_base_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["svr"], ensemble_dir / "svr_base_regressor.joblib")
    joblib.dump(ensemble.named_estimators_["cat"], ensemble_dir / "catboost_base_regressor.joblib")
    joblib.dump(ordinal_clf, ensemble_dir / "ordinal_classifier.joblib")
    joblib.dump(unique_classes, ensemble_dir / "ordinal_classes.joblib")
    print(f"✅ Models saved to {ensemble_dir}; plots to {results_dir}")


if __name__ == "__main__":
    train_advanced_models()
