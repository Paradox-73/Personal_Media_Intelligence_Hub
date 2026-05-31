import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def train_models():
    print("🤖 Starting Movie Model Training (v2)...")

    if not config.TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.TRAINING_DATA_PATH)

    X = df.drop(columns=['target_reg', 'target_class', 'target_ordinal'], errors='ignore')
    y_reg = df['target_reg']
    y_class = df['target_class']

    # 80/20 Split
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )

    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. Regressor ---
    print(f"   Training Regressor...")

    # Calculate sample weights to handle imbalanced ratings
    weight_classes = (y_reg_train * 2).round() / 2 # Round to nearest 0.5
    sample_weights = compute_sample_weight(class_weight='balanced', y=weight_classes)

    regressor = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=42
    )
    regressor.fit(X_train, y_reg_train, sample_weight=sample_weights)

    preds_reg_raw = regressor.predict(X_test)
    preds_reg = np.round(np.clip(preds_reg_raw, 0, 5) * 2) / 2

    mse = mean_squared_error(y_reg_test, preds_reg)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_reg_test, preds_reg)
    r2 = r2_score(y_reg_test, preds_reg)

    print("\n" + "="*50)
    print("📊 PERFORMANCE REPORT: Baseline XGBoost")
    print("="*50)
    print(f"   📉 [TEST SET] Regressor MSE:  {mse:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")

    # Detailed Diff Report
    diffs = np.abs(y_reg_test - preds_reg)
    total = len(diffs)
    print("   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40)
    print("="*50 + "\n")

    # --- 2. Visualization ---
    results_dir = config.BASE_DIR / "results" / "movies" / "baseline"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # KDE Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_reg_test, label='Actual', color='black', fill=True, alpha=0.1)
    sns.kdeplot(preds_reg, label='Baseline Predicted', color='orange', linestyle='--')
    plt.title("Baseline XGBoost: Rating Distribution (KDE)")
    plt.xticks(np.arange(0.5, 5.5, 0.5))
    plt.legend()
    plt.savefig(results_dir / "baseline_kde.png")
    plt.close()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    sns.histplot(y_reg_test, bins=np.arange(0.25, 5.75, 0.5), ax=axes[0], color='gray')
    axes[0].set_title("Actual Ratings")
    sns.histplot(preds_reg, bins=np.arange(0.25, 5.75, 0.5), ax=axes[1], color='orange')
    axes[1].set_title("Baseline Predicted")
    for ax in axes:
        ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    plt.suptitle("Baseline XGBoost: Rating Histograms")
    plt.savefig(results_dir / "baseline_histograms.png")
    plt.close()

    # --- 3. Classifier ---
    print(f"   Training Classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        objective='multi:softmax',
        num_class=3,
        random_state=42
    )
    classifier.fit(X_train, y_clf_train)

    acc = accuracy_score(y_clf_test, classifier.predict(X_test))
    print(f"   🎯 [TEST SET] Accuracy: {acc*100:.2f}%")

    joblib.dump(regressor, config.MODEL_REGRESSOR)
    joblib.dump(classifier, config.MODEL_CLASSIFIER)
    print(f"✅ Baseline models and graphs saved to {results_dir}")

if __name__ == "__main__":
    train_models()
