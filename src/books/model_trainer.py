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

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def train_models():
    print("🤖 Starting Book Model Training...")

    if not config.BOOKS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.BOOKS_TRAINING_DATA_PATH)

    X = df.drop(columns=['target_reg', 'target_class'], errors='ignore')
    y_reg = df['target_reg']
    y_class = df['target_class']

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. Regressor ---
    print(f"   Training Regressor (XGBoost)...")
    
    regressor = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=42
    )
    regressor.fit(X_train, y_train)

    preds_xgb = regressor.predict(X_test)
    
    # Rounding to 0.5 increments
    def round_h(x): return np.round(np.clip(x, 0, 5) * 2) / 2
    y_pred = round_h(preds_xgb)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("📊 PERFORMANCE REPORT: Book XGBoost Regressor")
    print("="*50)
    print(f"   📉 [TEST SET] MSE:  {mse:.4f}")
    print(f"   📉 [TEST SET] RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] R²:   {r2:.4f}")

    # Detailed Diff Report
    diffs = np.abs(y_test - y_pred)
    total = len(diffs)
    print("   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40)
    print("="*50 + "\n")

    # --- 2. Full Dataset Predictions for Visualization ---
    preds_full = round_h(regressor.predict(X))
    
    results_dir = config.BASE_DIR / "results" / "books"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # KDE Plot
    plt.figure(figsize=(12, 6))
    sns.kdeplot(y_reg, label='Actual (Whole Data)', color='black', fill=True, alpha=0.1)
    sns.kdeplot(preds_full, label='Predicted (Whole Data)', color='green', linestyle='--')
    plt.title("Book Rating Distribution (KDE) - Whole Data")
    plt.xticks(np.arange(0, 5.5, 0.5))
    plt.legend()
    plt.savefig(results_dir / "books_distribution_kde.png")
    plt.close()

    # Histogram
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    sns.histplot(y_reg, bins=np.arange(-0.25, 5.75, 0.5), ax=axes[0], color='gray')
    axes[0].set_title("Actual Ratings (Whole Data)")
    sns.histplot(preds_full, bins=np.arange(-0.25, 5.75, 0.5), ax=axes[1], color='green')
    axes[1].set_title("Predicted Ratings (Whole Data)")
    for ax in axes:
        ax.set_xticks(np.arange(0, 5.5, 0.5))
    plt.suptitle("Book Rating Histograms")
    plt.savefig(results_dir / "books_distribution_histograms.png")
    plt.close()

    # --- 3. Classifier ---
    classifier = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        objective='multi:softmax',
        num_class=3,
        random_state=42
    )
    y_class_train = y_class.iloc[X_train.index]
    classifier.fit(X_train, y_class_train)

    # Save Models
    joblib.dump(regressor, config.BOOKS_MODEL_REGRESSOR)
    joblib.dump(classifier, config.BOOKS_MODEL_CLASSIFIER)
    
    # Save results CSV
    try:
        df_orig = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except:
        df_orig = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
        
    df_results = pd.DataFrame({
        'title': df_orig['title'],
        'actual': y_reg,
        'predicted': preds_full
    })
    df_results.to_csv(results_dir / "book_predictions_full.csv", index=False)
    
    print(f"✅ Models and graphs saved to {results_dir}")

if __name__ == "__main__":
    train_models()
