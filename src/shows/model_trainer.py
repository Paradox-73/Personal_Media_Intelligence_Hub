import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def train_models():
    print("🤖 STARTING TV SHOW MODEL TRAINING (DEEPER LEARNING)...")

    if not config.TV_SHOWS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.TV_SHOWS_TRAINING_DATA_PATH)
    
    X = df.drop(columns=['target_reg', 'target_class'])
    y_reg = df['target_reg']
    y_class = df['target_class']

    # 80/20 Split
    X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=42)

    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. Regressor ---
    print("\n   📉 Training Regressor...")
    regressor = GradientBoostingRegressor(
        n_estimators=100,      # Back to 100 to let it learn nuanced patterns
        max_depth=4,           # INCREASED: Allows complex interactions (Network + Genre + Year)
        learning_rate=0.03,    # Slower learning for better generalization
        subsample=0.7,         # Aggressive regularization to prevent overfitting
        random_state=42
    )
    regressor.fit(X_train, y_train_reg)
    
    # Test Metrics
    preds_reg = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_reg, preds_reg))
    mae = mean_absolute_error(y_test_reg, preds_reg)
    print(f"      [TEST SET] RMSE: {rmse:.4f}")
    print(f"      [TEST SET] MAE:  {mae:.4f}")
    
    joblib.dump(regressor, config.TV_SHOWS_MODEL_REGRESSOR)

    # --- 2. Classifier ---
    print("\n   🎯 Training Classifier...")
    classifier = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.7,
        random_state=42
    )
    if len(np.unique(y_train_clf)) > 1:
        classifier.fit(X_train, y_train_clf)
        acc = accuracy_score(y_test_clf, classifier.predict(X_test))
        print(f"      [TEST SET] Accuracy: {acc*100:.2f}%")
        joblib.dump(classifier, config.TV_SHOWS_MODEL_CLASSIFIER)
    else:
        print("      ⚠️ Skipping Classifier (Single class)")

    print(f"\n✅ Models saved to {config.TV_SHOWS_MODEL_DIR}")

if __name__ == "__main__":
    train_models()