import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def train_models():
    print("🤖 STARTING TV SHOW MODEL TRAINING...")

    if not config.TV_SHOWS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found. Run feature_engineering.py first.")
        return

    df = pd.read_csv(config.TV_SHOWS_TRAINING_DATA_PATH)
    
    # Verify Columns
    required_cols = ['target_reg', 'target_class']
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: Missing columns {required_cols}. Run feature_engineering.py again.")
        return

    X = df.drop(columns=['target_reg', 'target_class'])
    y_reg = df['target_reg']
    y_class = df['target_class']

    print(f"   Training on {len(df)} samples...")

    # --- 1. Regressor ---
    print("\n   📉 Training Regressor...")
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    
    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    regressor.fit(X_train, y_train)
    
    rmse = np.sqrt(mean_squared_error(y_test, regressor.predict(X_test)))
    print(f"      Regressor RMSE: {rmse:.4f}")
    joblib.dump(regressor, config.TV_SHOWS_MODEL_REGRESSOR)

    # --- 2. Classifier ---
    print("\n   🎯 Training Classifier...")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    # SAFETY CHECK: Do we have at least 2 classes (0 and 1)?
    if len(np.unique(y_train_c)) < 2:
        print("      ⚠️ WARNING: Only 1 class found in training data (e.g., all 'Liked').")
        print("      Skipping Classifier training to prevent crash.")
        # We don't save a broken classifier, or we could save a dummy one.
        # For now, we just don't save a new one, or delete the old one to avoid confusion.
    else:
        classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        classifier.fit(X_train_c, y_train_c)
        
        acc = accuracy_score(y_test_c, classifier.predict(X_test_c))
        print(f"      Classifier Accuracy: {acc*100:.2f}%")
        joblib.dump(classifier, config.TV_SHOWS_MODEL_CLASSIFIER)

    print(f"\n✅ Training Complete. Artifacts in {config.TV_SHOWS_MODEL_DIR}")

if __name__ == "__main__":
    train_models()