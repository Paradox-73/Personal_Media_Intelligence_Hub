import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def train_models():
    print("🤖 Starting Movie Model Training...")
    
    if not config.TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.TRAINING_DATA_PATH)
    
    X = df.drop(columns=['target_reg', 'target_class'])
    y_reg = df['target_reg']
    y_class = df['target_class']
    
    # 80/20 Split
    # We split X, y_reg, and y_class all at once to ensure indices match
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )
    
    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. Regressor ---
    print(f"   Training Regressor...")
    regressor = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.04,
        max_depth=4, 
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )
    regressor.fit(X_train, y_reg_train)
    
    preds_reg = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_reg_test, preds_reg))
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    
    # --- 2. Classifier ---
    print(f"   Training Classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=3,
        random_state=42
    )
    # FIX: Use X_train, not X_clf_train
    classifier.fit(X_train, y_clf_train)
    
    acc = accuracy_score(y_clf_test, classifier.predict(X_test))
    print(f"   🎯 [TEST SET] Accuracy: {acc*100:.2f}%")
    
    joblib.dump(regressor, config.MODEL_REGRESSOR)
    joblib.dump(classifier, config.MODEL_CLASSIFIER)
    print("✅ Models saved.")

if __name__ == "__main__":
    train_models()