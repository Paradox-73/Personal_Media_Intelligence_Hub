import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np  # Added numpy
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

def train_models():
    print("🤖 Starting Model Training...")
    
    try:
        df = pd.read_csv(config.TRAINING_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: training_features.csv not found.")
        return
    
    X = df.drop(columns=['target_reg', 'target_class'])
    y_reg = df['target_reg']
    y_class = df['target_class']
    
    # Split Data
    X_train, X_test, y_reg_train, y_reg_test, y_class_train, y_class_test = train_test_split(
        X, y_reg, y_class, test_size=0.2, random_state=42
    )
    
    # --- 1. Train Regressor ---
    print(f"   Training Regressor...")
    regressor = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    regressor.fit(X_train, y_reg_train)
    
    # Evaluate (Fixed RMSE Calculation)
    preds_reg = regressor.predict(X_test)
    mse = mean_squared_error(y_reg_test, preds_reg)
    rmse = np.sqrt(mse)  # Manual calculation compatible with all versions
    print(f"   📉 Regressor RMSE: {rmse:.4f}")
    
    # --- 2. Train Classifier ---
    print(f"   Training Classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=4,
        objective='multi:softprob', num_class=3, random_state=42
    )
    classifier.fit(X_train, y_class_train)
    
    # Evaluate
    preds_class = classifier.predict(X_test)
    acc = accuracy_score(y_class_test, preds_class)
    print(f"   🎯 Classifier Accuracy: {acc*100:.2f}%")
    
    # Save
    joblib.dump(regressor, config.MODEL_REGRESSOR)
    joblib.dump(classifier, config.MODEL_CLASSIFIER)
    print("✅ Models saved.")

if __name__ == "__main__":
    train_models()