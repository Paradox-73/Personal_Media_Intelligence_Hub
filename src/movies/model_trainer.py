import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score

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
    
    preds_reg_raw = regressor.predict(X_test)
    preds_reg = np.round(np.clip(preds_reg_raw, 0, 5) * 2) / 2
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, preds_reg))
    mae = mean_absolute_error(y_reg_test, preds_reg)
    r2 = r2_score(y_reg_test, preds_reg)
    
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")

    # Detailed Diff Report
    diffs = np.abs(y_reg_test - preds_reg)
    total = len(diffs)
    print("\n" + "   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40 + "\n")
    
    # --- 2. Classifier ---
    print(f"   Training Classifier...")
    classifier = xgb.XGBClassifier(
        n_estimators=100, 
        learning_rate=0.05, 
        max_depth=3,
        random_state=42
    )
    classifier.fit(X_train, y_clf_train)
    
    acc = accuracy_score(y_clf_test, classifier.predict(X_test))
    print(f"   🎯 [TEST SET] Accuracy: {acc*100:.2f}%")
    
    joblib.dump(regressor, config.MODEL_REGRESSOR)
    joblib.dump(classifier, config.MODEL_CLASSIFIER)
    print("✅ Models saved.")

if __name__ == "__main__":
    train_models()