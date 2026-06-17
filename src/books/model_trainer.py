import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def get_unified_predictions(df, media_type='book'):
    """
    Helper to get predictions from the Unified Model to use as a feature (distillation).
    """
    try:
        ensemble_dir = config.UNIFIED_ENSEMBLE_DIR
        stack_path = ensemble_dir / "stacking_ensemble_regressor.joblib"
        state_path = config.UNIFIED_PREPROCESSOR_STATE
        
        if not stack_path.exists() or not state_path.exists():
            return None
            
        unified_results_path = config.UNIFIED_PREDICTIONS_DIR / "unified_predictions_ensemble.csv"
        if unified_results_path.exists():
            u_df = pd.read_csv(unified_results_path)
            mapping = u_df[u_df['media_type'] == media_type].set_index('display_name')['pred_Stacking'].to_dict()
            name_col = 'title' if 'title' in df.columns else 'name'
            return df[name_col].map(mapping).fillna(df['user_rating'].mean() if 'user_rating' in df else 3.0)
        
        return None
    except Exception as e:
        print(f"   ⚠️ Error getting unified predictions: {e}")
        return None

def calculate_skill_score(y_true, y_pred, y_train_mean):
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_baseline = mean_absolute_error(y_true, np.full_like(y_true, y_train_mean))
    if mae_baseline == 0: return 0
    return 1 - (mae_model / mae_baseline)

def train_models():
    print("🤖 Starting Book Model Training (Distillation Mode)...")

    if not config.BOOKS_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    # Train on the SAME engineered matrix that feature_engineering / predict_ratings
    # and the Oracle transform use, so model, preprocessor_state and Oracle agree.
    df = pd.read_csv(config.BOOKS_TRAINING_DATA_PATH)
    try:
        df_enriched = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except Exception:
        df_enriched = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    df_enriched = df_enriched.dropna(subset=['my_rating']).reset_index(drop=True)

    # 1. Distillation prior (dropped by default; inert unless the unified predictions
    #    file exposes a usable column — see the project's distillation DROP verdict).
    unified_preds = get_unified_predictions(df_enriched, 'book')

    X = df.drop(columns=['target_reg', 'target_class'], errors='ignore').fillna(0)
    if unified_preds is not None:
        print("   ✅ Fusing Unified Model Predictions as a feature.")
        X['unified_prior'] = unified_preds.values

    y_reg = df['target_reg']
    y_class = df['target_class'].astype(int)

    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. SVR (Distilled) ---
    # With N<100, we freeze hyperparameters as recommended
    print(f"   Training Distilled SVR (Frozen HPs: C=1.0, kernel=rbf)...")
    
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1))
    ])
    svr_pipe.fit(X_train, y_train)
    
    # --- CONFORMAL PREDICTION (Uncertainty) ---
    print("   Computing conformal intervals via 5-fold OOF residuals...")
    # cross_val_predict needs a clean partition (each item tested once); RepeatedKFold
    # reuses test items across repeats and raises "only works for partitions".
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = cross_val_predict(svr_pipe, X, y_reg, cv=kf)
    abs_residuals = np.abs(y_reg - oof_preds)
    
    # 80% coverage interval width
    q_80 = np.quantile(abs_residuals, 0.8)
    print(f"   ✅ Conformal Interval (80% coverage): ±{q_80:.3f}")

    y_pred_raw = svr_pipe.predict(X_test)

    # Rounding to 0.5 increments
    def round_h(x): return np.round(np.clip(x, 0, 5) * 2) / 2
    y_pred = round_h(y_pred_raw)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    skill_score = calculate_skill_score(y_test, y_pred, y_train.mean())

    print("\n" + "="*50)
    print("📊 PERFORMANCE REPORT: Distilled Book SVR")
    print("="*50)
    print(f"   📉 [TEST SET] MAE:   {mae:.4f}")
    print(f"   📈 [TEST SET] R²:    {r2:.4f}")
    print(f"   🎯 [TEST SET] Skill: {skill_score:.4f} (MAE vs. Global Mean)")
    print("="*50 + "\n")

    # Save Model and Interval Metadata
    joblib.dump(svr_pipe, config.BOOKS_MODEL_REGRESSOR)
    
    # Save metadata including the conformal width
    meta_path = Path(config.BOOKS_MODEL_REGRESSOR).parent / "model_meta.joblib"
    joblib.dump({
        'conformal_width_80': q_80,
        'skill_score': skill_score,
        'mae': mae,
        'r2': r2,
        'train_mean': y_train.mean()
    }, meta_path)
    
    # --- 2. Classifier ---
    classifier = xgb.XGBClassifier(
        n_estimators=50, learning_rate=0.05, max_depth=3,
        objective='multi:softmax', num_class=3, random_state=42
    )
    y_class_train = y_class.loc[X_train.index]
    classifier.fit(X_train, y_class_train)
    joblib.dump(classifier, config.BOOKS_MODEL_CLASSIFIER)

    # --- 3. Full Dataset Predictions for Visualization ---
    preds_full = round_h(svr_pipe.predict(X))
    results_dir = config.BASE_DIR / "results" / "books"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame({
        'title': df_enriched['title'] if 'title' in df_enriched.columns else range(len(y_reg)),
        'actual': y_reg.values,
        'predicted': preds_full
    })
    df_results.to_csv(results_dir / "book_predictions_full.csv", index=False)
    
    print(f"✅ Models and results saved to {results_dir}")

if __name__ == "__main__":
    train_models()
