import pandas as pd
import xgboost as xgb
import joblib
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.books.model_trainer import calculate_skill_score, refit_encodings
from src.games.feature_engineering import GAME_ENCODE, TE_PRIOR

# C=3.0, not the old frozen 1.0. Every domain measured ~95% bias-dominated, which
# is the regime where more capacity helps. Chosen by NESTED CV (hyperparameter
# selected on an inner 3-fold of each training fold, scored on the untouched outer
# fold) -- reports/capacity_tuning.json. A naive same-folds sweep picked C=10, but
# that was winner's curse; the honest optimum is 3.0.
SVR_C = 3.0

def train_models():
    print("🤖 Starting Game Model Training...")

    if not config.GAMES_TRAINING_DATA_PATH.exists():
        print("❌ Error: Training data not found.")
        return

    df = pd.read_csv(config.GAMES_TRAINING_DATA_PATH)
    df_enriched = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)

    X = df.drop(columns=['target_reg', 'target_class'], errors='ignore')
    y_reg = df['target_reg']
    y_class = df['target_class']

    # Separate training data from all data
    mask_train = y_reg.notna()
    X_train_full = X[mask_train].reset_index(drop=True)
    raw_train_full = df_enriched[mask_train].reset_index(drop=True)
    y_reg_train_full = y_reg[mask_train].reset_index(drop=True)
    y_class_train_full = y_class[mask_train].astype(int).reset_index(drop=True)

    # 80/20 Split, with target encodings refitted on the train rows only.
    tr, te = train_test_split(np.arange(len(X_train_full)), test_size=0.2, random_state=42)
    X_train, X_test = refit_encodings(X_train_full, raw_train_full, y_reg_train_full,
                                      tr, te, GAME_ENCODE, TE_PRIOR)
    y_train, y_test = y_reg_train_full.iloc[tr], y_reg_train_full.iloc[te]

    print(f"   Training Set: {len(X_train)} | Test Set: {len(X_test)}")

    # --- 1. SVR ---
    # With N<100, we freeze hyperparameters as recommended
    print(f"   Training SVR (nested-CV tuned: C={SVR_C}, kernel=rbf)...")
    
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=SVR_C, gamma='scale', epsilon=0.1))
    ])
    svr_pipe.fit(X_train, y_train)
    
    # --- CONFORMAL PREDICTION (Uncertainty) ---
    print("   Computing conformal intervals via 5-fold OOF residuals...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.empty(len(X_train_full))
    for f_tr, f_va in kf.split(X_train_full):
        X_a, X_b = refit_encodings(X_train_full, raw_train_full, y_reg_train_full,
                                   f_tr, f_va, GAME_ENCODE, TE_PRIOR)
        oof_preds[f_va] = clone(svr_pipe).fit(X_a, y_reg_train_full.iloc[f_tr]).predict(X_b)
    abs_residuals = np.abs(y_reg_train_full.values - oof_preds)
    
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
    print("📊 PERFORMANCE REPORT: Game SVR")
    print("="*50)
    print(f"   📉 [TEST SET] MAE:   {mae:.4f}")
    print(f"   📈 [TEST SET] R²:    {r2:.4f}")
    print(f"   🎯 [TEST SET] Skill: {skill_score:.4f} (MAE vs. Global Mean)")
    print("="*50 + "\n")

    # Save Model and Interval Metadata
    joblib.dump(svr_pipe, config.GAMES_MODEL_REGRESSOR)
    
    # Save metadata including the conformal width
    meta_path = Path(config.GAMES_MODEL_REGRESSOR).parent / "model_meta.joblib"
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
    y_class_train = y_class_train_full.iloc[tr]
    classifier.fit(X_train, y_class_train)
    joblib.dump(classifier, config.GAMES_MODEL_CLASSIFIER)

    # --- 3. Full Dataset Predictions for Visualization ---
    preds_full = round_h(svr_pipe.predict(X))
    results_dir = config.BASE_DIR / "results" / "games"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    df_results = pd.DataFrame({
        'name': df_enriched['name'],
        'actual': y_reg,
        'predicted': preds_full
    })
    df_results.to_csv(results_dir / "game_predictions_full.csv", index=False)
    
    print(f"✅ Models and results saved to {results_dir}")

if __name__ == "__main__":
    train_models()
