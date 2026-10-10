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
from src.common import taste_features as tf
from src.books.feature_engineering import BOOK_ENCODE, TE_PRIOR

# C=3.0, not the old frozen 1.0. Every domain measured ~95% bias-dominated, which
# is the regime where more capacity helps. Chosen by NESTED CV (hyperparameter
# selected on an inner 3-fold of each training fold, scored on the untouched outer
# fold) -- reports/capacity_tuning.json. A naive same-folds sweep picked C=10, but
# that was winner's curse; the honest optimum is 3.0.
SVR_C = 3.0

def calculate_skill_score(y_true, y_pred, y_train_mean):
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_baseline = mean_absolute_error(y_true, np.full_like(y_true, y_train_mean))
    if mae_baseline == 0: return 0
    return 1 - (mae_model / mae_baseline)


def refit_encodings(X, raw, y, tr, te, encode, m):
    """Rebuild the `{col}_encoded` / `{col}_n` columns of X for one split, fitted on
    the train rows only, and return (X[tr], X[te]).

    The training CSV's encodings came from a k-fold over ALL rows, so a train row's
    encoding already contains the test rows' ratings. Here train rows get an
    out-of-fold encoding within `tr`, and test rows get the train-only map: an
    entity seen only in `te` falls back to the train mean.
    `raw` holds the entity cells row-aligned with X; `tr` / `te` are positions.
    """
    X = X.reset_index(drop=True).copy()
    y = np.asarray(y, dtype=float)
    for col, limit in encode:
        if f'{col}_encoded' not in X.columns:
            continue
        cells = raw[col].tolist()
        enc, sup, art = tf.oof_entity_encode([cells[i] for i in tr], y[tr], m=m, limit=limit)
        X.loc[tr, f'{col}_encoded'] = enc
        X.loc[tr, f'{col}_n'] = sup
        for i in te:
            X.loc[i, [f'{col}_encoded', f'{col}_n']] = tf.apply_entity_map(
                cells[i], art['map'], art['counts'], art['mean'], limit)
    return X.loc[tr], X.loc[te]

def train_models():
    print("🤖 Starting Book Model Training...")

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
    # Same row filter as feature_engineering, so df_enriched lines up with df.
    df_enriched = df_enriched[pd.to_numeric(df_enriched['my_rating'], errors='coerce').notna()].reset_index(drop=True)
    assert len(df_enriched) == len(df), "enriched rows no longer line up with the training CSV"

    X = df.drop(columns=['target_reg', 'target_class'], errors='ignore').fillna(0)
    y_reg = df['target_reg']
    y_class = df['target_class'].astype(int)

    # 80/20 Split, with target encodings refitted on the train rows only.
    tr, te = train_test_split(np.arange(len(X)), test_size=0.2, random_state=42)
    X_train, X_test = refit_encodings(X, df_enriched, y_reg, tr, te, BOOK_ENCODE, TE_PRIOR)
    y_train, y_test = y_reg.iloc[tr], y_reg.iloc[te]

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
    oof_preds = np.empty(len(X))
    for f_tr, f_va in kf.split(X):
        X_a, X_b = refit_encodings(X, df_enriched, y_reg, f_tr, f_va, BOOK_ENCODE, TE_PRIOR)
        oof_preds[f_va] = clone(svr_pipe).fit(X_a, y_reg.iloc[f_tr]).predict(X_b)
    abs_residuals = np.abs(y_reg.values - oof_preds)
    
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
    print("📊 PERFORMANCE REPORT: Book SVR")
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
    y_class_train = y_class.iloc[tr]
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
