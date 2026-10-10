import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.feature_engineering import transform_single_movie
# Needed so joblib can unpickle the stacking/ordinal models, which embed an
# OrdinalExpectedValueRegressor instance (referenced as __main__ at train time).
from src.movies.advanced_movie_model_trainer import (
    OrdinalExpectedValueRegressor, MeanMetaRegressor, AnchoredRegressor)  # noqa: F401


def batch_predict_ratings():
    print("🚀 Starting Batch Prediction (Ensemble Mode)...")

    # --- 1. Load Models ---
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    model_paths = {
        "Baseline_XGB": config.MODEL_REGRESSOR,
        "XGB": ensemble_dir / "xgb_base_regressor.joblib",
        "SVR": ensemble_dir / "svr_base_regressor.joblib",
        "CatBoost": ensemble_dir / "catboost_base_regressor.joblib",
        "Stacking": ensemble_dir / "stacking_ensemble_regressor.joblib",
        "Ordinal_EV": ensemble_dir / "ordinal_classifier.joblib"
    }
    
    models = {}
    for name, path in model_paths.items():
        if not path.exists():
            print(f"❌ ERROR: Model file not found for '{name}' at {path}. Please run the advanced trainer first.")
            return
        models[name] = joblib.load(path)
    
    if not config.PREPROCESSOR_STATE.exists():
        print(f"❌ ERROR: Preprocessor State not found at {config.PREPROCESSOR_STATE}.")
        return
        
    state = joblib.load(config.PREPROCESSOR_STATE)
    print(f"✅ Loaded {len(models)} models and preprocessor state.")
    
    try:
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        print(f"   Loaded {len(df)} movies from enriched data.")
    except FileNotFoundError:
        print(f"❌ ERROR: Enriched data not found at {config.MOVIES_ENRICHED_DATA_PATH}.")
        return

    print("   Processing features...")
    
    # --- 2. Feature Engineering ---
    # The same per-row builder the Oracle uses, so batch scores match training parity.
    X_final = pd.concat([transform_single_movie(r, state) for r in df.to_dict('records')],
                        ignore_index=True)
    X_final.index = df.index

    # --- 3. Predict with all models ---
    print("   Predicting ratings with all models...")
    df['name'] = df['title'].fillna(df.get('letterboxd_name', 'Unknown'))
    
    # predict_proba columns are the classes present at train time, not all 10 buckets.
    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])[
        joblib.load(ensemble_dir / "ordinal_classes.joblib")]
    
    # Deployment calibration reference: the model's raw preds on rated movies vs ratings.
    from src.utils.deployment_calibration import calibrate_to_ratings
    _r = pd.to_numeric(df['user_rating'], errors='coerce') if 'user_rating' in df.columns else pd.Series(np.nan, index=df.index)
    _rated = _r.notna().values
    for name, model in models.items():
        if name == "Ordinal_EV":
            probs = model.predict_proba(X_final)
            # REVERTED: Using Expected Value instead of argmax
            raw = np.sum(probs * bucket_vals, axis=1)
        else:
            raw = np.asarray(model.predict(X_final), dtype=float)
        # Restamp onto the user's real rating distribution so the Oracle spans the full
        # range (incl. 5*) instead of regressing to the mode; ranking-preserving.
        raw = calibrate_to_ratings(raw, raw[_rated], _r.values[_rated])
        df[f'pred_{name}'] = np.round(np.clip(raw, 0.5, 5.0) * 2) / 2

    # --- 4. Performance Report ---
    results_dir = BASE_DIR / "results" / "movies"
    results_dir.mkdir(parents=True, exist_ok=True)

    if 'user_rating' in df.columns and not df['user_rating'].isna().all():
        eval_df = df.dropna(subset=['user_rating']).copy()
        y_true = eval_df['user_rating']
        
        print("\n" + "="*55)
        print("📊 ML ENSEMBLE IN-SAMPLE FIT (most rows were training rows, not a held-out score)")
        print("="*55)
        print(f"Total Evaluated: {len(eval_df)}")
        
        for name in models.keys():
            pred_col = f'pred_{name}'
            y_pred = eval_df[pred_col]
            
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            diffs = np.abs(y_true - y_pred)
            total = len(diffs)
            
            print("-" * 55)
            print(f"   Model: {name}")
            print(f"   📉 MSE: {mse:.4f} | 📉 RMSE: {rmse:.4f} | 📉 MAE: {mae:.4f} | 📈 R²: {r2:.4f}")
            print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
            print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
            print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
            print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
        print("="*55 + "\n")
        
        df['abs_diff_stacking'] = np.abs(df['user_rating'] - df['pred_Stacking'])

        # --- 5a. Visualization (KDE Plot) ---
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(16, 8))
        
        sns.kdeplot(data=eval_df, x='user_rating', ax=ax, label='Actual Ratings', color='black', linewidth=3, fill=True, alpha=0.1)
        
        for name in models.keys():
            sns.kdeplot(data=eval_df, x=f'pred_{name}', ax=ax, label=f'{name} Preds', linestyle='--', linewidth=2)
            
        ax.set_title('Comparison of Predicted Rating Distributions (Full Dataset - KDE)', fontsize=18, pad=20)
        ax.set_xlabel('Rating (0.5 - 5.0)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_xticks(np.arange(0.5, 5.5, 0.5))
        ax.legend(title='Model', fontsize='medium')
        plt.tight_layout()
        
        plot_path_kde = results_dir / "ratings_distribution_kde.png"
        plt.savefig(plot_path_kde, dpi=150)
        print(f"📈 KDE distribution plot saved to {plot_path_kde}")
        plt.close(fig) 

        # --- 5b. Visualization (Histogram Grid) ---
        data_cols = ['user_rating'] + [f'pred_{name}' for name in models.keys()]
        titles = ['Actual Ratings'] + [f'{name} Predictions' for name in models.keys()]
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharey=True)
        axes = axes.flatten()
        
        for i, col in enumerate(data_cols):
            sns.histplot(eval_df[col], bins=np.arange(0.25, 5.75, 0.5), 
                         ax=axes[i], kde=False, edgecolor='black')
            axes[i].set_title(titles[i])
            axes[i].set_xlabel('Rating (0.5 - 5.0)')
            axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
            axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Hide unused axes
        for i in range(len(data_cols), len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle('Histogram of Rating Distributions (Full Dataset)', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        plot_path_hist = results_dir / "ratings_distribution_histograms.png"
        plt.savefig(plot_path_hist, dpi=150)
        print(f"📈 Histogram grid plot saved to {plot_path_hist}")
        plt.close(fig)

    # --- 6. Save Results ---
    out_cols = ['name', 'year', 'user_rating'] + [f'pred_{name}' for name in models.keys()] + ['abs_diff_stacking', 'director', 'production']
    final_df = df[[c for c in out_cols if c in df.columns]].sort_values(by='pred_Stacking', ascending=False)
    
    out_path = results_dir / "predicted_ratings_ensemble.csv"
    final_df.to_csv(out_path, index=False)
    print(f"✅ Saved all model predictions to {out_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    batch_predict_ratings()
