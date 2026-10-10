import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
# safe_eval / parse_people are re-exported for src/evaluation/shows_active_learning_queue.py.
from src.shows.feature_engineering import (transform_shows, safe_eval, get_primary_network,  # noqa: F401
                                           parse_people)

def run_consolidated_predictions():
    print("🚀 STARTING CONSOLIDATED TV SHOW PREDICTIONS & EVALUATION...")

    # 1. Load Artifacts
    ensemble_dir = config.TV_SHOWS_MODEL_DIR / "ensemble"
    try:
        models = {
            "Baseline": joblib.load(config.TV_SHOWS_MODEL_REGRESSOR),
            "XGB": joblib.load(ensemble_dir / "xgb_base_regressor.joblib"),
            "SVR": joblib.load(ensemble_dir / "svr_base_regressor.joblib"),
            "CatBoost": joblib.load(ensemble_dir / "catboost_base_regressor.joblib"),
            "Stacking": joblib.load(ensemble_dir / "stacking_ensemble_regressor.joblib"),
            "Ordinal": joblib.load(ensemble_dir / "ordinal_classifier.joblib")
        }
        unique_classes = joblib.load(ensemble_dir / "ordinal_classes.joblib")
        state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
        print("   ✅ Loaded all models and preprocessor state.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        print("   Have you run src/shows/model_trainer.py?")
        return

    # 2. Load and Prepare Data
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    print(f"   Loaded {len(df)} shows for prediction.")
    
    X = transform_shows(df, state)
    df['primary_network'] = df['network'].apply(get_primary_network)

    # 3. Predict using all models
    print(f"📊 Predicting for {len(df)} shows...")
    
    bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
    present_bucket_vals = np.array([bucket_map[c] for c in unique_classes])

    for name, model in models.items():
        if name == "Ordinal":
            probs = model.predict_proba(X)
            preds = np.sum(probs * present_bucket_vals, axis=1)
        else:
            preds = model.predict(X)
        df[f'pred_{name.lower()}'] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2

    # Main prediction column (Using Stacking for best accuracy)
    df['predicted_rating'] = df['pred_stacking']

    # 4. Evaluation and Visuals
    results_dir = config.BASE_DIR / "results" / "shows"
    results_dir.mkdir(parents=True, exist_ok=True)

    if 'user_rating' in df.columns and not df['user_rating'].isna().all():
        eval_df = df.dropna(subset=['user_rating']).copy()
        y_true = eval_df['user_rating']
        
        print("\n" + "="*60)
        print("📊 SHOW ENSEMBLE PERFORMANCE REPORT (FULL DATASET)")
        print("="*60)
        
        for name in models.keys():
            y_pred = eval_df[f'pred_{name.lower()}']
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            diffs = np.abs(y_true - y_pred)
            
            print(f"Model: {name:<10} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
            print(f"   Accuracy -> Exact: {((diffs == 0.0).sum()/len(y_true))*100:.1f}% | ±0.5: {((diffs <= 0.5).sum()/len(y_true))*100:.1f}%")
        print("="*60 + "\n")

        # Visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # KDE Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.kdeplot(data=eval_df, x='user_rating', label='Actual', color='black', linewidth=3, fill=True, alpha=0.1)
        for name in ["Baseline", "Stacking", "Ordinal"]:
            sns.kdeplot(data=eval_df, x=f'pred_{name.lower()}', label=f'{name} Preds', linestyle='--')
        ax.set_title('TV Show Rating Distributions (KDE)', fontsize=16)
        ax.set_xticks(np.arange(0, 5.5, 0.5))
        plt.legend()
        plt.savefig(results_dir / "shows_distribution_kde.png", dpi=150)
        plt.close()

        # Histogram Grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()
        plot_models = ["Baseline", "XGB", "SVR", "CatBoost", "Stacking", "Ordinal"]
        for i, name in enumerate(plot_models):
            sns.histplot(eval_df[f'pred_{name.lower()}'], bins=np.arange(0.25, 5.75, 0.5), ax=axes[i], kde=False, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{name} Predictions')
            axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
        fig.suptitle('Histogram of TV Show Rating Predictions', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(results_dir / "shows_distribution_histograms.png", dpi=150)
        plt.close()

    # 5. Save Results
    # Dashboard view (Cleaned up)
    output_cols = ['name', 'year', 'user_rating', 'predicted_rating', 'primary_network']
    if 'user_rating' in df.columns:
        df['abs_diff'] = np.abs(df['user_rating'] - df['predicted_rating'])
        output_cols.append('abs_diff')
    
    final_df = df[output_cols].sort_values(by='predicted_rating', ascending=False)
    final_df.to_csv(config.TV_SHOWS_FULL_VIEW_PATH, index=False)
    
    # Full results
    full_results_path = results_dir / "show_predictions_full.csv"
    df.to_csv(full_results_path, index=False)
    
    print(f"✅ Dashboard view saved to: {config.TV_SHOWS_FULL_VIEW_PATH}")
    print(f"✅ Full results and graphs saved to: {results_dir}")

if __name__ == "__main__":
    run_consolidated_predictions()
