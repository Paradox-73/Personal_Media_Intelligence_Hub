import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_feature_engineering import transform_single_media
from src.movies.ingestion import get_movie_metadata

def calculate_metrics(y_true, y_pred):
    y_pred = np.round(np.clip(y_pred, 0.5, 5.0) * 2) / 2
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    diffs = np.abs(y_true - y_pred)
    total = len(diffs)
    return {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2,
        "Exact": (diffs == 0.0).sum() / total * 100,
        "Off_05": (diffs == 0.5).sum() / total * 100,
        "Off_10": (diffs == 1.0).sum() / total * 100,
        "Off_GT1": (diffs > 1.0).sum() / total * 100
    }

def print_metrics(label, metrics):
    print(f"\n--- {label} ---")
    print(f"   📉 MSE: {metrics['MSE']:.4f} | RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | R²: {metrics['R2']:.4f}")
    print(f"   🎯 Exact: {metrics['Exact']:.1f}% | ±0.5: {metrics['Off_05']:.1f}% | ±1.0: {metrics['Off_10']:.1f}% | >1.0: {metrics['Off_GT1']:.1f}%")

def run_comprehensive_evaluation(new_ratings_path=None):
    print("🚀 Initializing Master Evaluation Pipeline...")
    
    # 1. Load Data
    if not config.UNIFIED_TRAINING_DATA_PATH.exists():
        print("❌ Unified training data missing. Run feature engineering first.")
        return
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # 80/20 Split (Same seed as trainer)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"✅ Loaded Unified Data. Test set size: {len(test_df)}")

    # 2. Load Models
    # Unified Models
    uni_dir = config.UNIFIED_ENSEMBLE_DIR
    uni_models = {
        "Unified_Stacking": joblib.load(uni_dir / "stacking_ensemble_regressor.joblib"),
        "Unified_XGB": joblib.load(uni_dir / "xgb_base_regressor.joblib"),
        "Unified_CatBoost": joblib.load(uni_dir / "catboost_base_regressor.joblib"),
        "Unified_Ordinal_EV": joblib.load(uni_dir / "ordinal_classifier.joblib")
    }
    uni_state = joblib.load(config.UNIFIED_PREPROCESSOR_STATE)
    uni_classes = joblib.load(uni_dir / "ordinal_classes.joblib")
    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])[uni_classes]

    # Baseline Movie Model
    movie_baseline = joblib.load(config.MODEL_REGRESSOR)
    movie_state = joblib.load(config.PREPROCESSOR_STATE)

    # 3. Predict on Unified Test Set
    print("🧠 Predicting on Unified Test Set...")
    X_test = test_df[uni_state['training_columns']]
    y_true = test_df['target_reg']
    
    results = test_df.copy()
    for name, model in uni_models.items():
        if "Ordinal" in name:
            probs = model.predict_proba(X_test)
            results[f'pred_{name}'] = np.sum(probs * bucket_vals, axis=1)
        else:
            results[f'pred_{name}'] = model.predict(X_test)

    # 4. Report Metrics
    print("\n" + "="*60)
    print("📊 COMPREHENSIVE PERFORMANCE SUMMARY")
    print("="*60)

    # Slices
    slices = {
        "WHOLE UNIFIED TEST SET": results,
        "MOVIES ONLY (TEST)": results[results['is_tv_flag'] == 0],
        "SHOWS ONLY (TEST)": results[results['is_tv_flag'] == 1]
    }

    for label, slice_df in slices.items():
        if slice_df.empty: continue
        print(f"\n[{label}] (N={len(slice_df)})")
        for name in uni_models.keys():
            m = calculate_metrics(slice_df['target_reg'], slice_df[f'pred_{name}'])
            print_metrics(name, m)

    # 5. NEW RATINGS EVALUATION
    if new_ratings_path and Path(new_ratings_path).exists():
        print("\n" + "="*60)
        print(f"🔥 EVALUATING ON NEW RATINGS (50 MOVIES): {new_ratings_path}")
        print("="*60)
        df_new = pd.read_csv(new_ratings_path)
        # Assuming columns: Name, Year, Rating
        new_results = []
        for _, row in df_new.iterrows():
            meta = get_movie_metadata(row['Name'], row['Year'])
            if not meta or not meta.get('title'): continue
            
            # Transform for Unified
            feat = transform_single_media(meta, uni_state, is_tv_show=False)
            res = {'Name': row['Name'], 'Actual': row['Rating']}
            for name, model in uni_models.items():
                if "Ordinal" in name:
                    probs = model.predict_proba(feat)[0]
                    res[f'pred_{name}'] = np.sum(probs * bucket_vals)
                else:
                    res[f'pred_{name}'] = model.predict(feat)[0]
            new_results.append(res)
        
        if new_results:
            df_new_res = pd.DataFrame(new_results)
            print(f"N={len(df_new_res)}")
            for name in uni_models.keys():
                m = calculate_metrics(df_new_res['Actual'], df_new_res[f'pred_{name}'])
                print_metrics(name, m)

    # 6. VISUALIZATION
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # KDE
    sns.kdeplot(data=results, x='target_reg', ax=ax1, label='Actual', color='black', linewidth=3, fill=True, alpha=0.1)
    for name in uni_models.keys():
        sns.kdeplot(data=results, x=f'pred_{name}', ax=ax1, label=name, linestyle='--')
    ax1.set_title("Unified Test Set: Distribution Comparison (KDE)")
    ax1.legend()

    # Histograms
    sns.histplot(results['target_reg'], ax=ax2, color='black', alpha=0.3, label='Actual', bins=10)
    sns.histplot(results['pred_Unified_Stacking'], ax=ax2, color='orange', alpha=0.5, label='Stacking Preds', bins=10)
    ax2.set_title("Unified Test Set: Stacking vs Actual (Histogram)")
    ax2.legend()

    plt.tight_layout()
    plot_path = config.UNIFIED_PREDICTIONS_DIR / "comprehensive_eval_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n📈 Comprehensive plots saved to {plot_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--new_ratings", help="Path to the 50 new movies CSV", default=None)
    args = parser.parse_args()
    run_comprehensive_evaluation(args.new_ratings)
