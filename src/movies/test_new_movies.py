import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies.ingestion import get_movie_metadata
from src.movies.feature_engineering import transform_single_movie

def round_to_nearest_half(rating):
    """Rounds rating to nearest 0.5 in range [0.5, 5.0]."""
    rounded = np.round(rating * 2) / 2
    return np.clip(rounded, 0.5, 5.0)

def run_test_on_new_movies(csv_path):
    print(f"🚀 Loading new movies from: {csv_path} (Ensemble Mode)")

    # 1. Load Data
    try:
        df_new = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # Standardize column names
    required_cols = {'Name': 'Name', 'Year': 'Year', 'Rating': 'Rating'}
    df_new.rename(columns={c: required_cols.get(c.lower(), c) for c in df_new.columns if c.lower() in required_cols}, inplace=True)

    if not all(col in df_new.columns for col in required_cols.values()):
        print(f"❌ Missing one or more required columns: {list(required_cols.values())}")
        return

    # Clean and validate data
    initial_rows = len(df_new)
    df_new.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_new.dropna(subset=['Name', 'Year', 'Rating'], inplace=True)
    df_new['Year'] = pd.to_numeric(df_new['Year'], errors='coerce')
    df_new['Rating'] = pd.to_numeric(df_new['Rating'], errors='coerce')
    df_new.dropna(subset=['Year', 'Rating'], inplace=True)
    df_new['Year'] = df_new['Year'].astype(int)

    if len(df_new) < initial_rows:
        print(f"   Filtered out {initial_rows - len(df_new)} incomplete rows. Processing {len(df_new)} valid movies.")

    if df_new.empty:
        print("❌ No valid movie entries found after filtering.")
        return

    # 2. Load Models & State
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
            print(f"❌ ERROR: Model file not found for '{name}' at {path}.")
            return
        models[name] = joblib.load(path)

    if not config.PREPROCESSOR_STATE.exists():
        print(f"❌ ERROR: Preprocessor State not found at {config.PREPROCESSOR_STATE}.")
        return

    state = joblib.load(config.PREPROCESSOR_STATE)
    print(f"✅ Loaded {len(models)} models and preprocessor state.")

    # 3. Process each movie
    results = []
    METADATA_CACHE_FILE = config.CACHE_DIR / "movies_test_metadata_cache.csv"
    cached_metadata_df = pd.read_csv(METADATA_CACHE_FILE) if METADATA_CACHE_FILE.exists() else pd.DataFrame()

    print(f"🔍 Enriching and predicting for {len(df_new)} movies...")
    bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    for idx, row in df_new.iterrows():
        name, year, actual = str(row['Name']).strip(), int(row['Year']), float(row['Rating'])

        try:
            cached_entry_mask = (cached_metadata_df.get('title', pd.Series(dtype=str)).str.lower() == name.lower()) & \
                                (cached_metadata_df.get('year', pd.Series(dtype=int)) == year)
            cached_entry = cached_metadata_df[cached_entry_mask]

            if not cached_entry.empty:
                print(f"   [CACHE HIT] Using cached data for '{name} ({year})'")
                metadata = cached_entry.iloc[0].to_dict()
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try: metadata[key] = json.loads(value)
                        except: pass
            else:
                print(f"   [CACHE MISS] Fetching data for '{name} ({year})'")
                metadata = get_movie_metadata(name, year)
                if metadata and metadata.get('title'):
                    flattened_metadata = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in metadata.items()}
                    new_cache_entry = pd.DataFrame([flattened_metadata])
                    cached_metadata_df = pd.concat([cached_metadata_df, new_cache_entry], ignore_index=True)
                    cached_metadata_df.to_csv(METADATA_CACHE_FILE, index=False)

            if not metadata or not metadata.get('title'):
                print(f"⚠️ Skipping '{name} ({year})': Could not fetch metadata.")
                continue

            features = transform_single_movie(metadata, state)

            result_row = {'Name': name, 'Year': year, 'Actual': actual}
            for model_name, model in models.items():
                if model_name == "Ordinal_EV":
                    probs = model.predict_proba(features)[0]
                    # REVERTED: Expected Value
                    raw_pred = np.sum(probs * bucket_vals)
                else:
                    raw_pred = model.predict(features)[0]

                result_row[f'Pred_{model_name}'] = round_to_nearest_half(raw_pred)

            results.append(result_row)
        except Exception as e:
            print(f"⚠️ Error processing '{name} ({year})': {e}. Skipping.")
            continue

    if not results:
        print("❌ No movies were successfully processed.")
        return

    df_results = pd.DataFrame(results)
    print(f"✅ Successfully processed and predicted for {len(df_results)} movies.")

    # 4. Performance Reports for all models
    y_true = df_results['Actual']
    print("\n" + "="*55)
    print("📊 NEW MOVIES ENSEMBLE PERFORMANCE REPORT")
    print("="*55)

    for name in models.keys():
        y_pred = df_results[f'Pred_{name}']

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        diffs = np.abs(y_true - y_pred)
        total = len(diffs)

        print(f"   Model: {name}")
        print(f"   📉 MSE: {mse:.4f} | 📉 RMSE: {rmse:.4f} | 📉 MAE: {mae:.4f} | 📈 R²: {r2:.4f}")
        print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
        print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
        print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
        print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
        print("-" * 55)

    # 5a. Visualization (KDE Plot)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.kdeplot(data=df_results, x='Actual', ax=ax, label='Actual Ratings', color='black', linewidth=3, fill=True, alpha=0.1)

    for name in models.keys():
        sns.kdeplot(data=df_results, x=f'Pred_{name}', ax=ax, label=f'{name} Preds', linestyle='--', linewidth=2)

    ax.set_title('Comparison of Predicted Rating Distributions (New Movies - KDE)', fontsize=18, pad=20)
    ax.set_xlabel('Rating (0.5 - 5.0)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax.legend(title='Model', fontsize='medium')
    plt.tight_layout()

    config.MOVIES_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path_kde = config.MOVIES_PREDICTIONS_DIR / "new_movies_test_kde.png"
    plt.savefig(plot_path_kde, dpi=150)
    print(f"\n📈 KDE distribution plot saved to {plot_path_kde}")
    plt.close(fig)

    # 5b. Visualization (Histogram Grid)
    data_cols = ['Actual'] + [f'Pred_{name}' for name in models.keys()]
    titles = ['Actual Ratings'] + [f'{name} Predictions' for name in models.keys()]

    fig, axes = plt.subplots(3, 3, figsize=(20, 15), sharey=True)
    axes = axes.flatten()

    for i, col in enumerate(data_cols):
        sns.histplot(df_results[col], bins=np.arange(0.25, 5.75, 0.5), 
                     ax=axes[i], kde=False, edgecolor='black')
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Rating (0.5 - 5.0)')
        axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    for i in range(len(data_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Histogram of Rating Distributions (New Movies)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path_hist = config.MOVIES_PREDICTIONS_DIR / "new_movies_test_histograms.png"
    plt.savefig(plot_path_hist, dpi=150)
    print(f"📈 Histogram grid plot saved to {plot_path_hist}")
    plt.close(fig)

    # 6. Final Results & Misses
    df_results['Diff_Stacking'] = np.abs(df_results['Actual'] - df_results['Pred_Stacking'])
    print("\n🚩 Top 5 Prediction Misses (based on Stacking model):")
    print(df_results.sort_values(by='Diff_Stacking', ascending=False).head(5)[['Name', 'Actual', 'Pred_Stacking', 'Diff_Stacking']].to_string(index=False))

    results_csv_path = config.MOVIES_PREDICTIONS_DIR / "new_movies_test_results_ensemble.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n✅ Ensemble test results saved to {results_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test movie prediction model on new movies.")
    parser.add_argument("csv_path", help="Path to the CSV file with new movies (Name, Year, Rating columns)")
    args = parser.parse_args()

    run_test_on_new_movies(args.csv_path)