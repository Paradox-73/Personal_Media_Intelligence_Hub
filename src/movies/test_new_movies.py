import pandas as pd
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
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
    print(f"🚀 Loading new movies from: {csv_path}")
    
    # 1. Load Data
    try:
        df_new = pd.read_csv(csv_path)
        initial_rows = len(df_new)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    required_cols = ['Name', 'Year', 'Rating']
    for col in required_cols:
        if col not in df_new.columns:
            match = next((c for c in df_new.columns if c.lower() == col.lower()), None)
            if match:
                df_new.rename(columns={match: col}, inplace=True)
            else:
                print(f"❌ Missing required column: {col}")
                print(f"Found: {df_new.columns.tolist()}")
                return
    
    df_new.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_new.dropna(subset=['Name', 'Year', 'Rating'], inplace=True)
    df_new['Year'] = pd.to_numeric(df_new['Year'], errors='coerce')
    df_new.dropna(subset=['Year'], inplace=True)
    df_new['Year'] = df_new['Year'].astype(int)
    df_new['Rating'] = pd.to_numeric(df_new['Rating'], errors='coerce')
    df_new.dropna(subset=['Rating'], inplace=True)
    df_new['Rating'] = df_new['Rating'].astype(float)
    
    if len(df_new) < initial_rows:
        print(f"   Filtered out {initial_rows - len(df_new)} empty or incomplete rows. Processing {len(df_new)} valid movies.")

    if df_new.empty:
        print("❌ No valid movie entries found in the CSV after filtering.")
        return

    # 2. Load Model & State
    if not config.MODEL_REGRESSOR.exists() or not config.PREPROCESSOR_STATE.exists():
        print("❌ Model or Preprocessor State not found. Please train the model first.")
        return
        
    model = joblib.load(config.MODEL_REGRESSOR)
    state = joblib.load(config.PREPROCESSOR_STATE)

    # 3. Process each movie
    results = []
    processed_count = 0
    
    METADATA_CACHE_FILE = config.CACHE_DIR / "movies_test_metadata_cache.csv"
    cached_metadata_df = pd.DataFrame()

    if METADATA_CACHE_FILE.exists():
        try:
            cached_metadata_df = pd.read_csv(METADATA_CACHE_FILE)
            print(f"   Loaded {len(cached_metadata_df)} entries from metadata cache.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load metadata cache CSV from {METADATA_CACHE_FILE}: {e}. Starting with empty cache.")

    print(f"🔍 Enriching and predicting for {len(df_new)} movies... (CSV Cache enabled at {METADATA_CACHE_FILE})")
    
    for idx, row in df_new.iterrows():
        name = str(row['Name']).strip()
        year = int(row['Year'])
        actual = float(row['Rating'])
        
        if not name or year <= 1800:
            print(f"⚠️ Skipping invalid movie data: Name='{name}', Year='{year}'")
            continue

        metadata = {}
        try:
            cached_entry_mask = (cached_metadata_df.get('title', pd.Series(dtype=str)) == name) & \
                                (cached_metadata_df.get('year', pd.Series(dtype=int)) == year)
            cached_entry = cached_metadata_df[cached_entry_mask]
            
            if not cached_entry.empty:
                print(f"   [CACHE HIT] Using cached data for '{name} ({year})'")
                metadata = cached_entry.iloc[0].to_dict()
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try: metadata[key] = json.loads(value)
                        except json.JSONDecodeError: pass
            else:
                print(f"   [CACHE MISS] Fetching data for '{name} ({year})'")
                metadata = get_movie_metadata(name, year)
                if metadata and metadata.get('title'):
                    flattened_metadata = {k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in metadata.items()}
                    new_cache_entry = pd.DataFrame([flattened_metadata])
                    
                    cached_metadata_df = pd.concat([cached_metadata_df, new_cache_entry], ignore_index=True)
                    cached_metadata_df.to_csv(METADATA_CACHE_FILE, index=False)
            
            if not metadata or not metadata.get('title'):
                print(f"⚠️ Warning: Could not fetch comprehensive metadata for '{name} ({year})'. Skipping prediction.")
                continue
        except Exception as e:
            print(f"⚠️ Warning: Error during data fetching/caching for '{name} ({year})': {e}. Skipping.")
            continue
            
        try:
            features = transform_single_movie(metadata, state)
            raw_pred = model.predict(features)[0]
            rounded_pred = round_to_nearest_half(raw_pred)
            
            results.append({
                'Name': name,
                'Year': year,
                'Actual': actual,
                'Predicted': rounded_pred,
                'Raw_Pred': raw_pred
            })
            processed_count += 1
        except Exception as e:
            print(f"⚠️ Warning: Error processing or predicting for '{name} ({year})': {e}. Skipping this movie.")
            continue
            
    if not results:
        print("❌ No movies were successfully processed and predicted.")
        return

    df_results = pd.DataFrame(results)
    print(f"✅ Successfully processed and predicted for {processed_count} movies.")
    
    # 4. Statistical Analysis
    y_true = df_results['Actual']
    y_pred = df_results['Predicted']
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    diffs = np.abs(y_true - y_pred)
    
    # 5. Performance Report
    print("\n" + "="*45)
    print("📊 NEW MOVIES PERFORMANCE REPORT")
    print("="*45)
    print(f"Total Evaluated: {len(df_results)}")
    print("-" * 45)
    print(f"📉 MSE:  {mse:.4f}")
    print(f"📉 RMSE: {rmse:.4f}")
    print(f"📈 R²:   {r2:.4f}")
    print("-" * 45)
    
    total = len(diffs)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("="*45 + "\n")

    # 6. Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    # My Ratings Spread
    sns.histplot(df_results['Actual'], bins=np.arange(0.25, 5.75, 0.5), 
                 ax=axes[0], color='royalblue', kde=True)
    axes[0].set_title('My Actual Ratings Distribution')
    axes[0].set_xlabel('Rating (0.5 - 5.0)')
    axes[0].set_ylabel('Count')
    axes[0].set_xticks(np.arange(0.5, 5.5, 0.5))
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # ML's Ratings Spread
    sns.histplot(df_results['Predicted'], bins=np.arange(0.25, 5.75, 0.5), 
                 ax=axes[1], color='darkorange', kde=True)
    axes[1].set_title("ML's Predicted Ratings Distribution")
    axes[1].set_xlabel('Rating (0.5 - 5.0)')
    axes[1].set_ylabel('') # Hide y-axis label for clarity
    axes[1].set_xticks(np.arange(0.5, 5.5, 0.5))
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle('Comparison of Rating Distributions', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for the suptitle
    
    plot_path = Path("new_movies_eval_histogram.png")
    plt.savefig(plot_path)
    print(f"📈 Histogram saved to {plot_path.absolute()}")
    
    df_results['Diff'] = diffs
    print("\n🚩 Top 5 Prediction Misses:")
    print(df_results.sort_values(by='Diff', ascending=False).head(5)[['Name', 'Actual', 'Predicted', 'Diff']].to_string(index=False))

    # Save results to CSV
    results_csv_path = config.MOVIES_PREDICTIONS_DIR / "new_movies_test_results.csv"
    df_results.to_csv(results_csv_path, index=False)
    print(f"✅ Test results saved to {results_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test movie prediction model on new movies.")
    parser.add_argument("csv_path", help="Path to the CSV file with new movies (Name, Year, Rating columns)")
    args = parser.parse_args()
    
    run_test_on_new_movies(args.csv_path)