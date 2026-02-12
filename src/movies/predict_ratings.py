import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.feature_engineering import transform_dataframe_for_prediction

def batch_predict_ratings():
    print("🚀 Starting Batch Prediction...")

    # Ensure output directory exists
    predictions_dir = Path("data/predictions/movies")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Artifacts
    try:
        regressor = joblib.load(config.MODEL_REGRESSOR)
        preprocessor_state = joblib.load(config.PREPROCESSOR_STATE)
    except FileNotFoundError:
        print("❌ Error: Model or preprocessor state not found. Run `python src/movies/model_trainer.py` first.")
        return

    # 2. Load Enriched Data
    try:
        enriched_df = pd.read_csv(config.ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv not found.")
        return

    # 3. Feature Engineering
    # We drop rows where target might be missing if we want strict evaluation, 
    # but for prediction we keep them. We will separate evaluation logic.
    X_predict = transform_dataframe_for_prediction(enriched_df, preprocessor_state)
    print(f"📊 Features for prediction created. Shape: {X_predict.shape}")

    # 4. Predict
    predictions = regressor.predict(X_predict)
    
    # 5. Prepare Output DataFrame
    output_df = enriched_df.copy()
    
    # Use 'title' or fallback to 'letterboxd_name' if title is missing
    output_df['name'] = output_df['title'].fillna(output_df['letterboxd_name'])
    output_df['predicted_rating'] = predictions.round(1) # Round for readability

    # 6. Evaluation & Stats
    if 'user_rating' in enriched_df.columns and not enriched_df['user_rating'].isna().all():
        output_df['actual_rating'] = output_df['user_rating']
        
        # Filter for evaluation (only rows where we have an actual rating)
        eval_df = output_df.dropna(subset=['actual_rating'])
        
        if not eval_df.empty:
            y_true = eval_df['actual_rating']
            y_pred = eval_df['predicted_rating']
            
            # Calculate Absolute Differences
            eval_df['abs_diff'] = np.abs(y_true - y_pred).round(1)
            output_df['abs_diff'] = np.abs(output_df['actual_rating'] - output_df['predicted_rating']).round(1)

            # --- Metrics Calculation ---
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            
            # --- Binning Counts ---
            diffs = eval_df['abs_diff']
            count_0 = (diffs == 0.0).sum()
            count_small = ((diffs >= 0.1) & (diffs <= 0.4)).sum()
            count_med = ((diffs >= 0.5) & (diffs <= 0.9)).sum()
            count_large = (diffs >= 1.0).sum()
            total = len(diffs)

            # --- Terminal Output ---
            print("\n" + "="*40)
            print("📊  ML ENGINEER PERFORMANCE REPORT")
            print("="*40)
            print(f"Total Evaluated Samples: {total}")
            print("-" * 40)
            print(f"📉 MAE (Mean Abs Error):  {mae:.4f}")
            print(f"📉 RMSE (Root Mean Sq):   {rmse:.4f}")
            print(f"📈 R² Score:              {r2:.4f}")
            print("-" * 40)
            print("🎯 Error Distribution (Absolute Diff):")
            print(f"   Exact Match (0.0):     {count_0:3d}  ({(count_0/total)*100:.1f}%)")
            print(f"   Tiny Diff   (0.1-0.4): {count_small:3d}  ({(count_small/total)*100:.1f}%)")
            print(f"   Small Diff  (0.5-0.9): {count_med:3d}  ({(count_med/total)*100:.1f}%)")
            print(f"   Large Diff  (>= 1.0):  {count_large:3d}  ({(count_large/total)*100:.1f}%)")
            print("="*40 + "\n")
            
        else:
            print("⚠️ No valid actual ratings found for evaluation stats.")
    else:
        print("⚠️ 'user_rating' column missing. Cannot calculate ML stats.")

    # 7. Select & Save Columns
    # Selecting only specific columns as requested
    cols_to_save = ['name', 'year', 'actual_rating', 'predicted_rating', 'abs_diff']
    
    # Handle case where columns might be missing (e.g. predicting on new data without actuals)
    final_cols = [c for c in cols_to_save if c in output_df.columns]
    final_output = output_df[final_cols]

    output_file_path = predictions_dir / "predicted_ratings.csv"
    final_output.to_csv(output_file_path, index=False)
    print(f"🎉 Predicted ratings saved to: {output_file_path}")

if __name__ == "__main__":
    batch_predict_ratings()