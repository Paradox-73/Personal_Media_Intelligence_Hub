import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.feature_engineering import transform_dataframe_for_prediction

def batch_predict_ratings():
    print("🚀 Starting Batch Prediction...")

    # Ensure output directory exists
    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Artifacts
    try:
        regressor = joblib.load(config.MODEL_REGRESSOR)
        preprocessor_state = joblib.load(config.PREPROCESSOR_STATE)
    except FileNotFoundError:
        print("❌ Error: Model or preprocessor state not found. Run `python src/model_trainer.py` first.")
        return

    # 2. Load Enriched Data
    try:
        enriched_df = pd.read_csv(config.ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv not found.")
        return

    # Keep original user_rating for calculating absolute difference
    original_ratings = enriched_df['user_rating'] if 'user_rating' in enriched_df.columns else None

    # 3. Feature Engineering
    X_predict = transform_dataframe_for_prediction(enriched_df, preprocessor_state)
    print(f"📊 Features for prediction created. Shape: {X_predict.shape}")

    # 4. Predict
    predictions = regressor.predict(X_predict)
    print("✅ Predictions generated.")

    # 5. Prepare Output DataFrame
    output_df = enriched_df.copy()
    output_df['predicted_rating'] = predictions.round(1) # Round predictions to one decimal place

    if original_ratings is not None:
        output_df['actual_rating'] = original_ratings
        output_df['abs_diff'] = np.abs(output_df['predicted_rating'] - output_df['actual_rating']).round(1)
        print("✅ Absolute differences calculated.")
    else:
        print("⚠️ 'user_rating' column not found in enriched data. Skipping absolute difference calculation.")

    # 6. Save Results
    output_file_path = predictions_dir / "predicted_ratings.csv"
    output_df.to_csv(output_file_path, index=False)
    print(f"🎉 Predicted ratings saved to {output_file_path}")

if __name__ == "__main__":
    batch_predict_ratings()
