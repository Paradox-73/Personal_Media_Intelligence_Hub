import pandas as pd
import numpy as np
import joblib
import sys
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def safe_eval(val):
    if pd.isna(val) or val == "": return []
    try:
        if isinstance(val, list): return val
        return ast.literal_eval(str(val))
    except:
        return [x.strip() for x in str(val).split(',') if x.strip()]

def run_predictions():
    print("🚀 Starting Batch Prediction...")

    # 1. Load Artifacts
    try:
        model = joblib.load(config.TV_SHOWS_MODEL_REGRESSOR)
        state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    except FileNotFoundError:
        print("❌ Error: Model or state not found.")
        return

    # 2. Load Data
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    
    # 3. Preprocess (Apply Training Medians/Encoders)
    mlb_genres = state['mlb_genres']
    train_cols = state['features_columns']
    median_values = state['median_values']
    cat_cols = state['cat_cols']

    # Numericals
    for col, med_val in median_values.items():
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(med_val)

    # Genres
    df['genres_temp'] = df['genres'].apply(safe_eval)
    genres_encoded = mlb_genres.transform(df['genres_temp'])
    genres_df = pd.DataFrame(genres_encoded, columns=[f"genre_{c}" for c in mlb_genres.classes_], index=df.index)

    # Categoricals
    df_encoded_cats = pd.get_dummies(df[cat_cols], dummy_na=True)

    # Combine & Align
    X_full = pd.concat([df[list(median_values.keys())], genres_df, df_encoded_cats], axis=1)
    
    for col in train_cols:
        if col not in X_full.columns:
            X_full[col] = 0
            
    X_final = X_full[train_cols]
    print(f"📊 Features for prediction created. Shape: {X_final.shape}")

    # 4. Predict
    predictions = model.predict(X_final)
    df['predicted_rating'] = np.clip(np.round(predictions, 1), 1, 10)

    # 5. Statistical Analysis (The Report)
    # Only analyze rows that have a real User Rating
    analysis_df = df[df['user_rating'].notna()].copy()
    
    if not analysis_df.empty:
        y_true = analysis_df['user_rating']
        y_pred = analysis_df['predicted_rating']
        
        # Core Metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Error Distribution
        analysis_df['abs_diff'] = abs(analysis_df['user_rating'] - analysis_df['predicted_rating'])
        
        exact = len(analysis_df[analysis_df['abs_diff'] == 0])
        tiny = len(analysis_df[(analysis_df['abs_diff'] > 0) & (analysis_df['abs_diff'] < 0.5)])
        small = len(analysis_df[(analysis_df['abs_diff'] >= 0.5) & (analysis_df['abs_diff'] < 1.0)])
        large = len(analysis_df[analysis_df['abs_diff'] >= 1.0])
        total = len(analysis_df)

        print("\n========================================")
        print("📊  ML ENGINEER PERFORMANCE REPORT")
        print("========================================")
        print(f"Total Evaluated Samples: {total}")
        print("----------------------------------------")
        print(f"📉 MAE (Mean Abs Error):  {mae:.4f}")
        print(f"📉 RMSE (Root Mean Sq):   {rmse:.4f}")
        print(f"📈 R² Score:              {r2:.4f}")
        print("----------------------------------------")
        print("🎯 Error Distribution (Absolute Diff):")
        print(f"   Exact Match (0.0):      {exact}  ({(exact/total)*100:.1f}%)")
        print(f"   Tiny Diff   (0.1-0.4):  {tiny}  ({(tiny/total)*100:.1f}%)")
        print(f"   Small Diff  (0.5-0.9):  {small}  ({(small/total)*100:.1f}%)")
        print(f"   Large Diff  (>= 1.0):   {large}  ({(large/total)*100:.1f}%)")
        print("========================================\n")

    # 6. Save
    if 'genres_temp' in df.columns: del df['genres_temp']
    df.to_csv(config.TV_SHOWS_FULL_VIEW_PATH, index=False)
    print(f"✅ Full dashboard view saved to: {config.TV_SHOWS_FULL_VIEW_PATH}")

if __name__ == "__main__":
    run_predictions()