import pandas as pd
import numpy as np
import joblib
import sys
import ast
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def safe_eval(val):
    if pd.isna(val) or val == "": return []
    try:
        if isinstance(val, list): return val
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return [x.strip() for x in str(val).split(',') if x.strip()]

def get_primary_network(val):
    if pd.isna(val) or val == "": return "Other"
    try:
        if "[" in str(val):
            val_list = ast.literal_eval(str(val))
            if val_list: return val_list[0].strip()
        return str(val).split(',')[0].strip()
    except:
        return "Other"

def run_predictions():
    print("🚀 STARTING PREDICTION (NLP + NETWORK AWARE)...")

    # 1. Load Artifacts
    try:
        if not config.TV_SHOWS_MODEL_REGRESSOR.exists():
            print(f"❌ Error: Model not found at {config.TV_SHOWS_MODEL_REGRESSOR}")
            return
            
        model = joblib.load(config.TV_SHOWS_MODEL_REGRESSOR)
        state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        return

    # 2. Load Data
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    print(f"   Loaded {len(df)} shows for prediction.")
    
    # --- 3. Feature Engineering (Must Match Training EXACTLY) ---
    
    # A. Numericals
    df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(state['median_vote_avg'])
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(state['median_year'])
    df['season_count'] = pd.to_numeric(df['number_of_seasons'], errors='coerce').fillna(1)
    
    # B. Vibe Features (Network & Adult)
    df['is_adult'] = df['age_rating'].apply(lambda x: 1 if str(x) in ['TV-MA', 'R', '18+'] else 0)
    
    df['primary_network'] = df['network'].apply(get_primary_network)
    # Use the 'top_networks' list saved during training to encode
    df['network_clean'] = df['primary_network'].apply(lambda x: x if x in state['top_networks'] else "Other")
    
    # Manual One-Hot Encoding for Networks
    for net in state['top_networks']:
        df[f"net_{net}"] = (df['network_clean'] == net).astype(int)
    # Ensure net_Other exists if it was in training
    if "net_Other" not in df.columns:
        df["net_Other"] = (df['network_clean'] == "Other").astype(int)

    # C. Genres
    df['genres_list'] = df['genres'].apply(safe_eval)
    for col_name in state['kept_genres']:
        raw_genre = col_name.replace('g_', '')
        df[col_name] = df['genres_list'].apply(lambda x: 1 if raw_genre in x else 0)

    # D. NLP (Text Features)
    if 'tfidf_model' in state:
        print("   📚 Processing text features...")
        df['overview'] = df['overview'].fillna('')
        tfidf = state['tfidf_model']
        tfidf_matrix = tfidf.transform(df['overview'])
        
        # Create temporary DF for text features
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=state['tfidf_cols'], index=df.index)
        # Attach to main DF temporarily
        df = pd.concat([df, tfidf_df], axis=1)

    # E. Final Assembly
    # Create the X dataframe with ONLY the columns the model expects, in correct order
    train_cols = state['features_columns']
    X = pd.DataFrame(index=df.index)
    
    for col in train_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0 # Safety fill for missing columns
            
    print(f"📊 Features prepared. Shape: {X.shape}")

    # 4. Predict
    predictions = model.predict(X)
    df['predicted_rating'] = np.clip(predictions, 1, 10).round(1)

    # 5. Statistical Report (RESTORED)
    analysis_df = df[df['user_rating'].notna()].copy()
    
    if not analysis_df.empty:
        y_true = analysis_df['user_rating']
        y_pred = analysis_df['predicted_rating']
        
        # Calculate diffs
        analysis_df['abs_diff'] = abs(y_true - y_pred)
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Bins
        exact = len(analysis_df[analysis_df['abs_diff'] == 0])
        tiny = len(analysis_df[(analysis_df['abs_diff'] > 0) & (analysis_df['abs_diff'] <= 0.5)])
        small = len(analysis_df[(analysis_df['abs_diff'] > 0.5) & (analysis_df['abs_diff'] <= 1.0)])
        large = len(analysis_df[analysis_df['abs_diff'] > 1.0])
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
        print(f"   Exact Match (0.0):      {exact:<3} ({(exact/total)*100:.1f}%)")
        print(f"   Tiny Diff   (0.1-0.5):  {tiny:<3} ({(tiny/total)*100:.1f}%)")
        print(f"   Small Diff  (0.6-1.0):  {small:<3} ({(small/total)*100:.1f}%)")
        print(f"   Large Diff  (> 1.0):    {large:<3} ({(large/total)*100:.1f}%)")
        print("========================================\n")
        
        # Map diffs back to main dataframe
        df.loc[analysis_df.index, 'abs_diff'] = analysis_df['abs_diff']
    else:
        df['abs_diff'] = np.nan

    # 6. Save Dashboard View
    output_cols = ['name', 'year', 'user_rating', 'predicted_rating', 'abs_diff', 'primary_network']
    
    # Save sorted by biggest errors (so you can see what to fix next) or prediction
    # Let's save sorted by prediction for general use
    final_df = df[output_cols].sort_values(by='predicted_rating', ascending=False)
    
    final_df.to_csv(config.TV_SHOWS_FULL_VIEW_PATH, index=False)
    print(f"✅ Dashboard view saved to: {config.TV_SHOWS_FULL_VIEW_PATH}")

if __name__ == "__main__":
    run_predictions()