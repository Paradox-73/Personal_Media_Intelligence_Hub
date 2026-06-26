import pandas as pd
import joblib
import numpy as np
import sys
import re
import ast
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return 0
        s = str(val).strip()
        if '-' in s:
            parts = s.split('-')
            if len(parts[0]) == 4: return int(parts[0])
            if len(parts[-1]) == 4: return int(parts[-1])
        return int(float(s[:4]))
    except:
        return 0

def parse_list(x):
    if pd.isna(x) or x == "": return []
    try:
        if isinstance(x, list): return x
        s = str(x).strip()
        if s.startswith('[') and s.endswith(']'):
            return ast.literal_eval(s)
        return [item.strip() for item in s.split(',')]
    except:
        return [str(x)]

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

def batch_predict_games():
    print("🚀 Starting Game Batch Prediction...")

    # Load Model and State
    if not config.GAMES_MODEL_REGRESSOR.exists() or not config.GAMES_MODEL_PREPROCESSOR_STATE.exists():
        print("❌ Error: Model or Preprocessor State not found. Run trainer first.")
        return

    model = joblib.load(config.GAMES_MODEL_REGRESSOR)
    state = joblib.load(config.GAMES_MODEL_PREPROCESSOR_STATE)
    
    # Load Data
    try:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH, encoding='latin1')
    
    print(f"   Loaded {len(df)} games.")

    # Feature Engineering (Manual replication for inference)
    df['my_rating_clean'] = pd.to_numeric(df['my_rating'], errors='coerce').fillna(0)
    df['year'] = df['released'].apply(clean_year)
    
    num_cols = ['year', 'metacritic', 'rating', 'ratings_count', 'reviews_count']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    X_platform = pd.get_dummies(df['platform_from_text'], prefix='plat')
    
    df['genre_list'] = df['genres'].apply(parse_list)
    X_genre = pd.DataFrame(state['mlb_genre'].transform(df['genre_list']), 
                           columns=[f"gen_{sanitize_col(c)}" for c in state['mlb_genre'].classes_], 
                           index=df.index)
    
    df['dev_list'] = df['developers'].apply(parse_list)
    dev_encoded = state['mlb_dev'].transform(df['dev_list'])
    X_dev = pd.DataFrame(dev_encoded[:, state['valid_dev_indices']], 
                         columns=[f"dev_{sanitize_col(state['mlb_dev'].classes_[i])}" for i in state['valid_dev_indices']], 
                         index=df.index)

    print("   Generating text embeddings...")
    df['text_content'] = "Name: " + df['name'].fillna('') + \
                         ". Genres: " + df['genres'].fillna('') + \
                         ". Tags: " + df['tags'].fillna('') + \
                         ". Description: " + df['description_raw'].fillna('')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    X_text_pca = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # Combine and Align
    X_final = pd.concat([df[num_cols], X_platform, X_genre, X_dev, X_text], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    
    # Align with training columns
    X_inf = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    for col in state['training_columns']:
        if col in X_final.columns:
            X_inf[col] = X_final[col]
    
    # Predict
    preds = np.asarray(model.predict(X_inf), dtype=float)
    # Deployment calibration: restamp onto the user's real rating distribution so the
    # Oracle predicts the full range (incl. 5*) instead of regressing to the mode.
    from src.utils.deployment_calibration import calibrate_to_ratings
    _r = pd.to_numeric(df['my_rating'], errors='coerce')
    _rated = (_r > 0).values
    preds = calibrate_to_ratings(preds, preds[_rated], _r.values[_rated])
    df['predicted_rating'] = np.round(np.clip(preds, 0, 5) * 2) / 2
    
    # Save Results
    results_path = config.BASE_DIR / "results" / "games" / "game_predictions_detailed.csv"
    df[['name', 'platform_from_text', 'released', 'my_rating', 'predicted_rating', 'metacritic', 'genres']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_games()
