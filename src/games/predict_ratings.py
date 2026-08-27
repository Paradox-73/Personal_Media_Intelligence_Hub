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
from src.common.parsing import (  # canonical shared parsers
    parse_list, people_key, sanitize_col, clean_money)
from src.common import taste_features as tf
from src.common.text_embedding import encode_pooled
from src.games.feature_engineering import _game_header, _game_body

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
    
    # Developer / publisher / tag target encodings, through the same helper that
    # built the training columns. The old hand-rolled multi-hot here would silently
    # zero-fill every real feature once training moved to encodings.
    _arts = state.get('te_artifacts') or {}
    _enc_cols = {}
    for _col, _lim in state.get('game_encode', []):
        _art = _arts.get(_col)
        if not _art:
            continue
        _pairs = [tf.apply_entity_map(v, _art['map'], _art.get('counts', {}),
                                      _art['mean'], _art.get('limit', _lim))
                  for v in df.get(_col, pd.Series('', index=df.index))]
        _enc_cols[f'{_col}_encoded'] = [p[0] for p in _pairs]
        _enc_cols[f'{_col}_n'] = [p[1] for p in _pairs]
    _enc_cols['playtime'] = pd.to_numeric(df.get('playtime'), errors='coerce').fillna(0).values
    X_dev = pd.DataFrame(_enc_cols, index=df.index)

    print("   Generating text embeddings...")
    # MUST match feature_engineering.process_features(): the PCA in `state` was fit
    # on chunk-pooled, L2-normalised vectors. A plain `.encode()` here would score
    # every game in a different space from the one the model was trained in.
    _records = df.to_dict('records')
    df['text_header'] = [_game_header(r) for r in _records]
    _body = pd.Series([_game_body(r) for r in _records], index=df.index)
    df['text_content'] = df['text_header'] + " " + _body

    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)
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
    _missing = [c for c in state['training_columns'] if c not in X_final.columns]
    if _missing:
        print(f"   ⚠️ {len(_missing)} model feature(s) not built here, left at 0: "
              f"{_missing[:8]}{' ...' if len(_missing) > 8 else ''}")
    
    # Predict
    preds = np.asarray(model.predict(X_inf), dtype=float)
    # Deployment calibration: restamp onto the user's real rating distribution so the
    # Oracle predicts the full range (incl. 5*) instead of regressing to the mode.
    from src.utils.deployment_calibration import calibrate_to_ratings
    _r = pd.to_numeric(df['my_rating'], errors='coerce')
    _rated = (_r > 0).values
    preds = calibrate_to_ratings(preds, preds[_rated], _r.values[_rated])
    df['predicted_rating'] = np.round(np.clip(preds, 0.5, 5.0) * 2) / 2
    
    # Save Results
    results_path = config.BASE_DIR / "results" / "games" / "game_predictions_detailed.csv"
    df[['name', 'platform_from_text', 'released', 'my_rating', 'predicted_rating', 'metacritic', 'genres']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_games()
