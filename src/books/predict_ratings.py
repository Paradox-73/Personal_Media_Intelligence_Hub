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
from src.books.feature_engineering import _book_header, _book_body

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return 0
        s = str(val).strip()
        match = re.search(r'(\d{4})', s)
        if match: return int(match.group(1))
        return 0
    except:
        return 0



def batch_predict_books():
    print("🚀 Starting Book Batch Prediction...")

    # Load Model and State
    if not config.BOOKS_MODEL_REGRESSOR.exists() or not config.BOOKS_PREPROCESSOR_STATE.exists():
        print("❌ Error: Model or Preprocessor State not found. Run trainer first.")
        return

    model = joblib.load(config.BOOKS_MODEL_REGRESSOR)
    state = joblib.load(config.BOOKS_PREPROCESSOR_STATE)
    
    # Load Data
    try:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    
    print(f"   Loaded {len(df)} books.")

    # Feature Engineering
    df['my_rating_clean'] = pd.to_numeric(df['my_rating'], errors='coerce').fillna(0)
    df['year'] = df['publishedDate'].apply(clean_year)
    
    num_cols = ['year', 'pageCount', 'averageRating', 'ratingsCount']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Entity target encodings, rebuilt through the SAME helper feature_engineering
    # used. This block used to re-implement the author/category multi-hot by hand;
    # once training switched to target encodings it would have kept producing
    # columns the model no longer wants, and the alignment below would have filled
    # every real feature with a silent 0. Same failure that hit shows twice.
    _arts = state.get('te_artifacts') or {}
    _enc_cols = {}
    for _col, _lim in state.get('book_encode', []):
        _art = _arts.get(_col)
        if not _art:
            continue
        _pairs = [tf.apply_entity_map(v, _art['map'], _art.get('counts', {}),
                                      _art['mean'], _art.get('limit', _lim))
                  for v in df.get(_col, pd.Series('', index=df.index))]
        _enc_cols[f'{_col}_encoded'] = [p[0] for p in _pairs]
        _enc_cols[f'{_col}_n'] = [p[1] for p in _pairs]
    X_entity = pd.DataFrame(_enc_cols, index=df.index)

    print("   Generating text embeddings...")
    # MUST match feature_engineering.process_features(): the PCA in `state` was fit on
    # chunk-pooled, L2-normalised vectors built from `_book_header` + `_book_body`. A
    # plain `.encode()` here returned UNnormalised vectors of a differently-worded
    # string, truncated at 256 tokens -- so the scored embeddings landed in a different
    # space from the one the PCA and the model were trained on.
    _records = df.to_dict('records')
    df['text_header'] = [_book_header(r) for r in _records]
    _body = pd.Series([_book_body(r) for r in _records], index=df.index)
    df['text_content'] = df['text_header'] + " " + _body

    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = encode_pooled(df['text_header'].tolist(), _body.tolist(),
                                    transformer_model, show_progress_bar=True)
    X_text_pca = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # Combine and Align
    X_final = pd.concat([df[num_cols].reset_index(drop=True),
                         X_entity.reset_index(drop=True),
                         X_text.reset_index(drop=True)], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]

    # Align with training columns
    X_inf = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    for col in state['training_columns']:
        if col in X_final.columns:
            X_inf[col] = X_final[col]
    # Silent zero-fill is how a serve/train mismatch hides. Say it out loud.
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
    results_path = config.BASE_DIR / "results" / "books" / "book_predictions_detailed.csv"
    df[['title', 'authors', 'publishedDate', 'my_rating', 'predicted_rating', 'averageRating', 'categories']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_books()
