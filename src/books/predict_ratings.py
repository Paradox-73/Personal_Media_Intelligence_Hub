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
        match = re.search(r'(\d{4})', s)
        if match: return int(match.group(1))
        return 0
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
    
    df['author_list'] = df['authors'].apply(parse_list)
    author_encoded = state['mlb_author'].transform(df['author_list'])
    X_author = pd.DataFrame(author_encoded[:, state['valid_author_indices']], 
                            columns=[f"aut_{sanitize_col(state['mlb_author'].classes_[i])}" for i in state['valid_author_indices']], 
                            index=df.index)
    
    df['cat_list'] = df['categories'].apply(parse_list)
    X_cat = pd.DataFrame(state['mlb_cat'].transform(df['cat_list']), 
                         columns=[f"cat_{sanitize_col(c)}" for c in state['mlb_cat'].classes_], 
                         index=df.index)

    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('') + \
                         ". Authors: " + df['authors'].fillna('') + \
                         ". Categories: " + df['categories'].fillna('') + \
                         ". Description: " + df['description'].fillna('')
    
    transformer_model = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    X_text_pca = state['pca'].transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # Combine and Align
    X_final = pd.concat([df[num_cols].reset_index(drop=True), 
                         X_author.reset_index(drop=True), 
                         X_cat.reset_index(drop=True), 
                         X_text.reset_index(drop=True)], axis=1)
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    
    # Align with training columns
    X_inf = pd.DataFrame(0, index=df.index, columns=state['training_columns'])
    for col in state['training_columns']:
        if col in X_final.columns:
            X_inf[col] = X_final[col]
    
    # Predict
    preds = model.predict(X_inf)
    df['predicted_rating'] = np.round(np.clip(preds, 0, 5) * 2) / 2
    
    # Save Results
    results_path = config.BASE_DIR / "results" / "books" / "book_predictions_detailed.csv"
    df[['title', 'authors', 'publishedDate', 'my_rating', 'predicted_rating', 'averageRating', 'categories']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_books()
