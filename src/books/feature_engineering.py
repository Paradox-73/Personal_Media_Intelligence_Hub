import pandas as pd
import numpy as np
import joblib
import ast
import sys
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# --- CONFIG ---
PCA_COMPONENTS = 15 # Reduced for small dataset

def clean_year(val):
    try:
        if pd.isna(val) or val == '': return 0
        s = str(val).strip()
        # Handle formats like DD-MM-YYYY, YYYY-MM-DD, or just YYYY
        match = re.search(r'(\d{4})', s)
        if match:
            return int(match.group(1))
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

def process_features():
    print("🛠️ Starting Book Feature Engineering...")

    if not config.BOOKS_ENRICHED_DATA_PATH.exists():
        print(f"❌ Error: Enriched data not found at {config.BOOKS_ENRICHED_DATA_PATH}")
        return

    try:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    except:
        df = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH, encoding='latin1')
    
    # 1. Clean Target
    df = df.dropna(subset=['my_rating']).copy()
    df['my_rating_clean'] = pd.to_numeric(df['my_rating'], errors='coerce').fillna(0)
    
    # 2. Extract Year
    df['year'] = df['publishedDate'].apply(clean_year)
    
    # 3. Numeric Features
    num_cols = ['year', 'pageCount', 'averageRating', 'ratingsCount']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 4. Authors Multi-Hot (Top ones to avoid explosion)
    df['author_list'] = df['authors'].apply(parse_list)
    mlb_author = MultiLabelBinarizer()
    author_encoded = mlb_author.fit_transform(df['author_list'])
    # Only keep authors that appear at least twice
    author_counts = author_encoded.sum(axis=0)
    valid_author_indices = np.where(author_counts >= 2)[0]
    X_author = pd.DataFrame(author_encoded[:, valid_author_indices], 
                            columns=[f"aut_{sanitize_col(mlb_author.classes_[i])}" for i in valid_author_indices], 
                            index=df.index)
    
    # 5. Categories Multi-Hot
    df['cat_list'] = df['categories'].apply(parse_list)
    mlb_cat = MultiLabelBinarizer()
    X_cat = pd.DataFrame(mlb_cat.fit_transform(df['cat_list']), 
                         columns=[f"cat_{sanitize_col(c)}" for c in mlb_cat.classes_], 
                         index=df.index)
    
    # 6. Text Embeddings
    print("   Generating text embeddings...")
    df['text_content'] = "Title: " + df['title'].fillna('') + \
                         ". Authors: " + df['authors'].fillna('') + \
                         ". Categories: " + df['categories'].fillna('') + \
                         ". Description: " + df['description'].fillna('')
    
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=True)
    
    print("   Applying PCA to embeddings...")
    pca = PCA(n_components=min(PCA_COMPONENTS, len(df)-1))
    X_text_pca = pca.fit_transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(X_text_pca.shape[1])])

    # 7. Combine
    X_final = pd.concat([df[num_cols].reset_index(drop=True), 
                         X_author.reset_index(drop=True), 
                         X_cat.reset_index(drop=True), 
                         X_text.reset_index(drop=True)], axis=1)
    
    X_final.columns = [sanitize_col(col) for col in X_final.columns]
    X_final = X_final.loc[:, ~X_final.columns.duplicated()]
    
    # Targets
    X_final['target_reg'] = df['my_rating_clean'].values
    X_final['target_class'] = pd.cut(df['my_rating_clean'], bins=[-1, 2.5, 4.5, 5.1], labels=[0, 1, 2]).astype(int)

    # Save
    X_final.to_csv(config.BOOKS_TRAINING_DATA_PATH, index=False)
    
    state = {
        'mlb_author': mlb_author,
        'valid_author_indices': valid_author_indices,
        'mlb_cat': mlb_cat,
        'pca': pca,
        'median_values': df[num_cols].median().to_dict(),
        'training_columns': X_final.drop(columns=['target_reg', 'target_class']).columns.tolist(),
        'sentence_transformer': 'all-MiniLM-L6-v2'
    }
    joblib.dump(state, config.BOOKS_PREPROCESSOR_STATE)
    print(f"✅ Features processed. Shape: {X_final.shape}")

if __name__ == "__main__":
    process_features()
