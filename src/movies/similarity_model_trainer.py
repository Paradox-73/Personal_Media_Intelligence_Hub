import pandas as pd
import numpy as np
import joblib
import sys
import ast
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def parse_list(x):
    if pd.isna(x) or x == "": return []
    try:
        if isinstance(x, list): return x
        if "[" in str(x): return ast.literal_eval(str(x))
        return [str(x)]
    except: return []

def train_content_similarity_model():
    """
    Builds a content-based item-item similarity model.
    The similarity between movies is based on their features (genre, director, etc.),
    not collaborative rating patterns. This allows us to predict for any movie.
    """
    print("🤝 Starting Content-Based Similarity Model Training...")

    # 1. Load Data & Preprocessing Artifacts
    try:
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH).set_index('letterboxd_uri', drop=False)
        state = joblib.load(config.PREPROCESSOR_STATE)
        print(f"   Loaded {len(df)} movies and preprocessor state.")
    except FileNotFoundError as e:
        print(f"❌ Error: Missing required file: {e.filename}. Run previous steps.")
        return

    # 2. Engineer Features for Similarity
    # We'll create a feature matrix using a subset of features from the main model
    
    # A. Genre Features
    mlb_genre = state['mlb_genre']
    genre_features = mlb_genre.transform(df['genre'].apply(parse_list))
    genre_df = pd.DataFrame(genre_features, index=df.index, columns=mlb_genre.classes_)

    # B. Text Features (using the existing PCA representation)
    df['text_content'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    tfidf_mat = state['tfidf'].transform(df['text_content'])
    pca_mat = state['pca'].transform(tfidf_mat.toarray())
    text_df = pd.DataFrame(pca_mat, index=df.index, columns=[f'pca_{i}' for i in range(pca_mat.shape[1])])

    # C. Key Categorical Features (Director, Primary Actor)
    def create_dummies_for_similarity(series, valid_list, prefix):
        # Get the primary value (e.g., first director)
        cleaned_series = series.apply(lambda x: parse_list(x)[0] if parse_list(x) else 'Unknown')
        # Mark values not in the original training list as 'Other'
        final_series = cleaned_series.apply(lambda x: x if x in valid_list else 'Other')
        # Create dummies, ensuring all original columns are present
        all_categories = sorted(list(set(valid_list) | {'Other'}))
        cat_type = pd.CategoricalDtype(categories=all_categories, ordered=False)
        return pd.get_dummies(final_series.astype(cat_type), prefix=prefix)

    dir_df = create_dummies_for_similarity(df['director'], state['valid_directors'], 'dir')
    act_df = create_dummies_for_similarity(df['actors'], state['valid_actors'], 'act')
    
    # Align indices
    dir_df.index = df.index
    act_df.index = df.index

    # 3. Assemble the Full Content Feature Matrix
    feature_matrix = pd.concat([genre_df, text_df, dir_df, act_df], axis=1)
    print(f"   Assembled content feature matrix with shape: {feature_matrix.shape}")

    # 4. Compute Cosine Similarity
    # This matrix shows how similar each movie is to every other based on content
    content_similarity_matrix = cosine_similarity(feature_matrix.fillna(0))
    print(f"   Computed similarity matrix with shape: {content_similarity_matrix.shape}")

    # 5. Get User Ratings
    rated_df = df.dropna(subset=['user_rating']).copy()
    user_ratings = rated_df[['user_rating']].copy() # Keep as DataFrame with URI index

    if len(rated_df) < 5:
        print("⚠️ Warning: Not enough rated movies to provide meaningful evaluation.")
        test_df = pd.DataFrame()
    else:
        # Split rated data for evaluation
        _, test_df = train_test_split(rated_df, test_size=0.2, random_state=42)
        print(f"   Using {len(test_df)} ratings for evaluation.")

    # 6. Save Artifacts
    # The URI-to-index mapping now covers ALL movies
    movie_uri_to_idx = {uri: i for i, uri in enumerate(df.index)}
    
    artifacts = {
        'similarity_matrix': content_similarity_matrix,
        'movie_uri_to_idx': movie_uri_to_idx,
        'user_ratings': user_ratings
    }
    
    config.MOVIES_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, config.SIMILARITY_MATRIX)
    
    print(f"✅ Content-based similarity model artifacts saved to {config.SIMILARITY_MATRIX}")

    # 7. Evaluate on the test set
    if not test_df.empty:
        evaluate_model(artifacts, test_df)

def evaluate_model(artifacts, test_df):
    """Calculates the RMSE for the test set using the new content-based model."""
    print("   Evaluating model performance on test set...")
    predictions = []
    actuals = []
    
    for uri, row in test_df.iterrows():
        actual_rating = row['user_rating']
        pred, _ = predict_single_rating(artifacts, uri)
        if pred is not None:
            predictions.append(pred)
            actuals.append(actual_rating)

    if not predictions:
        print("   Could not generate any predictions for the test set.")
        return

    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
    print(f"   📉 [TEST SET] Content-Similarity Model RMSE: {rmse:.4f}")

def predict_single_rating(artifacts, target_movie_uri, top_n=15):
    """Predicts a single movie's rating based on content similarity."""
    sim_matrix = artifacts['similarity_matrix']
    uri_to_idx = artifacts['movie_uri_to_idx']
    user_ratings = artifacts['user_ratings'] # DF with user_rating col and URI index

    if target_movie_uri not in uri_to_idx:
        return None, "Movie not found in dataset."

    target_idx = uri_to_idx[target_movie_uri]
    sim_scores = sim_matrix[target_idx]

    # Get indices of movies the user has rated
    rated_uris = set(user_ratings.index)
    
    # Find similarity scores for rated movies
    numerator = 0
    denominator = 0
    
    # Get sorted similarities
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    count = 0
    idx_to_uri = {i: uri for uri, i in uri_to_idx.items()}

    for idx in sorted_indices:
        # Stop after finding top_n neighbors
        if count >= top_n:
            break
        # Skip the movie itself
        if idx == target_idx:
            continue
        
        movie_uri = idx_to_uri.get(idx)
        if movie_uri in rated_uris:
            rating = user_ratings.loc[movie_uri, 'user_rating']
            similarity = sim_scores[idx]
            
            numerator += similarity * rating
            denominator += similarity
            count += 1
            
    if denominator == 0:
        return None, "No similar rated movies found."

    prediction = numerator / denominator
    reason = f"Based on {count} most similar rated movies."
    return prediction, reason

if __name__ == "__main__":
    train_content_similarity_model()
