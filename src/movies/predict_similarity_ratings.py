import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def predict_similarity_ratings_batch():
    """
    Generates predictions for all movies using the item-similarity model
    and provides a performance report for movies that already have a user rating.
    """
    print("BATCH PREDICTION: Generating ratings with Content-Similarity Model...")

    # 1. Load Artifacts
    try:
        artifacts = joblib.load(config.SIMILARITY_MATRIX)
        sim_matrix = artifacts['similarity_matrix']
        uri_to_idx = artifacts['movie_uri_to_idx']
        user_ratings = artifacts['user_ratings'] # FIX: Correct key is 'user_ratings'
        idx_to_uri = {i: uri for uri, i in uri_to_idx.items()}
        print("   ✅ Loaded similarity model artifacts.")
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ Error loading artifacts: {e}.")
        print("   Please run `python src/movies/similarity_model_trainer.py` first.")
        return

    # 2. Load all movie data
    try:
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        print(f"   Loaded {len(df)} total movies.")
    except FileNotFoundError:
        print(f"❌ Error: Enriched movie data not found at {config.MOVIES_ENRICHED_DATA_PATH}")
        return

    # 3. Predict for all movies
    tqdm.pandas(desc="   Predicting ratings for all movies")
    df['predicted_rating_similarity'] = df['letterboxd_uri'].progress_apply(
        lambda uri: predict_single_rating_for_batch(uri, sim_matrix, uri_to_idx, idx_to_uri, user_ratings)
    )

    # 4. Performance Report
    eval_df = df.dropna(subset=['user_rating', 'predicted_rating_similarity']).copy()
    if not eval_df.empty:
        y_true = eval_df['user_rating']
        y_pred = eval_df['predicted_rating_similarity']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        diffs = np.abs(y_true - y_pred)
        
        print("\n" + "="*45)
        print("📊 CONTENT-SIMILARITY PERFORMANCE REPORT")
        print("="*45)
        print(f"Total Evaluated: {len(eval_df)}")
        print("-" * 45)
        print(f"📉 MAE:  {mae:.4f}")
        print(f"📉 RMSE: {rmse:.4f}")
        print(f"📈 R²:   {r2:.4f}")
        print("-" * 45)
        print(f"   Exact (0.0):    {(diffs == 0).sum():<3} ({((diffs == 0).sum()/len(diffs))*100:.1f}%)")
        print(f"   Tiny (0.1-0.4): {((diffs > 0) & (diffs <= 0.4)).sum():<3} ({(((diffs > 0) & (diffs <= 0.4)).sum()/len(diffs))*100:.1f}%)")
        print(f"   Small (0.5-0.9):{((diffs >= 0.5) & (diffs <= 0.9)).sum():<3} ({(((diffs >= 0.5) & (diffs <= 0.9)).sum()/len(diffs))*100:.1f}%)")
        print(f"   Large (>= 1.0): {(diffs >= 1.0).sum():<3} ({((diffs >= 1.0).sum()/len(diffs))*100:.1f}%)")
        print("="*45 + "\n")
        
        df.loc[eval_df.index, 'abs_diff_similarity'] = diffs

    # 5. Format and Save Results
    output_df = df.dropna(subset=['predicted_rating_similarity']).copy()
    output_df['predicted_rating_similarity'] = output_df['predicted_rating_similarity'].round(2)
    
    out_cols = [
        'title', 'year', 'user_rating', 'predicted_rating_similarity', 'abs_diff_similarity'
    ]
    final_df = output_df[[c for c in out_cols if c in output_df.columns]].sort_values(
        by='predicted_rating_similarity', ascending=False
    )

    config.MOVIES_PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(config.SIMILARITY_PREDICTIONS_PATH, index=False)
    
    print(f"✅ Saved {len(final_df)} similarity predictions to {config.SIMILARITY_PREDICTIONS_PATH}")


def predict_single_rating_for_batch(target_movie_uri, sim_matrix, uri_to_idx, idx_to_uri, user_ratings, top_n=15):
    """
    Predicts a rating for a single movie based on content similarity to rated movies.
    """
    if target_movie_uri not in uri_to_idx:
        return None 
        
    target_idx = uri_to_idx[target_movie_uri]
    sim_scores = sim_matrix[target_idx]
    
    rated_uris = set(user_ratings.index)
    
    numerator = 0
    denominator = 0
    count = 0
    
    # Get sorted similarities to find top N neighbors
    sorted_indices = np.argsort(sim_scores)[::-1]

    for idx in sorted_indices:
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
        # Return the user's average rating if no similar movies are found
        return user_ratings['user_rating'].mean()

    return numerator / denominator


if __name__ == "__main__":
    predict_similarity_ratings_batch()

