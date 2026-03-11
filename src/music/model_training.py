import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MUSIC_TRAINING_DATA_PATH, MUSIC_TASTE_PROFILE

def create_taste_profile():
    print("Loading processed audio embeddings...")
    df = pd.read_csv(MUSIC_TRAINING_DATA_PATH)
    
    # Extract only the 512 dimensions
    embedding_cols = [f'clap_dim_{i}' for i in range(512)]
    X = df[embedding_cols].values
    
    print("Calculating Taste Centroid...")
    # Calculate the mean vector across all songs
    taste_centroid = np.mean(X, axis=0)
    
    # Save the profile
    joblib.dump(taste_centroid, MUSIC_TASTE_PROFILE)
    print(f"✅ Taste Profile generated and saved to {MUSIC_TASTE_PROFILE}")
    print("Your Oracle is ready.")

if __name__ == "__main__":
    create_taste_profile()