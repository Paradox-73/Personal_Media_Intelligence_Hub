import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def create_frozen_folds():
    print("🧊 Creating Frozen Fold Registry...")
    
    # 1. Load Universal Dataset (to get all item IDs)
    try:
        df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading unified training data: {e}")
        return
        
    if 'source_id' not in df.columns or 'media_type' not in df.columns:
        print("❌ 'source_id' or 'media_type' not found in training data.")
        return

    # We need a stable unique identifier for each item. 
    # 'media_type' + '_' + 'source_id' is one way, or if we have a title.
    # Let's use index if source_id is just index, but index changes!
    # Let's see if we can use a hash or unique identifier. 
    # If source_id is not stable, we'll use an MD5 hash of some stable columns in the future.
    # For now, let's assume 'source_id' is stable per media_type.
    
    df['global_id'] = df['media_type'] + "_" + df['source_id'].astype(str)
    
    registry = {}
    
    rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
    
    for split_idx, (train_idx, test_idx) in enumerate(rkf.split(df)):
        fold_id = split_idx
        for idx in test_idx:
            item_id = df.iloc[idx]['global_id']
            if item_id not in registry:
                registry[item_id] = []
            registry[item_id].append(fold_id)
            
    registry_path = config.UNIFIED_MODEL_DIR / "fold_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f)
        
    print(f"✅ Frozen folds saved to {registry_path} for {len(registry)} items.")

if __name__ == "__main__":
    create_frozen_folds()
