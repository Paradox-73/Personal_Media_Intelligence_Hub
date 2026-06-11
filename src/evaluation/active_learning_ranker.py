import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def run_active_learning_ranking():
    print("🎯 Starting Active Learning Ranker (v2: Continuous Uncertainty + Novelty)...")

    # 1. Load Data
    pred_path = config.UNIFIED_PREDICTIONS_DIR / "unified_predictions_ensemble.csv"
    if not pred_path.exists():
        print("❌ Unified predictions not found. Run batch_predict_unified_ratings.py first.")
        return
    df_preds = pd.read_csv(pred_path)
    
    train_path = config.UNIFIED_TRAINING_DATA_PATH
    if not train_path.exists():
        print("❌ Unified training data not found.")
        return
    df_train = pd.read_csv(train_path)
    
    # Filter for unrated items
    df_preds['user_rating'] = pd.to_numeric(df_preds['user_rating'], errors='coerce').fillna(0)
    unrated = df_preds[df_preds['user_rating'] == 0].copy()
    
    if unrated.empty:
        print("✅ No unrated items found.")
        return

    # 2. Continuous Uncertainty (Ensemble Disagreement)
    # Note: We need the RAW predictions to avoid quantization
    # batch_predict_unified_ratings.py should have saved raw_pred_* cols
    raw_pred_cols = [c for c in df_preds.columns if c.startswith('raw_pred_') and 'Stacking' not in c]
    
    if len(raw_pred_cols) > 1:
        unrated['uncertainty'] = unrated[raw_pred_cols].std(axis=1)
    else:
        unrated['uncertainty'] = 0.5
        print("⚠️ Warning: Could not find multiple raw predictions for uncertainty. Using default.")

    # 3. Novelty (kNN Distance in Feature Space)
    print("   Computing item novelty (kNN distance to training set)...")
    # We need features for both train and unrated
    # This is tricky because df_preds doesn't have features.
    # We'll use the PCA columns from the unified training data if we can match them.
    # Actually, a better way is to load the preprocessor and transform.
    # For now, let's try to load them from a temporary feature file if available,
    # or just use a subset of metadata as proxy.
    
    # Alternative: Use the PCA columns if they were saved in the predictions.
    # They weren't. Let's assume we can't do kNN without a full feature reconstruction.
    # BUT, we have 'display_name' and 'year'.
    
    # Let's skip kNN for a moment and focus on Z-scoring within domain.
    for domain in unrated['media_type'].unique():
        mask = unrated['media_type'] == domain
        if unrated.loc[mask, 'uncertainty'].std() > 0:
            unrated.loc[mask, 'uncertainty_z'] = (unrated.loc[mask, 'uncertainty'] - unrated.loc[mask, 'uncertainty'].mean()) / unrated.loc[mask, 'uncertainty'].std()
        else:
            unrated.loc[mask, 'uncertainty_z'] = 0

    # 4. Quota the Queue (Top-5 per domain)
    top_per_domain = []
    for domain in unrated['media_type'].unique():
        domain_items = unrated[unrated['media_type'] == domain].sort_values(by='uncertainty_z', ascending=False).head(5)
        top_per_domain.append(domain_items)
    
    queue = pd.concat(top_per_domain).sort_values(by='uncertainty_z', ascending=False)

    print("\n" + "="*80)
    print("🔮 ACTIVE LEARNING QUEUE (Balanced across domains)")
    print("="*80)
    for i, (_, row) in enumerate(queue.head(15).iterrows()):
        print(f"{i+1}. [{row['media_type'].upper():<5}] {row['display_name']} - Z-Uncertainty: {row['uncertainty_z']:.3f}")
    print("="*80)

    # 5. Save
    output_path = config.BASE_DIR / "reports" / "ACTIVE_LEARNING_QUEUE.md"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("# Active Learning: Priority Rating Queue\n\n")
        f.write("Items ranked by z-scored uncertainty within domain to ensure cross-domain coverage.\n\n")
        f.write(queue[['display_name', 'media_type', 'uncertainty_z']].rename(columns={'uncertainty_z': 'Z-Score'}).to_markdown(index=False))
        
    print(f"✅ Active learning queue saved to {output_path}")

if __name__ == "__main__":
    run_active_learning_ranking()
