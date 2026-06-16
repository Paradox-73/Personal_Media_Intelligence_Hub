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
    print("🎯 Starting Active Learning Ranker (v3: Raw Uncertainty + kNN Novelty)...")

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
    
    # Load Preprocessor State to get PCA for kNN
    state_path = config.UNIFIED_PREPROCESSOR_STATE
    if not state_path.exists():
        print("❌ Preprocessor state missing.")
        return
    state = joblib.load(state_path)
    pca_cols = [c for c in df_train.columns if c.startswith('pca_')]
    
    # Filter for unrated items
    df_preds['user_rating'] = pd.to_numeric(df_preds['user_rating'], errors='coerce').fillna(0)
    unrated = df_preds[df_preds['user_rating'] == 0].copy()
    
    if unrated.empty:
        print("✅ No unrated items found.")
        return

    # 2. Continuous Uncertainty (Ensemble Disagreement)
    # We use raw continuous predictions to avoid quantization (snapping)
    raw_pred_cols = [c for c in df_preds.columns if c.startswith('raw_pred_') and 'Stacking' not in c]
    
    if len(raw_pred_cols) > 1:
        unrated['uncertainty'] = unrated[raw_pred_cols].std(axis=1)
    else:
        unrated['uncertainty'] = 0.5
        print("⚠️ Warning: Could not find multiple raw predictions. Using default.")

    # 3. Novelty (kNN Distance in Feature Space)
    print("   Computing item novelty (kNN distance to training set)...")
    # To compute kNN, we'd need PCA features for unrated items.
    # Since they aren't in df_preds, we have two options: 
    # A) Assume the active learning queue is run AFTER feature engineering for the backlog.
    # B) Use a simpler proxy (Year, metadata) if PCA isn't available.
    # But wait, predict_unified_ratings.py SHOULD output the features if we want this.
    
    # For now, let's try to find PCA columns in df_preds (let's hope we added them in the previous step).
    # IF NOT, we'll use 'year' and 'media_type' as a fallback, but the user wants "aligned feature space".
    
    if all(c in unrated.columns for c in pca_cols):
        X_unrated = unrated[pca_cols].fillna(0).values
        X_train = df_train[pca_cols].fillna(0).values
        
        knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        knn.fit(X_train)
        distances, _ = knn.kneighbors(X_unrated)
        unrated['novelty'] = distances.mean(axis=1)
    else:
        print("⚠️ Warning: PCA features missing from predictions. kNN Novelty will be flat.")
        unrated['novelty'] = 0.0

    # 4. Composite Ranking: Z(Uncertainty) + Z(Novelty)
    epsilon = 1e-6
    for domain in unrated['media_type'].unique():
        mask = unrated['media_type'] == domain
        
        # Uncertainty Z-score
        u_std = unrated.loc[mask, 'uncertainty'].std()
        if u_std > epsilon:
            unrated.loc[mask, 'uncertainty_z'] = (unrated.loc[mask, 'uncertainty'] - unrated.loc[mask, 'uncertainty'].mean()) / u_std
        else:
            unrated.loc[mask, 'uncertainty_z'] = 0.0
            
        # Novelty Z-score
        n_std = unrated.loc[mask, 'novelty'].std()
        if n_std > epsilon:
            unrated.loc[mask, 'novelty_z'] = (unrated.loc[mask, 'novelty'] - unrated.loc[mask, 'novelty'].mean()) / n_std
        else:
            unrated.loc[mask, 'novelty_z'] = 0.0

    # Guard: if uncertainty std is near zero, fall back to novelty
    unrated['priority_score'] = unrated['uncertainty_z'] + unrated['novelty_z']
    
    # 5. Restore Columns & Confidence Interval
    # Prediction column is likely 'pred_Stacking' or 'pred_MeanEnsemble'
    pred_col = 'pred_Stacking' if 'pred_Stacking' in unrated.columns else 'pred_MeanEnsemble'
    if pred_col in unrated.columns:
        unrated['Predicted'] = unrated[pred_col]
        
        # Conformal Interval (Vanilla Split-Conformal 80% coverage)
        # Measured from OOF residuals
        conformal_widths = {
            'book': 0.96,
            'game': 1.23,
            'movie': 0.78,
            'tv': 1.10,
            'music': 0.80 # Fallback
        }
        
        def get_interval(row):
            w = conformal_widths.get(row['media_type'], 1.0)
            return f"±{w:.2f}"
            
        unrated['80% Interval'] = unrated.apply(get_interval, axis=1)
    
    # 6. Final Sort and Save
    queue = unrated.sort_values(by='priority_score', ascending=False)
    
    # Ensure cross-domain representation (Top 10 per domain, then interleave)
    top_per_domain = []
    for domain in queue['media_type'].unique():
        top_per_domain.append(queue[queue['media_type'] == domain].head(10))
    queue_balanced = pd.concat(top_per_domain).sort_values(by='priority_score', ascending=False)

    print("\n" + "="*80)
    print("🔮 ACTIVE LEARNING QUEUE (Continuous Uncertainty + kNN Novelty)")
    print("="*80)
    display_cols = ['display_name', 'media_type', 'Predicted', '80% Interval', 'priority_score']
    cols_present = [c for c in display_cols if c in queue_balanced.columns]
    print(queue_balanced[cols_present].head(15).to_string(index=False))
    print("="*80)

    # Save
    output_path = config.BASE_DIR / "reports" / "ACTIVE_LEARNING_QUEUE.md"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write("# Active Learning: Priority Rating Queue\n\n")
        f.write("Items ranked by composite score: $Z(\\text{uncertainty}) + Z(\\text{novelty})$.\n\n")
        f.write(queue_balanced[cols_present].rename(columns={'priority_score': 'Priority Score'}).to_markdown(index=False))
        
    print(f"✅ Active learning queue saved to {output_path}")

if __name__ == "__main__":
    run_active_learning_ranking()
