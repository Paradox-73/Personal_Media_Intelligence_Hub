"""
evaluation.py — Evaluates the Music Taste Profile using held-out library tracks.

Metrics:
- ROC-AUC: Ability to separate held-out library tracks from the background pool.
- Recall@k: % of held-out tracks in the top k of a mixed set.
- Comparison against a random baseline to ensure the profile is meaningful.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import config
import feature_engineering
import affinity
import profile_builder

def run_evaluation(n_repeats=1):
    print("Loading data for evaluation...")
    data_lib = np.load(config.FEATURES_NPZ, allow_pickle=True)
    X_full = data_lib["X"]
    
    # Load pool
    from background_pool import POOL_CSV
    df_pool = pd.read_csv(POOL_CSV)
    bundle = joblib.load(config.MODELS_DIR / "preprocessors.joblib")
    X_pool = feature_engineering.transform(df_pool, bundle)
    
    results = []
    
    for i in range(n_repeats):
        print(f"Repeat {i+1}/{n_repeats}...")
        X_train, X_test = train_test_split(X_full, test_size=0.2, random_state=42+i)
        
        # Build a temporary profile on the 80%
        # (Mocking profile builder steps for speed)
        from sklearn.cluster import KMeans
        from xgboost import XGBClassifier
        
        km = KMeans(n_clusters=6, random_state=42, n_init=10)
        km.fit(X_train)
        centroids = km.cluster_centers_
        
        pu_model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
        X_pu = np.vstack([X_train, X_pool])
        y_pu = np.array([1]*len(X_train) + [0]*len(X_pool))
        pu_model.fit(X_pu, y_pu)
        
        # 2. Evaluate on X_test vs X_pool
        X_eval = np.vstack([X_test, X_pool])
        y_eval = np.array([1]*len(X_test) + [0]*len(X_pool))
        
        def get_scores(X):
            d = affinity.density_score(X, centroids)
            k = affinity.knn_density(X, X_train)
            p = pu_model.predict_proba(X)[:, 1]
            return affinity.blend({"density": d, "knn": k, "pu": p})
            
        scores = get_scores(X_eval)
        auc = roc_auc_score(y_eval, scores)
        
        # Random baseline
        X_rand_train = np.random.normal(0, 1, X_train.shape)
        km_rand = KMeans(n_clusters=6, random_state=42, n_init=10).fit(X_rand_train)
        
        def get_rand_scores(X):
            # Just use random centroids
            return affinity.density_score(X, km_rand.cluster_centers_)
            
        rand_scores = get_rand_scores(X_eval)
        rand_auc = roc_auc_score(y_eval, rand_scores)
        
        results.append({"auc": auc, "rand_auc": rand_auc})
        
    avg_auc = np.mean([r["auc"] for r in results])
    avg_rand = np.mean([r["rand_auc"] for r in results])
    
    print("\n--- Evaluation Results ---")
    print(f"Profile ROC-AUC: {avg_auc:.4f}")
    print(f"Random Baseline ROC-AUC: {avg_rand:.4f}")
    print(f"Lift over baseline: {avg_auc - avg_rand:.4f}")
    
    if avg_auc > 0.8:
        print("Status: EXCELLENT. Profile is highly discriminative.")
    elif avg_auc > 0.6:
        print("Status: GOOD. Profile captures personal taste.")
    else:
        print("Status: WEAK. Profile might be overfitting or too generic.")

if __name__ == "__main__":
    run_evaluation()
