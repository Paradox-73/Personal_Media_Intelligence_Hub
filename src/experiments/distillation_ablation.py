import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def calculate_skill_score(y_true, y_pred, y_train_mean):
    mae_model = mean_absolute_error(y_true, y_pred)
    mae_baseline = mean_absolute_error(y_true, np.full_like(y_true, y_train_mean))
    if mae_baseline == 0: return 0
    return 1 - (mae_model / mae_baseline)

def run_distillation_ablation(domain='game'):
    print(f"🧪 Starting Distillation Ablation for {domain.upper()}...")

    train_path = config.GAMES_TRAINING_DATA_PATH if domain == 'game' else config.BOOKS_TRAINING_DATA_PATH
    df = pd.read_csv(train_path)
    
    # Load Unified Data & Registry
    df_uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    
    # 1. Inject correct global_id into local df
    # Since row order is identical between local training_features and unified's slice of that domain
    domain_uni = df_uni[df_uni['media_type'] == domain].copy()
    if len(domain_uni) != len(df):
        print(f"❌ Error: Unified has {len(domain_uni)} {domain}s, but local has {len(df)} {domain}s. Run feature engineering again.")
        return
        
    df['source_id'] = domain_uni['source_id'].values
    
    # Task 1.1: Only rated items
    df = df[df['target_reg'].notna()].copy()
    print(f"   Filtered to RATED items: N={len(df)}")
    
    df['global_id'] = domain + "_" + df['source_id'].astype(str)
    
    df_uni['global_id'] = df_uni['media_type'] + "_" + df_uni['source_id'].astype(str)
    
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json", "r") as f:
        registry = json.load(f)

    # 2. Get OOF Unified Priors
    print("   Loading OOF Unified Priors from cache...")
    oof_path = Path("reports/oof_predictions.csv")
    if not oof_path.exists():
        print("❌ Error: reports/oof_predictions.csv missing. Run comprehensive_evaluator.py first.")
        return
        
    oof_df = pd.read_csv(oof_path)
    # Filter for this domain and average repeats
    domain_oof = oof_df[oof_df['media_type'] == domain].copy()
    avg_priors_df = domain_oof.groupby('source_id')[['pred', 'target_reg']].mean().sort_index()
    
    # Task 1.1: Only keep rated items in OOF priors to match local data
    avg_priors_df = avg_priors_df[avg_priors_df['target_reg'] > 0]
    
    if len(avg_priors_df) != len(df):
        print(f"⚠️ Shape mismatch. OOF priors: {len(avg_priors_df)}, Local data: {len(df)}. Aligning indices...")
        
    df.set_index('source_id', inplace=True, drop=False)
    common_idx = df.index.intersection(avg_priors_df.index)
    df = df.loc[common_idx].copy()
    avg_priors_df = avg_priors_df.loc[common_idx].copy()
    print(f"   Aligned to {len(df)} common items.")
        
    # Validation: Check if targets align to ensure we matched the right items
    target_diff = np.abs(avg_priors_df['target_reg'].values - df['target_reg'].values).sum()
    if target_diff > 1e-5:
        print(f"❌ Error: Target alignment failed (diff={target_diff:.4f}). Matching logic is invalid.")
        return
        
    df['unified_prior'] = avg_priors_df['pred'].values
    # Crucial: Use the source_id from the universal dataset for global_id to match the registry
    df['global_id'] = domain + "_" + avg_priors_df.index.astype(str)
    print(f"   ✅ Unified priors and global_ids matched for {len(df)} items.")

    # 2. Local SVR Evaluation
    X_base = df.drop(columns=['target_reg', 'target_class', 'target_ordinal', 'source_id', 'global_id', 'unified_prior'], errors='ignore')
    y = df['target_reg']
    
    def evaluate_local(use_prior=False):
        all_true = []
        all_pred = []
        for fold_id in range(50):
            test_mask = df['global_id'].apply(lambda x: fold_id in registry.get(x, []))
            if not test_mask.any(): continue
            train_mask = ~test_mask
            
            X_fold = X_base.copy()
            if use_prior:
                X_fold['unified_prior'] = df['unified_prior']
                
            X_tr, X_te = X_fold[train_mask], X_fold[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]
            
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
            ])
            model.fit(X_tr, y_tr)
            
            all_true.extend(y_te.tolist())
            all_pred.extend(model.predict(X_te).tolist())
            
        y_true = np.array(all_true)
        y_pred = np.round(np.clip(np.array(all_pred), 0.5, 5.0) * 2) / 2
        mae = mean_absolute_error(y_true, y_pred)
        skill = calculate_skill_score(y_true, y_pred, y.mean())
        return mae, skill, y_pred, y_true

    print("   Evaluating Without Prior...")
    mae_no, skill_no, pred_no, true_no = evaluate_local(use_prior=False)
    print(f"      MAE: {mae_no:.4f} | Skill: {skill_no:.4f}")
    
    print("   Evaluating With Prior...")
    mae_wi, skill_wi, pred_wi, true_wi = evaluate_local(use_prior=True)
    print(f"      MAE: {mae_wi:.4f} | Skill: {skill_wi:.4f}")
    
    # Significance
    err_no = np.abs(true_no - pred_no)
    err_wi = np.abs(true_wi - pred_wi)
    _, p_val = wilcoxon(err_no, err_wi)
    
    verdict = "KEEP" if (mae_wi < mae_no and p_val < 0.05) else "DROP"
    print(f"   Verdict: {verdict} (p={p_val:.4f})")
    
    return {
        "Domain": domain,
        "MAE_NoPrior": mae_no,
        "Skill_NoPrior": skill_no,
        "MAE_WithPrior": mae_wi,
        "Skill_WithPrior": skill_wi,
        "p_value": p_val,
        "Verdict": verdict
    }

if __name__ == "__main__":
    results = []
    results.append(run_distillation_ablation('game'))
    results.append(run_distillation_ablation('book'))
    
    res_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("📊 DISTILLATION PRIOR ABLATION SUMMARY")
    print("="*60)
    print(res_df.to_string(index=False))
    
    # Save results for renderer
    with open("reports/distillation_ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)
