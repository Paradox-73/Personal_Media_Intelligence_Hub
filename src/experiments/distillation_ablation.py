import pandas as pd
import numpy as np
import joblib
import sys
import json
from pathlib import Path
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import wilcoxon

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.books.model_trainer import calculate_skill_score
from src.reporting.standalone_benchmarks import (DOMAIN_FIT, _round_half, load_registry,
                                               registry_folds, set_calibration)

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
    
    registry = load_registry()

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
    
    # The DEPLOYED local model (C, balanced weights, calibration), not a stand-in SVR.
    set_calibration(domain)
    fit_fn = DOMAIN_FIT[domain]

    def evaluate_local(use_prior=False):
        parts = []
        for fold_id, test_mask in registry_folds(df, registry):
            train_mask = ~test_mask

            X_fold = X_base.copy()
            if use_prior:
                X_fold['unified_prior'] = df['unified_prior']

            X_tr, X_te = X_fold[train_mask], X_fold[test_mask]
            parts.append(pd.DataFrame({'global_id': df.loc[test_mask, 'global_id'].values,
                                       'y': y[test_mask].values,
                                       'pred': np.asarray(fit_fn(X_tr, y[train_mask], X_te))}))

        # One row per item (mean of its 5 repeat predictions), as the benchmark scores it,
        # so the paired test below does not count each item 5 times.
        per_item = pd.concat(parts).groupby('global_id').agg(y=('y', 'first'), pred=('pred', 'mean'))
        y_true = per_item['y'].values
        y_pred = _round_half(per_item['pred'].values)
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
