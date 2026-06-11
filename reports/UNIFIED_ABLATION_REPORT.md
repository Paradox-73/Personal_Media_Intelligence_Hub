# Unified Model Ablation & Decomposition Report

## 1. Component Attribution
This table decomposes the performance gains from each architectural component using 5-fold cross-validation.

| Protocol | MAE | R² |
| :--- | :--- | :--- |
| 1. Base (XGB, Aligned) | 0.5752 | 0.3611 |
| 2. Mean Ensemble (XGB+Cat) | 0.5578 | 0.4059 |
| 3. Ridge Stacking (XGB+Cat+SVR) | 0.5609 | 0.3837 |
| 4. Full Model (+Residual Heads) | 0.6088 | 0.2688 |

## 2. Statistical Significance (Wilcoxon signed-rank on MAE)
- **MeanStack vs Base:** p=0.0004 ✅ Significant
- **RidgeStack vs Base:** p=0.0703 ❌ Not Significant
- **FullModel vs Base:** p=0.0014 ✅ Significant (but negatively impactful)

## 3. Findings
- ⚠️ **Stacking Pathology confirmed:** The Ridge Stacker (R² 0.3837) underperforms the simple Mean Ensemble (R² 0.4059). The meta-learner overfits the limited OOF pool (N≈1200).
- ⚠️ **Residual Heads Pathology:** Adding domain-specific residual heads dropped R² significantly to 0.2688. With small-N domains like Games (N=62) and Books (N=63), fitting local linear corrections on residuals introduces more noise than signal.
- ✅ **Mean Ensemble is optimal:** Combining XGBoost and CatBoost with equal weights provides the most robust cross-domain performance.

## 4. Tuned Hyperparameters
- **Optimal λ (Temporal Decay):** 0.000429
- **Taste Half-life:** 1,615 days (~4.4 years)
- **Asymmetric Penalties:** alpha_hi=0.178, alpha_lo=0.057 (Penalizing over-predictions more heavily at the top edge).

*Results generated on 2026-06-11 23:32:00*
