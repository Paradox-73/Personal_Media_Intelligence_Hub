# Unified Model Ablation & Decomposition Report (RATED-ONLY)

| Protocol                                           |      MAE |       R2 | EffectSize        | p            |
|:---------------------------------------------------|---------:|---------:|:------------------|:-------------|
| Base XGB (Aligned)                                 | 0.550475 | 0.422544 | ref               | ref          |
| Mean Ensemble (XGB+Cat)                            | 0.543908 | 0.43836  | d=0.084 (trivial) | 0.0000       |
| Ridge Stack (removed — legacy 5-fold protocol)     | 0.5609   | 0.3837   | n/a (legacy)      | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088   | 0.2688   | n/a (legacy)      | n/a (legacy) |

*Results generated on 2026-06-17 17:25:25*
