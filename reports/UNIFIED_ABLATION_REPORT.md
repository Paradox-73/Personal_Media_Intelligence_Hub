# Unified Model Ablation & Decomposition Report (RATED-ONLY)

| Protocol                                           |      MAE |       R2 | EffectSize        | p            |
|:---------------------------------------------------|---------:|---------:|:------------------|:-------------|
| Base XGB (Aligned)                                 | 0.536989 | 0.455603 | ref               | ref          |
| Mean Ensemble (XGB+Cat)                            | 0.529425 | 0.466303 | d=0.077 (trivial) | 0.0000       |
| Ridge Stack (removed — legacy 5-fold protocol)     | 0.5609   | 0.3837   | n/a (legacy)      | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088   | 0.2688   | n/a (legacy)      | n/a (legacy) |

*Results generated on 2026-06-26 18:11:52*
