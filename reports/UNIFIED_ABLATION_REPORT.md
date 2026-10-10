# Unified Model Ablation & Decomposition Report (RATED-ONLY)

| Protocol                                           |      MAE |       R2 | EffectSize        | p            |
|:---------------------------------------------------|---------:|---------:|:------------------|:-------------|
| Base XGB (Aligned)                                 | 0.542999 | 0.436445 | ref               | ref          |
| Mean Ensemble (XGB+Cat)                            | 0.533289 | 0.452637 | d=0.131 (trivial) | 0.0000       |
| Ridge Stack (removed — legacy 5-fold protocol)     | 0.5609   | 0.3837   | n/a (legacy)      | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088   | 0.2688   | n/a (legacy)      | n/a (legacy) |

*Results generated on 2026-10-02 21:01:08*
