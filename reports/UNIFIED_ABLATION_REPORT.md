# Unified Model Ablation & Decomposition Report (RATED-ONLY)

| Protocol                                           |      MAE |       R2 | EffectSize        | p            |
|:---------------------------------------------------|---------:|---------:|:------------------|:-------------|
| Base XGB (Aligned)                                 | 0.537492 | 0.436275 | ref               | ref          |
| Mean Ensemble (XGB+Cat)                            | 0.524575 | 0.456349 | d=0.076 (trivial) | 0.0026       |
| Ridge Stack (removed — legacy 5-fold protocol)     | 0.5609   | 0.3837   | n/a (legacy)      | n/a (legacy) |
| +Residual Heads (removed — legacy 5-fold protocol) | 0.6088   | 0.2688   | n/a (legacy)      | n/a (legacy) |

*Results generated on 2026-11-18 14:42:55*
