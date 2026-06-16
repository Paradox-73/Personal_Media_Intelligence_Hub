# Pipeline Run Log

Captured terminal outputs from running the full metrics + experiment pipeline on the frozen registry folds. Regenerate with the commands shown under each section (venv: `content_rec/`, prefix `PYTHONUTF8=1`).

Generated: 2026-06-12 (run on branch `movies`).

## 1. Master evaluation — `python -m src.unified_model.comprehensive_evaluator`
Produces `reports/latest_metrics.json` (benchmarks, slices, ablation, distillation, params). Standalone benchmarks + unified registry OOF + full-pool secondary + rated-only 4-row ablation.
```
🚀 Running Master Evaluation Pipeline (Phase 0 corrected)...
📊 Reusing cached unified registry OOF (reports/oof_predictions.csv)
📊 Unified (full pool incl. music pseudo-labels) -- secondary CV...
📊 Standalone per-domain benchmarks (registry folds)...
   [standalone] reusing cached per-item OOF <- reports/standalone_oof.csv
✅ Assertions passed: benchmarks != slices; N == unique items (no 5x inflation).
🧪 Starting Unified Model Ablation Study (RATED-ONLY, Frozen Folds)...
📊 Filtered to RATED-ONLY items: N=1264

🚀 Evaluating Protocol: Base
   Processed 10 folds...
   Processed 20 folds...
   Processed 30 folds...
   Processed 40 folds...
   Processed 50 folds...

🚀 Evaluating Protocol: MeanEnsemble
   Processed 10 folds...
   Processed 20 folds...
   Processed 30 folds...
   Processed 40 folds...
   Processed 50 folds...
   Ensemble vs Base: dMAE=+0.0066  Cohen's d=0.0836  p=7.431e-09  -> significant-but-trivial
✅ Ablation report saved to reports\UNIFIED_ABLATION_REPORT.md
✅ Metrics successfully written to reports\latest_metrics.json
🏁 Master evaluation complete.
```

## 2. Doc renderer — `python -m src.reporting.render_docs`
```
✅ Updated README.md
✅ Updated reports/FINAL_ML_TECHNICAL_REPORT.md
```

## 3. Distillation-prior ablation — `python -m src.experiments.distillation_ablation`
```
##### DISTILLATION ABLATION #####
🧪 Starting Distillation Ablation for GAME...
   Filtered to RATED items: N=55
   Loading OOF Unified Priors from cache...
   Aligned to 55 common items.
   ✅ Unified priors and global_ids matched for 55 items.
   Evaluating Without Prior...
      MAE: 0.6582 | Skill: 0.2253
   Evaluating With Prior...
      MAE: 0.6509 | Skill: 0.2339
   Verdict: DROP (p=0.2059)
🧪 Starting Distillation Ablation for BOOK...
   Filtered to RATED items: N=63
   Loading OOF Unified Priors from cache...
   Aligned to 63 common items.
   ✅ Unified priors and global_ids matched for 63 items.
   Evaluating Without Prior...
      MAE: 0.5476 | Skill: 0.1078
   Evaluating With Prior...
      MAE: 0.5587 | Skill: 0.0897
   Verdict: DROP (p=0.1779)

============================================================
📊 DISTILLATION PRIOR ABLATION SUMMARY
============================================================
Domain  MAE_NoPrior  Skill_NoPrior  MAE_WithPrior  Skill_WithPrior  p_value Verdict
  game     0.658182       0.225292       0.650909         0.233852 0.205903    DROP
  book     0.547619       0.107759       0.558730         0.089655 0.177932    DROP
```

## 4. Transfer grid (pilot) — `python -m src.experiments.transfer_study pilot`
Full grid: `... transfer_study full` (50 folds, 300 trees).
```
PILOT grid (6 folds, 120 trees, top-40 genres, rated-domain sources) -- documented reduction.
Shared-space features: 58 (genre one-hots capped at 40)
  target=movie: cumulative rows=203
  target=tv: cumulative rows=406
  target=game: cumulative rows=609
  target=book: cumulative rows=812
✅ grid -> reports\transfer_grid_results.csv (812 rows)
```

## 5. Transfer analysis / verdict, entity links, bridge eval, active-learning queue
```
########## TRANSFER ANALYSIS ##########
✅ summary -> reports/transfer_grid_summary.json
VERDICT: NULL: in the shared feature space, no domain combination beats the local model (Protocol A). Cross-domain taste signal, if it exists, must be item-level (entity bridges), not feature-level pooling.
########## ENTITY LINKS (cached resolve) ##########
Resolving 584 items to Wikidata QIDs (movies capped at 300)...
✅ 1173 links -> E:\Personal_Media_Intelligence_Hub\data\processed\entity_links.csv
link_type
same_franchise    983
shared_creator    185
adaptation          5

For manual verification (Task 3.1): eyeball the 'adaptation' rows below.
                                  title_a media_a                title_b media_b
                                Star Trek   movie Star Trek: Lower Decks      tv
                        The Suicide Squad   movie             Peacemaker      tv
                   Cyberpunk: Edgerunners      tv         Cyberpunk 2077    game
                        The Cuphead Show!      tv                Cuphead    game
Harry Potter and the Order of the Phoenix   movie   Order of the Phoenix    book
########## BRIDGE EVAL ##########
Linked rated items: 293
{
  "n_linked": 293,
  "mae_without_bridge": 0.5427,
  "mae_with_bridge": 0.5222,
  "mae_delta": 0.0205,
  "delta_ci95": [
    -0.0171,
    0.058
  ],
  "wilcoxon_p": 0.2772,
  "verdict": "no significant bridge effect at this N (pilot)"
}
✅ -> E:\Personal_Media_Intelligence_Hub\reports\entity_bridge_results.json
########## ACTIVE LEARNING RANKER ##########
🎯 Starting Active Learning Ranker (v3: Raw Uncertainty + kNN Novelty)...
   Computing item novelty (kNN distance to training set)...

================================================================================
🔮 ACTIVE LEARNING QUEUE (Continuous Uncertainty + kNN Novelty)
================================================================================
     display_name media_type  Predicted 80% Interval  priority_score
    Expedition 33       game        2.5        ±1.23        1.841739
 Sunset Overdrive       game        3.5        ±1.23        1.358904
Black Myth Wukong       game        2.5        ±1.23        0.012996
         TEKKEN 6       game        3.0        ±1.23       -0.367489
   Cyberpunk 2077       game        3.5        ±1.23       -0.721672
    Hollow Knight       game        3.5        ±1.23       -0.956614
  Alien Isolation       game        3.5        ±1.23       -1.167864
================================================================================
✅ Active learning queue saved to E:\Personal_Media_Intelligence_Hub\reports\ACTIVE_LEARNING_QUEUE.md
```
