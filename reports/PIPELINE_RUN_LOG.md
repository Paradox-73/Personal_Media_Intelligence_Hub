# Pipeline Run Log

Captured terminal outputs from running the full metrics + experiment pipeline on the frozen registry folds. Regenerate with the commands shown under each section (venv: `content_rec/`, prefix `PYTHONUTF8=1`).

Generated: 2026-06-17 (branch `main`, git `af3b9fd`). This was a **from-scratch run**: the OOF caches (`oof_predictions.csv`, `standalone_oof.csv`) were deleted first, so every model below was **refit** on the 50 frozen folds (the master log shows `Unified (rated) OOF on registry folds…` and `[standalone] movie …`, not a cache reuse). Numbers match the prior 2026-06-12 run — the expected result of determinism (fixed splits + `random_state=42`), not of cache reuse.

> **Scope caveat (important).** The `standalone` rows in §1 are the **benchmark estimators** in `standalone_benchmarks.py`, which match the production model for **Games/Books** (SVR, frozen HPs) but are **simplified proxies for Movies** (plain 200-tree XGB) and **TV** (manual simplex). **As of the 2026-06-17b revision below, the published benchmark table reports the production Movies/TV ensembles instead** (`production_benchmarks.py`); the §1 proxy numbers are retained here only as the pre-fix baseline.

---

## 2026-06-17b revision — production benchmarks, uncapped bridges, full grid

This second pass addresses the proxy-model gap and re-runs the item-level experiments at full scope. Commands and captured outputs:

**Production Movies & TV** — `python -m src.reporting.production_benchmarks` (Optuna-tuned edge-penalty alphas persisted to `models/<domain>/best_params.json`, reused across the 50 folds; hand-rolled leakage-safe stack):
```
[movie] tuned edge alphas: alpha_hi=0.0597 alpha_lo=0.0788 -> models/movie/best_params.json
[movie] PRODUCTION  N=980  MAE=0.4776  R2=0.5584  Acc=79.1%
[tv]    tuned edge alphas: alpha_hi=0.1679 alpha_lo=0.0503 -> models/tv/best_params.json
[tv]    PRODUCTION  N=159  MAE=0.8270  R2=0.0522  Acc=53.5%
```
Movies climbs from the proxy's 0.524/0.502 to **0.558/0.478** (the proxy understated it); the honest 50-fold figure is below the ~0.61 a single 80/20 split once suggested. TV is unchanged within noise (0.052/0.827 vs proxy 0.063/0.814) — the heavy ensemble does not rescue a low-signal domain. The master evaluation was re-run to fold these into `latest_metrics.json` (assertions pass; ablation reproduced ΔMAE +0.0066).

**Uncapped entity links** — `python -m src.linking.build_entity_links full`:
```
Resolving 1264 items to Wikidata QIDs...
✅ 2539 links -> data/processed/entity_links.csv
  same_franchise 2085 | shared_creator 441 | adaptation 13   (was 983 / 185 / 5)
```

**Bridge eval (pooled + adaptation-only)** — `python -m src.linking.bridge_features all`. *First run had a domain-identity leak in the base features (see diagnostics below); numbers here are the corrected **domain-blind** re-run:*
```
[all]         N=617  MAE 0.523 -> 0.506  ΔMAE +0.017  CI[-0.004,+0.040]  p=0.182  -> non-significant nudge
[adaptation]  N=22   MAE 0.955 -> 0.955  ΔMAE  0.000  p=NaN              -> degenerate (OOF features constant at this N)
```
(The leaky run reported ΔMAE +0.032 / p=0.004 "significant" — an artifact; removing `has_*_feats` from the shared-space base features drops it to non-significant.)

**Latent-space & transfer wiring diagnostics** — `python -m src.experiments.latent_space_diagnostics` (6 checks + music):
```
Check 1 embeddings populated   PASS   (no silent zeroing; music ‖vec‖≈0.42)
Check 2 text-shape parity      WARN   (median chars movie 214 / tv 252 / book 763 / game 1074)
Check 3 CORAL applied          PASS   (raw separability 1.25 -> centroid-aligned ~0)
Check 4 domain-identity leak   FAIL   (has_*_feats in grid+bridge features) -> FIXED (domain-blind)
Check 5 unified genre vocab    PASS   (gen_Action fires movie/tv/game; 16 genres span >=2 domains)
Check 6 taste consistency      INFO   (rating means 3.24-3.48, spread 0.23 stars)
Check M music lyrics embedded  PASS   (lyric_embed_text non-trivial in 93% of tracks)
```

**Full transfer grid (rated-only, 50 folds, 300 trees, DOMAIN-BLIND after the Check-4 fix)** — `transfer_study.run_grid(include_music=False)` then `python -m src.experiments.transfer_analysis`:
```
[checkpoint] 1800 rows after target=movie ... 3600 (tv) ... 5400 (game) ... 7200 (book)
✅ grid -> reports/transfer_grid_results.csv (7200 rows)
✅ summary -> reports/transfer_grid_summary.json
VERDICT: TASTE TRANSFERS — at least one target shows a positive transfer finding
         (augmented lift > 0, paired p < 0.05, at >= 2 fractions).

positive findings (ALL into target=game):
  movie+book   -> game   lift@100=+0.121  sig_fracs=3/4  (zero-shot +0.175 p=0.0004; 50% p=0.035; 100% p=0.0015)
  movie+tv+book-> game   lift@100=+0.094  sig_fracs=3/4
  tv           -> game   lift@100=+0.102  sig_fracs=3/4
  movie+tv     -> game   lift@100=+0.092  sig_fracs=3/4
  movie        -> game   lift@100=+0.110  sig_fracs=2/4
no positive finding for movie / tv / book targets.
zero-shot affinity: movie->tv +0.20, movie->game +0.16, tv->game +0.10 (books poor).
similarity<->transfer Spearman = -0.18 (n=12): geometry does not predict transfer.
```
**This reverses the 6-fold pilot's NULL** — driven by full statistical power (50 vs 6 folds), and the positive finding *survives* the domain-blind fix (so it is content transfer, not the leak). See report §5c.

**Stress controls on the Games finding** (err.txt round 3 — before promoting the tag):
```
# prior-vs-transfer  (transfer_prior_control.py)  -- is it content, or a better prior than games' N=55 mean?
  movie+book -> game   MAE 1.081   vs constants: game-mean 1.240 / source-mean 1.230 / global-mean 1.230
  Spearman(pred vs true game rating) = +0.368   -> CONTENT TRANSFER (beats every constant AND ranks games)

# text-normalization (text_norm_transfer_control.py) -- survive a length-matched re-embedding (Check 2 fix)?
  re-embedded all domains to ~232 median chars (was 214/252/763/1074)
  movie+book   -> game   lift@100 +0.121  sig 2/4  zeroshot-rho +0.36   <= SURVIVES (POSITIVE)
  tv           -> game   sig 0/4   |  movie+tv 1/4  |  movie+tv+book 1/4  |  movie 1/4   (broad set weakens)
  VERDICT: GAMES TRANSFER SURVIVES text-normalization; robust claim narrows to movie+book.
```
Both controls passed for `movie+book→game`; the broad 5-config result was partly text-shape-inflated. v0.9-rc → **v1.0** promoted after this gate.

## 1. Master evaluation — `python -m src.unified_model.comprehensive_evaluator`
Produces `reports/latest_metrics.json` (benchmarks, slices, ablation, distillation, params). Standalone benchmarks + unified registry OOF + full-pool secondary + rated-only 4-row ablation.
```
🚀 Running Master Evaluation Pipeline (Phase 0 corrected)...
📊 Unified (rated) OOF on registry folds...
      unified registry fold 10/50
      unified registry fold 20/50
      unified registry fold 30/50
      unified registry fold 40/50
      unified registry fold 50/50
📊 Unified (full pool incl. music pseudo-labels) -- secondary CV...
📊 Standalone per-domain benchmarks (registry folds)...
   [standalone] movie ...
      N=980  MAE=0.5015  R2=0.5238  Acc=77.9%
   [standalone] tv ...
      N=159  MAE=0.8145  R2=0.0630  Acc=52.8%
   [standalone] game ...
      (game: 7 unrated/Incomplete rows excluded -- no target)
      N=55  MAE=0.6364  R2=0.3641  Acc=61.8%
   [standalone] book ...
      N=63  MAE=0.5476  R2=0.1822  Acc=76.2%
   [standalone] per-item OOF saved -> reports/standalone_oof.csv
✅ Assertions passed: benchmarks != slices; N == unique items (no 5x inflation).
🧪 Starting Unified Model Ablation Study (RATED-ONLY, Frozen Folds)...
📊 Filtered to RATED-ONLY items: N=1264

🚀 Evaluating Protocol: Base
   Processed 10 folds... (… 50 folds)

🚀 Evaluating Protocol: MeanEnsemble
   Processed 10 folds... (… 50 folds)
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

> ⚠️ **SUPERSEDED.** Sections 4–5 below are the original **2026-06-12 pilot** captures (6-fold, with the domain-identity leak still present). Their **"VERDICT: NULL" and the N=293/1,173-link bridge numbers are obsolete** — see the *2026-06-17b revision* section near the top of this file for the authoritative full-grid result (**verdict: TASTE TRANSFERS into Games**), the uncapped 2,539-link / N=617 bridge eval, and the domain-blind fix. Kept here only as the historical pilot record.

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
