# The complete transfer grid (Spearman rank correlation, 50 frozen folds)

Every target against every subset of the other domains. **0%** is zero-shot: the
model is trained on the source domains only and has never seen a single rating
from the target. The remaining columns add that fraction of the target's own
training data on top of the source data.

Reported as Spearman, not MAE-skill: 62% of the movie corpus sits in three
adjacent half-star bins, where a constant 3.5-star predictor already scores MAE
0.759, so error-based skill cannot separate ordering from calibration.

## Target: Movies (107 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.677** (SE 0.008). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `tv+game+book` | +0.490 | +0.631 | +0.653 | **+0.683** |
| `tv` | +0.430 | +0.624 | +0.650 | **+0.680** |
| `tv+book` | +0.420 | +0.637 | +0.656 | +0.677 |
| `tv+game` | +0.373 | +0.621 | +0.649 | +0.674 |
| `game+book` | +0.320 | +0.626 | +0.650 | +0.676 |
| `book` | +0.137 | +0.628 | +0.658 | **+0.685** |
| `game` | +0.091 | +0.617 | +0.648 | **+0.677** |

## Target: TV (18 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.426** (SE 0.029). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.524** | **+0.433** | +0.424 | **+0.465** |
| `movie+game+book` | **+0.471** | **+0.433** | **+0.464** | **+0.493** |
| `movie+book` | **+0.453** | **+0.433** | **+0.470** | **+0.518** |
| `movie+game` | **+0.435** | **+0.468** | **+0.437** | **+0.493** |
| `game+book` | +0.331 | +0.322 | +0.367 | **+0.468** |
| `book` | +0.150 | +0.331 | +0.379 | **+0.483** |
| `game` | +0.149 | +0.246 | +0.316 | +0.419 |

## Target: Games (7 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.307** (SE 0.059). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.599** | **+0.586** | **+0.612** | **+0.602** |
| `movie+book` | **+0.584** | **+0.591** | **+0.601** | **+0.619** |
| `movie+tv` | **+0.541** | **+0.543** | **+0.569** | **+0.538** |
| `movie+tv+book` | **+0.508** | **+0.551** | **+0.535** | **+0.554** |
| `tv+book` | **+0.468** | **+0.474** | **+0.481** | **+0.507** |
| `tv` | **+0.404** | **+0.431** | **+0.487** | **+0.551** |
| `book` | **+0.333** | **+0.426** | **+0.442** | **+0.490** |

## Target: Books (11 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.635** (SE 0.029). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+tv` | +0.600 | +0.626 | **+0.642** | **+0.676** |
| `movie+tv+game` | +0.575 | +0.621 | **+0.658** | **+0.689** |
| `movie` | +0.570 | +0.617 | **+0.649** | **+0.684** |
| `movie+game` | +0.562 | **+0.636** | **+0.653** | **+0.683** |
| `tv` | +0.514 | +0.550 | +0.629 | +0.628 |
| `tv+game` | +0.510 | +0.576 | +0.612 | **+0.655** |
| `game` | +0.223 | +0.481 | +0.578 | **+0.636** |
