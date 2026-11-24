# The complete transfer grid (Spearman rank correlation, 50 frozen folds)

Every target against every subset of the other domains. **0%** is zero-shot: the
model is trained on the source domains only and has never seen a single rating
from the target. The remaining columns add that fraction of the target's own
training data on top of the source data.

Reported as Spearman, not MAE-skill: 62% of the movie corpus sits in three
adjacent half-star bins, where a constant 3.5-star predictor already scores MAE
0.759, so error-based skill cannot separate ordering from calibration.

## Target: Movies (119 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.669** (SE 0.007). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `tv+game+book` | +0.496 | +0.633 | +0.643 | +0.661 |
| `tv+book` | +0.492 | +0.626 | +0.652 | **+0.672** |
| `tv` | +0.445 | +0.621 | +0.645 | +0.665 |
| `game+book` | +0.323 | +0.622 | +0.651 | +0.664 |
| `tv+game` | +0.311 | +0.616 | +0.647 | +0.666 |
| `book` | +0.189 | +0.631 | +0.652 | +0.668 |
| `game` | +0.088 | +0.622 | +0.642 | +0.669 |

## Target: TV (18 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.417** (SE 0.024). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie` | **+0.448** | +0.347 | +0.363 | +0.394 |
| `movie+book` | **+0.430** | +0.353 | +0.397 | +0.410 |
| `movie+game+book` | +0.380 | +0.334 | +0.374 | +0.369 |
| `movie+game` | +0.338 | +0.343 | +0.370 | +0.399 |
| `game+book` | +0.337 | +0.127 | +0.191 | +0.242 |
| `book` | +0.122 | +0.165 | +0.212 | +0.249 |
| `game` | +0.114 | +0.079 | +0.117 | +0.114 |

## Target: Games (7 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.401** (SE 0.042). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+book` | **+0.553** | **+0.493** | **+0.481** | **+0.452** |
| `movie` | **+0.516** | +0.396 | **+0.421** | **+0.454** |
| `movie+tv+book` | **+0.512** | **+0.481** | **+0.501** | **+0.498** |
| `movie+tv` | **+0.473** | +0.378 | +0.395 | **+0.462** |
| `tv` | **+0.412** | **+0.519** | **+0.528** | **+0.522** |
| `tv+book` | **+0.408** | +0.173 | +0.253 | +0.250 |
| `book` | +0.255 | +0.261 | +0.298 | +0.310 |

## Target: Books (12 held-out items per fold)

A target-only XGB on the same shared features (not the deployed model) scores **0.593** (SE 0.031). Bold = above that point estimate (no significance test).

| Source domains | 0% (zero-shot) | 25% | 50% | 100% |
| :--- | ---: | ---: | ---: | ---: |
| `movie+tv+game` | **+0.599** | +0.474 | +0.479 | +0.523 |
| `movie` | +0.592 | +0.483 | +0.460 | +0.487 |
| `movie+game` | +0.539 | +0.455 | +0.470 | +0.501 |
| `movie+tv` | +0.532 | +0.477 | +0.471 | +0.482 |
| `tv` | +0.452 | +0.483 | +0.530 | +0.585 |
| `tv+game` | +0.387 | +0.502 | +0.555 | +0.583 |
| `game` | +0.226 | +0.393 | +0.552 | **+0.652** |
