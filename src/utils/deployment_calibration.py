"""Deployment-time rank/quantile calibration.

Mirrors the benchmark calibration (src/reporting/standalone_benchmarks._maybe_calibrate)
for the live Oracle: map a model's raw predictions onto the empirical distribution of the
user's actual ratings, via each prediction's RANK among the model's predictions on the
rated items. Ranking-preserving, so the model's ordering is untouched -- only the values
are restamped onto the real rating curve, letting the deployed app predict the full
0.5-5.0 range (including 5 stars) instead of regressing to the mode.

Applied per domain for Movies / Games / Books. NOT TV: its rank correlation is too weak
(Spearman ~0.45), so calibration would place 5-star predictions on the wrong shows.
"""
import numpy as np

MIN_REF = 20  # need enough rated items for a stable empirical map


def calibrate_to_ratings(raw_preds, ref_preds, ref_ratings):
    """Map ``raw_preds`` onto the ``ref_ratings`` distribution by rank.

    ref_preds / ref_ratings: the model's RAW (continuous, pre-rounded) predictions on the
    rated items and their true user ratings -- the calibration reference. Returns
    continuous calibrated predictions (caller rounds to the 0.5 grid). If too few rated
    items exist, returns the raw predictions unchanged.
    """
    raw = np.asarray(raw_preds, dtype=float)
    rp = np.asarray(ref_preds, dtype=float)
    ry = np.asarray(ref_ratings, dtype=float)
    mask = ~np.isnan(rp) & ~np.isnan(ry)
    rp, ry = rp[mask], ry[mask]
    if len(rp) < MIN_REF:
        return raw
    sp = np.sort(rp)
    q = np.interp(raw, sp, np.linspace(0.0, 1.0, len(sp)))
    return np.quantile(ry, q)
