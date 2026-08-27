"""
Baseline sanity / "is the model cheating?" diagnostic.

The worry (a sharp one): in domains where ratings cluster in a narrow band, a model
that just predicts *near the mean/median* — never committing to a 5★ or a 1★ — can
post a decent MAE and a high ±0.5★ accuracy for free. So MAE and ±0.5★ accuracy
alone CANNOT prove the model learned taste. This script measures, per deployed
model, whether it actually beats the trivial baselines and — crucially — whether it
ORDERS items correctly (something a mean-hugger mathematically cannot do).

For each model's cached registry OOF (no retraining) it reports:
  * MAE_model vs MAE(predict global mean) vs MAE(predict global median)
  * Skill vs mean / median = 1 - MAE_model/MAE_baseline  (>0 ⇒ beats the constant)
  * ±0.5★ acc of the model vs ±0.5★ acc of predicting the single most common rating
    (the "mode baseline" — how much of the accuracy is FREE in a clustered domain)
  * Spearman ρ (+p): does the prediction *order* track the true order? A constant /
    near-constant predictor has ρ≈0; ρ≫0 with tiny p is the signal a mean-hugger
    cannot fake.
  * Spread ratio predSD/trueSD and % of predictions within ±0.5 of the global mean
    — direct "is it collapsing toward the center?" indicators. Honest by design: it
    exposes the small domains' compression, not just the wins.

Sources (deployed models, one row per item): production_oof.csv (movie/tv),
standalone_oof.csv (game/book), oof_predictions.csv deduped (unified).
Music is excluded (no ground-truth rating).

Output: reports/baseline_sanity.json  (rendered into the report/README).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

MODEL_ORDER = ["movie", "tv", "game", "book", "unified"]
PRETTY = {"movie": "Movies", "tv": "TV Shows", "game": "Games",
          "book": "Books", "unified": "Unified"}


def _rh(x):
    return np.round(np.clip(np.asarray(x, float), 0.5, 5.0) * 2) / 2


def _frames():
    prod = pd.read_csv("reports/production_oof.csv")
    stand = pd.read_csv("reports/standalone_oof.csv")
    uni = pd.read_csv("reports/oof_predictions.csv")
    uni = (uni.groupby(["media_type", "source_id"])
              .agg(target_reg=("target_reg", "first"), pred=("pred", "mean"))
              .reset_index())
    return {"movie": prod[prod.media_type == "movie"],
            "tv": prod[prod.media_type == "tv"],
            "game": stand[stand.media_type == "game"],
            "book": stand[stand.media_type == "book"],
            "unified": uni}


def _acc(yt, yp):
    return float((np.abs(yt - yp) <= 0.5).mean() * 100)


def main():
    out = {"models": {}}
    print("=" * 108)
    print("BASELINE SANITY — is the model beating predict-the-mean, or just hugging it?")
    print("=" * 108)
    hdr = (f"{'model':8} {'N':>4} {'MAE':>5} {'MAEmean':>7} {'MAEmed':>6} "
           f"{'skMean':>6} {'skMed':>6} {'acc':>5} {'accMode':>7} {'accLift':>7} "
           f"{'Spearman':>8} {'p':>8} {'predSD/trueSD':>13} {'%±0.5·mean':>10}")
    print(hdr)
    frames = _frames()
    for m in MODEL_ORDER:
        d = frames[m]
        yt = np.asarray(d.target_reg, float)
        ypr = np.asarray(d.pred, float)
        yp = _rh(ypr)
        mean_b = float(yt.mean())
        med_b = float(np.median(yt))
        mode_b = float(pd.Series(_rh(yt)).mode().iloc[0])

        mae_m = float(mean_absolute_error(yt, yp))
        mae_mean = float(mean_absolute_error(yt, _rh(np.full_like(yt, mean_b))))
        mae_med = float(mean_absolute_error(yt, np.full_like(yt, med_b)))
        sk_mean = 1 - mae_m / mae_mean
        sk_med = 1 - mae_m / mae_med

        acc_m = _acc(yt, yp)
        acc_mode = _acc(yt, np.full_like(yt, mode_b))     # predict the single most common rating
        acc_lift = acc_m - acc_mode

        rho, p = spearmanr(yt, ypr)
        spread = float(yp.std() / yt.std()) if yt.std() else float("nan")
        pct_near_mean = float(np.mean(np.abs(yp - _rh(np.array([mean_b]))[0]) <= 0.5) * 100)

        rec = {"N": int(len(d)), "mean": round(mean_b, 3), "median": round(med_b, 3),
               "mode": round(mode_b, 2),
               "MAE_model": round(mae_m, 4), "MAE_mean": round(mae_mean, 4),
               "MAE_median": round(mae_med, 4),
               "skill_vs_mean": round(float(sk_mean), 4),
               "skill_vs_median": round(float(sk_med), 4),
               "acc_model": round(acc_m, 1), "acc_mode_baseline": round(acc_mode, 1),
               "acc_lift_over_mode": round(acc_lift, 1),
               "spearman": round(float(rho), 4), "spearman_p": float(p),
               "spread_ratio": round(spread, 3),
               "pct_within_half_of_mean": round(pct_near_mean, 1)}
        out["models"][m] = rec
        print(f"{PRETTY[m]:8} {rec['N']:>4} {mae_m:>5.3f} {mae_mean:>7.3f} {mae_med:>6.3f} "
              f"{sk_mean:>6.3f} {sk_med:>6.3f} {acc_m:>4.0f}% {acc_mode:>6.0f}% {acc_lift:>+6.0f}% "
              f"{rho:>8.3f} {p:>8.1e} {spread:>13.2f} {pct_near_mean:>9.0f}%")

    print("\nRead: skill>0 ⇒ beats predict-the-mean; accLift ⇒ ±0.5★ accuracy ABOVE the free "
          "mode-baseline;\n  Spearman ρ≫0 (tiny p) ⇒ predictions ORDER items correctly — a "
          "mean-hugger has ρ≈0 and cannot fake this;\n  spread<1 & high %±0.5·mean ⇒ the model "
          "compresses toward the center (honest flag for the small domains).")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_sanity.json").write_text(json.dumps(out, indent=2))
    print("\nWrote reports/baseline_sanity.json")


if __name__ == "__main__":
    main()
