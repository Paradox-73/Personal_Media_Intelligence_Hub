"""
Prior-vs-transfer control for the Games positive finding (err.txt round 3).

The Games zero-shot lift is measured as *skill vs games' own training mean*. At
N=55 that mean is noisily estimated, so "source helps" could mean either:
  (a) genuine content transfer — movie/book taste informs game ratings, or
  (b) regression to a better prior — the source pool just supplies a better central
      tendency than games' 55 own points, with no taste content crossing over.

Disambiguation (two independent tests):
  1. RANK test — do the zero-shot predictions track which games I rate high vs low?
     A constant prior has Spearman 0; real content transfer has Spearman > 0.
  2. FEATURELESS-BASELINE test — does movie+book->game (with features) beat a
     *constant* predictor (global mean / source mean / games mean) on MAE? If it
     only beats games-own-mean but not a global constant, it's a prior, not transfer.

Run on the same 50 frozen registry folds, predictions rounded to 0.5 (project rule).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_utils import DomainAligner
from src.experiments.transfer_study import select_features

SEED = 42


def _round(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def run(target="game", source=("movie", "book")):
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)  # domain-blind (post-fix)
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)

    tdf = df[df.media_type == target].copy()
    sdf = df[df.media_type.isin(source)].copy()
    global_mean = df[df.media_type != "music"]["target_reg"].mean()
    source_mean = sdf["target_reg"].mean()

    rows = []
    for fold in range(50):
        te = tdf[tdf.global_id.apply(lambda g: fold in registry.get(g, []))]
        if te.empty:
            continue
        tr = tdf[~tdf.global_id.isin(te.global_id)]
        y = te["target_reg"].values
        game_mean = tr["target_reg"].mean()

        # zero-shot transfer model: train on source only, predict target test
        al = DomainAligner(method="coral")
        pca = [c for c in feat if c.startswith("pca_")]
        Xs, Xt = sdf[feat].copy(), te[feat].copy()
        al.fit(Xs[pca].values, sdf.media_type.values)
        Xs.loc[:, pca] = al.transform(Xs[pca].values, sdf.media_type.values)
        Xt.loc[:, pca] = al.transform(Xt[pca].values, te.media_type.values)
        m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="reg:absoluteerror", random_state=SEED)
        m.fit(Xs, sdf["target_reg"])
        pred = _round(m.predict(Xt))

        rows.append({
            "fold": fold, "n": len(y),
            "mae_transfer": mean_absolute_error(y, pred),
            "mae_const_game_mean": mean_absolute_error(y, _round(np.full_like(y, game_mean))),
            "mae_const_source_mean": mean_absolute_error(y, _round(np.full_like(y, source_mean))),
            "mae_const_global_mean": mean_absolute_error(y, _round(np.full_like(y, global_mean))),
            "spearman": spearmanr(y, pred).correlation if len(np.unique(pred)) > 1 else 0.0,
        })
    R = pd.DataFrame(rows)
    sp = R["spearman"].dropna()
    print(f"\n=== Prior-vs-transfer control: {'+'.join(source)} -> {target} (50 folds) ===")
    print(f"  MAE  transfer model (movie+book features) : {R.mae_transfer.mean():.4f}")
    print(f"  MAE  constant = games own mean            : {R.mae_const_game_mean.mean():.4f}")
    print(f"  MAE  constant = source (movie+book) mean  : {R.mae_const_source_mean.mean():.4f}")
    print(f"  MAE  constant = global cross-domain mean  : {R.mae_const_global_mean.mean():.4f}")
    print(f"  Spearman(pred vs true game rating)        : {sp.mean():+.4f}  (0 => flat prior; >0 => tracks content)")
    beats_all_constants = (R.mae_transfer.mean() <
                           min(R.mae_const_source_mean.mean(), R.mae_const_global_mean.mean()))
    print()
    if beats_all_constants and sp.mean() > 0.15:
        print("  VERDICT: CONTENT TRANSFER. The feature model beats every featureless constant "
              "baseline AND its predictions rank-track true game ratings — not a mere prior.")
    elif sp.mean() > 0.15:
        print("  VERDICT: MOSTLY CONTENT. Predictions track game ratings (Spearman>0.15), but the "
              "feature model does not clearly beat a constant on MAE — partial transfer + prior.")
    else:
        print("  VERDICT: PRIOR, NOT TRANSFER. Flat predictions / no MAE gain over a constant — the "
              "'lift' is regression to a better mean, not taste crossing over.")
    return R


if __name__ == "__main__":
    run()
