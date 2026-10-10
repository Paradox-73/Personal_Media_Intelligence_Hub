import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.common.parsing import clean_year, read_media_csv
from src.unified_model.unified_feature_engineering import transform_single_media
from src.unified_model.unified_oracle import _load, _to_backbone

# (enriched CSV, rating column) per domain. Music is de-rated and not in the model.
SOURCES = {
    "movie": (config.MOVIES_ENRICHED_DATA_PATH, "user_rating"),
    "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "user_rating"),
    "game": (config.GAMES_ENRICHED_DATA_PATH, "my_rating"),
    "book": (config.BOOKS_ENRICHED_DATA_PATH, "my_rating"),
}


def build_library_features(state):
    """Every library row through the Oracle's own path (_to_backbone ->
    transform_single_media). A separate batch re-implementation of that path drifted
    twice and once served 21 of the model's 32 columns as 0."""
    rows, X = [], []
    for mt, (path, rating_col) in SOURCES.items():
        df = read_media_csv(path)
        print(f"   {mt}: {len(df)} rows")
        for raw in df.to_dict("records"):
            meta = _to_backbone(raw, mt)
            X.append(transform_single_media(meta, state, media_type=mt))
            rows.append({"display_name": meta.get("title") or raw.get("letterboxd_name") or "Unknown",
                         "year": clean_year(meta.get("year") or meta.get("released")),
                         "media_type": mt,
                         "user_rating": raw.get(rating_col)})
    return pd.DataFrame(rows), pd.concat(X, ignore_index=True)


def batch_predict_unified_ratings():
    print("🚀 Starting Batch Prediction (Unified Ensemble Mode)...")
    c = _load()
    df, X_final = build_library_features(c["state"])

    # Same three scores the Oracle's predict_unified computes, vectorised.
    p_xgb = c["xgb"].predict(X_final)
    p_cat = c["cat"].predict(X_final)
    preds = {"XGB_Base": p_xgb, "CatBoost_Base": p_cat,
             "MeanEnsemble": (c["mean"].predict(np.column_stack([p_xgb, p_cat]))
                              if c.get("mean") is not None else (p_xgb + p_cat) / 2)}
    for name, raw_preds in preds.items():
        df[f"raw_pred_{name}"] = raw_preds
        df[f"pred_{name}"] = np.round(np.clip(raw_preds, 0.5, 5.0) * 2) / 2

    # Include PCA features for the ranker
    pca_cols = [col for col in X_final.columns if col.startswith("pca_")]
    df = pd.concat([df, X_final[pca_cols]], axis=1)

    df["user_rating_numeric"] = pd.to_numeric(df["user_rating"], errors="coerce")
    eval_df = df.dropna(subset=["user_rating_numeric"])
    if not eval_df.empty:
        y_true = eval_df["user_rating_numeric"]
        print("\n" + "=" * 55 + "\n📊 UNIFIED ML ENSEMBLE PERFORMANCE REPORT (in-sample)\n" + "=" * 55)
        for name in preds:
            y_pred = eval_df[f"pred_{name}"]
            print(f"   Model: {name:<15} | MAE: {mean_absolute_error(y_true, y_pred):.4f} | "
                  f"R²: {r2_score(y_true, y_pred):.4f} | "
                  f"±0.5 Acc: {((y_true - y_pred).abs() <= 0.5).mean() * 100:.1f}%")
        print("=" * 55 + "\n")

    out_dir = config.UNIFIED_PREDICTIONS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = (["display_name", "year", "media_type", "user_rating"]
                + [f"pred_{n}" for n in preds] + [f"raw_pred_{n}" for n in preds] + pca_cols)
    df[out_cols].to_csv(out_dir / "unified_predictions_ensemble.csv", index=False)
    print(f"✅ Saved unified predictions to {out_dir / 'unified_predictions_ensemble.csv'}")


if __name__ == "__main__":
    batch_predict_unified_ratings()
