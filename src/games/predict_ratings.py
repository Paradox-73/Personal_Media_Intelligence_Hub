import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.games.feature_engineering import transform_single_game, _load_enriched_games
from src.utils.deployment_calibration import calibrate_to_ratings


def library_reference(model, state):
    """Raw model output for every library game, built through the Oracle's own
    `transform_single_game`, plus the calibration reference (raw preds and true
    ratings on the rated rows). The batch CSV and the Oracle page both use this,
    so the dashboard and the Oracle cannot show different numbers for one game.

    Returns (library df, raw preds, ref preds, ref ratings).
    """
    df = _load_enriched_games()
    X = pd.concat([transform_single_game(r, state) for r in df.to_dict('records')],
                  ignore_index=True)
    raw = np.asarray(model.predict(X), dtype=float)
    r = pd.to_numeric(df['my_rating'], errors='coerce').values
    rated = r > 0
    return df, raw, raw[rated], r[rated]


def calibrated_rating(raw, ref_preds, ref_ratings):
    """Restamp raw SVR output onto the user's rating distribution, on the 0.5 grid."""
    return np.round(np.clip(calibrate_to_ratings(raw, ref_preds, ref_ratings), 0.5, 5.0) * 2) / 2


def batch_predict_games():
    print("🚀 Starting Game Batch Prediction...")

    if not config.GAMES_MODEL_REGRESSOR.exists() or not config.GAMES_MODEL_PREPROCESSOR_STATE.exists():
        print("❌ Error: Model or Preprocessor State not found. Run trainer first.")
        return

    model = joblib.load(config.GAMES_MODEL_REGRESSOR)
    state = joblib.load(config.GAMES_MODEL_PREPROCESSOR_STATE)

    df, raw, ref_p, ref_y = library_reference(model, state)
    print(f"   Scored {len(df)} games.")
    df['predicted_rating'] = calibrated_rating(raw, ref_p, ref_y)

    results_path = config.BASE_DIR / "results" / "games" / "game_predictions_detailed.csv"
    df[['name', 'platform_from_text', 'released', 'my_rating', 'predicted_rating', 'metacritic', 'genres']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_games()
