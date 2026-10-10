import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.books.feature_engineering import transform_single_book, _load_enriched_books
from src.utils.deployment_calibration import calibrate_to_ratings


def library_reference(model, state):
    """Raw model output for every library book, built through the Oracle's own
    `transform_single_book`, plus the calibration reference (raw preds and true
    ratings on the rated rows). The batch CSV and the Oracle page both use this,
    so the dashboard and the Oracle cannot show different numbers for one book.

    Returns (library df, raw preds, ref preds, ref ratings).
    """
    df = _load_enriched_books()
    X = pd.concat([transform_single_book(r, state) for r in df.to_dict('records')],
                  ignore_index=True)
    raw = np.asarray(model.predict(X), dtype=float)
    r = pd.to_numeric(df['my_rating'], errors='coerce').values
    rated = r > 0
    return df, raw, raw[rated], r[rated]


def calibrated_rating(raw, ref_preds, ref_ratings):
    """Restamp raw SVR output onto the user's rating distribution, on the 0.5 grid."""
    return np.round(np.clip(calibrate_to_ratings(raw, ref_preds, ref_ratings), 0.5, 5.0) * 2) / 2


def batch_predict_books():
    print("🚀 Starting Book Batch Prediction...")

    if not config.BOOKS_MODEL_REGRESSOR.exists() or not config.BOOKS_PREPROCESSOR_STATE.exists():
        print("❌ Error: Model or Preprocessor State not found. Run trainer first.")
        return

    model = joblib.load(config.BOOKS_MODEL_REGRESSOR)
    state = joblib.load(config.BOOKS_PREPROCESSOR_STATE)

    df, raw, ref_p, ref_y = library_reference(model, state)
    print(f"   Scored {len(df)} books.")
    df['predicted_rating'] = calibrated_rating(raw, ref_p, ref_y)

    results_path = config.BASE_DIR / "results" / "books" / "book_predictions_detailed.csv"
    df[['title', 'authors', 'publishedDate', 'my_rating', 'predicted_rating', 'averageRating', 'categories']].to_csv(results_path, index=False)
    print(f"✅ Predictions saved to {results_path}")

if __name__ == "__main__":
    batch_predict_books()
