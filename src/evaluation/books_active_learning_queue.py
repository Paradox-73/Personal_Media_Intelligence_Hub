"""
Books active-learning queue — from the Hardcover *watchlist* (Want-to-Read).

The rated-books model can only improve where you rate the *most informative*
unread books. This queue pulls your Hardcover "Want to Read" shelf (unrated), runs
each book through the deployed books feature pipeline + regressor, and ranks them
by **acquisition value**:

  * Novelty   — kNN cosine distance from the book to your rated library in the
                book PCA/feature space. A book unlike anything you've rated teaches
                the model the most, so novelty is the primary acquisition signal
                for a single-regressor model (no ensemble variance to exploit).
  * Predicted rating ± 80% conformal interval (width read from the trained
    books model, not hardcoded) — so you
                can see where the model is guessing and how confidently.

Mirror of reports/MOVIE_ACTIVE_LEARNING_QUEUE.md, for books.
Output: reports/BOOKS_ACTIVE_LEARNING_QUEUE.md

Run:  python -m src.evaluation.books_active_learning_queue
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.books import ingestion as books_ing
from src.books.feature_engineering import transform_single_book

def _book_conformal_width(default=1.01):
    """80% conformal half-width, read from the trained books model.

    This used to be hardcoded at 1.01. The trainer recomputes the width from OOF
    residuals on every run, so the constant went stale the moment the library
    grew (after the 2026-08-14 retrain at N=93 the real width was 0.92, and the
    queue was advertising an interval ~10% too wide). Read it from the artifact
    the trainer writes so the two can never disagree again.
    """
    try:
        meta = joblib.load(config.BOOKS_MODEL_DIR / "model_meta.joblib")
        w = float(meta["conformal_width_80"])
        if w > 0:
            return w
    except Exception:  # noqa: BLE001 - fall back to the documented default
        pass
    return default


BOOK_CONFORMAL_WIDTH = _book_conformal_width()

# Want-to-Read shelf: status_id = 1 in Hardcover, no rating yet.
WATCHLIST_QUERY = """
query($uid: Int!, $limit: Int!, $offset: Int!) {
  user_books(where: {user_id: {_eq: $uid}, rating: {_is_null: true}, status_id: {_eq: 1}},
             order_by: {id: asc}, limit: $limit, offset: $offset) {
    rating
    status_id
    edition { isbn_13 pages release_date edition_format publisher { name } language { language } }
    book {
      title slug release_year pages rating ratings_count description
      image { url } cached_tags contributions { author { name } }
      editions(where: {publisher_id: {_is_null: false}}, order_by: {users_count: desc}, limit: 1) {
        isbn_13 publisher { name }
      }
    }
  }
}
"""


def fetch_watchlist(page_size=100):
    me = books_ing._gql(books_ing.ME_QUERY)
    if not me or not me.get("me"):
        print("❌ Could not resolve Hardcover user (check HARDCOVER_API_KEY).")
        return []
    uid = me["me"][0]["id"]
    rows, offset = [], 0
    while True:
        data = books_ing._gql(WATCHLIST_QUERY, {"uid": uid, "limit": page_size, "offset": offset})
        if not data:
            break
        batch = data.get("user_books", [])
        if not batch:
            break
        rows.extend(books_ing._shape_user_book(ub) for ub in batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def _snap(x):
    return float(np.round(np.clip(x, 0.5, 5.0) * 2) / 2)


def build_books_queue():
    print("📚 Building Books active-learning queue from the Hardcover watchlist...")

    # 1) artifacts
    for p in (config.BOOKS_PREPROCESSOR_STATE, config.BOOKS_MODEL_REGRESSOR,
              config.BOOKS_TRAINING_DATA_PATH):
        if not Path(p).exists():
            print(f"❌ Missing {p}. Run the books pipeline first.")
            return
    state = joblib.load(config.BOOKS_PREPROCESSOR_STATE)
    train = pd.read_csv(config.BOOKS_TRAINING_DATA_PATH)
    pca_cols = [c for c in train.columns if c.startswith("pca_")]
    # Predictions come from the deployed TEXT-SIMILARITY model (embedding-kNN), which
    # scores skill 0.43 for Books vs 0.24 for the crowd-mixing SVR — and produces real,
    # spread predictions instead of everything hugging 3.5. See TEXT_VS_CROWD_ANALYSIS.md.
    from src.books.text_similarity import BooksTextOracle
    text_oracle = BooksTextOracle()

    # 2) watchlist
    watch = fetch_watchlist()
    if not watch:
        print("✅ No unrated Want-to-Read books on the shelf — nothing to queue.")
        return
    print(f"   {len(watch)} watchlist books fetched.")

    # 3) featurize + predict + novelty
    knn = NearestNeighbors(n_neighbors=min(5, len(train)), metric="cosine")
    knn.fit(train[pca_cols].fillna(0).values)

    recs = []
    for raw in watch:
        try:
            X = transform_single_book(raw, state)
            pred_raw = text_oracle.predict(raw, state)
            novelty = float(knn.kneighbors(X[pca_cols].fillna(0).values)[0].mean()) if pca_cols else 0.0
        except Exception as e:
            print(f"   ⚠️ skipped '{raw.get('title')}': {e}")
            continue
        recs.append({
            "Title": raw.get("title") or "(untitled)",
            "Authors": raw.get("authors") or "",
            "Predicted": _snap(pred_raw),
            "80% Interval": f"±{BOOK_CONFORMAL_WIDTH:.2f}",
            "Novelty": round(novelty, 4),
        })

    if not recs:
        print("❌ No watchlist books could be featurized.")
        return

    q = pd.DataFrame(recs).sort_values("Novelty", ascending=False).reset_index(drop=True)
    q.insert(0, "Rank", q.index + 1)
    q["Predicted (±0.5★)"] = q.apply(
        lambda r: f"{r['Predicted']:.1f} {r['80% Interval']}", axis=1)

    # 4) console + markdown
    show = q[["Rank", "Title", "Authors", "Predicted (±0.5★)", "Novelty"]]
    print("\n" + "=" * 80)
    print("🔮 BOOKS ACTIVE-LEARNING QUEUE (rank by novelty; rate the top ones first)")
    print("=" * 80)
    print(show.to_string(index=False))

    out = config.BASE_DIR / "reports" / "BOOKS_ACTIVE_LEARNING_QUEUE.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("# Books — Active Learning: Priority Rating Queue\n\n")
        f.write(f"Your Hardcover **Want-to-Read** shelf ({len(q)} unrated books), ranked by "
                "**novelty** — kNN cosine distance from each book to your rated library in the "
                "book feature space. Rating the most novel books first teaches the model the most "
                "per rating. `Predicted` is the deployed **text-similarity** model (embedding-kNN "
                "over plot/author/description — skill 0.43 vs 0.24 for the old crowd-feature SVR) "
                f"snapped to the 0.5★ grid, with its 80% conformal interval "
                f"(±{BOOK_CONFORMAL_WIDTH:.2f}, from the trained model's OOF residuals).\n\n")
        f.write(show.to_markdown(index=False))
        f.write("\n")
    print("=" * 80)
    print(f"✅ Saved -> {out}")


if __name__ == "__main__":
    build_books_queue()
