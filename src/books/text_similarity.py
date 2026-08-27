"""
Books text-similarity model — the deployed Books rating predictor.

Finding (reports/TEXT_VS_CROWD_ANALYSIS.md): for Books your taste lives in the
plot/author/description **text**, not crowd scores. A pure embedding-kNN predictor
— "the mean rating of the books whose descriptions most resemble this one" —
scores **skill 0.43** on repeated 5×5 CV, vs **0.24** for the full SVR that mixes
in crowd features (Goodreads avg / ratings-count). Blending the two *hurts* (≈0.39),
so the honest best model is text-similarity alone.

This trains/saves a KNeighborsRegressor over the book vibe-PCA embedding (cosine,
distance-weighted) and exposes:
  * predict(book_raw)         -> rating from nearest rated books
  * similar(book_raw, n)      -> the neighbour books + why (the explanation)
Deployed artifact: models/books/text_knn.joblib  (bundles the fitted kNN + the
rated-library frame so it needs nothing else at inference).

Run:  python -m src.books.text_similarity          # train + report CV skill
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

KNN_PATH = config.BOOKS_MODEL_DIR / "text_knn.joblib" if hasattr(config, "BOOKS_MODEL_DIR") \
    else Path(config.BOOKS_MODEL_REGRESSOR).parent / "text_knn.joblib"
DEFAULT_K = 8


def _build(k, n_train):
    return Pipeline([("s", StandardScaler()),
                     ("m", KNeighborsRegressor(n_neighbors=min(k, max(1, n_train - 1)),
                                               weights="distance", metric="cosine"))])


def _skill(yt, yp):
    mae = mean_absolute_error(yt, np.clip(yp, 0.5, 5.0))
    base = mean_absolute_error(yt, np.full_like(yt, yt.mean()))
    return 1 - mae / base


def train(k=DEFAULT_K):
    df = pd.read_csv(config.BOOKS_TRAINING_DATA_PATH)
    df = df[df["target_reg"].notna()].reset_index(drop=True)
    pca = [c for c in df.columns if c.startswith("pca_")]
    X, y = df[pca].fillna(0).values, df["target_reg"].astype(float).values

    # honest CV skill of the deployed design
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    pred = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in rkf.split(X):
        m = _build(k, len(tr)); m.fit(X[tr], y[tr]); pred[te] += m.predict(X[te]); cnt[te] += 1
    pred /= np.maximum(cnt, 1)
    sk = _skill(y, pred)
    mae = mean_absolute_error(y, np.clip(np.round(pred * 2) / 2, 0.5, 5.0))
    print(f"📚 Books text-similarity (embedding-kNN, k={k}) — 5×5 CV: "
          f"skill={sk:.3f}  MAE(0.5★)={mae:.3f}  (full-SVR baseline skill≈0.24)")

    # fit on the whole rated library and persist (kNN stores the training rows)
    model = _build(k, len(X)); model.fit(X, y)
    # keep titles/authors for the "because you rated…" explanation (rated enriched rows
    # are positionally aligned with the training matrix, same as standalone_benchmarks).
    enr = pd.read_csv(config.BOOKS_ENRICHED_DATA_PATH)
    enr = enr[enr["my_rating"].notna()].reset_index(drop=True) if "my_rating" in enr.columns else enr.reset_index(drop=True)
    titles = enr["title"].astype(str).values[:len(df)] if "title" in enr.columns else np.array([""] * len(df))
    authors = enr["authors"].astype(str).values[:len(df)] if "authors" in enr.columns else np.array([""] * len(df))

    joblib.dump({"model": model, "pca_cols": pca, "y": y,
                 "titles": titles, "authors": authors,
                 "cv_skill": round(float(sk), 4), "k": k}, KNN_PATH)
    print(f"✅ Saved -> {KNN_PATH}")
    return sk


class BooksTextOracle:
    """Query wrapper: predict a rating and explain via nearest rated books."""

    def __init__(self):
        b = joblib.load(KNN_PATH)
        self.model = b["model"]; self.pca_cols = b["pca_cols"]
        self.y = np.asarray(b["y"]); self.titles = b["titles"]; self.authors = b["authors"]

    def _vec(self, book_raw, state):
        from src.books.feature_engineering import transform_single_book
        X = transform_single_book(book_raw, state)
        return X.reindex(columns=self.pca_cols, fill_value=0.0).fillna(0).values

    def predict(self, book_raw, state):
        v = self._vec(book_raw, state)
        return float(np.clip(self.model.predict(v)[0], 0.5, 5.0))

    def similar(self, book_raw, state, n=5):
        v = self.model.named_steps["s"].transform(self._vec(book_raw, state))
        knn = self.model.named_steps["m"]
        dist, idx = knn.kneighbors(v, n_neighbors=min(n, len(self.y)))
        out = []
        for d, i in zip(dist[0], idx[0]):
            out.append({"title": str(self.titles[i]), "authors": str(self.authors[i]),
                        "your_rating": float(self.y[i]), "similarity": float(1 - d)})
        return out


if __name__ == "__main__":
    train()
