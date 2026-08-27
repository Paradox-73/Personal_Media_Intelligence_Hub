"""
Can we lift the TEXT-similarity signal for Movies / TV?

text_vs_crowd.py showed movie/TV text-only skill ≈ 0.05 (near noise). Two very
different explanations — and different fixes:
  (a) the embedded text is too THIN (movies embed only Title+Director+Plot — no
      cast/writer/genre/tagline/overview) and too COMPRESSED (15 PCA comps), so a
      real personal signal is being thrown away  → fixable;
  (b) your movie/TV taste ≈ critic consensus, so the personal-beyond-crowd signal
      the text would carry barely exists  → not fixable by any text trick.

This distinguishes them. For Movies and TV we re-embed under:
   CUR  = the current text recipe
   RICH = Title+Genre+Director+Writer+Cast+Tagline+Plot+Overview (adds crew/cast/theme)
and score text-only skill four ways:
   PCA15 + SVR   (current-style compression)
   PCA50 + SVR   (keep more components)
   RAW384 + SVR  (no compression)
   PLS15         (TARGET-SUPERVISED projection — keeps only the embedding directions
                  that correlate with YOUR rating; the acid test for "is any taste
                  signal in the text that PCA is discarding?")
   kNN-cosine    (purest plot-similarity)
Crowd-only skill is printed for reference. Metric = skill vs predict-the-mean (5x5 CV).

Output: reports/text_lift.json
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config


def _s(x):
    return "" if pd.isna(x) else str(x)


def _texts(domain):
    if domain == "movie":
        df = pd.read_csv(config.MOVIES_PROCESSED_DIR / "enriched_data.csv")
        df = df[df["user_rating"].notna()].reset_index(drop=True)
        cur = ("Title: " + df["title"].map(_s) + ". Directed by: " + df["director"].map(_s)
               + ". " + df["plot"].map(_s))
        rich = ("Title: " + df["title"].map(_s) + ". Genre: " + df["genre"].map(_s)
                + ". Directed by: " + df["director"].map(_s) + ". Written by: " + df["writer"].map(_s)
                + ". Starring: " + df["actors"].map(_s) + ". Tagline: " + df["tagline"].map(_s)
                + ". " + df["plot"].map(_s) + " " + df["overview"].map(_s))
        crowd = df[["imdb_rating", "vote_average", "metascore", "rotten_tomatoes_rating"]].apply(
            pd.to_numeric, errors="coerce").fillna(0).values
        y = df["user_rating"].astype(float).values
    else:  # tv
        df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        df = df[df["user_rating"].notna()].reset_index(drop=True)
        cur = ("Title: " + df["name"].map(_s) + ". Network: " + df["network"].map(_s)
               + ". Creator: " + df["created_by"].map(_s) + ". Cast: " + df["actors"].map(_s)
               + ". " + df["overview"].map(_s))
        rich = ("Title: " + df["name"].map(_s) + ". Genre: " + df["genres"].map(_s)
                + ". Network: " + df["network"].map(_s) + ". Creator: " + df["created_by"].map(_s)
                + ". Writer: " + df["writer"].map(_s) + ". Cast: " + df["actors"].map(_s)
                + ". Tagline: " + df["tagline"].map(_s) + ". " + df["overview"].map(_s))
        crowd = df[["vote_average", "imdb_rating"]].apply(
            pd.to_numeric, errors="coerce").fillna(0).values
        y = df["user_rating"].astype(float).values
    return cur.tolist(), rich.tolist(), crowd, y


def _skill(y, pred):
    mae = mean_absolute_error(y, np.clip(pred, 0.5, 5.0))
    base = mean_absolute_error(y, np.full_like(y, y.mean()))
    return 1 - mae / base


def _cv(emb, y, mode):
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    pred = np.zeros(len(y)); cnt = np.zeros(len(y))
    for tr, te in rkf.split(emb):
        Xtr, Xte, ytr = emb[tr], emb[te], y[tr]
        if mode.startswith("pca"):
            k = int(mode[3:]); k = min(k, Xtr.shape[1], len(tr) - 1)
            p = PCA(n_components=k, random_state=0).fit(Xtr)
            Ztr, Zte = p.transform(Xtr), p.transform(Xte)
            m = Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
            m.fit(Ztr, ytr); pred[te] += m.predict(Zte)
        elif mode == "raw":
            m = Pipeline([("s", StandardScaler()), ("m", SVR(kernel="rbf", C=1.0, epsilon=0.1))])
            m.fit(Xtr, ytr); pred[te] += m.predict(Xte)
        elif mode == "pls15":
            k = min(15, Xtr.shape[1])
            m = PLSRegression(n_components=k); m.fit(Xtr, ytr)
            pred[te] += m.predict(Xte).ravel()
        elif mode == "knn":
            k = max(3, min(10, len(tr) // 8))
            m = Pipeline([("s", StandardScaler()),
                          ("m", KNeighborsRegressor(n_neighbors=k, weights="distance", metric="cosine"))])
            m.fit(Xtr, ytr); pred[te] += m.predict(Xte)
        cnt[te] += 1
    pred /= np.maximum(cnt, 1)
    return round(float(_skill(y, pred)), 3)


def main():
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    out = {"models": {}}
    print("=" * 92)
    print("TEXT-LIFT for Movies/TV — text-only skill under richer text + more comps + supervised proj")
    print("=" * 92)
    print(f"{'domain':7} {'text':5} {'PCA15':>6} {'PCA50':>6} {'RAW384':>7} {'PLS15':>6} {'kNN':>6} "
          f"| {'crowd-only':>10}")
    for d in ["movie", "tv"]:
        cur, rich, crowd, y = _texts(d)
        crowd_skill = _cv(crowd.astype(float), y, "raw")
        rec = {}
        for label, txt in [("CUR", cur), ("RICH", rich)]:
            emb = enc.encode(txt, show_progress_bar=False, batch_size=64,
                             normalize_embeddings=True).astype(float)
            row = {m: _cv(emb, y, m) for m in ["pca15", "pca50", "raw", "pls15", "knn"]}
            rec[label] = row
            print(f"{d:7} {label:5} {row['pca15']:>6.3f} {row['pca50']:>6.3f} {row['raw']:>7.3f} "
                  f"{row['pls15']:>6.3f} {row['knn']:>6.3f} | {crowd_skill:>10.3f}")
        rec["crowd_only"] = crowd_skill
        out["models"][d] = rec
    Path("reports/text_lift.json").write_text(json.dumps(out, indent=2))
    print("\nRead: if RICH/PLS15/RAW lift text skill toward crowd-only, text was under-used (fixable).")
    print("      if all stay ~0.05 << crowd, movie/TV taste ≈ consensus — text can't rescue it.")
    print("Wrote reports/text_lift.json")


if __name__ == "__main__":
    main()
