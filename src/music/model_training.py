"""
model_trainer.py — Stage 5 of the Music pipeline.

Trains a personalised model that predicts your star rating for any track, then
exposes an Oracle-style recommender (same idea as the Movies/Books Oracles):

  * predict_rating(track_id)  -> your likely 1-5 rating for a track
  * similar_to(track_id, n)   -> n most similar tracks you own, with why
  * discover(seed_ids, n)     -> NEW tracks via ReccoBeats recommendation API
                                 (fills the gap left by Spotify's removed
                                  /v1/recommendations endpoint)

Model: a Stacking Ensemble (XGBoost + SVR, plus CatBoost when installed) with a
Ridge meta-learner — consistent with the Unified model's stacking approach.

Outputs:
  models/music/model.joblib       trained ensemble
  prints a performance report in the project's standard format

Run:
    python src/music/model_trainer.py
    python src/music/model_trainer.py --similar 3n3Ppam7vgaVa1iaRUc9Lp
    python src/music/model_trainer.py --predict 3n3Ppam7vgaVa1iaRUc9Lp
    python src/music/model_trainer.py --discover 3n3Ppam7vgaVa1iaRUc9Lp,7ouMYWpwJ422jRcDASZB7P
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from xgboost import XGBRegressor

import config

MODEL_PATH = config.MODELS_DIR / "model.joblib"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_features():
    if not config.FEATURES_NPZ.exists():
        raise SystemExit("Run feature_engineering.py first (music_features.npz missing).")
    d = np.load(config.FEATURES_NPZ, allow_pickle=True)
    return d["X"], d["y"], list(d["feature_names"])


def build_estimator():
    base = [
        ("xgb", XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=42, n_jobs=-1)),
        ("svr", SVR(C=3.0, epsilon=0.1, kernel="rbf")),
    ]
    try:
        from catboost import CatBoostRegressor
        base.append(("cat", CatBoostRegressor(
            iterations=400, depth=5, learning_rate=0.05, verbose=0, random_state=42)))
    except ImportError:
        print("(CatBoost not installed — using XGBoost + SVR only.)")
    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0), n_jobs=-1)


# --------------------------------------------------------------------------- #
# Train + report
# --------------------------------------------------------------------------- #
def train():
    X, y, names = load_features()
    print(f"Training on {X.shape[0]} tracks x {X.shape[1]} features\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_estimator()
    model.fit(X_tr, y_tr)

    pred = np.clip(model.predict(X_te), 1.0, 5.0)
    mae = mean_absolute_error(y_te, pred)
    rmse = np.sqrt(mean_squared_error(y_te, pred))
    r2 = r2_score(y_te, pred)
    err = np.abs(pred - y_te)
    exact = np.mean(err <= 1e-6) * 100
    half = np.mean(err <= 0.5) * 100
    one = np.mean((err > 0.5) & (err <= 1.0)) * 100

    print("📊 PERFORMANCE REPORT: 🎵 MUSIC STACKING ENSEMBLE 🎵")
    print("=" * 50)
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")
    print("   " + "-" * 40)
    print(f"   Exact (0.0):  {exact:.1f}%")
    print(f"   ±0.5 Stars:   {half:.1f}%")
    print(f"   ±1.0 Stars:   {one:.1f}%")
    print("=" * 50)

    # Refit on all data before saving (more signal for inference)
    model.fit(X, y)
    joblib.dump({"model": model, "feature_names": names}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


# --------------------------------------------------------------------------- #
# Recommender (Oracle)
# --------------------------------------------------------------------------- #
class Oracle:
    def __init__(self):
        d = np.load(config.FEATURES_NPZ, allow_pickle=True)
        self.X = d["X"]
        self.df = pd.read_csv(config.MASTER_CSV)
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        # L2-normalise rows once so dot product == cosine similarity
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self._Xn = self.X / np.clip(norms, 1e-9, None)
        self._idx = {tid: i for i, tid in enumerate(self.df["track_id"])}

    def predict_rating(self, track_id):
        i = self._idx.get(track_id)
        if i is None:
            raise KeyError(f"{track_id} not in library")
        pred = float(np.clip(self.model.predict(self.X[i:i + 1])[0], 1.0, 5.0))
        row = self.df.iloc[i]
        print(f"\n🎯 {row['name']} — {row['artists']}")
        print(f"   Predicted rating: {pred:.2f} / 5.00   (current: {row['rating']:.2f})")
        return pred

    def similar_to(self, track_id, n=10):
        i = self._idx.get(track_id)
        if i is None:
            raise KeyError(f"{track_id} not in library")
        sims = self._Xn @ self._Xn[i]
        order = np.argsort(-sims)
        order = [j for j in order if j != i][:n]
        seed = self.df.iloc[i]
        print(f"\n🔮 Because you have: {seed['name']} — {seed['artists']}\n")
        for j in order:
            r = self.df.iloc[j]
            shared = self._explain(i, j)
            print(f"  {sims[j]:.3f}  {r['name']} — {r['artists']}"
                  + (f"   [{shared}]" if shared else ""))
        return self.df.iloc[order]

    def _explain(self, i, j):
        """Cheap, readable reason: overlapping genre tags."""
        def tags(idx):
            t = str(self.df.iloc[idx].get("artist_genres", "")) + "," + \
                str(self.df.iloc[idx].get("mb_tags", ""))
            return {x.strip().lower() for x in t.replace(";", ",").split(",") if x.strip()}
        shared = tags(i) & tags(j)
        return ", ".join(sorted(shared)[:3])

    def discover(self, seed_ids, n=15):
        """
        Cold-start discovery of tracks OUTSIDE your library using ReccoBeats'
        recommendation endpoint (Spotify's own /recommendations is deprecated).
        seed_ids: list of Spotify track IDs.
        """
        try:
            r = requests.get(
                f"{config.RECCOBEATS_BASE}/track/recommendation",
                params={"seeds": ",".join(seed_ids), "size": n},
                headers={"Accept": "application/json", "User-Agent": config.USER_AGENT},
                timeout=20)
            r.raise_for_status()
            content = r.json().get("content", [])
        except Exception as e:
            print(f"ReccoBeats discovery unavailable ({e}).")
            return []
        print(f"\n🌱 Discovery from {len(seed_ids)} seeds:")
        for obj in content:
            title = obj.get("trackTitle") or obj.get("name")
            artists = obj.get("artists")
            if isinstance(artists, list):
                artists = ", ".join(a.get("name", "") for a in artists)
            print(f"  • {title} — {artists}")
        return content


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--similar")
    ap.add_argument("--predict")
    ap.add_argument("--discover", help="comma-separated Spotify track IDs")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    if args.similar:
        Oracle().similar_to(args.similar, args.n)
    elif args.predict:
        Oracle().predict_rating(args.predict)
    elif args.discover:
        Oracle().discover(args.discover.split(","), args.n)
    else:
        train()


if __name__ == "__main__":
    main()