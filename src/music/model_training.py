"""
model_training.py — Stage 5 of the Music pipeline: the Taste / Similarity Oracle.

Music has NO star ratings and we deliberately do not synthesise one. Instead of a
rating regressor, this exposes an unsupervised taste engine over the engineered
feature matrix:

  * similar_to(track_id, n)   -> n most similar tracks you own (cosine), with why
  * cluster_of(track_id)      -> which taste "persona" (KMeans cluster) a track
                                 belongs to, and the cluster's signature genres
  * discover(seed_ids, n)     -> NEW tracks via ReccoBeats recommendation API
                                 (fills the gap left by Spotify's removed
                                  /v1/recommendations endpoint)

The clustering + PU-affinity taste profile itself is built by
`src/music/profile_builder.py` (saved to models/music/profile.joblib); this module
is the thin query layer the app's Music Oracle page talks to. There is no
model.joblib rating regressor anymore.

Run:
    python -m src.music.model_training                 # library + profile summary
    python -m src.music.model_training --similar <id>
    python -m src.music.model_training --cluster <id>
    python -m src.music.model_training --discover <id1>,<id2>
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
import requests

from src.music import config

PROFILE_PATH = config.MODELS_DIR / "profile.joblib"


@dataclass
class MusicProfile:
    """Container for the taste profile (mirror of profile_builder.MusicProfile).

    Defined here too so the pickled profile.joblib loads regardless of how it was
    built. profile_builder.py is usually run as a script, so its objects pickle
    under ``__main__.MusicProfile``; we register this compatible class into
    ``__main__`` before joblib.load so unpickling resolves it (see Oracle.__init__)."""
    centroids: Any = None
    cluster_labels: Any = None
    cluster_meta: Any = None
    X_lib: Any = None
    pu_model: Any = None
    pool_score_dist: Any = None
    top_genres: Any = None
    audio_fingerprint: Any = None
    feature_names: Any = None
    feature_groups: Any = None


def _load_profile():
    """Load the taste profile, tolerating the __main__/module pickle-path quirk."""
    if not PROFILE_PATH.exists():
        return None
    main_mod = sys.modules.get("__main__")
    injected = main_mod is not None and not hasattr(main_mod, "MusicProfile")
    if injected:
        main_mod.MusicProfile = MusicProfile
    try:
        return joblib.load(PROFILE_PATH)
    except Exception:
        return None
    finally:
        if injected:
            del main_mod.MusicProfile


# --------------------------------------------------------------------------- #
# Taste / Similarity Oracle (unsupervised — no rating model)
# --------------------------------------------------------------------------- #
class Oracle:
    def __init__(self):
        if not config.FEATURES_NPZ.exists():
            raise SystemExit("Run feature_engineering.py first (music_features.npz missing).")
        d = np.load(config.FEATURES_NPZ, allow_pickle=True)
        self.X = d["X"]
        self.feature_names = list(d["feature_names"])
        self.df = pd.read_csv(config.MASTER_CSV)
        # L2-normalise rows once so dot product == cosine similarity
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self._Xn = self.X / np.clip(norms, 1e-9, None)
        self._idx = {tid: i for i, tid in enumerate(self.df["track_id"])}
        # Optional taste profile (clusters/personas) — degrade gracefully if absent.
        self.profile = _load_profile()

    # ---- song-to-song similarity ------------------------------------------
    def similar_to(self, track_id, n=10):
        i = self._idx.get(track_id)
        if i is None:
            raise KeyError(f"{track_id} not in library")
        sims = self._Xn @ self._Xn[i]
        order = [j for j in np.argsort(-sims) if j != i][:n]
        out = self.df.iloc[order].copy()
        out["similarity"] = sims[order]
        out["shared_tags"] = [self._explain(i, j) for j in order]
        seed = self.df.iloc[i]
        print(f"\n🔮 Because you have: {seed['name']} — {seed['artists']}\n")
        for j in order:
            r = self.df.iloc[j]
            shared = self._explain(i, j)
            print(f"  {sims[j]:.3f}  {r['name']} — {r['artists']}"
                  + (f"   [{shared}]" if shared else ""))
        return out

    def _explain(self, i, j):
        """Cheap, readable reason: overlapping genre tags."""
        def tags(idx):
            t = str(self.df.iloc[idx].get("artist_genres", "")) + "," + \
                str(self.df.iloc[idx].get("mb_tags", ""))
            return {x.strip().lower() for x in t.replace(";", ",").split(",") if x.strip()}
        return ", ".join(sorted(tags(i) & tags(j))[:3])

    # ---- taste-cluster ("persona") assignment -----------------------------
    def cluster_of(self, track_id):
        """Assign a track to its nearest taste-profile cluster and describe it.
        Returns a dict {id, label, genres, similarity} or None if no profile."""
        if self.profile is None:
            return None
        i = self._idx.get(track_id)
        if i is None:
            raise KeyError(f"{track_id} not in library")
        from src.music import affinity
        sims = affinity.cosine_similarity(self.X[i:i + 1], self.profile.centroids)[0]
        cid = int(np.argmax(sims))
        meta = self.profile.cluster_meta[cid]
        row = self.df.iloc[i]
        print(f"\n🧬 {row['name']} — {row['artists']}")
        print(f"   Taste persona: {meta['label']}  (cluster {cid}, match {sims[cid]:.2f})")
        print(f"   Signature genres: {', '.join(meta['genres'])}")
        return {"id": cid, "label": meta["label"], "genres": meta["genres"],
                "similarity": float(sims[cid])}

    def taste_personas(self):
        """The full set of taste clusters (personas) from the profile."""
        if self.profile is None:
            return []
        return self.profile.cluster_meta

    # ---- cold-start discovery outside the library -------------------------
    def discover(self, seed_ids, n=15):
        """Discovery of tracks OUTSIDE your library via ReccoBeats' recommendation
        endpoint (Spotify's own /recommendations is deprecated). seed_ids: track IDs."""
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


def _summary():
    """Default action: report the library + taste-profile state (nothing to 'train'
    — the similarity index is the feature matrix and the clusters live in the profile)."""
    o = Oracle()
    print(f"🎵 Music taste engine ready: {len(o.df)} tracks × {o.X.shape[1]} features.")
    if o.profile is not None:
        print(f"   Taste personas ({len(o.profile.cluster_meta)} clusters):")
        for m in o.profile.cluster_meta:
            print(f"     - {m['label']}  ({m['size']} tracks; {', '.join(m['genres'][:3])})")
    else:
        print("   No taste profile yet — run: python -m src.music.profile_builder")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--similar")
    ap.add_argument("--cluster")
    ap.add_argument("--discover", help="comma-separated Spotify track IDs")
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    if args.similar:
        Oracle().similar_to(args.similar, args.n)
    elif args.cluster:
        Oracle().cluster_of(args.cluster)
    elif args.discover:
        Oracle().discover(args.discover.split(","), args.n)
    else:
        _summary()


if __name__ == "__main__":
    main()
