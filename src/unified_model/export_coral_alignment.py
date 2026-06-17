"""Export a global domain-alignment transform for the Latent Space Explorer.

The training pipeline (advanced_unified_model_trainer / comprehensive_evaluator)
fits a `DomainAligner` *per fold* on the PCA vibe block to avoid leakage when
*scoring*. The Latent Space Explorer, by contrast, is a purely descriptive
visualization — there is no held-out metric to leak into — so a single global
centroid alignment fitted on all rated items is the right artifact to ship.

This writes `models/coral_alignment.joblib` in the dict form the explorer
understands: {'centroids': {media: mean_vec}, 'global_centroid': vec, 'vibe_cols': [...]}.
The explorer then centers every domain's vibe cloud on the shared global centroid
(the same mean-shift `DomainAligner(method='centroid')` performs), so cross-domain
proximity reflects semantic content rather than per-domain offset.

Run:
    python -m src.unified_model.export_coral_alignment
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# Must match the explorer's VIBE_PREFIXES / EXCLUDE rules.
VIBE_PREFIXES = ("vibe_", "pca_", "emb_", "semantic_")
EXCLUDE_EXACT = {"target_reg", "target_ordinal", "rating_date", "source_id",
                 "media_type", "title"}
OUT_PATH = config.MODEL_DIR / "coral_alignment.joblib"


def main():
    path = config.UNIFIED_TRAINING_DATA_PATH
    if not Path(path).exists():
        print(f"❌ Unified training data not found at {path}. "
              f"Run unified_feature_engineering.py first.")
        return

    df = pd.read_csv(path)
    if "media_type" not in df.columns:
        print("❌ media_type column missing from unified training data.")
        return

    vibe_cols = [c for c in df.columns
                 if c.startswith(VIBE_PREFIXES) and c not in EXCLUDE_EXACT]
    if not vibe_cols:
        print(f"❌ No vibe columns matched {VIBE_PREFIXES}.")
        return

    X = np.nan_to_num(df[vibe_cols].to_numpy(dtype=float))
    media = df["media_type"].astype(str)

    global_centroid = X.mean(axis=0)
    centroids = {}
    for m in sorted(media.unique()):
        mask = (media == m).to_numpy()
        if mask.sum() >= 2:  # need a couple of items to estimate a centroid
            centroids[m] = X[mask].mean(axis=0)

    transform = {
        "method": "centroid",
        "vibe_cols": vibe_cols,
        "global_centroid": global_centroid,
        "centroids": centroids,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(transform, OUT_PATH)
    print(f"✅ Exported domain-alignment transform -> {OUT_PATH}")
    print(f"   vibe cols ({len(vibe_cols)}): {vibe_cols}")
    print(f"   domains: {', '.join(f'{m} (n={ (media==m).sum() })' for m in centroids)}")


if __name__ == "__main__":
    main()
