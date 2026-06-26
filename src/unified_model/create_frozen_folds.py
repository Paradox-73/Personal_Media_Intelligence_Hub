import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
import sys

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

N_SPLITS = 10
N_REPEATS = 5
RANDOM_STATE = 42


def _stratum_labels(df, rating_col, n_splits):
    """Build a stratification label per row = '<media_type>|<rating_bin>'.

    Ratings are binned into low (<=2.5) / mid / high (>4.5) — the project's
    target_class convention. Stratifying on (domain x rating bin) keeps every
    fold's domain mix AND rating curve close to the full dataset.

    Small domains can't support 10-fold on a rare bin (StratifiedKFold needs
    >= n_splits members per class), so within each domain we iteratively merge
    the smallest bin into the largest until every kept bin has >= n_splits.
    Domains all have >> n_splits items, so this always terminates feasibly.
    """
    r = pd.to_numeric(df[rating_col], errors="coerce")
    base = pd.cut(r, bins=[-np.inf, 2.5, 4.5, np.inf], labels=["lo", "mid", "hi"]).astype("object")
    base = base.where(r.notna(), other="na")  # keep unrated rows in the universe

    labels = pd.Series(index=df.index, dtype="object")
    for dom, idx in df.groupby("media_type").groups.items():
        lab = base.loc[idx].copy()
        while True:
            vc = lab.value_counts()
            if len(vc) == 1 or vc.min() >= n_splits:
                break
            smallest = vc.idxmin()
            target = vc.drop(smallest).idxmax()  # merge rare bin into the largest remaining
            lab[lab == smallest] = target
        labels.loc[idx] = dom + "|" + lab.astype(str)
    return labels


def create_frozen_folds():
    print("🧊 Creating Frozen Fold Registry (stratified by domain x rating)...")

    try:
        df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    except Exception as e:
        print(f"❌ Error loading unified training data: {e}")
        return

    if "source_id" not in df.columns or "media_type" not in df.columns:
        print("❌ 'source_id' or 'media_type' not found in training data.")
        return

    rating_col = next((c for c in ["target_reg", "target_ordinal", "my_rating", "user_rating"]
                       if c in df.columns), None)
    if rating_col is None:
        print("❌ No rating/target column found for stratification.")
        return

    # The registry covers the RATED items only (movie/tv/game/book). Music carries
    # PU pseudo-labels, has no frozen folds, and is evaluated separately, so it is
    # excluded here — the registry must contain exactly the rated universe.
    n_before = len(df)
    df = df[df["media_type"] != "music"].reset_index(drop=True)
    print(f"   Rated universe: {len(df)} items (excluded {n_before - len(df)} music rows).")

    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    strata = _stratum_labels(df, rating_col, N_SPLITS)
    print(f"   Stratifying on '{rating_col}' across {strata.nunique()} (domain x rating) strata; "
          f"smallest stratum = {strata.value_counts().min()} items.")

    registry = {}
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    for fold_id, (train_idx, test_idx) in enumerate(rskf.split(df, strata)):
        for idx in test_idx:
            item_id = df.iloc[idx]["global_id"]
            registry.setdefault(item_id, []).append(fold_id)

    registry_path = config.UNIFIED_MODEL_DIR / "fold_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f)

    print(f"✅ Frozen folds saved to {registry_path} for {len(registry)} items "
          f"({N_SPLITS}-fold x {N_REPEATS} repeats, stratified).")


if __name__ == "__main__":
    create_frozen_folds()
