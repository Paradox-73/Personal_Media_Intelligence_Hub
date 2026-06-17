"""
Text-normalization control for the Games transfer finding (err.txt round 3, #1).

Check 2 of the wiring audit flagged that the shared vibe space is distorted by
text LENGTH (movie plots ~214 chars vs game descriptions ~1074). The Games
transfer verdict rests on that space. This script tests whether the finding
SURVIVES a length-normalized re-embedding:

  * For every rated item, build a length-matched, identically-templated string:
        "<title>. Genre: <genres>. <first 40 words of description>"
    so a game and a movie become the same shape of text.
  * Re-embed (MiniLM) and refit a shared PCA(10) -> new pca_ columns.
  * Swap ONLY the pca_ (vibe) channel into the unified matrix; keep genre/year/
    popularity/critic unchanged. Re-run the Games-target transfer grid (domain-blind,
    50 folds) for the five source configs that were positive, plus the prior control.

If the lift survives/strengthens -> genuine content transfer (text artifact was not
driving it). If it collapses -> the original verdict was a text-shape artifact.
"""
import sys
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon, spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_utils import DomainAligner
from src.experiments.transfer_study import select_features

SEED = 42
# enriched file, title col, genre col, description col (same mapping as the pipelines)
DOMAINS = {
    "movie": (config.MOVIES_ENRICHED_DATA_PATH, "title", "genre", "plot"),
    "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "name", "genres", "overview"),
    "game": (config.GAMES_ENRICHED_DATA_PATH, "name", "genres", "description_raw"),
    "book": (config.BOOKS_ENRICHED_DATA_PATH, "title", "categories", "description"),
}
WORDS = 40  # length-normalize every description to its first 40 words


def _round(x):
    return np.round(np.clip(x, 0.5, 5.0) * 2) / 2


def first_words(s, n=WORDS):
    return " ".join(str(s).split()[:n])


def build_normalized_pca(df):
    """Return a {source_id: [pca_0..9]} frame from length-normalized templated text."""
    texts, sids = [], []
    for dom, (path, tcol, gcol, ocol) in DOMAINS.items():
        try:
            enr = pd.read_csv(path)
        except Exception:
            enr = pd.read_csv(path, encoding="latin1")
        uni_slice = df[df.media_type == dom].reset_index(drop=True)
        if len(enr) != len(uni_slice):
            raise ValueError(f"{dom}: enriched {len(enr)} != unified slice {len(uni_slice)}")
        enr = enr.reset_index(drop=True)
        def col(c):
            return enr[c].fillna("").astype(str) if c in enr.columns else pd.Series([""] * len(enr))
        # identical template + length-normalized description -> comparable text shape
        t = ("Title: " + col(tcol).str.slice(0, 120)
             + ". Genre: " + col(gcol).str.replace(r"[\[\]']", "", regex=True).str.slice(0, 80)
             + ". " + col(ocol).apply(first_words))
        texts.extend(t.tolist())
        sids.extend(uni_slice["source_id"].tolist())
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
    pca = PCA(n_components=10, random_state=SEED)
    red = pca.fit_transform(emb)
    out = pd.DataFrame(red, columns=[f"pca_{i}" for i in range(10)])
    out["source_id"] = sids
    # report the new text-length parity
    lens = pd.Series([len(t) for t in texts])
    print(f"  normalized text median chars (all domains): {lens.median():.0f} "
          f"(was 214/252/763/1074 by domain)")
    return out


def swap_pca(df, newpca):
    df = df[df.media_type != "music"].copy()
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    df = df.drop(columns=pca_cols).merge(newpca, on="source_id", how="inner")
    return df


def _fit_zero_or_aug(sdf, t_aug, te, feat, pca):
    train = pd.concat([sdf, t_aug], ignore_index=True) if len(t_aug) else sdf
    Xtr, Xte = train[feat].copy(), te[feat].copy()
    al = DomainAligner(method="coral")
    al.fit(Xtr[pca].values, train.media_type.values)
    Xtr.loc[:, pca] = al.transform(Xtr[pca].values, train.media_type.values)
    Xte.loc[:, pca] = al.transform(Xte[pca].values, te.media_type.values)
    m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, objective="reg:absoluteerror", random_state=SEED)
    m.fit(Xtr, train["target_reg"])
    return _round(m.predict(Xte))


def run():
    df0 = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df0["global_id"] = df0["media_type"] + "_" + df0["source_id"].astype(str)
    print("Re-embedding with length-normalized templated text...")
    newpca = build_normalized_pca(df0)
    df = swap_pca(df0, newpca)
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)
    pca = [c for c in feat if c.startswith("pca_")]
    with open(config.UNIFIED_MODEL_DIR / "fold_registry.json") as f:
        registry = json.load(f)

    target = "game"
    tdf = df[df.media_type == target].copy()
    SRcfgs = [["movie", "book"], ["tv"], ["movie", "tv"], ["movie", "tv", "book"], ["movie"]]
    fractions = [0.0, 0.25, 0.5, 1.0]

    # collect per-fold MAE for protocol A and each (source, fraction)
    A_mae, cells = {}, {}
    spear = {}
    for fold in range(50):
        te = tdf[tdf.global_id.apply(lambda g: fold in registry.get(g, []))]
        if te.empty:
            continue
        tr = tdf[~tdf.global_id.isin(te.global_id)]
        y = te["target_reg"].values
        predA = _fit_zero_or_aug(tr, tr.iloc[0:0], te, feat, pca)
        A_mae[fold] = mean_absolute_error(y, predA)
        for src in SRcfgs:
            sdf = df[df.media_type.isin(src)].copy()
            for frac in fractions:
                if frac == 0.0:
                    t_aug = tr.iloc[0:0]
                else:
                    rng = np.random.RandomState(SEED + fold)
                    n = max(1, int(round(frac * len(tr))))
                    t_aug = tr.iloc[rng.choice(len(tr), n, replace=False)]
                pred = _fit_zero_or_aug(sdf, t_aug, te, feat, pca)
                cells.setdefault(("+".join(src), frac), {})[fold] = mean_absolute_error(y, pred)
                if frac == 0.0:
                    spear.setdefault("+".join(src), []).append(
                        spearmanr(y, pred).correlation if len(np.unique(pred)) > 1 else 0.0)

    print(f"\n=== Games-target transfer on LENGTH-NORMALIZED embeddings (50 folds, domain-blind) ===")
    print(f"  {'source':14s} {'lift@100':>9s} {'sig_fracs':>9s} {'zeroshot_spearman':>18s}")
    any_positive = False
    for src in SRcfgs:
        lbl = "+".join(src)
        sig = 0
        lift100 = None
        for frac in fractions:
            cell = cells[(lbl, frac)]
            folds = sorted(set(cell) & set(A_mae))
            a = np.array([A_mae[f] for f in folds]); c = np.array([cell[f] for f in folds])
            lift = a.mean() - c.mean()
            try:
                p = wilcoxon(a, c).pvalue
            except ValueError:
                p = 1.0
            if frac == 1.0:
                lift100 = lift
            if lift > 0 and p < 0.05:
                sig += 1
        positive = sig >= 2
        any_positive |= positive
        sp = np.nanmean(spear[lbl])
        print(f"  {lbl:14s} {lift100:+9.4f} {sig:>9d} {sp:>18.3f}   {'<= POSITIVE' if positive else ''}")

    print(f"\n  VERDICT: {'GAMES TRANSFER SURVIVES text-normalization — genuine content transfer.' if any_positive else 'GAMES TRANSFER COLLAPSES under text-normalization — the original lift was a text-shape artifact.'}")
    return any_positive


if __name__ == "__main__":
    run()
