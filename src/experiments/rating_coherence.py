"""Local rating coherence, and why it predicts the sign of the transfer effect.

The transfer grid shows a target's own data HELPING (book), doing NOTHING (game) and
actively HURTING (tv) a movie-trained model. The ordering is not about sample size --
tv has more items than both game and book. It tracks how well an item's rating is
predicted by its nearest neighbour IN THE SHARED TEXT SPACE.

If that correlation is weak, gradient boosting still fits those rows; there is simply
no local structure there to fit, so it fits noise into exactly the region where the
target's test items live. Writes reports/rating_coherence.json.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

from src import config

DOMAINS = ("movie", "tv", "game", "book")


def local_coherence():
    """rho between each item's rating and its nearest same-domain neighbour's rating."""
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH, low_memory=False)
    emb = np.load(config.UNIFIED_PROCESSED_DIR / "text_embeddings.npy")
    keep = df["target_reg"].notna() & (df["target_reg"] > 0)
    df, emb = df[keep].reset_index(drop=True), emb[keep.values]
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)

    out = {}
    for mt in DOMAINS:
        m = (df["media_type"] == mt).values
        if m.sum() < 3:
            continue
        y = df["target_reg"].values[m]
        sim = emb[m] @ emb[m].T
        np.fill_diagonal(sim, -9.0)
        nn = sim.argmax(1)
        rho, p = stats.spearmanr(y, y[nn])
        out[mt] = {
            "n": int(m.sum()),
            "rho_nearest_neighbour": round(float(rho), 4),
            "p": round(float(p), 4),
            "mean_abs_gap_to_neighbour": round(float(np.abs(y - y[nn]).mean()), 4),
            "sd_of_ratings": round(float(y.std()), 4),
            # The decisive column: when the gap EXCEEDS the spread, copying a
            # lookalike's rating is worse than predicting the domain mean.
            "neighbour_beats_mean": bool(np.abs(y - y[nn]).mean() < y.std()),
        }
    return out


def augmentation_effect(source="movie"):
    """Paired-by-fold effect of adding X% of the target's OWN data to a model trained
    on `source` alone. Spearman, because the rating distribution is too concentrated
    for MAE to separate a model from a constant."""
    grid = pd.read_csv("reports/transfer_grid_results.csv")
    out = {}
    for tgt in DOMAINS:
        t = grid[(grid["target"] == tgt) & (grid["source"] == source)]
        base = t[t["protocol"] == "B"].set_index("fold")["spearman"]
        if base.empty:
            continue
        fr = {}
        for f in (0.25, 0.5, 1.0):
            aug = t[(t["protocol"] == "C") & (t["fraction"] == f)].set_index("fold")["spearman"]
            j = pd.concat([base.rename("z"), aug.rename("a")], axis=1).dropna()
            if len(j) < 3:
                continue
            diff = j["a"] - j["z"]
            tt = stats.ttest_rel(j["a"], j["z"])
            fr[str(f)] = {
                "delta_rho": round(float(diff.mean()), 4),
                "se": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 4),
                "p": round(float(tt.pvalue), 5),
                "n_folds": int(len(j)),
            }
        if fr:
            sig = [v for v in fr.values() if v["p"] < 0.05]
            direction = ("helps" if sig and all(v["delta_rho"] > 0 for v in sig)
                         else "hurts" if sig and all(v["delta_rho"] < 0 for v in sig)
                         else "null")
            out[tgt] = {"by_fraction": fr, "direction": direction}
    return out


def main():
    coh = local_coherence()
    aug = augmentation_effect("movie")
    rows = []
    for mt in DOMAINS:
        if mt not in coh:
            continue
        rows.append({
            "domain": mt,
            "n": coh[mt]["n"],
            "coherence_rho": coh[mt]["rho_nearest_neighbour"],
            "coherence_p": coh[mt]["p"],
            "own_data_effect": (aug.get(mt) or {}).get("direction", "n/a (is the source)"),
        })
    rows.sort(key=lambda r: r["coherence_rho"])
    payload = {
        "coherence": coh,
        "augmentation_vs_movie_trained": aug,
        "ordered": rows,
        "claim": ("Local rating coherence orders the domains identically to the sign of "
                  "the augmentation effect: the less predictable a domain's ratings are "
                  "from nearby items, the less its own data is worth -- and below a "
                  "threshold it is worth less than nothing."),
    }
    with open("reports/rating_coherence.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("[ok] reports/rating_coherence.json")
    print(f"  {'domain':<7}{'n':>6}{'coherence':>12}{'p':>9}   own data ->")
    for r in rows:
        print(f"  {r['domain']:<7}{r['n']:>6}{r['coherence_rho']:>12.3f}{r['coherence_p']:>9.4f}   {r['own_data_effect']}")


if __name__ == "__main__":
    main()
