"""Label-permutation placebo control for the movie->game transfer headline.

Skeptical null: Games (N=66, var~1.98) is the smallest, highest-variance domain, so it
may benefit from *any* added structure regardless of whether movie taste maps to game
taste. To rule this out we re-run the augmented protocol with the source (movie) RATINGS
SHUFFLED, keeping the source features intact, on the same 50 frozen folds and fractions.
Lifts and tests use one averaged out-of-fold prediction per game.

Interpretation:
  - If the shuffled-source lift ~= the real lift  -> variance reduction, NOT taste transfer
    (the headline collapses).
  - If the shuffled-source lift ~= 0 while real > 0 -> genuine taste transfer (headline holds).

Writes reports/transfer_placebo_control.json.
"""
import sys, json, pathlib
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from src import config
from src.reporting.standalone_benchmarks import load_registry, map_folds, registry_folds
import src.experiments.transfer_study as ts
from src.experiments.transfer_study import select_features, _fit_predict, SEED

ts.N_ESTIMATORS = 300
FRACS = (0.0, 0.25, 0.5, 1.0)
TARGET, SOURCE = "game", ["movie"]


def main():
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df = df[df["target_reg"] > 0].copy()  # unrated rows are stored as 0.0, not as ratings
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)
    registry = load_registry()
    tdf = df[df["media_type"] == TARGET].copy()
    sdf_base = df[df["media_type"].isin(SOURCE)].copy()

    def one_fold(fold_id, tm):
        ts.N_ESTIMATORS = 300  # workers import transfer_study afresh
        tr_t, te_t = tdf[~tm], tdf[tm]
        fit = lambda tr: _fit_predict(tr[feat], tr["target_reg"], tr["media_type"],
                                      te_t[feat], te_t["media_type"])
        rec = {"global_id": te_t["global_id"].values, "y": te_t["target_reg"].values}
        frames = [pd.DataFrame({**rec, "frac": -1.0, "arm": "A", "pred": fit(tr_t)})]
        for frac in FRACS:
            if frac == 0.0:
                t_aug = tr_t.iloc[0:0]
            else:
                rr = np.random.RandomState(SEED + fold_id * 100)
                n = max(1, int(round(frac * len(tr_t))))
                t_aug = tr_t.iloc[rr.choice(len(tr_t), size=n, replace=False)]
            # placebo: shuffle source ratings (features intact)
            sdf_sh = sdf_base.copy()
            sh = np.random.RandomState(SEED + fold_id * 7 + int(frac * 10))
            sdf_sh["target_reg"] = sh.permutation(sdf_sh["target_reg"].values)
            for arm, src in (("real", sdf_base), ("placebo", sdf_sh)):
                frames.append(pd.DataFrame({**rec, "frac": frac, "arm": arm,
                                            "pred": fit(pd.concat([src, t_aug], ignore_index=True))}))
        return pd.concat(frames)

    # One row per game: its mean per-fold error across the 5 repeats. The tests pair
    # items, not the 50 overlapping folds.
    r = pd.concat(map_folds(one_fold, list(registry_folds(tdf, registry))))
    err = r.assign(e=(r.y - r.pred).abs()).pivot_table(
        index="global_id", columns=["frac", "arm"], values="e", aggfunc="mean")
    eA = err[(-1.0, "A")]
    out = {}
    print("\n=== PLACEBO CONTROL: movie->game (real vs shuffled-source) ===")
    for frac in FRACS:
        e_real, e_plac = err[(frac, "real")], err[(frac, "placebo")]
        lr, lp = float((eA - e_real).mean()), float((eA - e_plac).mean())
        d = e_plac - e_real
        p = float(wilcoxon(d).pvalue) if d.any() else 1.0
        out[str(frac)] = dict(lift_real=round(lr, 4), lift_placebo=round(lp, 4),
                              p_real_vs_placebo=round(p, 4), n_items=int(len(d)))
        print(f"frac {frac}: lift_real {lr:+.4f} | lift_placebo {lp:+.4f} | "
              f"p(real!=placebo) {p:.4f} | n {len(d)}")
    json.dump(out, open(config.BASE_DIR / "reports" / "transfer_placebo_control.json", "w"), indent=1)
    print("\nwrote reports/transfer_placebo_control.json")


if __name__ == "__main__":
    main()
