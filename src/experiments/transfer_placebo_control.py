"""Label-permutation placebo control for the movie->game transfer headline.

Skeptical null: Games (N=66, var~1.98) is the smallest, highest-variance domain, so it
may benefit from *any* added structure regardless of whether movie taste maps to game
taste. To rule this out we re-run the augmented protocol with the source (movie) RATINGS
SHUFFLED, keeping the source features intact, on the same 50 frozen folds and fractions.

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
import src.experiments.transfer_study as ts
from src.experiments.transfer_study import select_features, _fit_predict, SEED

ts.N_ESTIMATORS = 300
FRACS = (0.0, 0.25, 0.5, 1.0)
TARGET, SOURCE = "game", ["movie"]


def main():
    df = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    df["global_id"] = df["media_type"] + "_" + df["source_id"].astype(str)
    feat = select_features(df, max_genres=40)
    registry = json.load(open(config.UNIFIED_MODEL_DIR / "fold_registry.json"))
    tdf = df[df["media_type"] == TARGET].copy()
    sdf_base = df[df["media_type"].isin(SOURCE)].copy()

    rows = []
    for fold_id in range(50):
        tm = tdf["global_id"].apply(lambda g: fold_id in registry.get(g, []))
        if not tm.any():
            continue
        tr_t, te_t = tdf[~tm], tdf[tm]
        y_te = te_t["target_reg"].values
        pA = _fit_predict(tr_t[feat], tr_t["target_reg"], tr_t["media_type"],
                          te_t[feat], te_t["media_type"])
        maeA = float(np.abs(y_te - pA).mean())
        for frac in FRACS:
            if frac == 0.0:
                t_aug = tr_t.iloc[0:0]
            else:
                rr = np.random.RandomState(SEED + fold_id * 100)
                n = max(1, int(round(frac * len(tr_t))))
                t_aug = tr_t.iloc[rr.choice(len(tr_t), size=n, replace=False)]
            # real source
            comb = pd.concat([sdf_base, t_aug], ignore_index=True)
            pC = _fit_predict(comb[feat], comb["target_reg"], comb["media_type"],
                              te_t[feat], te_t["media_type"])
            mae_real = float(np.abs(y_te - pC).mean())
            # placebo: shuffle source ratings (features intact)
            sdf_sh = sdf_base.copy()
            sh = np.random.RandomState(SEED + fold_id * 7 + int(frac * 10))
            sdf_sh["target_reg"] = sh.permutation(sdf_sh["target_reg"].values)
            comb_sh = pd.concat([sdf_sh, t_aug], ignore_index=True)
            pCs = _fit_predict(comb_sh[feat], comb_sh["target_reg"], comb_sh["media_type"],
                               te_t[feat], te_t["media_type"])
            mae_plac = float(np.abs(y_te - pCs).mean())
            rows.append(dict(fold=fold_id, frac=frac, maeA=maeA,
                             liftReal=maeA - mae_real, liftPlac=maeA - mae_plac))
        print(f"  fold {fold_id} done", flush=True)

    r = pd.DataFrame(rows)
    out = {}
    print("\n=== PLACEBO CONTROL: movie->game (real vs shuffled-source) ===")
    for frac in FRACS:
        s = r[r["frac"] == frac]
        lr, lp = s["liftReal"].mean(), s["liftPlac"].mean()
        try:
            p = float(wilcoxon(s["liftReal"], s["liftPlac"]).pvalue)
        except Exception:
            p = float("nan")
        out[str(frac)] = dict(lift_real=round(float(lr), 4), lift_placebo=round(float(lp), 4),
                              p_real_vs_placebo=round(p, 4), n_folds=int(len(s)))
        print(f"frac {frac}: lift_real {lr:+.4f} | lift_placebo {lp:+.4f} | "
              f"p(real!=placebo) {p:.4f} | n {len(s)}")
    json.dump(out, open(config.BASE_DIR / "reports" / "transfer_placebo_control.json", "w"), indent=1)
    print("\nwrote reports/transfer_placebo_control.json")


if __name__ == "__main__":
    main()
