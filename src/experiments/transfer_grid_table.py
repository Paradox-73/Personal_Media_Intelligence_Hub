"""The complete transfer grid, every target x every source subset, in rank correlation.

The summary artifact keeps only the BEST source per target, which hides the shape of the
result: that Movies is the strongest source into all three other domains, that Books is
the weakest lone source everywhere, and that the sign of the augmentation effect flips
between targets. This renders the whole grid.

Spearman, not MAE-skill. The rating distribution is concentrated enough that a constant
3.5-star predictor scores MAE 0.759, so error-based skill barely separates any model from
a constant and conflates ordering with calibration. Rank correlation asks the question the
study is about: does taste in one domain ORDER another.

Writes reports/transfer_grid_full.md and paper/tables/transfer_full_grid.tex.
"""
import numpy as np
import pandas as pd
from pathlib import Path

GRID = Path("reports/transfer_grid_results.csv")
MD_OUT = Path("reports/transfer_grid_full.md")
TEX_OUT = Path("paper/tables/transfer_full_grid.tex")
TARGETS = ("movie", "tv", "game", "book")
TITLE = {"movie": "Movies", "tv": "TV", "game": "Games", "book": "Books"}
FRACTIONS = (0.25, 0.5, 1.0)


def build():
    d = pd.read_csv(GRID)
    out = {}
    for tgt in TARGETS:
        t = d[d["target"] == tgt]
        if t.empty:
            continue
        own = t.loc[t["protocol"] == "A", "spearman"]
        rows = []
        for src, g in t[t["protocol"] == "B"].groupby("source"):
            rec = {"source": src, "zero_shot": g["spearman"].mean(),
                   "zero_shot_se": g["spearman"].std(ddof=1) / np.sqrt(len(g))}
            for f in FRACTIONS:
                c = t[(t["protocol"] == "C") & (t["source"] == src) & (t["fraction"] == f)]
                rec[f] = c["spearman"].mean() if len(c) else np.nan
            rows.append(rec)
        # Strongest zero-shot source first: that ordering is the finding.
        rows.sort(key=lambda r: -r["zero_shot"])
        out[tgt] = {"own": own.mean(), "own_se": own.std(ddof=1) / np.sqrt(len(own)),
                    "n_items": int(t["N"].max()), "rows": rows}
    return out


def to_markdown(grid):
    L = ["# The complete transfer grid (Spearman rank correlation, 50 frozen folds)",
         "",
         "Every target against every subset of the other domains. **0%** is zero-shot: the",
         "model is trained on the source domains only and has never seen a single rating",
         "from the target. The remaining columns add that fraction of the target's own",
         "training data on top of the source data.",
         "",
         "Reported as Spearman, not MAE-skill: 62% of the movie corpus sits in three",
         "adjacent half-star bins, where a constant 3.5-star predictor already scores MAE",
         "0.759, so error-based skill cannot separate ordering from calibration.",
         ""]
    for tgt, blk in grid.items():
        L += [f"## Target: {TITLE[tgt]} ({blk['n_items']} held-out items per fold)", "",
              f"A target-only XGB on the same shared features (not the deployed model) "
              f"scores **{blk['own']:.3f}** (SE {blk['own_se']:.3f}). Bold = above that "
              f"point estimate (no significance test).", "",
              "| Source domains | 0% (zero-shot) | 25% | 50% | 100% |",
              "| :--- | ---: | ---: | ---: | ---: |"]
        for r in blk["rows"]:
            cells = []
            for key in ("zero_shot", 0.25, 0.5, 1.0):
                v = r[key]
                cells.append("-" if pd.isna(v) else
                             (f"**{v:+.3f}**" if v > blk["own"] else f"{v:+.3f}"))
            L.append(f"| `{r['source']}` | " + " | ".join(cells) + " |")
        L.append("")
    return "\n".join(L)


def to_latex(grid):
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Target & Source domains & $0\%$ & $25\%$ & $50\%$ & $100\%$ \\",
         r"\midrule"]
    for tgt, blk in grid.items():
        L.append(rf"\multicolumn{{6}}{{l}}{{\emph{{{TITLE[tgt]}}} --- target-only XGB, same shared "
                 rf"features $\rho={blk['own']:.3f}$}} \\")
        for r in blk["rows"]:
            cells = []
            for key in ("zero_shot", 0.25, 0.5, 1.0):
                v = r[key]
                if pd.isna(v):
                    cells.append("--")
                else:
                    s = f"{v:+.3f}"
                    cells.append(rf"\textbf{{{s}}}" if v > blk["own"] else s)
            src = r["source"].replace("_", r"\_")
            L.append(rf" & \src{{{src}}} & " + " & ".join(f"${c}$" if not c.startswith(r"\text")
                                                          else c for c in cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    return "\n".join(L)


def main():
    grid = build()
    MD_OUT.write_text(to_markdown(grid), encoding="utf-8")
    print(f"[ok] {MD_OUT}")
    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text(to_latex(grid), encoding="utf-8")
    print(f"[ok] {TEX_OUT}")
    for tgt, blk in grid.items():
        best = blk["rows"][0]
        flag = "  <-- beats target-only XGB" if best["zero_shot"] > blk["own"] else ""
        print(f"  {TITLE[tgt]:7} own={blk['own']:.3f}  best zero-shot={best['source']:16} "
              f"{best['zero_shot']:+.3f}{flag}")


if __name__ == "__main__":
    main()
