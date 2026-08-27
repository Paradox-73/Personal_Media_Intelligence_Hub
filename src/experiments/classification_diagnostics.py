"""
Classification diagnostics — confusion matrices + precision/recall/F1/accuracy
for every DEPLOYED rating model, derived from the cached registry OOF predictions
(no retraining). This is the classification-framed complement to the regression
benchmarks: the same predictions the README benchmark table scores on MAE/R2 are
here re-scored as a discrete "Dislike / Neutral / Like" decision problem.

Why this exists: MAE tells you how far off a prediction is; a confusion matrix
tells you *which taste decisions the model gets right or wrong* (does it mistake a
5-star favourite for a pass?). Both are reported so the docs answer "how good is
the model" in the two languages reviewers ask it in.

Class convention (the deployed movies `target_class` bins, applied uniformly so
every model is comparable):
    Dislike (0):  rating <= 2.5
    Neutral (1):  2.5 < rating <= 3.5
    Like    (2):  rating > 3.5
A second, recommendation-relevant BINARY view ("Like" = rating >= 3.5) is also
scored, since "would I recommend this?" is the decision the Oracle actually makes.

Sources (deployed models, one row per unique item):
    movie, tv  <- reports/production_oof.csv   (production stacking / simplex ensembles)
    game, book <- reports/standalone_oof.csv   (deployed local SVR)
    unified    <- reports/oof_predictions.csv   (Mean Ensemble; repeats averaged per item)
Music is excluded: it has no ground-truth rating (PU pseudo-labels only), so a
confusion matrix against "truth" would be meaningless.

Outputs: reports/classification_diagnostics.json + reports/confusion_<model>.png
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, accuracy_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
)

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

CLASS_BINS = [-0.001, 2.5, 3.5, 5.0001]
CLASS_LABELS = ["Dislike", "Neutral", "Like"]      # rating <=2.5 / (2.5,3.5] / >3.5
LIKE_THRESHOLD = 3.5                               # binary "would recommend"

MODEL_ORDER = ["movie", "tv", "game", "book", "unified"]
PRETTY = {"movie": "Movies", "tv": "TV Shows", "game": "Games",
          "book": "Books", "unified": "Unified"}
DEPLOYED = {
    "movie": "Prod Stacking (edge+Cat+SVR+Ord)",
    "tv": "Prod Simplex-Stack (edge+Cat+SVR)",
    "game": "Local SVR",
    "book": "Local SVR",
    "unified": "Mean Ensemble (XGB+Cat)",
}


def _round_half(x):
    return np.round(np.clip(np.asarray(x, dtype=float), 0.5, 5.0) * 2) / 2


def _to_class(ratings):
    return pd.cut(pd.Series(np.asarray(ratings, dtype=float)),
                  bins=CLASS_BINS, labels=[0, 1, 2], right=True).astype(int).to_numpy()


def _load_oof_frames():
    prod = pd.read_csv("reports/production_oof.csv")       # movie, tv (deployed ensembles)
    stand = pd.read_csv("reports/standalone_oof.csv")      # game, book (deployed SVR)
    uni = pd.read_csv("reports/oof_predictions.csv")       # unified, 5 repeats/item
    uni = (uni.groupby(["media_type", "source_id"])
              .agg(target_reg=("target_reg", "first"), pred=("pred", "mean"))
              .reset_index())
    return {
        "movie": prod[prod.media_type == "movie"],
        "tv": prod[prod.media_type == "tv"],
        "game": stand[stand.media_type == "game"],
        "book": stand[stand.media_type == "book"],
        "unified": uni,
    }


def _per_class_scores(y_true, y_pred, labels):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    return [{"class": lab, "precision": round(float(p[i]), 4),
             "recall": round(float(r[i]), 4), "f1": round(float(f[i]), 4),
             "support": int(s[i])} for i, lab in enumerate(labels)]


def _avg(y_true, y_pred, avg):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average=avg, zero_division=0)
    return {"precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f), 4)}


def _save_cm_png(cm, model, acc, out_path):
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_LABELS); ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{PRETTY[model]} — confusion matrix\n(3-class, acc={acc*100:.1f}%)")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _ascii_cm(cm):
    head = "            " + "".join(f"{c:>10}" for c in CLASS_LABELS)
    lines = [head, "            " + "  (predicted) ->"]
    for i, lab in enumerate(CLASS_LABELS):
        lines.append(f"  {lab:>8} | " + "".join(f"{cm[i, j]:>10}" for j in range(3)))
    return "\n".join(lines)


def main():
    Path("reports").mkdir(exist_ok=True)
    frames = _load_oof_frames()
    out = {"class_bins": CLASS_BINS, "class_labels": CLASS_LABELS,
           "like_threshold": LIKE_THRESHOLD, "models": {}}

    print("=" * 74)
    print("CLASSIFICATION DIAGNOSTICS — confusion matrix + P/R/F1/acc per model")
    print("Classes: Dislike (<=2.5)  Neutral (2.5-3.5]  Like (>3.5) ;  0.5-rounded preds")
    print("=" * 74)

    for model in MODEL_ORDER:
        d = frames[model]
        yt_r = np.asarray(d.target_reg, dtype=float)
        yp_r = _round_half(d.pred.values)
        yt, yp = _to_class(yt_r), _to_class(yp_r)
        labels = [0, 1, 2]

        cm = confusion_matrix(yt, yp, labels=labels)
        acc = float(accuracy_score(yt, yp))
        bacc = float(balanced_accuracy_score(yt, yp))
        kappa = float(cohen_kappa_score(yt, yp, labels=labels))
        try:
            mcc = float(matthews_corrcoef(yt, yp))
        except Exception:
            mcc = float("nan")

        # binary "Like" (>= 3.5) — the recommend/not decision
        bt = (yt_r >= LIKE_THRESHOLD).astype(int)
        bp = (yp_r >= LIKE_THRESHOLD).astype(int)
        bp_, br_, bf_, _ = precision_recall_fscore_support(
            bt, bp, average="binary", pos_label=1, zero_division=0)
        binary = {
            "accuracy": round(float(accuracy_score(bt, bp)), 4),
            "precision": round(float(bp_), 4),
            "recall": round(float(br_), 4),
            "f1": round(float(bf_), 4),
            "n_liked_true": int(bt.sum()),
        }

        rec = {
            "model": DEPLOYED[model],
            "N": int(len(d)),
            "confusion_matrix": cm.tolist(),
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bacc, 4),
            "cohen_kappa": round(kappa, 4),
            "mcc": round(mcc, 4),
            "macro_avg": _avg(yt, yp, "macro"),
            "weighted_avg": _avg(yt, yp, "weighted"),
            "per_class": _per_class_scores(yt, yp, labels),
            "binary_like": binary,
        }
        out["models"][model] = rec
        _save_cm_png(cm, model, acc, f"reports/confusion_{model}.png")

        # console
        print(f"\n### {PRETTY[model]}  (N={rec['N']}, model={DEPLOYED[model]})")
        print(_ascii_cm(cm))
        print(f"  accuracy={acc*100:5.1f}%   balanced_acc={bacc*100:5.1f}%   "
              f"kappa={kappa:.3f}   MCC={mcc:.3f}")
        print(f"  macro    P={rec['macro_avg']['precision']:.3f} "
              f"R={rec['macro_avg']['recall']:.3f} F1={rec['macro_avg']['f1']:.3f}")
        print(f"  weighted P={rec['weighted_avg']['precision']:.3f} "
              f"R={rec['weighted_avg']['recall']:.3f} F1={rec['weighted_avg']['f1']:.3f}")
        for c in rec["per_class"]:
            print(f"    {CLASS_LABELS[c['class']]:<8} "
                  f"P={c['precision']:.3f} R={c['recall']:.3f} "
                  f"F1={c['f1']:.3f} (n={c['support']})")
        print(f"  binary 'Like' (>=3.5): P={binary['precision']:.3f} "
              f"R={binary['recall']:.3f} F1={binary['f1']:.3f} "
              f"acc={binary['accuracy']*100:.1f}% (n_liked={binary['n_liked_true']})")

    Path("reports/classification_diagnostics.json").write_text(
        json.dumps(out, indent=2))
    print("\nWrote reports/classification_diagnostics.json + reports/confusion_<model>.png")


if __name__ == "__main__":
    main()
