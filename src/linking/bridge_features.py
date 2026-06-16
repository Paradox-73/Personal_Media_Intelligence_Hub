"""
Phase 3 — Bridge features + paired evaluation (Task 3.2).

"Does knowing I loved the book predict I'll love the show?"

For each library item that participates in a cross-domain entity link, build:
  * linked_item_rating  — my rating of the linked partner, OOF-SAFE (only counted
                          when the partner is in the current training fold).
  * has_link            — 1 if the item has any cross-domain link.
  * franchise_mean_rating — mean of my ratings over same_franchise partners (OOF-safe).
  * shared_creator_te   — K-fold target-encoding of shared creators across domains.

Evaluation is on the LINKED SUBSET only: predict rating with vs without the bridge
features, paired Wilcoxon on absolute errors, with a bootstrap CI on the MAE delta.
N is tiny -> framed explicitly as a pilot (effect size + CI, not a headline metric).
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.experiments.transfer_study import shared_space_columns

LINKS = config.PROCESSED_DIR / "entity_links.csv"
SEED = 42


def build():
    if not LINKS.exists():
        print(f"❌ {LINKS} missing -- run build_entity_links.py first.")
        return None
    links = pd.read_csv(LINKS)
    if links.empty:
        print("No links found; bridge evaluation skipped (report as: no cross-domain links surfaced).")
        return None

    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    uni = uni[uni.media_type != "music"].copy()
    rating = dict(zip(uni.source_id, uni.target_reg))
    feat = shared_space_columns(uni.columns.tolist())

    # Adjacency: source_id -> list of (partner_id, link_type)
    adj = {}
    for _, r in links.iterrows():
        adj.setdefault(r.source_id_a, []).append((r.source_id_b, r.link_type))
        adj.setdefault(r.source_id_b, []).append((r.source_id_a, r.link_type))

    linked_ids = [s for s in adj if s in rating and pd.notna(rating[s])]
    sub = uni[uni.source_id.isin(linked_ids)].reset_index(drop=True)
    print(f"Linked rated items: {len(sub)}")
    if len(sub) < 8:
        print("⚠️ Linked subset too small for a paired test; reporting as descriptive pilot.")

    X_base = sub[feat].copy()
    y = sub["target_reg"].values
    sids = sub["source_id"].values

    kf = KFold(n_splits=min(5, len(sub)), shuffle=True, random_state=SEED)
    err_base, err_bridge = np.zeros(len(sub)), np.zeros(len(sub))

    for tr, te in kf.split(sub):
        train_ids = set(sids[tr])
        train_rating = {s: rating[s] for s in train_ids}
        gmean = np.mean([train_rating[s] for s in train_ids]) if train_ids else 3.0

        def bridge_row(sid):
            partners = adj.get(sid, [])
            # OOF-safe: only partners present in THIS training fold contribute.
            lr = [rating[p] for p, _ in partners if p in train_rating]
            fr = [rating[p] for p, lt in partners if lt == "same_franchise" and p in train_rating]
            return {
                "linked_item_rating": np.mean(lr) if lr else gmean,
                "has_link": 1.0,
                "franchise_mean_rating": np.mean(fr) if fr else gmean,
            }

        bf = pd.DataFrame([bridge_row(s) for s in sids], index=sub.index)
        Xb = pd.concat([X_base, bf], axis=1)

        def fit_pred(Xtr, Xte):
            m = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05, max_depth=4,
                                 random_state=SEED, objective="reg:absoluteerror")
            m.fit(Xtr.iloc[tr], y[tr])
            return np.round(np.clip(m.predict(Xte.iloc[te]), 0.5, 5.0) * 2) / 2

        err_base[te] = np.abs(y[te] - fit_pred(X_base, X_base))
        err_bridge[te] = np.abs(y[te] - fit_pred(Xb, Xb))

    mae_base, mae_bridge = err_base.mean(), err_bridge.mean()
    delta = mae_base - mae_bridge  # positive => bridges help
    try:
        p = float(wilcoxon(err_base, err_bridge).pvalue)
    except ValueError:
        p = 1.0
    # Bootstrap CI on the MAE delta.
    rng = np.random.RandomState(SEED)
    boots = []
    for _ in range(2000):
        idx = rng.randint(0, len(sub), len(sub))
        boots.append(err_base[idx].mean() - err_bridge[idx].mean())
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    result = {
        "n_linked": int(len(sub)),
        "mae_without_bridge": round(float(mae_base), 4),
        "mae_with_bridge": round(float(mae_bridge), 4),
        "mae_delta": round(float(delta), 4),
        "delta_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "wilcoxon_p": round(p, 4),
        "verdict": ("bridges help (significant)" if (delta > 0 and p < 0.05)
                    else "no significant bridge effect at this N (pilot)"),
    }
    out = config.BASE_DIR / "reports" / "entity_bridge_results.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"✅ -> {out}")
    return result


if __name__ == "__main__":
    build()
