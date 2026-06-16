import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Transfer Atlas", page_icon="🧭", layout="wide")

st.header("🧭 Transfer Atlas — does taste cross domains?")
st.markdown("""
The centerpiece experiment. In the **shared feature space** (aligned vibe PCA + unified
genre encoding + critic-average + year + log-popularity + missingness masks), we ask:
**which source-domain combinations transfer into which targets?** — zero-shot and
augmented — measured against each target's own local model on the frozen registry folds.
""")

SUMMARY = Path("reports/transfer_grid_summary.json")
RESULTS = Path("reports/transfer_grid_results.csv")

if not SUMMARY.exists():
    st.error("Run `python -m src.experiments.transfer_study` then "
             "`python -m src.experiments.transfer_analysis` to populate this page.")
    st.stop()

summary = json.loads(SUMMARY.read_text())
DOMAINS = ["movie", "tv", "game", "book"]

# ---- Verdict banner ----
if summary.get("verdict_is_positive"):
    st.success(f"**Verdict:** {summary['verdict']}")
else:
    st.warning(f"**Verdict (pre-registered):** {summary['verdict']}")

# ---- Affinity heatmap (zero-shot single-source skill) ----
st.subheader("Zero-shot affinity — single source → target (skill score)")
aff = summary.get("affinity", {})
mat = np.full((len(DOMAINS), len(DOMAINS)), np.nan)
for ti, t in enumerate(DOMAINS):
    for si, s in enumerate(DOMAINS):
        v = aff.get(t, {}).get(s)
        if v is not None:
            mat[si, ti] = v
fig = go.Figure(data=go.Heatmap(
    z=mat, x=[f"→{d}" for d in DOMAINS], y=[f"{d}" for d in DOMAINS],
    colorscale="RdBu", zmid=0, text=np.round(mat, 3), texttemplate="%{text}",
    colorbar=dict(title="skill")))
fig.update_layout(height=480, xaxis_title="Target", yaxis_title="Source",
                  template="plotly_white",
                  title="Positive = source alone beats the target's mean-rating baseline")
st.plotly_chart(fig, use_container_width=True)

# ---- Best-subset table ----
st.subheader("Best augmented source per target (lift vs local model @ 100% target data)")
rows = []
for t in DOMAINS:
    b = summary.get("best_subset", {}).get(t)
    if not b or b == "none":
        rows.append({"Target": t, "Best source": "none", "Lift@100%": None,
                     "# sig. fractions": 0, "Positive finding?": "—"})
    else:
        rows.append({"Target": t, "Best source": b["source"] + (" +music" if b.get("music") else ""),
                     "Lift@100%": round(b["lift@100"], 4), "# sig. fractions": b["frac_significant"],
                     "Positive finding?": "✅" if b.get("positive") else "❌"})
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption("Positive transfer finding = augmented lift > 0 AND paired Wilcoxon p < 0.05 at ≥ 2 target fractions.")

# ---- Learning curves ----
st.subheader("Learning curves — skill vs target-fraction (best source vs target-alone)")
lc = summary.get("learning_curves", {})
if lc:
    cols = st.columns(2)
    for i, t in enumerate(DOMAINS):
        c = lc.get(t)
        if not c:
            continue
        sk = c["skill_by_fraction"]
        xs = sorted(float(k) for k in sk)
        ys = [sk[str(x)] for x in xs]
        f = go.Figure()
        f.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers",
                               name=f"best src ({c['best_source']})"))
        f.add_hline(y=c["target_alone_skill"], line_dash="dash",
                    annotation_text="target-alone (Protocol A)")
        f.update_layout(title=f"{t}", xaxis_title="fraction of target training data",
                        yaxis_title="skill", height=320, template="plotly_white")
        cols[i % 2].plotly_chart(f, use_container_width=True)

# ---- Similarity vs transfer ----
st.subheader("Similarity ↔ transfer correlation (aligned space)")
sc = summary.get("similarity_corr", {})
if sc:
    c1, c2, c3 = st.columns(3)
    c1.metric("Spearman ρ (centroid dist vs zero-shot skill)",
              sc.get("spearman_centroid_dist_vs_zeroshot_skill"))
    c2.metric("Spearman ρ (MMD vs zero-shot skill)",
              sc.get("spearman_mmd_vs_zeroshot_skill"))
    c3.metric("# domain pairs", sc.get("n_pairs"))
    st.caption("Tests whether domains that sit closer in the aligned space transfer better. "
               "This is the 'relations between domains' deliverable.")

if RESULTS.exists():
    with st.expander("Raw grid results"):
        st.dataframe(pd.read_csv(RESULTS), use_container_width=True, hide_index=True)
