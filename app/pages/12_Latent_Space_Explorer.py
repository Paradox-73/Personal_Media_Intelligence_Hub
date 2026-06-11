"""Latent Space Explorer — corrected projection logic.

Fixes vs previous version:
  1. Default projection uses ONLY the shared semantic (vibe) components, cosine
     metric, with optional CORAL alignment — so cross-domain proximity is
     meaningful rather than an artifact of feature coverage.
  2. Full-feature diagnostic view standardizes features and EXCLUDES media-type
     indicators and has_{domain} missingness masks (which previously guaranteed
     domain separation by construction).
  3. Titles are joined on `source_id` (never positionally).
  4. Continuous features are z-scored before UMAP (kills the year-filament
     pathology caused by unscaled euclidean distances).
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Latent Space Explorer", page_icon="🌌", layout="wide")

# ---------------------------------------------------------------------------
# Configuration — ADJUST THESE to match unified_feature_engineering.py output
# ---------------------------------------------------------------------------
# Column-name prefixes identifying the shared semantic components.
VIBE_PREFIXES = ("vibe_", "pca_", "emb_", "semantic_")
# Columns that encode domain identity / feature coverage. These must NEVER
# enter the projection: they separate domains by construction.
EXCLUDE_PREFIXES = ("has_", "media_type", "mask_", "is_movie", "is_tv",
                    "is_game", "is_book", "is_music")
EXCLUDE_EXACT = {"target_reg", "target_ordinal", "rating_date", "source_id",
                 "media_type", "title"}
# Optional: a saved per-domain alignment transform (e.g. CORAL / centroid
# shift) fitted during training. Applied to vibe columns if present.
ALIGNMENT_TRANSFORM_PATH = getattr(config, "CORAL_TRANSFORM_PATH",
                                   Path("models/coral_alignment.joblib"))


def _is_vibe_col(col: str) -> bool:
    return col.startswith(VIBE_PREFIXES)


def _is_excluded(col: str) -> bool:
    return col in EXCLUDE_EXACT or col.startswith(EXCLUDE_PREFIXES)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_title_lookup() -> pd.DataFrame | None:
    """Build a (source_id, title) lookup from each enriched dataset.

    Joins later happen strictly on source_id — positional alignment between
    independently loaded CSVs is never assumed.
    """
    frames = []
    specs = [
        (getattr(config, "MOVIES_ENRICHED_DATA_PATH", None), "title", "movie"),
        (getattr(config, "TV_SHOWS_ENRICHED_DATA_PATH", None), "name", "tv"),
        (getattr(config, "GAMES_ENRICHED_DATA_PATH", None), "name", "game"),
        (getattr(config, "BOOKS_ENRICHED_DATA_PATH", None), "title", "book"),
    ]
    for path, title_col, media in specs:
        if path is None or not Path(path).exists():
            continue
        try:
            d = pd.read_csv(path)
        except UnicodeDecodeError:
            d = pd.read_csv(path, encoding="latin1")
        if "source_id" not in d.columns or title_col not in d.columns:
            continue
        sub = d[["source_id", title_col]].rename(columns={title_col: "title"})
        sub["media_type"] = media
        frames.append(sub)

    music_path = getattr(config, "MUSIC_ENRICHED_DATA_PATH", None)
    if music_path is not None and Path(music_path).exists():
        d = pd.read_csv(music_path)
        if "source_id" in d.columns and {"name", "artists"} <= set(d.columns):
            sub = d[["source_id"]].copy()
            sub["title"] = d["name"].astype(str) + " — " + d["artists"].astype(str)
            sub["media_type"] = "music"
            frames.append(sub)

    if not frames:
        return None
    lookup = pd.concat(frames, ignore_index=True)
    return lookup.drop_duplicates(subset=["source_id", "media_type"])


@st.cache_data
def load_unified() -> pd.DataFrame | None:
    path = config.UNIFIED_TRAINING_DATA_PATH
    if not Path(path).exists():
        return None
    df = pd.read_csv(path)
    lookup = load_title_lookup()
    if lookup is not None and "source_id" in df.columns:
        df = df.merge(lookup, on=["source_id", "media_type"], how="left")
        df["title"] = df["title"].fillna(df["source_id"].astype(str))
    else:
        df["title"] = df.get("source_id", pd.Series(range(len(df)))).astype(str)
    return df


def apply_alignment(X: np.ndarray, media_types: pd.Series) -> tuple[np.ndarray, bool]:
    """Apply a saved per-domain alignment transform to the vibe block, if one
    exists. Falls back to identity (and reports it) rather than silently
    showing an unaligned space as if it were aligned."""
    if Path(ALIGNMENT_TRANSFORM_PATH).exists():
        transform = joblib.load(ALIGNMENT_TRANSFORM_PATH)
        # Expected interface: transform.apply(X, media_types) -> aligned X.
        # Adapt to however the training pipeline serializes it.
        if hasattr(transform, "apply"):
            return transform.apply(X, media_types), True
        if isinstance(transform, dict) and "centroids" in transform:
            Xa = X.copy()
            global_centroid = transform.get("global_centroid", X.mean(axis=0))
            for media, centroid in transform["centroids"].items():
                m = (media_types == media).values
                if m.any():
                    Xa[m] = Xa[m] - np.asarray(centroid) + np.asarray(global_centroid)
            return Xa, True
    return X, False


@st.cache_data
def compute_projection(view: str, align: bool, n_neighbors: int, min_dist: float):
    df = load_unified()
    if df is None:
        return None, None

    if view == "Shared vibe space (semantic only)":
        cols = [c for c in df.columns if _is_vibe_col(c) and not _is_excluded(c)]
        if not cols:
            return df, "NO_VIBE_COLS"
        X = np.nan_to_num(df[cols].to_numpy(dtype=float))
        aligned = False
        if align:
            X, aligned = apply_alignment(X, df["media_type"])
        df.attrs["alignment_applied"] = aligned
        metric = "cosine"
    else:  # Full feature space (diagnostic)
        cols = [c for c in df.columns
                if not _is_excluded(c) and df[c].dtype.kind in "fiub"]
        X = np.nan_to_num(df[cols].to_numpy(dtype=float))
        X = StandardScaler().fit_transform(X)  # kill scale domination
        df.attrs["alignment_applied"] = False
        metric = "euclidean"

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                        metric=metric, random_state=42)
    emb = reducer.fit_transform(X)
    df["umap_1"], df["umap_2"] = emb[:, 0], emb[:, 1]
    df.attrs["n_features_used"] = len(cols)
    return df, None


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌌 Latent Space Explorer")

view = st.radio(
    "Projection space",
    ["Shared vibe space (semantic only)", "Full feature space (diagnostic)"],
    horizontal=True,
)

if view.startswith("Shared"):
    st.markdown(
        "**Semantic view.** Only the shared vibe components enter this map, "
        "under a cosine metric, with per-domain alignment if available. "
        "Cross-domain proximity here is *meaningful*: a movie next to a song "
        "genuinely shares semantic content. Domain mixing in this view is a "
        "direct visual measurement of cross-domain transferability."
    )
else:
    st.warning(
        "**Diagnostic view.** This projects the model's full (standardized) "
        "input space. Domains will form islands largely because of feature "
        "coverage (missingness masks zero out other domains' blocks). Do NOT "
        "read cross-domain distance here as taste similarity — use it to "
        "inspect what geometry the model actually sees."
    )

c1, c2 = st.columns([4, 1])
with c2:
    st.subheader("Map Controls")
    align = st.checkbox("Apply domain alignment (CORAL)", value=True,
                        disabled=not view.startswith("Shared"))
    n_neighbors = st.slider("UMAP n_neighbors", 5, 60, 15)
    min_dist = st.slider("UMAP min_dist", 0.0, 0.9, 0.1)

df, err = compute_projection(view, align, n_neighbors, min_dist)

if df is None:
    st.error("Unified training data not found. Run feature engineering first.")
    st.code("python src/unified_model/unified_feature_engineering.py")
    st.stop()
if err == "NO_VIBE_COLS":
    st.error(
        f"No columns matched VIBE_PREFIXES={VIBE_PREFIXES}. Open this file and "
        "set VIBE_PREFIXES to the semantic column names produced by "
        "unified_feature_engineering.py."
    )
    st.stop()

with c2:
    color_by = st.selectbox("Color By", ["media_type", "target_reg", "year"])
    available = sorted(df["media_type"].unique())
    media_filter = st.multiselect("Media Types", available, default=available)
    min_rating = st.slider("Min rating / affinity*", 0.0, 5.0, 0.0, 0.5)
    st.caption("*For music this is the PU-calibrated affinity pseudo-label, "
               "not an explicit rating.")
    if view.startswith("Shared"):
        applied = df.attrs.get("alignment_applied", False)
        if align and not applied:
            st.warning("Alignment transform not found at "
                       f"`{ALIGNMENT_TRANSFORM_PATH}` — showing UNALIGNED "
                       "embeddings. Export the fitted CORAL transform from "
                       "the training pipeline to enable it.")
    st.info(f"{df.attrs.get('n_features_used', '?')} features in projection · "
            f"{(df['media_type'].isin(media_filter)).sum()} items shown")

mask = df["media_type"].isin(media_filter) & (df["target_reg"] >= min_rating)
fdf = df[mask]

hover_cols = [c for c in ("title", "media_type", "target_reg", "year")
              if c in fdf.columns]
fig = px.scatter(
    fdf, x="umap_1", y="umap_2", color=color_by,
    hover_name="title", hover_data={c: True for c in hover_cols},
    title=f"Cross-Domain Latent Universe — {view}",
    color_continuous_scale="RdYlGn" if color_by == "target_reg" else "Viridis",
    height=800, template="plotly_dark",
)
fig.update_traces(marker=dict(size=6, opacity=0.65,
                              line=dict(width=0.3, color="white")))
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)
with c1:
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Quantitative companion: domain separability score
# ---------------------------------------------------------------------------
with st.expander("📐 Domain separability (quantifies what the map shows)"):
    st.write(
        "Mean pairwise centroid distance between domains in the projected "
        "space, normalized by mean within-domain spread. **Lower = domains "
        "mix more = stronger shared semantic structure.** Compare this number "
        "between aligned/unaligned and semantic/full views."
    )
    cents = fdf.groupby("media_type")[["umap_1", "umap_2"]].mean()
    spreads = fdf.groupby("media_type")[["umap_1", "umap_2"]].std().mean(axis=1)
    if len(cents) > 1:
        dists = []
        doms = cents.index.tolist()
        for i in range(len(doms)):
            for j in range(i + 1, len(doms)):
                dists.append(np.linalg.norm(cents.iloc[i] - cents.iloc[j]))
        score = float(np.mean(dists) / max(spreads.mean(), 1e-9))
        st.metric("Separability ratio (between / within)", f"{score:.2f}")

if st.checkbox("Show Data Table (Sample)"):
    st.dataframe(fdf[["title", "media_type", "target_reg"]]
                 .sample(min(100, len(fdf))))