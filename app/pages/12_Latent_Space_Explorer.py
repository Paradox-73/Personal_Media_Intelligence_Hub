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
import plotly.graph_objects as go
import streamlit as st
import umap
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="Latent Space Explorer", page_icon="🌌", layout="wide")
apply_mobile_css()

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
# Per-domain enriched CSV, its title column, and whether that domain drops
# unrated rows — this MUST mirror build_universal_dataset() in
# unified_feature_engineering.py, because `source_id` there is simply the row
# position in the movie→tv→game→book→music concatenation (df.index), and the
# enriched CSVs carry no source_id column of their own. Movies/TV drop rows with
# a null user_rating; games/books/music keep every row.
DOMAIN_TITLE_SPECS = [
    ("movie", getattr(config, "MOVIES_ENRICHED_DATA_PATH", None), "title", True),
    ("tv", getattr(config, "TV_SHOWS_ENRICHED_DATA_PATH", None), "name", True),
    ("game", getattr(config, "GAMES_ENRICHED_DATA_PATH", None), "name", False),
    ("book", getattr(config, "BOOKS_ENRICHED_DATA_PATH", None), "title", False),
]


@st.cache_data
def load_domain_titles() -> dict[str, list[str]]:
    """{media_type: [titles in build order]} reconstructing the exact row set/order
    unified_feature_engineering used, so the k-th unified item of a domain (in
    source_id order) is the k-th title here. Robust to later CSV growth because
    new items are appended after the frozen rows."""
    out: dict[str, list[str]] = {}
    for media, path, col, drop_unrated in DOMAIN_TITLE_SPECS:
        if path is None or not Path(path).exists():
            continue
        try:
            d = pd.read_csv(path)
        except UnicodeDecodeError:
            d = pd.read_csv(path, encoding="latin1")
        if drop_unrated and "user_rating" in d.columns:
            d = d.dropna(subset=["user_rating"])
        if col in d.columns:
            out[media] = d[col].astype(str).tolist()

    music_path = getattr(config, "MUSIC_ENRICHED_DATA_PATH", None)
    if music_path is not None and Path(music_path).exists():
        d = pd.read_csv(music_path)
        if {"name", "artists"} <= set(d.columns):
            out["music"] = (d["name"].astype(str) + " — " + d["artists"].astype(str)).tolist()
    return out


@st.cache_data
def load_music_points() -> pd.DataFrame | None:
    """Project the music library into the SAME 10-D vibe basis as the other
    domains, using the LYRICS as each track's description. Embeds lyrics with the
    unified pipeline's saved MiniLM model, transforms with its fitted PCA, and
    centres music on the shared global centroid so it lands in the same space —
    all read-only, without regenerating the frozen training features."""
    mpath = getattr(config, "MUSIC_ENRICHED_DATA_PATH", None)
    if mpath is None or not Path(mpath).exists() or not Path(config.UNIFIED_PREPROCESSOR_STATE).exists():
        return None
    state = joblib.load(config.UNIFIED_PREPROCESSOR_STATE)
    pca = state.get("pca")
    st_name = state.get("sentence_transformer", "all-MiniLM-L6-v2")
    if pca is None:
        return None

    d = pd.read_csv(mpath)
    if "lyric_embed_text" not in d.columns:
        return None
    d = d[d["lyric_embed_text"].astype(str).str.len() > 3].copy()
    if d.empty:
        return None

    # Same text recipe as the frozen space (Title + Genre + description), with the
    # LYRICS standing in as the description.
    gen = (d.get("artist_genres", pd.Series([""] * len(d), index=d.index)).fillna("")
           .astype(str).str.replace(r"[\[\]']", "", regex=True).str.slice(0, 80))
    desc = d["lyric_embed_text"].fillna("").astype(str).apply(lambda s: " ".join(s.split()[:40]))
    txt = ("Title: " + d["name"].fillna("").astype(str).str.slice(0, 120)
           + ". Genre: " + gen + ". " + desc)

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(st_name).encode(txt.tolist(), normalize_embeddings=True,
                                               show_progress_bar=False)
    vibe = pca.transform(emb)  # (n, 10) in the shared basis

    # Centre music on the shared global centroid (CORAL-style) so cross-domain
    # proximity is meaningful rather than a per-domain offset.
    if Path(ALIGNMENT_TRANSFORM_PATH).exists():
        tf = joblib.load(ALIGNMENT_TRANSFORM_PATH)
        gc = np.asarray(tf.get("global_centroid", vibe.mean(axis=0))) if isinstance(tf, dict) else vibe.mean(axis=0)
    else:
        gc = vibe.mean(axis=0)
    vibe = vibe - vibe.mean(axis=0) + gc

    out = pd.DataFrame(vibe, columns=[f"pca_{i}" for i in range(vibe.shape[1])])
    out["media_type"] = "music"
    out["title"] = d["name"].astype(str).values + " — " + d["artists"].astype(str).values
    out["target_reg"] = np.nan  # music is de-rated: no explicit rating
    yr = pd.to_numeric(d.get("release_year", d.get("year")), errors="coerce")
    out["year"] = yr.values if yr is not None else np.nan
    return out


@st.cache_data
def load_unified(include_music: bool = False) -> pd.DataFrame | None:
    path = config.UNIFIED_TRAINING_DATA_PATH
    if not Path(path).exists():
        return None
    df = pd.read_csv(path).reset_index(drop=True)
    # Fallback label = raw source_id; overwritten below where a name resolves.
    df["title"] = df.get("source_id", pd.Series(range(len(df)))).astype(str)

    if {"media_type", "source_id"} <= set(df.columns):
        titles_by_domain = load_domain_titles()
        unmatched = 0
        for media, titles in titles_by_domain.items():
            # Rows of this domain, in source_id (= build) order.
            order = df.index[df["media_type"] == media]
            order = df.loc[order].sort_values("source_id").index
            for pos, ridx in enumerate(order):
                if pos < len(titles):
                    df.at[ridx, "title"] = titles[pos]
                else:
                    unmatched += 1
        df.attrs["unmatched_titles"] = unmatched

    if include_music:
        music = load_music_points()
        if music is not None and len(music):
            df = pd.concat([df, music], ignore_index=True)
            df.attrs["n_music"] = len(music)
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
def compute_projection(view: str, align: bool, n_neighbors: int, min_dist: float,
                       include_music: bool = False):
    """Returns (df_with_umap, error, hidim) where hidim = {'X', 'metric'} is the
    HIGH-DIMENSIONAL matrix actually fed to UMAP. Similarity should be read off
    hidim['X'] (faithful), not the 2-D umap coords (UMAP distorts global distance)."""
    df = load_unified(include_music=include_music)
    if df is None:
        return None, None, None

    if view == "Shared vibe space (semantic only)":
        cols = [c for c in df.columns if _is_vibe_col(c) and not _is_excluded(c)]
        if not cols:
            return df, "NO_VIBE_COLS", None
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
    return df, None, {"X": X, "metric": metric}


def _pairwise_distance(X, metric):
    """Full pairwise distance matrix in the ORIGINAL feature space."""
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        return 1.0 - (Xn @ Xn.T)  # cosine distance in [0, 2]
    diff = X[:, None, :] - X[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def _neighbors_in_true_space(X, metric, query_pos, k=8):
    """Indices (into X) of the k nearest items to row `query_pos`, true space."""
    if metric == "cosine":
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        d = 1.0 - Xn @ Xn[query_pos]
    else:
        d = np.linalg.norm(X - X[query_pos], axis=1)
    order = np.argsort(d)
    order = order[order != query_pos]
    return order[:k], d


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
    with st.expander("ℹ️ What do these controls do?"):
        st.markdown(
            "- **Projection space** *(above the map)* — *Shared vibe* maps only the "
            "semantic embedding components (cross-domain distance = taste similarity); "
            "*Full feature* maps every model input (domains island by feature coverage — diagnostic only).\n"
            "- **Apply domain alignment (CORAL)** — re-centers each media type's vibe cloud "
            "on the shared global centroid, so a movie and a song that are *semantically* alike "
            "sit together instead of being pushed apart by per-domain offset. Off = raw embeddings.\n"
            "- **UMAP n_neighbors** — neighborhood size. *Low* (5–15) preserves fine local clusters; "
            "*high* (30–60) emphasizes global structure. It trades local vs. global faithfulness.\n"
            "- **UMAP min_dist** — how tightly points may pack. *0.0* = dense, clumpy clusters; "
            "*0.9* = points spread evenly apart. Purely visual spacing, not distance.\n"
            "- **Color By** — recolor points by media type, your rating (`target_reg`), or year.\n"
            "- **Media Types** — show/hide whole domains.\n"
            "- **Min rating / affinity** — hide items rated below the threshold "
            "(music uses its PU-calibrated affinity pseudo-label).\n\n"
            "UMAP layout is stochastic but seeded (`random_state=42`), so the same settings "
            "reproduce the same map."
        )
    align = st.checkbox("Apply domain alignment (CORAL)", value=True,
                        disabled=not view.startswith("Shared"),
                        help="Center each domain's vibe cloud on the shared centroid so "
                             "cross-domain proximity reflects semantic content, not domain offset.")
    n_neighbors = st.slider("UMAP n_neighbors", 5, 60, 15,
                            help="Neighborhood size: low preserves local clusters, high emphasizes global structure.")
    min_dist = st.slider("UMAP min_dist", 0.0, 0.9, 0.1,
                         help="Minimum spacing between points: 0 = tight clumps, 0.9 = spread out.")
    include_music = st.checkbox("🎵 Include music (lyrics as description)", value=False,
                                help="Project your music library into the same vibe space, using each "
                                     "track's lyrics as its description. ~3,700 tracks — first run embeds "
                                     "them (slower), then it's cached. Music has no rating.")

with st.spinner("Embedding lyrics + projecting…" if include_music else "Projecting…"):
    df, err, hidim = compute_projection(view, align, n_neighbors, min_dist, include_music)

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
    search = st.text_input(
        "🔎 Find & highlight items",
        placeholder="God of War, Percy Jackson",
        help="Comma-separated title fragments. Every match (across all domains) is "
             "ringed and labelled on the map so you can see whether they cluster "
             "together or sit apart.")

# Music is unrated (NaN target_reg); the rating threshold must not silently drop it.
rating_ok = (df["target_reg"] >= min_rating) | df["target_reg"].isna()
mask = df["media_type"].isin(media_filter) & rating_ok
fdf = df[mask].copy()

# Resolve search terms -> matched rows (case-insensitive substring, any domain).
terms = [t.strip().lower() for t in search.split(",") if t.strip()] if search else []
if terms:
    tl = fdf["title"].astype(str).str.lower()
    fdf["_match"] = tl.apply(lambda t: next((q for q in terms if q in t), None))
    matched = fdf[fdf["_match"].notna()]
else:
    matched = fdf.iloc[0:0]

hover_cols = [c for c in ("title", "media_type", "target_reg", "year")
              if c in fdf.columns]
fig = px.scatter(
    fdf, x="umap_1", y="umap_2", color=color_by,
    hover_name="title", hover_data={c: True for c in hover_cols},
    title=f"Cross-Domain Latent Universe — {view}",
    color_continuous_scale="RdYlGn" if color_by == "target_reg" else "Viridis",
    height=800, template="plotly_dark",
)
base_opacity = 0.18 if len(matched) else 0.65
fig.update_traces(marker=dict(size=6, opacity=base_opacity,
                              line=dict(width=0.3, color="white")))
if len(matched):
    fig.add_trace(go.Scatter(
        x=matched["umap_1"], y=matched["umap_2"],
        mode="markers+text",
        text=matched["title"], textposition="top center",
        textfont=dict(size=11, color="#FFD166"),
        marker=dict(size=15, color="rgba(255,90,90,0.95)",
                    line=dict(width=2, color="white"), symbol="star"),
        name="🔎 matches",
        hovertext=matched["title"] + "  (" + matched["media_type"] + ")",
        hoverinfo="text",
    ))
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
)
with c1:
    st.plotly_chart(fig, use_container_width=True)

    if terms:
        if len(matched):
            st.markdown(f"**{len(matched)} item(s) matched** `{', '.join(terms)}`:")
            st.dataframe(
                matched[["title", "media_type", "target_reg"]]
                .rename(columns={"target_reg": "your_rating"}),
                use_container_width=True, hide_index=True)

            metric = hidim["metric"] if hidim else "cosine"
            sim_word = "cosine distance (0 = identical)" if metric == "cosine" else "euclidean distance"

            # TRUE-SPACE pairwise distance — the faithful "close or apart?" read.
            # (The 2-D map distorts global distance; this uses the high-dim vectors.)
            if len(matched) > 1 and hidim is not None:
                pos = matched.index.to_numpy()          # positional rows into X
                sub = hidim["X"][pos]
                dmat = _pairwise_distance(sub, metric)
                dtab = pd.DataFrame(dmat, index=matched["title"], columns=matched["title"]).round(3)
                st.caption(f"**True-space {sim_word}** between your matches — smaller = "
                           "genuinely more similar (computed in the full embedding, not the 2-D map):")
                st.dataframe(dtab, use_container_width=True)

            # Nearest neighbours in TRUE space for each match — what actually
            # clusters with it, across every domain.
            if hidim is not None:
                st.caption("**Nearest neighbours in the true embedding space** "
                           "(what each match is genuinely closest to, any domain):")
                titles_all = df["title"].to_numpy()
                mtypes_all = df["media_type"].to_numpy()
                for ridx in matched.index:
                    nb, dist = _neighbors_in_true_space(hidim["X"], metric, ridx, k=6)
                    label = f"{titles_all[ridx]} ({mtypes_all[ridx]})"
                    neigh = ", ".join(f"{titles_all[j]} [{mtypes_all[j]}] · {dist[j]:.2f}"
                                      for j in nb)
                    st.markdown(f"- **{label}** → {neigh}")
        else:
            st.warning(f"No items matched `{', '.join(terms)}`. Note the map shows the "
                       "**frozen** unified set (movies=1000, TV=162, books=88, games=72) — "
                       "items added after the last unified feature-engineering run won't appear. "
                       "Also check the **Media Types** filter and **Min rating** slider aren't "
                       "hiding them, and press **R** / clear cache if the app predates this update.")

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