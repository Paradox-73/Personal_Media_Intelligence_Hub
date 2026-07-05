import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import ast
import re
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from itertools import combinations

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from src import config

from mobile import apply_mobile_css
st.set_page_config(page_title="Collaboration Network", page_icon="🕸️", layout="wide")
apply_mobile_css()

# --- CUSTOM STYLING (matches the rest of the hub) ---
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #14171C;
        color: #E0E0E0;
        font-family: 'Graphik', 'Helvetica', sans-serif;
    }
    [data-testid="stSidebar"] { background-color: #2C343F; }
    h1, h2, h3, h4, h5, h6 {
        color: #40bcf4 !important;
        font-family: 'Graphik', 'Helvetica', sans-serif;
    }
    .stMetric {
        background-color: #2C343F;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #556678;
    }
    div[data-testid="metric-container"] label { color: #00e054 !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #ff8000 !important; }
    hr { border-top: 1px solid #556678; }
</style>
""", unsafe_allow_html=True)

pio.templates["hub_theme"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0", family="Graphik, Helvetica, sans-serif"),
    )
)
pio.templates.default = "hub_theme"

DOMAIN_COLORS = {"Movies": "#ff8000", "TV": "#40bcf4", "Music": "#00e054"}


# ---------------------------------------------------------------------------
# Data loading — build a flat list of "credits" per title, once.
# Each credit = (domain, title, rating, ((role, person), ...))
# ---------------------------------------------------------------------------
def _as_list(val):
    """Parse a stringified python list; fall back to [] on anything odd."""
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    try:
        parsed = ast.literal_eval(val)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _split_names(val):
    """Plain comma-separated name string -> list of clean names."""
    if not isinstance(val, str) or not val.strip():
        return []
    return [n.strip() for n in val.split(",") if n.strip()]


@st.cache_data(show_spinner=False)
def load_credits():
    """Return one record per title: (domain, title, rating, [(role, name), ...])."""
    records = []

    # ---- Movies: director / writer / actors are stringified lists ----
    try:
        mv = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        for _, r in mv.iterrows():
            people = []
            for d in _as_list(r.get("director")):
                people.append(("director", str(d).strip()))
            for w in _as_list(r.get("writer")):
                people.append(("writer", str(w).strip()))
            for a in _as_list(r.get("actors")):
                people.append(("actor", str(a).strip()))
            if people:
                records.append(("Movies", str(r.get("title", "")),
                                r.get("user_rating"), people))
    except FileNotFoundError:
        pass

    # ---- TV: created_by is a list; actors/writer are comma strings ----
    try:
        tv = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
        for _, r in tv.iterrows():
            people = []
            for c in _as_list(r.get("created_by")):
                people.append(("creator", str(c).strip()))
            for w in _split_names(r.get("writer")):
                people.append(("writer", w))
            for a in _split_names(r.get("actors")):
                people.append(("actor", a))
            if people:
                records.append(("TV", str(r.get("name", "")),
                                r.get("user_rating"), people))
    except FileNotFoundError:
        pass

    # ---- Music: artist_ids are authoritative; map ids -> display names ----
    try:
        mu = pd.read_csv(config.MUSIC_ENRICHED_DATA_PATH)
        id_to_name = {}
        # First pass: learn a reliable id->name map from clean rows
        # (rows where the comma-split name count matches the id count).
        for _, r in mu.iterrows():
            ids = _split_names(r.get("artist_ids"))
            names = _split_names(r.get("artists"))
            if ids and len(ids) == len(names):
                for i, n in zip(ids, names):
                    id_to_name.setdefault(i, n)
            prim_id = str(r.get("artist_ids", "")).split(",")[0].strip()
            if prim_id and isinstance(r.get("primary_artist"), str):
                id_to_name.setdefault(prim_id, r["primary_artist"].strip())

        for _, r in mu.iterrows():
            ids = _split_names(r.get("artist_ids"))
            if len(ids) < 2:
                continue  # need a collaboration
            people = [("artist", id_to_name.get(i, i)) for i in ids]
            records.append(("Music", str(r.get("name", "")),
                            r.get("rating"), people))
    except FileNotFoundError:
        pass

    return records


# ---------------------------------------------------------------------------
# Graph construction from credits, given the active filters.
# ---------------------------------------------------------------------------
def build_graph(records, domains, roles, cast_cap, min_titles, min_weight, min_degree):
    # node -> stats
    titles_of = defaultdict(set)          # person -> set of title keys
    ratings_of = defaultdict(list)        # person -> [ratings]
    domains_of = defaultdict(set)         # person -> {domains}
    roles_of = defaultdict(set)           # person -> {roles}
    edge_weight = defaultdict(int)        # (a,b) sorted -> shared-title count

    for domain, title, rating, people in records:
        if domain not in domains:
            continue
        # keep only requested roles, then cap the cast (actors are billing-ordered)
        kept, n_actor = [], 0
        for role, name in people:
            if role not in roles or not name:
                continue
            if role == "actor":
                if n_actor >= cast_cap:
                    continue
                n_actor += 1
            kept.append(name)
        kept = list(dict.fromkeys(kept))  # dedupe, keep order
        if len(kept) < 1:
            continue

        tkey = (domain, title)
        try:
            rating_val = float(rating)
        except (TypeError, ValueError):
            rating_val = None
        for name in kept:
            titles_of[name].add(tkey)
            domains_of[name].add(domain)
            if rating_val is not None:
                ratings_of[name].append(rating_val)
        for role, name in people:
            if name in kept:
                roles_of[name].add(role)

        for a, b in combinations(sorted(kept), 2):
            edge_weight[(a, b)] += 1

    # node filter: appears in >= min_titles titles
    keep_nodes = {p for p, ts in titles_of.items() if len(ts) >= min_titles}

    edges = [(a, b, w) for (a, b), w in edge_weight.items()
             if w >= min_weight and a in keep_nodes and b in keep_nodes]

    # drop isolated nodes (no surviving edge) so the picture stays about collaboration
    connected = set()
    for a, b, _ in edges:
        connected.add(a)
        connected.add(b)
    keep_nodes &= connected

    # k-core trim: iteratively remove people with fewer than `min_degree`
    # collaborators. Removing one lowers its neighbours' counts, so repeat
    # until stable — this peels away the sparse fringe and keeps dense clusters.
    if min_degree > 1:
        changed = True
        while changed:
            deg = defaultdict(int)
            for a, b, _ in edges:
                deg[a] += 1
                deg[b] += 1
            drop = {p for p in keep_nodes if deg[p] < min_degree}
            if drop:
                keep_nodes -= drop
                edges = [(a, b, w) for a, b, w in edges
                         if a in keep_nodes and b in keep_nodes]
            else:
                changed = False

    nodes = []
    for p in keep_nodes:
        rts = ratings_of[p]
        nodes.append({
            "name": p,
            "titles": len(titles_of[p]),
            "avg_rating": float(np.mean(rts)) if rts else None,
            "domains": sorted(domains_of[p]),
            "roles": sorted(roles_of[p]),
        })
    nodes.sort(key=lambda d: d["titles"], reverse=True)
    return nodes, edges


# ---------------------------------------------------------------------------
# Compact Fruchterman-Reingold force-directed layout (numpy only).
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def spring_layout(node_names, edge_index, edge_weights, anchors, spread=1.0, seed=42):
    n = len(node_names)
    rng = np.random.default_rng(seed)
    # `anchors` = a per-node (x, y) pull target (or None for plain centre gravity).
    # When present it seeds the start position too, so domains land in their region.
    if anchors:
        anchor_arr = np.array(anchors, dtype=float)
        pos = anchor_arr + rng.uniform(-0.05, 0.05, size=(n, 2))
        gravity = 0.16          # strong pull so groups actually separate into regions
    else:
        anchor_arr = np.zeros((n, 2))
        pos = rng.uniform(-1, 1, size=(n, 2)).astype(float)
        gravity = 0.03          # gentle pull to centre so clusters don't drift off
    if n <= 1:
        return pos

    # More iterations for small graphs (they converge to something tidy),
    # fewer for large ones (each step is O(n^2)).
    iterations = int(np.clip(6000 / np.sqrt(n), 150, 500))
    k = spread / np.sqrt(n)        # ideal edge length — bigger `spread` = airier
    edges = np.array(edge_index) if edge_index else np.empty((0, 2), int)
    # Stronger collaborations pull a bit harder, so frequent co-workers cluster.
    w = np.array(edge_weights, dtype=float) if edge_weights else np.empty(0)
    if len(w):
        w = 0.5 + w / w.max()      # map weights into ~[0.5, 1.5]
    t = 0.1                        # initial "temperature" (max displacement)

    for _ in range(iterations):
        disp = np.zeros((n, 2))
        # Repulsion (all pairs, vectorised)
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9
        rep = (k * k) / dist
        np.fill_diagonal(rep, 0)
        disp += (delta * (rep / dist)[..., None]).sum(axis=1)
        # Attraction (along edges, scaled by collaboration strength)
        if len(edges):
            d = pos[edges[:, 0]] - pos[edges[:, 1]]
            dl = np.sqrt((d ** 2).sum(-1)) + 1e-9
            force = (dl * dl) / k * w
            move = d * (force / dl)[:, None]
            np.add.at(disp, edges[:, 0], -move)
            np.add.at(disp, edges[:, 1], move)
        # Gravity toward each node's anchor (centre by default)
        disp -= (pos - anchor_arr) * gravity
        # Limit displacement by temperature, then cool
        length = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += disp * (np.minimum(length, t) / length)[:, None]
        t *= 0.98
    # normalise to a nice square
    pos -= pos.mean(axis=0)
    span = np.abs(pos).max() + 1e-9
    return pos / span


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🕸️ Collaboration Network")
st.markdown(
    "People you've watched and listened to, wired together by who they worked with. "
    "**Nodes** are people; an **edge** links two people who share a title "
    "(a movie, show, or track). Edge thickness = number of shared titles; "
    "node size = how many of your titles they appear in."
)

records = load_credits()

role_opts = {
    "Actors / Artists": {"actor", "artist"},
    "Directors": {"director"},
    "Writers": {"writer"},
    "Creators (TV)": {"creator"},
}

with st.expander("⚙️  Filters & layout", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        domains = st.multiselect(
            "Domains", ["Movies", "TV", "Music"],
            default=["Movies", "TV", "Music"],
        )
    with c2:
        role_labels = st.multiselect(
            "Roles", list(role_opts.keys()),
            default=["Actors / Artists", "Directors", "Creators (TV)"],
        )
    roles = set().union(*[role_opts[l] for l in role_labels]) if role_labels else set()

    c3, c4, c5 = st.columns(3)
    with c3:
        cast_cap = st.slider(
            "Top-billed cast per title", 2, 20, 6,
            help="Movies list 50-100 actors. Cap to the top N (by billing) so the "
                 "graph shows lead collaborations, not every extra.",
        )
    with c4:
        min_titles = st.slider(
            "Min. titles to appear in graph", 1, 10, 3,
            help="A person must appear in at least this many of your titles to show up.",
        )
    with c5:
        min_weight = st.slider(
            "Min. collaborations per edge", 1, 5, 1,
            help="An edge shows only if the two people share at least this many titles.",
        )

    st.markdown("**Trim & readability**")
    c6, c7, c8 = st.columns(3)
    with c6:
        min_degree = st.slider(
            "Min. connections per person", 1, 8, 1,
            help="Trim the graph: keep only people with at least this many "
                 "collaborators (a k-core). Raise it to peel away the sparse fringe "
                 "and reveal the dense clusters.",
        )
    with c7:
        spread = st.slider(
            "Node spacing", 0.5, 3.0, 1.4, step=0.1,
            help="How airy the layout is. Higher = nodes pushed further apart, "
                 "less overlap.",
        )
    with c8:
        label_top = st.slider(
            "Label the top N hubs", 0, 60, 20,
            help="Only the most-connected people get a name label, to avoid clutter.",
        )

    c8b, c9 = st.columns([1.5, 1.5])
    with c8b:
        layout_mode = st.selectbox(
            "Layout", ["Force (organic)", "Clustered by domain", "Circle (sorted)"],
            help="Force = organic blob. Clustered = each domain pulled into its own "
                 "region. Circle = a sorted ring (chord diagram), ordered by domain "
                 "then connections — the most structured view.",
        )
    with c9:
        color_by = st.radio("Colour nodes by", ["Domain", "Avg. rating"], horizontal=True)

    c10, c11 = st.columns([1.5, 1.5])
    with c10:
        max_nodes = st.slider("Max nodes (for readability)", 30, 500, 200, step=10)
    with c11:
        edge_curve = st.slider(
            "Edge curve", 0.0, 0.5, 0.2, step=0.05,
            help="Bend edges into arcs so overlapping lines fan apart. "
                 "0 = straight lines.",
        )

if not domains or not roles:
    st.info("Pick at least one domain and one role above.")
    st.stop()

nodes, edges = build_graph(records, set(domains), roles, cast_cap,
                           min_titles, min_weight, min_degree)

if not nodes:
    st.warning("No collaborations match these filters. Try lowering the minimums.")
    st.stop()

# cap node count for readability — keep the most-connected people
truncated = False
if len(nodes) > max_nodes:
    truncated = True
    keep = {d["name"] for d in nodes[:max_nodes]}
    nodes = nodes[:max_nodes]
    edges = [(a, b, w) for a, b, w in edges if a in keep and b in keep]

# Truncation (or any filter above) can leave a person whose only collaborators
# were dropped — an isolated node. Keep only people that still have an edge.
connected = set()
for a, b, _ in edges:
    connected.add(a)
    connected.add(b)
nodes = [d for d in nodes if d["name"] in connected]

if not nodes:
    st.warning("No collaborations match these filters. Try lowering the minimums.")
    st.stop()

name_to_idx = {d["name"]: i for i, d in enumerate(nodes)}
edge_index = [(name_to_idx[a], name_to_idx[b]) for a, b, w in edges]

c1, c2, c3 = st.columns(3)
c1.metric("People", len(nodes))
c2.metric("Collaborations", len(edges))
c3.metric("Strongest link", max((w for *_, w in edges), default=0))
if truncated:
    st.caption(f"Showing the {max_nodes} most-connected people. "
               "Raise *Max nodes* or the minimums to change what's shown.")

edge_wts = [w for *_, w in edges]

# ---- node degrees (for hub labels, sorting + a "top collaborator" hover) ----
degree = defaultdict(int)
neighbours = defaultdict(list)
for (a, b, w) in edges:
    degree[a] += 1
    degree[b] += 1
    neighbours[a].append((w, b))
    neighbours[b].append((w, a))

# ---- layout: give the graph structure per the chosen mode ----
DOMAIN_RANK = {"Movies": 0, "TV": 1, "Music": 2, "Multi": 3}


def _group(d):
    return "Multi" if len(d["domains"]) > 1 else d["domains"][0]


groups = [_group(d) for d in nodes]

if layout_mode == "Circle (sorted)":
    # Deterministic ring: order by domain, then by connections (hubs together).
    order = sorted(range(len(nodes)),
                   key=lambda i: (DOMAIN_RANK.get(groups[i], 9),
                                  -degree[nodes[i]["name"]], nodes[i]["name"]))
    ang = np.zeros(len(nodes))
    for rank, i in enumerate(order):
        ang[i] = 2 * np.pi * rank / max(1, len(nodes))
    pos = np.column_stack([np.cos(ang), np.sin(ang)])
else:
    anchors = None
    if layout_mode == "Clustered by domain":
        # One anchor per domain, spaced evenly around a circle.
        uniq = sorted(set(groups), key=lambda g: DOMAIN_RANK.get(g, 9))
        m = len(uniq)
        centres = {g: (0.75 * np.cos(2 * np.pi * j / m),
                       0.75 * np.sin(2 * np.pi * j / m)) if m > 1 else (0.0, 0.0)
                   for j, g in enumerate(uniq)}
        anchors = tuple(centres[g] for g in groups)
    pos = spring_layout(tuple(d["name"] for d in nodes),
                        tuple(edge_index), tuple(edge_wts), anchors, spread)

# ---- edges, drawn in strength tiers so strong links read louder ----
# Faint thin lines for one-offs, brighter thicker lines for frequent pairs.
if edge_wts:
    q_mid, q_hi = np.quantile(edge_wts, [0.6, 0.9])
else:
    q_mid = q_hi = 1
tiers = [
    (q_hi, "rgba(64,188,244,0.55)", 2.2),   # strong  -> hub blue
    (q_mid, "rgba(140,160,180,0.32)", 1.2),  # medium
    (0, "rgba(85,102,120,0.14)", 0.6),       # weak
]

# Draw each edge as a quadratic-bezier arc. A control point offset perpendicular
# to the A->B line bends the edge, so lines that would overlap fan into strands.
# The bend direction is keyed on the sorted endpoint order, so it's stable and
# reciprocal pairs separate instead of stacking. curve=0 -> straight segments.
_t = np.linspace(0, 1, 16)
_one_minus = (1 - _t)


def _arc(p0, p1, curve):
    if curve <= 0:
        return [p0[0], p1[0], None], [p0[1], p1[1], None]
    mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    cx, cy = mx - dy * curve, my + dx * curve   # control point off the midpoint
    bx = _one_minus ** 2 * p0[0] + 2 * _one_minus * _t * cx + _t ** 2 * p1[0]
    by = _one_minus ** 2 * p0[1] + 2 * _one_minus * _t * cy + _t ** 2 * p1[1]
    return list(bx) + [None], list(by) + [None]


# Assign each edge to exactly one tier (the highest it qualifies for), then
# group same-styled edges into one Scatter trace (plotly needs one width/trace).
grouped = defaultdict(lambda: ([], []))
for (a, b, w), (ia, ib) in zip(edges, edge_index):
    for lo, color, width in tiers:
        if w >= lo:
            xs, ys = grouped[(color, width)]
            ax, ay = _arc(pos[ia], pos[ib], edge_curve)
            xs += ax
            ys += ay
            break
edge_go = [
    go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width,
               shape="spline"),
               hoverinfo="none", showlegend=False)
    for (color, width), (xs, ys) in grouped.items()
]

sizes = np.array([d["titles"] for d in nodes], dtype=float)
sizes = 7 + 22 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1e-9)

# ---- hub labels: only the most-connected people get a visible name ----
hub_names = {d["name"] for d in
             sorted(nodes, key=lambda d: degree[d["name"]], reverse=True)[:label_top]}
node_labels = [d["name"] if d["name"] in hub_names else "" for d in nodes]

hover, custom = [], []
for d in nodes:
    top = sorted(neighbours[d["name"]], reverse=True)[:3]
    top_str = ", ".join(f"{nm} ({w})" for w, nm in top) or "—"
    rating_str = f"{d['avg_rating']:.1f}" if d["avg_rating"] is not None else "n/a"
    hover.append(
        f"<b>{d['name']}</b><br>"
        f"Titles: {d['titles']} &nbsp;|&nbsp; Avg rating: {rating_str}<br>"
        f"Domains: {', '.join(d['domains'])}<br>"
        f"Roles: {', '.join(d['roles'])}<br>"
        f"Collaborators: {degree[d['name']]}<br>"
        f"Top links: {top_str}"
    )

if color_by == "Domain":
    def node_color(d):
        if len(d["domains"]) > 1:
            return "#c084fc"  # multi-domain
        return DOMAIN_COLORS.get(d["domains"][0], "#E0E0E0")
    marker = dict(
        size=sizes,
        color=[node_color(d) for d in nodes],
        line=dict(color="#14171C", width=1),
    )
    legend_note = (
        "🟠 Movies &nbsp; 🔵 TV &nbsp; 🟢 Music &nbsp; 🟣 Multi-domain"
    )
else:
    marker = dict(
        size=sizes,
        color=[d["avg_rating"] if d["avg_rating"] is not None else np.nan for d in nodes],
        colorscale="Viridis",
        colorbar=dict(title="Avg<br>rating", tickfont=dict(color="#E0E0E0")),
        line=dict(color="#14171C", width=1),
    )
    legend_note = "Node colour = your average rating of that person's titles."

node_trace = go.Scatter(
    x=pos[:, 0], y=pos[:, 1],
    mode="markers+text" if label_top else "markers",
    marker=marker,
    text=node_labels,
    textposition="top center",
    textfont=dict(size=9, color="#cbd5e1"),
    customdata=hover,
    hovertemplate="%{customdata}<extra></extra>",
    showlegend=False,
)

fig = go.Figure(data=edge_go + [node_trace])
fig.update_layout(
    height=760,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
    hoverlabel=dict(bgcolor="#2C343F", font=dict(color="#E0E0E0")),
)
st.plotly_chart(fig, use_container_width=True)
st.caption(
    legend_note
    + "  •  Brighter/thicker edges = more shared titles."
    + "  •  Only the top hubs are labelled — hover any node for its name & details;"
    + " scroll/drag to zoom and pan."
)

# ---- strongest collaborations table ----
st.subheader("Strongest collaborations")
top_edges = sorted(edges, key=lambda e: e[2], reverse=True)[:25]
st.dataframe(
    pd.DataFrame(
        [{"Person A": a, "Person B": b, "Shared titles": w} for a, b, w in top_edges]
    ),
    use_container_width=True, hide_index=True,
)

# ---- most-connected people ----
st.subheader("Most-connected people")
top_people = sorted(nodes, key=lambda d: degree[d["name"]], reverse=True)[:25]
st.dataframe(
    pd.DataFrame([{
        "Person": d["name"],
        "Collaborators": degree[d["name"]],
        "Titles": d["titles"],
        "Avg rating": round(d["avg_rating"], 1) if d["avg_rating"] is not None else None,
        "Domains": ", ".join(d["domains"]),
    } for d in top_people]),
    use_container_width=True, hide_index=True,
)
