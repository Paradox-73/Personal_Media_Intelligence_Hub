"""Shared ridgeline (joyplot) builder for "rating distribution by genre".

Renders one smoothed rating-density curve per genre, stacked vertically with a
slight overlap: the genre label sits on the left, the gaussian-like curve runs
across the middle, and the item count is printed on the right. Each domain
dashboard feeds it (genre, rating) pairs; the hub landing page (main.py) reuses
it in cross-domain form by passing the domain name in place of the genre.

Plotly-only and numpy-only so it drops into every page without new deps. The KDE
is a hand-rolled sum-of-gaussians with a bandwidth floor, so even a genre where
every item shares one rating still renders as a readable bump.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Endpoints of the ridge fill gradient: harsher (lower avg) -> kinder (higher avg).
_LOW_RGB = (85, 102, 120)    # #556678
_HIGH_RGB = (255, 128, 0)    # #ff8000


def _lerp_rgb(t):
    """Interpolate _LOW_RGB -> _HIGH_RGB; t in [0, 1]. Returns an (r, g, b) tuple."""
    t = 0.0 if (t is None or np.isnan(t)) else float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(lo + (hi - lo) * t)) for lo, hi in zip(_LOW_RGB, _HIGH_RGB))


def _kde(values, grid, bw_floor=0.32):
    """Gaussian KDE evaluated on `grid`. Manual so it survives zero-variance input."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.zeros_like(grid)
    std = v.std()
    h = 1.06 * std * v.size ** (-0.2) if std > 0 else 0.0
    h = max(h, bw_floor)
    u = (grid[None, :] - v[:, None]) / h
    return np.exp(-0.5 * u * u).sum(axis=0) / (v.size * h * np.sqrt(2 * np.pi))


def genre_rating_ridgeline(
    pairs,
    *,
    genre_col="genre",
    rating_col="rating",
    title=None,
    rating_label="Your Rating",
    count_label="items",
    top_n=12,
    min_count=5,
    rating_range=(0.5, 5.0),
    overlap=1.15,
    height_per_row=48,
):
    """Build a ridgeline figure of rating distribution per genre.

    pairs: long-form DataFrame with one row per (item, genre) and a numeric
    rating. Genres are ranked by volume (top_n kept) then stacked from your
    lowest- to highest-rated on average. Returns a plotly Figure, or None when
    there isn't enough data to draw anything meaningful.
    """
    if pairs is None or len(pairs) == 0:
        return None
    d = pairs[[genre_col, rating_col]].copy()
    d[rating_col] = pd.to_numeric(d[rating_col], errors="coerce")
    d = d.dropna(subset=[genre_col, rating_col])
    d[genre_col] = d[genre_col].astype(str).str.strip()
    d = d[~d[genre_col].str.lower().isin(["", "nan", "none", "unknown"])]
    lo, hi = rating_range
    d = d[(d[rating_col] >= lo) & (d[rating_col] <= hi)]
    if d.empty:
        return None

    g = d.groupby(genre_col)[rating_col]
    summary = pd.DataFrame({"count": g.size(), "mean": g.mean()})
    keep = summary[summary["count"] >= min_count]
    if keep.empty:  # small libraries: relax so we still show the busiest genres
        keep = summary[summary["count"] >= 2]
    if keep.empty:
        return None

    # Pick the top_n genres by volume, then order by mean rating (low at bottom).
    keep = keep.sort_values("count", ascending=False).head(top_n)
    keep = keep.sort_values("mean", ascending=True)
    genres = keep.index.tolist()

    grid = np.linspace(lo, hi, 220)
    mean_lo, mean_hi = float(keep["mean"].min()), float(keep["mean"].max())
    span = (mean_hi - mean_lo) or 1.0
    right_x = hi + (hi - lo) * 0.04

    fig = go.Figure()
    baselines = []
    for i, genre in enumerate(genres):
        vals = d.loc[d[genre_col] == genre, rating_col].values
        dens = _kde(vals, grid)
        peak = dens.max() or 1.0
        dens = dens / peak  # normalize each ridge to unit height for readability
        baseline = float(i)
        baselines.append(baseline)
        y = baseline + dens * overlap

        cnt = int(keep.loc[genre, "count"])
        avg = float(keep.loc[genre, "mean"])
        r, gg, b = _lerp_rgb((avg - mean_lo) / span)
        fill_c = f"rgba({r},{gg},{b},0.80)"
        line_c = f"rgb({min(r + 35, 255)},{min(gg + 35, 255)},{min(b + 35, 255)})"

        # Closed polygon: out along the curve, back along the baseline.
        fig.add_trace(go.Scatter(
            x=np.concatenate([grid, grid[::-1]]),
            y=np.concatenate([y, np.full_like(grid, baseline)]),
            fill="toself", mode="lines", line=dict(color=line_c, width=1.4),
            fillcolor=fill_c, hoverinfo="skip", showlegend=False,
        ))
        # Transparent overlay carrying the hover stats.
        fig.add_trace(go.Scatter(
            x=grid, y=y, mode="lines", line=dict(width=0.1, color="rgba(0,0,0,0)"),
            showlegend=False,
            hovertemplate=f"<b>{genre}</b><br>{cnt} {count_label} · avg {avg:.2f}★<extra></extra>",
        ))
        # The "number on the right".
        fig.add_annotation(x=right_x, y=baseline + 0.12, text=f"{cnt}", showarrow=False,
                           xanchor="left", font=dict(color="#99AABB", size=12))

    n = len(genres)
    fig.add_annotation(x=right_x, y=n - 1 + overlap + 0.25, text=count_label, showarrow=False,
                       xanchor="left", font=dict(color="#556678", size=10))
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        xaxis=dict(title=rating_label, range=[lo - 0.25, hi + (hi - lo) * 0.11],
                   tickmode="linear", tick0=lo, dtick=0.5,
                   gridcolor="#2C343F", zerolinecolor="#2C343F"),
        yaxis=dict(tickmode="array", tickvals=baselines, ticktext=genres,
                   showgrid=False, zeroline=False,
                   range=[-0.4, n - 1 + overlap + 0.5], tickfont=dict(size=12)),
        height=max(320, int(n * height_per_row + 130)),
        margin=dict(l=10, r=44, t=50 if title else 16, b=44),
        showlegend=False,
    )
    return fig
