"""Shared "me vs. the critics" block -- BUCKET distributions, not title rankings.

WHAT THIS IS NOT
    It is not a ranked list of your biggest disagreements. Each dashboard already
    has one of those ("Your Hot Takes (vs General Consensus)"), which prints the
    top-10 titles in each direction. Repeating that as a bar chart adds nothing.

WHAT THIS IS
    Both charts bucket on the half-star grid you actually rate on and plot COUNTS.
    The question they answer is distributional -- *what shape is my rating curve
    against theirs*, and *how big are my disagreements usually* -- which the
    per-title view cannot show and the Hot Takes list deliberately does not.

  1. `render_score_comparison` -- one bucket per half-star (0.5 ... 5.0), bars
     showing how many titles YOU put in that bucket beside how many each critic
     source put there. Hovering a bar lists the titles in it.

  2. `render_delta_chart` -- (you - critic average) snapped to the nearest half
     star, one bar per delta bucket, diverging around zero. Hovering lists the
     titles at that level of disagreement.

THE DENOMINATOR PROBLEM, AND WHY SIDE-BY-SIDE IS HONEST HERE
    `rating_distribution.py` deliberately refuses to draw user and critic
    histograms side by side, because critic columns carry their own missing values
    and a bar-height difference would really be a different denominator.

    That objection is real, and this module answers it rather than ignoring it:
    `build_comparison_frame` drops every row missing either a user rating or a
    critic average BEFORE anything is counted, so every series here is drawn over
    the identical set of titles. Same N, same rows, so a height difference is a
    real difference in how the scores are spread. The per-source bars are the one
    place this loosens -- a title with no Metacritic score is absent from the
    Metacritic bars -- so each source's own N is printed in the legend.

SCALES
    Each source arrives as (label, series, scale_max) and is mapped to 0-5 by
    `value / scale_max * 5`, then snapped with `round_half`, the same convention
    `rating_distribution.to_stars` uses. `scale_max` is the source's own maximum
    (10 for IMDb/TMDB, 100 for Metacritic/RT, 5 for RAWG and Hardcover), NOT a
    guess from the observed data.

A source is dropped rather than drawn empty when its column is absent or all
null -- but only AFTER `to_five` has had a chance to parse it. `rotten_tomatoes_rating`
reads as 0 usable values under a plain `pd.to_numeric` and 949 under the percent-
aware parse. `tvdb_score`, by contrast, looks like a rating and is not one: it
holds vote COUNTS up to 9.1 million, so it is never passed in as a critic source.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.common.parsing import round_half

USER_COLOR = "#00e054"     # your own stars -- the same green every dashboard uses
CRITIC_AVG_COLOR = "#ff8000"
SOURCE_COLORS = ["#40bcf4", "#a78bfa", "#f472b6", "#facc15", "#556678"]
ABOVE_COLOR = "#00e054"    # you rated it higher than the critics
BELOW_COLOR = "#ff8000"    # critics rated it higher than you
NEUTRAL_COLOR = "#556678"

MAX_HOVER_TITLES = 14


def to_five(series, scale_max):
    """Rescale one critic column onto the 0-5 star axis. 0 means 'unscored'.

    Strips a trailing '%' before coercing: OMDb stores Rotten Tomatoes as the
    STRING "92%", so a plain `pd.to_numeric` turns all 949 scored films into NaN
    and the source looks empty when it is actually well covered.
    """
    s = pd.Series(series)
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype("string").str.strip().str.rstrip("%")
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)
    return s / float(scale_max) * 5.0


def build_comparison_frame(df, title_col, user_col, critic_sources):
    """-> (frame, [source labels]), one row per title, one column per critic source.

    `critic_avg` is the mean of whatever sources actually scored that title, so a
    film with only an IMDb score is not dragged toward the middle by counting the
    missing Metacritic as anything at all.
    """
    out = pd.DataFrame({
        "title": df[title_col].astype(str).values,
        "user": pd.to_numeric(df[user_col], errors="coerce").values,
    }, index=df.index)

    labels = []
    for label, series, scale_max in critic_sources:
        if series is None:
            continue
        col = to_five(pd.Series(series).reindex(df.index), scale_max)
        if col.notna().sum() == 0:
            continue          # absent or entirely empty -- drop, never draw empty
        out[label] = col.values
        labels.append(label)

    if not labels:
        return out.iloc[0:0], []

    out["critic_avg"] = out[labels].mean(axis=1, skipna=True)
    out["delta"] = out["user"] - out["critic_avg"]
    return out.dropna(subset=["user", "critic_avg"]), labels


def _esc(s):
    """Titles land inside a plotly hovertemplate, which renders HTML."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _hover_list(titles):
    """A bucket's titles as hover text, capped so the tooltip stays on screen."""
    titles = sorted(titles, key=lambda t: str(t).lower())
    shown = [_esc(t) for t in titles[:MAX_HOVER_TITLES]]
    extra = len(titles) - len(shown)
    if extra > 0:
        shown.append(f"<i>...and {extra} more</i>")
    return "<br>".join(shown)


def _bucket_counts(values, titles, grid, lo=0.5, hi=5.0):
    """Snap to the half-star grid -> (counts per bucket, hover text per bucket).

    `lo`/`hi` MUST be widened for deltas: `round_half` defaults to clipping into
    [0.5, 5.0] because there is no 0-star Letterboxd rating, and a delta of -1.5
    silently clipped to +0.5 would put every disagreement in the wrong direction.
    """
    snapped = pd.Series(round_half(pd.Series(values).astype(float), lo=lo, hi=hi))
    counts, hovers = [], []
    for b in grid:
        mask = (snapped == b).values
        counts.append(int(mask.sum()))
        hovers.append(_hover_list(pd.Series(titles)[mask].tolist()) if mask.any() else "")
    return counts, hovers


def render_score_comparison(frame, labels, key):
    """Chart 1 -- how many titles you put in each half-star bucket vs the critics."""
    if frame.empty:
        st.info("No critic scores available to compare.")
        return

    c1, c2 = st.columns([1.1, 2])
    with c1:
        mode = st.radio("Bars", ["Grouped", "Stacked"], horizontal=True, key=f"{key}_mode",
                        help="Grouped compares bucket heights directly. Stacked shows "
                             "the combined critic mass per bucket.")
    with c2:
        picked = st.multiselect(
            "Critic sources to show", ["Critic avg"] + labels,
            default=["Critic avg"], key=f"{key}_srcs",
            help="Every series is drawn over the same titles; a per-source bar is "
                 "missing only where that source never scored the title.")

    grid = np.round(np.arange(0.5, 5.001, 0.5), 2)
    titles = frame["title"].tolist()

    fig = go.Figure()
    ucounts, uhov = _bucket_counts(frame["user"], titles, grid)
    fig.add_bar(x=grid, y=ucounts, name=f"You (n={len(frame)})",
                marker_color=USER_COLOR, customdata=uhov,
                hovertemplate="<b>%{x} stars — you rated %{y} title(s)</b>"
                              "<br>%{customdata}<extra></extra>")

    series = []
    if "Critic avg" in picked:
        series.append(("Critic avg", frame["critic_avg"], CRITIC_AVG_COLOR))
    for i, lbl in enumerate(labels):
        if lbl in picked:
            series.append((lbl, frame[lbl], SOURCE_COLORS[i % len(SOURCE_COLORS)]))

    for name, col, colour in series:
        sub = col.dropna()
        cnt, hov = _bucket_counts(sub, frame.loc[sub.index, "title"], grid)
        fig.add_bar(x=grid, y=cnt, name=f"{name} (n={len(sub)})", marker_color=colour,
                    customdata=hov,
                    hovertemplate="<b>%{x} stars — " + _esc(name) +
                                  " scored %{y} title(s) here</b><br>%{customdata}<extra></extra>")

    fig.update_layout(
        barmode="group" if mode == "Grouped" else "stack",
        bargap=0.18, height=430, margin=dict(l=0, r=10, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        xaxis=dict(title="Rating bucket (stars, every source rescaled to 0-5)",
                   tickmode="linear", tick0=0.5, dtick=0.5, range=[0.25, 5.25]),
        yaxis_title="Titles in bucket", hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_cmp")
    st.caption(
        f"Every series is counted over the **same {len(frame)} titles** — those with "
        f"both your rating and at least one critic score — so a height difference is a "
        f"real difference in spread, not a different denominator. **Hover any bar to "
        f"see which titles are in that bucket.**"
    )


def render_delta_chart(frame, key):
    """Chart 2 -- disagreement size as a distribution, snapped to the half-star grid."""
    if frame.empty:
        st.info("No critic scores available to compare.")
        return

    above = int((frame["delta"] > 0).sum())
    below = int((frame["delta"] < 0).sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean gap", f"{frame['delta'].mean():+.2f}",
              help="Stars. Positive = you rate this library above the critics on average.")
    m2.metric("Median gap", f"{frame['delta'].median():+.2f}")
    m3.metric("You rate higher", f"{above} ({above / len(frame):.0%})")
    m4.metric("Critics rate higher", f"{below} ({below / len(frame):.0%})")

    snapped = pd.Series(round_half(frame["delta"].astype(float), lo=-5.0, hi=5.0)).values
    lo, hi = float(np.min(snapped)), float(np.max(snapped))
    grid = np.round(np.arange(lo, hi + 0.001, 0.5), 2)
    counts, hovers = _bucket_counts(frame["delta"], frame["title"].tolist(), grid,
                                    lo=-5.0, hi=5.0)

    colours = [NEUTRAL_COLOR if abs(b) < 1e-9 else (ABOVE_COLOR if b > 0 else BELOW_COLOR)
               for b in grid]
    fig = go.Figure()
    fig.add_bar(x=grid, y=counts, marker_color=colours, customdata=hovers,
                text=[c if c else "" for c in counts], textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{x:+.1f} stars — %{y} title(s)</b>"
                              "<br>%{customdata}<extra></extra>")
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#8899aa")
    fig.update_layout(
        height=430, margin=dict(l=0, r=10, t=30, b=0), showlegend=False,
        bargap=0.12, hovermode="closest",
        xaxis=dict(title="Your rating minus critic average (stars, snapped to half-star buckets)",
                   tickmode="linear", tick0=0.0, dtick=0.5,
                   range=[lo - 0.4, hi + 0.4], zeroline=False),
        yaxis_title="Titles in bucket",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_delta")
    st.caption(
        "Green = you rated it **above** the critic average · orange = **below** · "
        "grey = level. **Hover any bar to see which titles disagree by that much.** "
        "Buckets are half a star wide, so the grey `0.0` bar is everything within "
        "a quarter star either way — near-agreement, not exact agreement, which is "
        "why it holds more titles than the metrics above would suggest. "
        f"A symmetric shape centred on 0 would mean you and the critics differ at "
        f"random; the centre here sits at {frame['delta'].median():+.2f} stars, so "
        f"there is a systematic offset on top of the spread."
    )


def render_critic_delta(df, title_col, user_col, critic_sources, key):
    """Both bucket charts: the shape of the two rating curves, then the gap between them."""
    frame, labels = build_comparison_frame(df, title_col, user_col, critic_sources)
    if frame.empty:
        st.info("No critic scores available to compare against your ratings.")
        return
    st.markdown("#### Rating curve — yours vs. the critics', bucket by bucket")
    render_score_comparison(frame, labels, key=f"{key}_cmp")
    st.divider()
    st.markdown("#### How big are the disagreements?")
    render_delta_chart(frame, key=f"{key}_d")
