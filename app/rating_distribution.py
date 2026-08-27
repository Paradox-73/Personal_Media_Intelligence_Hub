"""Shared "my ratings vs. the critics" distribution block.

Every domain already showed how the user's own stars are spread. The obvious
follow-up question - *is that how the critics see the same library?* - was only
answerable through the scatter plots further down each page, which show alignment
per title but never the shape of the two distributions.

The obstacle is that no two critic sources share a scale: IMDb and TMDB are 0-10,
Metacritic and Rotten Tomatoes are 0-100, RAWG and Hardcover are already 0-5. A
critic histogram is only readable against a Letterboxd-style star chart if it is
rescaled onto the same 0.5-5 axis and snapped to the same half-star grid, so that
is what `to_stars` does.

The two views deliberately never render side by side: they answer the same
question on the same axis, and putting them next to each other invites reading a
bar-height difference that is really just a different denominator (critic columns
have their own missing values). Tabs keep the axis shared and the comparison
honest, with the mean delta called out in words underneath.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.common.parsing import round_half


def parse_percent(x):
    """'92%' -> 92.0. Rotten Tomatoes is the only source stored as a string."""
    if pd.isna(x):
        return None
    s = str(x).strip().rstrip('%')
    try:
        return float(s)
    except ValueError:
        return None


def to_stars(series, scale_max):
    """Rescale a critic column onto the 0.5-5 star grid the user rates on."""
    s = pd.to_numeric(series, errors='coerce').dropna()
    s = s[s > 0]  # 0 means "nobody scored it", not "everybody hated it"
    if s.empty:
        return s
    return pd.Series(round_half(s / float(scale_max) * 5.0), index=s.index)


def _bar(counts, color, style_fn=None, show_text=True):
    fig = px.bar(counts.sort_values('Rating'), x='Rating', y='Count',
                 text='Count' if show_text else None,
                 color_discrete_sequence=[color])
    if style_fn is not None:
        style_fn(fig)
    if show_text:
        fig.update_traces(textposition='outside')
    fig.update_layout(xaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5, range=[0.25, 5.25]),
                      margin=dict(l=0, r=0, t=10, b=0))
    return fig


def _counts(values):
    c = pd.Series(values).value_counts().reset_index()
    c.columns = ['Rating', 'Count']
    return c


def render_rating_distribution(user_ratings, critic_sources, key,
                               user_color='#00e054', critic_color='#40bcf4',
                               style_fn=None, show_text=True):
    """Two tabs over one axis: the user's own stars, then any critic source.

    critic_sources: list of (label, series, scale_max). A source whose column is
    absent or entirely empty is dropped rather than shown as an empty chart.
    """
    user_ratings = pd.to_numeric(pd.Series(user_ratings), errors='coerce').dropna()

    usable = []
    for label, series, scale_max in critic_sources:
        if series is None:
            continue
        stars = to_stars(series, scale_max)
        if not stars.empty:
            usable.append((label, stars))

    t_mine, t_critics = st.tabs(["⭐ My Ratings", "🎬 Critics"])

    with t_mine:
        st.plotly_chart(_bar(_counts(user_ratings), user_color, style_fn, show_text),
                        use_container_width=True, key=f"{key}_mine")

    with t_critics:
        if not usable:
            st.info("No critic scores available for this library.")
            return
        if len(usable) > 1:
            pick = st.radio("Critic source", [lbl for lbl, _ in usable],
                            horizontal=True, key=f"{key}_src", label_visibility="collapsed")
        else:
            pick = usable[0][0]
        stars = dict(usable)[pick]

        st.plotly_chart(_bar(_counts(stars), critic_color, style_fn, show_text),
                        use_container_width=True, key=f"{key}_critics")
        delta = user_ratings.mean() - stars.mean()
        direction = "higher" if delta >= 0 else "lower"
        st.caption(
            f"**{pick}**, rescaled to stars and snapped to the half-star grid "
            f"({len(stars)} of {len(user_ratings)} titles scored). "
            f"Critic mean ⭐{stars.mean():.2f} · your mean ⭐{user_ratings.mean():.2f} — "
            f"you rate this library **{abs(delta):.2f} stars {direction}** on average."
        )
