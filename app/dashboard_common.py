"""Theme and small helpers shared by the Movies, TV, Games and Collaboration pages."""
import math
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

HUB_PALETTE = ["#ff8000", "#00e054", "#40bcf4", "#556678", "#2C343F"]

_HUB_CSS = """
<style>
    /* Font Fallbacks */
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
    .stTabs [data-baseweb="tab-list"] { background-color: #14171C; }
    .stTabs [data-baseweb="tab"] { color: #E0E0E0; }
    .stTabs [aria-selected="true"] {
        color: #ff8000 !important;
        border-bottom-color: #ff8000 !important;
    }
    hr { border-top: 1px solid #556678; }
</style>
"""


def apply_hub_theme(bargap=None):
    """Page CSS plus the default Plotly template. bargap=0.05 joins histogram bars."""
    st.markdown(_HUB_CSS, unsafe_allow_html=True)
    pio.templates["hub_theme"] = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E0E0E0", family="Graphik, Helvetica, sans-serif"),
            colorway=HUB_PALETTE,
            xaxis=dict(gridcolor="#2C343F", zerolinecolor="#2C343F", tickfont=dict(color="#556678")),
            yaxis=dict(gridcolor="#2C343F", zerolinecolor="#2C343F", tickfont=dict(color="#556678")),
            bargap=bargap,
        )
    )
    pio.templates.default = "hub_theme"


def update_bar_trace(fig):
    fig.update_traces(marker_line_width=1, marker_line_color="#14171C")
    return fig


def star_text(rating):
    """Plain-text Letterboxd stars ("★★★½"); "Not Rated" for a missing rating."""
    if pd.isna(rating):
        return "Not Rated"
    return "★" * int(rating) + ("½" if (rating % 1) >= 0.5 else "")


def render_stars(rating):
    """Letterboxd-style stars as an HTML span; empty for a missing rating."""
    if pd.isna(rating):
        return ""
    return f"<span style='color:#00e054; font-family: courier; font-size: 1.2em;'>{star_text(rating)}</span>"


def lb_metric(label, value, help_text=None):
    tooltip = f'title="{help_text}"' if help_text else ""
    cursor = "help" if help_text else "default"
    st.markdown(f"""
    <div {tooltip} style="background-color: #2C343F; padding: 10px 15px; border-radius: 4px; border-bottom: 2px solid #556678; cursor: {cursor};">
        <div style="color: #99AABB; font-size: 0.8em; text-transform: uppercase; font-weight: bold; letter-spacing: 0.05em;">{label}</div>
        <div style="color: #FFFFFF; font-size: 1.8em; font-weight: 500;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


_BLANK = ['unknown', 'nan', 'none', '']


def calculate_entropy(series):
    """Shannon entropy in bits over list or scalar cells, ignoring blank labels."""
    items = []
    for val in series.dropna():
        vals = val if isinstance(val, list) else [val]
        items.extend(str(i).strip() for i in vals if str(i).strip().lower() not in _BLANK)
    if not items:
        return 0
    total = len(items)
    return -sum((c / total) * math.log2(c / total) for c in Counter(items).values())


def _exploded(df, col_name):
    d = df.explode(col_name)
    return d[~d[col_name].isin(['Unknown', 'nan', np.nan, ''])]


def get_frequent_items_with_avg(df, col_name, top_n=10):
    stats = _exploded(df, col_name).groupby(col_name).agg(
        Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('Count', ascending=False).head(top_n)


def _rated_items(df, col_name, top_n, min_count, ascending):
    stats = _exploded(df, col_name).groupby(col_name).agg(
        avg_rating=('user_rating', 'mean'), count=('user_rating', 'count')).reset_index()
    stats = stats[stats['count'] >= min_count]
    stats['avg_rating'] = stats['avg_rating'].round(3)
    return stats.sort_values('avg_rating', ascending=ascending).head(top_n)


def get_highly_rated_items(df, col_name, top_n=10, min_count=3):
    return _rated_items(df, col_name, top_n, min_count, ascending=False)


def get_lowest_rated_items(df, col_name, top_n=10, min_count=3):
    return _rated_items(df, col_name, top_n, min_count, ascending=True)


def render_people_tab(df, col_list, label, min_count_threshold, noun):
    """Most-watched, highest- and lowest-rated people for one credit column."""
    ent = calculate_entropy(df[col_list])
    st.caption(f"**{label} Diversity (Entropy):** {ent:.2f}")

    c_freq, c_high, c_low = st.columns(3)
    with c_freq:
        st.caption(f"**Most Watched {label}**")
        freq = get_frequent_items_with_avg(df, col_list)
        fig = px.bar(freq, x='Count', y=col_list, orientation='h', color='Count', hover_data={'avg_rating': ':.3f'},
                     color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]])
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        update_bar_trace(fig)
        st.plotly_chart(fig, use_container_width=True)
    with c_high:
        st.caption(f"**Highest Rated {label}** (Min {min_count_threshold} {noun})")
        rated = get_highly_rated_items(df, col_list, min_count=min_count_threshold)
        if not rated.empty:
            fig = px.bar(rated, x='avg_rating', y=col_list, orientation='h', color='avg_rating',
                         color_continuous_scale=[[0, HUB_PALETTE[1]], [1, HUB_PALETTE[2]]], range_x=[0, 5],
                         hover_data={'count': True})
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data")
    with c_low:
        st.caption(f"**Lowest Rated {label}** (Min {min_count_threshold} {noun})")
        low_rated = get_lowest_rated_items(df, col_list, min_count=min_count_threshold)
        if not low_rated.empty:
            fig = px.bar(low_rated, x='avg_rating', y=col_list, orientation='h', color='avg_rating',
                         color_continuous_scale=[[0, "#ff4b4b"], [1, HUB_PALETTE[0]]], range_x=[0, 5],
                         hover_data={'count': True})
            fig.update_layout(yaxis={'categoryorder': 'total descending'})
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data")
