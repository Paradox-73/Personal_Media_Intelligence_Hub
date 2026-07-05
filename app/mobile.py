"""Shared mobile/responsive CSS for every page in the hub.

Streamlit's `layout="wide"` keeps multi-column rows side-by-side on phones, which
squeezes metric cards, the 3-column Oracle layout, and 2-up chart rows down to
unreadable widths. `apply_mobile_css()` injects one media-query block (≤640px) that
stacks every column to full width, trims page padding, shrinks headings, and lets
the many-tab dashboards scroll horizontally instead of cramming.

Call it once per page, right after `st.set_page_config(...)` (it uses `st.markdown`,
so it must come after set_page_config but before the rest of the page). Test-ids
(`stHorizontalBlock`, `stColumn`, `stMainBlockContainer`) are verified against the
installed Streamlit 1.54 build.
"""
import streamlit as st

_MOBILE_CSS = """
<style>
/* Touch-scroll fix (all touch devices, not just phones): when a Plotly chart fills
   the screen, a vertical finger-drag is otherwise captured by the chart as a
   pan/box-select/zoom, so the page won't scroll past it. `touch-action: pan-y`
   hands vertical swipes back to the browser — but Plotly attaches its own touch
   handlers that call preventDefault(), so this hint alone isn't always honored.
   The phone rule below (pointer-events:none) is the guaranteed fix. */
.stPlotlyChart, .js-plotly-plot, .js-plotly-plot .plot-container,
.js-plotly-plot .nsewdrag, .js-plotly-plot .drag {
    touch-action: pan-y !important;
}

@media (max-width: 640px) {
    /* GUARANTEED scroll-past-graph fix: make Plotly charts inert to touch on phones
       so a swipe falls straight through to the page scroller instead of being
       captured by the chart as zoom/pan/box-select. Plotly's own preventDefault()
       beats `touch-action`, so we drop pointer-events entirely here. Cost: charts
       are static images on a phone (no tap-hover/zoom); desktop stays fully
       interactive since this is inside the ≤640px media query. */
    .stPlotlyChart, .stPlotlyChart *,
    [data-testid="stPlotlyChart"], [data-testid="stPlotlyChart"] * {
        pointer-events: none !important;
        touch-action: pan-y !important;
    }

    /* Reclaim the side padding so content uses the full screen width. */
    [data-testid="stMainBlockContainer"] {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
    }

    /* Stack every column layout vertically: charts, the Oracle's image/meta/verdict
       split, search-result rows and metric grids all get the full width. */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0.6rem !important;
    }
    [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Smaller, tighter headings so titles don't wrap to four lines. */
    h1 { font-size: 1.55rem !important; line-height: 1.25 !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.08rem !important; }
    h4 { font-size: 1.0rem !important; }

    /* Many-tab bars (Movies/TV dashboards) scroll sideways instead of squashing. */
    [data-baseweb="tab-list"] {
        overflow-x: auto !important;
        flex-wrap: nowrap !important;
        scrollbar-width: thin;
    }
    [data-baseweb="tab-list"] [data-baseweb="tab"] {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
    }

    /* Keep wide tables and charts inside the viewport. */
    [data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto !important; }
    .stPlotlyChart, .js-plotly-plot, [data-testid="stImage"] img { max-width: 100% !important; }

    /* Metric numbers that would otherwise clip on a narrow card. */
    [data-testid="stMetricValue"] { font-size: 1.4rem !important; }
}
</style>
"""


def apply_mobile_css():
    """Inject the responsive stylesheet. Safe to call once per page run."""
    st.markdown(_MOBILE_CSS, unsafe_allow_html=True)
