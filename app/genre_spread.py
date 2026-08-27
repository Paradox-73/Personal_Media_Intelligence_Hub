"""Per-genre rating spread, shared by every domain dashboard.

Two changes over the plain `px.box` these pages used to draw:

1. **Violin with an inner box.** Ratings here are half-star and lumpy - a genre can
   hold a fat 4.0 pile and a thin 2.5 tail and still report the same quartiles as a
   flat spread. The violin shows that shape; keeping `box=True` means nothing the
   box plot told you was lost. `spanmode='hard'` stops the kernel from drawing
   density past 0.5 and 5, which are hard limits, not tails.

2. **Outliers that name themselves.** Plotly's own outlier dots carry no identity,
   so "which film is that 0.5 in Comedy?" was unanswerable. These are recomputed
   with the same 1.5 x IQR fence rule and drawn as a separate hoverable layer.
   Titles sharing a genre and a rating land on the same dot - they are listed
   together rather than jittered apart, because jitter would put a dot at a rating
   the title does not have.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

OUTLIER_FILL = "#EAF0F6"
OUTLIER_LINE = "#2C343F"
MAX_TITLES_PER_DOT = 10


def iqr_outliers(exploded, genre_col, rating_col):
    """Rows beyond the 1.5 x IQR fences within their own genre.

    Same rule and same quartile interpolation Plotly uses for the whiskers, so
    these dots sit exactly where Plotly would have drawn its own.
    """
    frames = []
    for _, grp in exploded.groupby(genre_col):
        q1, q3 = grp[rating_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        frames.append(grp[(grp[rating_col] < lo) | (grp[rating_col] > hi)])
    return pd.concat(frames) if frames else exploded.iloc[0:0]


def _label(row, title_col, year_col):
    name = str(row[title_col])
    if year_col and year_col in row.index and pd.notna(row[year_col]):
        try:
            return f"{name} ({int(row[year_col])})"
        except (TypeError, ValueError):
            pass
    return name


def genre_spread_figure(exploded, genre_col, rating_col, title_col, palette,
                        min_count=5, x_title="Genre", y_title="Your Rating",
                        year_col=None, height=740, noun="titles", max_genres=20):
    """Violin + inner box + hoverable outlier dots. None when nothing qualifies."""
    exploded = exploded.dropna(subset=[genre_col, rating_col]).copy()
    exploded = exploded[~exploded[genre_col].astype(str).isin(['', 'nan', 'None', 'Unknown'])]
    if exploded.empty:
        return None

    counts = exploded[genre_col].value_counts()
    # Violins need horizontal room. Book categories in particular run to 50+ labels,
    # which squeezes every violin into an unreadable sliver, so keep the busiest.
    keep = counts[counts >= min_count]
    if max_genres:
        keep = keep.head(max_genres)
    data = exploded[exploded[genre_col].isin(keep.index)]
    if data.empty:
        return None

    order = list(keep.index)
    fig = px.violin(data, x=genre_col, y=rating_col, color=genre_col, box=True, points=False,
                    color_discrete_sequence=palette, category_orders={genre_col: order})
    # Ratings cannot run past the ends of the scale, so neither should the density.
    fig.update_traces(spanmode='hard', width=0.85, meanline_visible=True,
                      hoveron='violins', scalemode='width')

    outliers = iqr_outliers(data, genre_col, rating_col)
    if not outliers.empty and title_col in outliers.columns:
        outliers = outliers.assign(
            _label=outliers.apply(lambda r: _label(r, title_col, year_col), axis=1))

        def stack(labels):
            names = sorted(dict.fromkeys(labels))
            shown = "<br>".join(f"• {n}" for n in names[:MAX_TITLES_PER_DOT])
            extra = len(names) - MAX_TITLES_PER_DOT
            return shown + (f"<br>• +{extra} more" if extra > 0 else "")

        stacked = (outliers.groupby([genre_col, rating_col])['_label']
                   .agg([('names', stack), ('n', 'size')]).reset_index())
        fig.add_trace(go.Scatter(
            x=stacked[genre_col], y=stacked[rating_col], mode='markers',
            marker=dict(size=(7 + 2.5 * (stacked['n'] - 1)).clip(upper=18),
                        color=OUTLIER_FILL, line=dict(width=1.4, color=OUTLIER_LINE),
                        opacity=0.95),
            customdata=stacked[['names', 'n']].values,
            hovertemplate=("<b>%{x}</b> · ⭐ %{y}<br>"
                           "%{customdata[1]} outlier " + noun + "<br>%{customdata[0]}<extra></extra>"),
            hoverlabel=dict(align="left"), showlegend=False, name="Outliers",
        ))

    fig.update_layout(showlegend=False, xaxis_title=x_title, yaxis_title=y_title,
                      height=height, violingap=0.25, margin=dict(l=0, r=0, t=40, b=0))
    return fig


SPREAD_CAPTION = ("Width shows where your ratings pile up; the inner box keeps the median and "
                  "quartiles. Pale dots are outliers (beyond 1.5 × IQR) — hover one to see which "
                  "{noun} it is.")
