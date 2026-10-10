"""Movies dashboard: your Letterboxd lists tagged director / franchise / studio."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from dashboard_common import HUB_PALETTE, update_bar_trace, lb_metric


@st.cache_data
def load_tagged_lists():
    """One row per film per list. The list's tag ("director", "franchise",
    "studio") sits in the metadata row above the film table of each export."""
    rows = []
    for path in sorted((config.DATA_DIR / "raw" / "movies" / "lists").glob("*.csv")):
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        header = next((i for i, ln in enumerate(lines) if ln.startswith('Position,')), None)
        if header is None or len(lines) < 3:
            continue
        meta = pd.read_csv(path, nrows=1, skiprows=1)
        tag = str(meta['Tags'].iloc[0]).strip().lower() if 'Tags' in meta else ''
        if tag not in ('director', 'franchise', 'studio'):
            continue
        films = pd.read_csv(path, skiprows=header)
        films['list_name'] = str(meta['Name'].iloc[0]).strip()
        films['list_tag'] = tag
        rows.append(films[['list_name', 'list_tag', 'Position', 'Name', 'Year', 'URL']])
    return pd.concat(rows, ignore_index=True) if rows else None


def render_ranked_lists(df):
    df_lists = load_tagged_lists()
    st.subheader("🏆 Director, Franchise & Studio Lists")
    if df_lists is None:
        st.info("No lists tagged director / franchise / studio found in `data/raw/movies/lists/`.")
    else:
        dl = df_lists.merge(df[['letterboxd_uri', 'user_rating', 'is_liked', 'poster']],
                            left_on='URL', right_on='letterboxd_uri', how='left')
        list_stats = (dl.groupby(['list_tag', 'list_name'])
                        .agg(films=('Name', 'size'), rated=('user_rating', 'count'),
                             avg=('user_rating', 'mean'), median=('user_rating', 'median'),
                             min=('user_rating', 'min'), max=('user_rating', 'max'),
                             liked=('is_liked', 'mean'),
                             number_one=('Name', lambda s: s.iloc[0]))
                        .reset_index())
        # Does your ranking agree with your stars? Spearman between list position and
        # rating; negative = films you rank higher (position 1) also get more stars.
        list_stats['rank_vs_stars'] = [
            (-g['Position'].corr(g['user_rating'], method='spearman')
             if g['user_rating'].notna().sum() >= 3 and g['user_rating'].nunique() > 1 else np.nan)
            for _, g in dl.groupby(['list_tag', 'list_name'])]

        def best_of(tag):
            part = list_stats[(list_stats['list_tag'] == tag) & (list_stats['rated'] >= 2)]
            if part.empty:
                return "N/A", None
            top = part.sort_values(['avg', 'rated'], ascending=False).iloc[0]
            return top['list_name'], f"{top['avg']:.2f} avg over {int(top['rated'])} films"

        lk1, lk2, lk3, lk4 = st.columns(4)
        with lk1:
            name, tip = best_of('franchise')
            lb_metric("Highest Rated Franchise", name, help_text=tip)
        with lk2:
            name, tip = best_of('director')
            lb_metric("Highest Rated Director", name, help_text=tip)
        with lk3:
            name, tip = best_of('studio')
            lb_metric("Highest Rated Studio", name, help_text=tip)
        with lk4:
            lb_metric("Lists Ranked", len(list_stats),
                      help_text=f"{dl['URL'].nunique()} distinct films across them")

        tab_lfr, tab_ldir, tab_lstu, tab_lone = st.tabs(["Franchises", "Directors", "Studios", "One List"])
        for tab, tag in ((tab_lfr, 'franchise'), (tab_ldir, 'director'), (tab_lstu, 'studio')):
            with tab:
                part = list_stats[list_stats['list_tag'] == tag].sort_values('avg')
                if part.empty:
                    st.info(f"No lists tagged {tag}.")
                    continue
                fig = go.Figure()
                for _, r in part.iterrows():
                    fig.add_shape(type='line', x0=r['min'], x1=r['max'], y0=r['list_name'], y1=r['list_name'],
                                  line=dict(color="#556678", width=4))
                fig.add_trace(go.Scatter(
                    x=part['avg'], y=part['list_name'], mode='markers+text', name="Average",
                    text=[f"{a:.2f}" for a in part['avg']], textposition='middle right',
                    marker=dict(size=13, color=HUB_PALETTE[0], line=dict(width=2, color="#14171C")),
                    customdata=part[['rated', 'films', 'median', 'min', 'max', 'number_one']].values,
                    hovertemplate="<b>%{y}</b><br>Avg %{x:.2f} · median %{customdata[2]}"
                                  "<br>Range %{customdata[3]}–%{customdata[4]}"
                                  "<br>%{customdata[0]} of %{customdata[1]} films rated"
                                  "<br>Your #1: %{customdata[5]}<extra></extra>"))
                fig.add_trace(go.Scatter(x=part['median'], y=part['list_name'], mode='markers', name="Median",
                                         marker=dict(size=9, symbol='diamond-open', color=HUB_PALETTE[2],
                                                     line=dict(width=2, color=HUB_PALETTE[2])),
                                         hovertemplate="%{y}<br>Median %{x}<extra></extra>"))
                fig.update_layout(title=f"Your rating by {tag} list (dot = average, line = min to max)",
                                  xaxis=dict(range=[0, 5.5], title="Your rating"), yaxis_title=None,
                                  height=max(320, 32 * len(part) + 110), legend=dict(orientation='h', y=-0.12))
                st.plotly_chart(fig, use_container_width=True)
                tbl = part.sort_values('avg', ascending=False)[
                    ['list_name', 'films', 'rated', 'avg', 'median', 'min', 'max', 'liked', 'number_one', 'rank_vs_stars']].copy()
                tbl['liked'] = (tbl['liked'] * 100).round(0)
                st.dataframe(tbl.rename(columns={
                    'list_name': tag.title(), 'films': "Films", 'rated': "Rated", 'avg': "Avg ★",
                    'median': "Median ★", 'min': "Min ★", 'max': "Max ★", 'liked': "% Liked",
                    'number_one': "Your #1", 'rank_vs_stars': "Rank agrees with stars"}).round(2),
                    hide_index=True, use_container_width=True)
                st.caption("\"Rank agrees with stars\" runs from −1 to +1: +1 means your list order matches "
                           "your star ratings exactly, 0 means no relation. Needs 3+ rated films.")

        with tab_lone:
            pick = st.selectbox("List", sorted(list_stats['list_name']), key="ranked_list_pick")
            one = dl[dl['list_name'] == pick].sort_values('Position')
            fig = go.Figure(go.Bar(
                x=one['Position'].astype(str) + ". " + one['Name'], y=one['user_rating'],
                marker_color=HUB_PALETTE[1],
                text=one['user_rating'].map(lambda v: f"{v:g}" if pd.notna(v) else "–"), textposition='outside',
                hovertemplate="%{x}<br>Your rating %{y}<extra></extra>"))
            fig.update_layout(title=f"{pick}: your ranking (left = #1) and your stars",
                              yaxis=dict(range=[0, 5.5], title="Your rating"), xaxis_title=None)
            st.plotly_chart(update_bar_trace(fig), use_container_width=True)
