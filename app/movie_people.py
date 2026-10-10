"""Movies dashboard: cast, crew and studio analysis (moved out of 1_Movies_Dashboard.py)."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src.common.parsing import parse_list
from dashboard_common import HUB_PALETTE, update_bar_trace, lb_metric, render_people_tab


# Every credited person -> the films they worked on and in which role. Uses the
# full cast list (actors_list is cut to the first 10 billed for the charts), plus
# the crew columns ingestion stores as plain strings.
PEOPLE_ROLES = {'Director': 'director', 'Actor': 'actors', 'Writer': 'writer', 'Composer': 'composer',
                'Cinematographer': 'cinematographer', 'Editor': 'editor', 'Producer': 'producer'}

@st.cache_data
def build_people_index(frame):
    rows = []
    for role, col in PEOPLE_ROLES.items():
        if col not in frame.columns:
            continue
        for idx, val in frame[col].items():
            for name in dict.fromkeys(n.strip() for n in parse_list(val) if str(n).strip()):
                rows.append((name, role, idx))
    return pd.DataFrame(rows, columns=['person', 'role', 'idx'])


def render_movie_people(df):
    st.subheader("👥 Cast, Crew & Studios Analysis")
    tab_search, tab_dir, tab_act, tab_wri, tab_stu = st.tabs(
        ["🔎 Search Cast & Crew", "🎬 Directors", "🎭 Actors", "✍️ Writers", "🏢 Studios"])

    with tab_search:
        people = build_people_index(df[list({c for c in PEOPLE_ROLES.values() if c in df.columns})])
        film_counts = people.groupby('person')['idx'].nunique().sort_values(ascending=False)
        # One searchable dropdown: Streamlit filters the options as you type, so the
        # list narrows letter by letter instead of search-then-pick.
        pick = st.selectbox(
            "Search cast & crew (actor, director, writer, composer, cinematographer, editor, producer)",
            film_counts.index.tolist(), index=None, key="people_pick",
            placeholder="Start typing a name, e.g. Ryan Gosling",
            format_func=lambda n: f"{n}  ({film_counts[n]} films)")
        if pick is None:
            st.caption(f"{len(film_counts):,} people across your {len(df):,} films. Most frequent: "
                       + ", ".join(f"{n} ({c})" for n, c in film_counts.head(8).items()))
        else:
            mine = people[people['person'] == pick]
            films = df.loc[mine['idx'].unique()].copy()
            films['roles'] = films.index.map(mine.groupby('idx')['role'].apply(lambda s: ", ".join(sorted(set(s)))))
            rated = films.dropna(subset=['user_rating'])
            overall = df['user_rating'].mean()

            pk1, pk2, pk3, pk4, pk5, pk6 = st.columns(6)
            with pk1: lb_metric("Films", len(films), help_text=f"{len(rated)} rated")
            with pk2: lb_metric("Avg Rating", f"{rated['user_rating'].mean():.2f}" if not rated.empty else "N/A",
                                help_text=f"{rated['user_rating'].mean() - overall:+.2f} vs your overall {overall:.2f}"
                                if not rated.empty else None)
            with pk3: lb_metric("Median", f"{rated['user_rating'].median():g}" if not rated.empty else "N/A")
            with pk4: lb_metric("Min / Max", f"{rated['user_rating'].min():g} / {rated['user_rating'].max():g}"
                                if not rated.empty else "N/A")
            with pk5: lb_metric("Liked", f"{films['is_liked'].fillna(0).mean() * 100:.0f}%",
                                help_text=f"{int(films['is_liked'].fillna(0).sum())} of {len(films)} films hearted")
            with pk6:
                imdb5 = rated['imdb_rating'] / 2
                gap = (rated['user_rating'] - imdb5).mean()
                lb_metric("You vs IMDb", f"{gap:+.2f}" if imdb5.notna().any() else "N/A",
                          help_text="Your stars minus IMDb ÷ 2, averaged over their films. Positive = you like them more than the crowd.")

            yrs = films['year'].dropna()
            # Rank among everyone with at least as many rated films, so one 5★ film
            # does not top the chart.
            rated_people = (people.merge(df[['user_rating']], left_on='idx', right_index=True)
                                  .dropna(subset=['user_rating']).drop_duplicates(['person', 'idx'])
                                  .groupby('person')['user_rating'].agg(['mean', 'size']))
            peers = rated_people[rated_people['size'] >= max(3, min(len(rated), 5))]
            rank_txt = ""
            if pick in peers.index and len(peers) > 1:
                pct = (peers['mean'] < peers.loc[pick, 'mean']).mean() * 100
                rank_txt = f" · rated higher than {pct:.0f}% of the {len(peers):,} people with {peers['size'].min()}+ rated films"
            best = rated.loc[rated['user_rating'].idxmax()] if not rated.empty else None
            worst = rated.loc[rated['user_rating'].idxmin()] if not rated.empty else None
            st.caption(
                (f"Active {int(yrs.min())}–{int(yrs.max())} in your library" if not yrs.empty else "")
                + (f" · best: **{best['title']}** ({best['user_rating']:g}★) · worst: **{worst['title']}** ({worst['user_rating']:g}★)"
                   if best is not None else "") + rank_txt)

            pa, pb = st.columns([3, 2])
            with pa:
                fig = px.scatter(rated, x='year', y='user_rating', hover_name='title', color='roles',
                                 custom_data=['imdb_rating', 'roles'],
                                 labels={'year': "Release year", 'user_rating': "Your rating", 'roles': "Role"},
                                 title=f"{pick}: your ratings by release year",
                                 color_discrete_sequence=HUB_PALETTE)
                fig.update_traces(marker=dict(size=11, line=dict(width=2, color="#14171C")),
                                  hovertemplate="<b>%{hovertext}</b> (%{x})<br>You %{y}★ · IMDb %{customdata[0]}"
                                                "<br>%{customdata[1]}<extra></extra>")
                fig.add_hline(y=overall, line=dict(color="#556678", dash='dot'),
                              annotation_text=f"your average {overall:.2f}", annotation_position="bottom right")
                fig.update_yaxes(range=[0, 5.3])
                st.plotly_chart(fig, use_container_width=True)
            with pb:
                by_role = (mine.merge(df[['user_rating']], left_on='idx', right_index=True)
                               .groupby('role')['user_rating'].agg(['size', 'mean', 'min', 'max']).reset_index())
                st.markdown("**By role**")
                st.dataframe(by_role.rename(columns={'role': "Role", 'size': "Films", 'mean': "Avg ★",
                                                     'min': "Min ★", 'max': "Max ★"}).round(2),
                             hide_index=True, use_container_width=True)
                dist = rated['user_rating'].value_counts().sort_index().reset_index()
                dist.columns = ['Rating', 'Films']
                fig = px.bar(dist, x='Rating', y='Films', title="Rating spread", color_discrete_sequence=[HUB_PALETTE[1]])
                fig.update_layout(height=260, margin=dict(t=40, b=10), xaxis=dict(dtick=0.5))
                st.plotly_chart(update_bar_trace(fig), use_container_width=True)

            show_cols = {'title': "Film", 'year': "Year", 'roles': "Role", 'user_rating': "You",
                         'imdb_rating': "IMDb", 'is_liked': "Liked", 'genre': "Genre"}
            st.dataframe(films.sort_values(['user_rating', 'year'], ascending=[False, True])[list(show_cols)]
                         .assign(is_liked=lambda t: t['is_liked'].map({1: "♥", 1.0: "♥"}).fillna(""),
                                 genre=lambda t: t['genre'].apply(lambda g: ", ".join(parse_list(g))),
                                 year=lambda t: t['year'].astype('Int64'))
                         .rename(columns=show_cols), hide_index=True, use_container_width=True)

    with tab_dir: render_people_tab(df, 'director_list', "Directors", 4, "Movies")
    with tab_act: render_people_tab(df, 'actors_list', "Actors", 7, "Movies")
    with tab_wri: render_people_tab(df, 'writer_list', "Writers", 3, "Movies")
    with tab_stu: render_people_tab(df, 'production_list', "Studios", 8, "Movies")
