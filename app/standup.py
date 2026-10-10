"""Movies dashboard: the Standup Specials section."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config
from dashboard_common import HUB_PALETTE, update_bar_trace, lb_metric


@st.cache_data
def load_standup():
    """Standup Specials list, written by scripts/fetch_standup_specials.py (your
    rating + TMDB comedian birthday/birthplace + platform). The page never calls
    TMDB itself; a missing CSV means that script has not been run."""
    path = config.MOVIES_PROCESSED_DIR / "standup_specials.csv"
    if not path.exists():
        return None
    d = pd.read_csv(path)
    d['watched_date'] = pd.to_datetime(d['watched_date'], errors='coerce')
    d['platform'] = d['platform'].replace('Unknown', 'Not listed on TMDB')
    d['comedian_country'] = d['comedian_country'].fillna('Unknown')
    # IMDb is out of 10; halve it so it sits on the same 0.5-5 axis as your stars.
    d['imdb_5'] = pd.to_numeric(d['imdb_rating'], errors='coerce') / 2
    d['delta'] = d['user_rating'] - d['imdb_5']
    return d


def standup_group_bar(d, col, title):
    g = (d.dropna(subset=['user_rating']).groupby(col)
          .agg(avg=('user_rating', 'mean'), n=('user_rating', 'size'),
               titles=('name', lambda s: "<br>".join(s)))
          .reset_index().sort_values('avg'))
    fig = go.Figure(go.Bar(
        x=g['avg'], y=g[col], orientation='h', marker_color=HUB_PALETTE[1],
        text=[f"{a:.2f} · {n}" for a, n in zip(g['avg'], g['n'])], textposition='outside',
        customdata=g[['n', 'titles']],
        hovertemplate="<b>%{y}</b><br>Avg %{x:.2f} over %{customdata[0]} special(s)"
                      "<br>%{customdata[1]}<extra></extra>"))
    fig.update_layout(title=title, xaxis=dict(range=[0, 5.4], title="Your average rating"),
                      yaxis_title=None, height=max(320, 26 * len(g) + 90),
                      margin=dict(l=10, r=10, t=50, b=40))
    return update_bar_trace(fig)


def render_standup():
    df_standup = load_standup()
    st.subheader("🎤 Standup Specials")

    if df_standup is None or df_standup.empty:
        st.info("Standup data not found. Run `python scripts/fetch_standup_specials.py`.")
    else:
        su = df_standup
        su_rated = su.dropna(subset=['user_rating'])

        sk1, sk2, sk3, sk4, sk5 = st.columns(5)
        with sk1: lb_metric("Specials", len(su))
        with sk2: lb_metric("Comedians", su['comedian'].nunique())
        with sk3: lb_metric("Avg Rating", f"{su_rated['user_rating'].mean():.2f}" if not su_rated.empty else "N/A")
        with sk4: lb_metric("Laughing Time", f"{su['runtime'].sum() / 60:.0f} hrs",
                            help_text=f"{su['runtime'].sum():.0f} minutes")
        with sk5:
            su_top = (su_rated.groupby('comedian')['user_rating'].agg(['mean', 'size'])
                              .query('size >= 2').sort_values(['mean', 'size'], ascending=False))
            lb_metric("Top Comedian (2+)", su_top.index[0] if not su_top.empty else "N/A",
                      help_text=f"{su_top['mean'].iloc[0]:.2f} avg over {int(su_top['size'].iloc[0])}"
                      if not su_top.empty else None)

        s_com, s_crowd, s_plat, s_age, s_all = st.tabs(
            ["Comedians", "You vs IMDb", "Platform & Country", "Age & Era", "All Specials"])

        with s_com:
            st.plotly_chart(standup_group_bar(su, 'comedian', "Average rating by comedian (label: avg · specials)"),
                            use_container_width=True)

        with s_crowd:
            su_c = su_rated.dropna(subset=['imdb_5'])
            if su_c.empty:
                st.info("No IMDb ratings for these specials.")
            else:
                fig = px.scatter(su_c, x='imdb_5', y='user_rating', hover_name='name',
                                 custom_data=['imdb_rating', 'delta'],
                                 labels={'imdb_5': "IMDb rating (÷2)", 'user_rating': "Your rating"},
                                 title="Your stars vs IMDb (above the line = you liked it more)")
                fig.update_traces(marker=dict(size=11, color=HUB_PALETTE[1], line=dict(width=2, color="#14171C")),
                                  hovertemplate="<b>%{hovertext}</b><br>You %{y} · IMDb %{customdata[0]}"
                                                "<br>Gap %{customdata[1]:+.2f}<extra></extra>")
                fig.add_shape(type='line', x0=0.5, y0=0.5, x1=5, y1=5,
                              line=dict(color="#556678", dash='dot', width=2))
                fig.update_xaxes(range=[1.5, 5.1]); fig.update_yaxes(range=[1.5, 5.1])
                st.plotly_chart(fig, use_container_width=True)
                gap_cols = {'name': "Special", 'user_rating': "You", 'imdb_rating': "IMDb", 'delta': "Gap"}
                sa, sb = st.columns(2)
                with sa:
                    st.markdown("**You rated higher**")
                    st.dataframe(su_c.nlargest(5, 'delta')[list(gap_cols)].rename(columns=gap_cols).round(2),
                                 hide_index=True, use_container_width=True)
                with sb:
                    st.markdown("**You rated lower**")
                    st.dataframe(su_c.nsmallest(5, 'delta')[list(gap_cols)].rename(columns=gap_cols).round(2),
                                 hide_index=True, use_container_width=True)

        with s_plat:
            sa, sb = st.columns(2)
            with sa:
                st.plotly_chart(standup_group_bar(su, 'platform', "By platform"), use_container_width=True)
            with sb:
                st.plotly_chart(standup_group_bar(su, 'comedian_country', "By comedian's country"),
                                use_container_width=True)
            st.caption("Platform is the network in TMDB's production companies (HBO, Netflix) when named, "
                       "otherwise where the special streams now (India first, then US). "
                       "Country is the comedian's birthplace, or the special's origin country when TMDB has no birthplace.")

        with s_age:
            sa, sb = st.columns(2)
            with sa:
                su_a = su_rated.dropna(subset=['comedian_age_at_release'])
                fig = px.scatter(su_a, x='comedian_age_at_release', y='user_rating', hover_name='name',
                                 labels={'comedian_age_at_release': "Comedian's age at release", 'user_rating': "Your rating"},
                                 title="Does the comedian's age matter?")
                fig.update_traces(marker=dict(size=11, color=HUB_PALETTE[1], line=dict(width=2, color="#14171C")))
                if len(su_a) >= 3:
                    r = su_a['comedian_age_at_release'].corr(su_a['user_rating'])
                    fig.add_annotation(text=f"r = {r:.2f}", xref='paper', yref='paper', x=0.98, y=0.04,
                                       showarrow=False, font=dict(color="#99AABB"))
                st.plotly_chart(fig, use_container_width=True)
            with sb:
                fig = px.scatter(su_rated, x='year', y='user_rating', size='runtime', hover_name='name',
                                 labels={'year': "Release year", 'user_rating': "Your rating", 'runtime': "Runtime (min)"},
                                 title="Release year (dot size = runtime)")
                fig.update_traces(marker=dict(color=HUB_PALETTE[2], line=dict(width=2, color="#14171C")))
                st.plotly_chart(fig, use_container_width=True)

        with s_all:
            su_show = su.sort_values('user_rating', ascending=False)[
                ['name', 'comedian', 'year', 'user_rating', 'imdb_rating', 'runtime', 'platform',
                 'comedian_country', 'watched_date']].copy()
            su_show['watched_date'] = su_show['watched_date'].dt.date
            st.dataframe(su_show.rename(columns={
                'name': "Special", 'comedian': "Comedian", 'year': "Year", 'user_rating': "You",
                'imdb_rating': "IMDb", 'runtime': "Min", 'platform': "Platform",
                'comedian_country': "Country", 'watched_date': "Watched"}),
                hide_index=True, use_container_width=True)
            st.markdown("**Top rated**")
            su_posters = su_rated.dropna(subset=['poster']).sort_values(['user_rating', 'imdb_rating'], ascending=False).head(6)
            for col, (_, r) in zip(st.columns(6), su_posters.iterrows()):
                with col:
                    st.image("https://image.tmdb.org/t/p/w342" + r['poster'], use_container_width=True)
                    st.caption(f"{r['name']} · ★{r['user_rating']}")
