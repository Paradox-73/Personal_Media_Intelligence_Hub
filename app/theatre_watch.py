"""Movies dashboard: the Theatre Watch section (list join, per-trip rows, tabs)."""
import re
from collections import Counter

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from theatre_tags import render_theatre_tag_analytics
from release_gap import render_release_gap
from dashboard_common import (HUB_PALETTE, update_bar_trace, lb_metric, render_stars, star_text,
                              calculate_entropy, get_frequent_items_with_avg)


@st.cache_data
def load_theatre_watch():
    try:
        path = config.DATA_DIR / "raw" / "movies" / "lists" / "theatre-watch.csv"
        if not path.exists(): return None
        # Letterboxd list exports put list metadata above the film table. The
        # separator line between them is blank in some exports and ",,,," in
        # others, so a fixed skiprows lands on the wrong row half the time —
        # find the real header instead.
        with open(path, encoding='utf-8') as fh:
            lines = fh.readlines()
        header = next((i for i, ln in enumerate(lines) if ln.startswith('Position,')), 3)
        return pd.read_csv(path, skiprows=header)
    except Exception as e:
        print(f"Error loading theatre watch list: {e}")
        return None


def render_theatre_watch(df, df_diary, df_diary_tags, stops):
    df_theatre_raw = load_theatre_watch()
    # Prepare Theatre Watch DF by joining with main df to get ratings and genres
    df_theatre = None
    if df_theatre_raw is not None:
        # Use URL as the primary key for joining
        df_theatre = pd.merge(df_theatre_raw, df[['letterboxd_uri', 'user_rating', 'genre_list', 'poster', 'title',
                                                  'director_list', 'actors_list', 'writer_list', 'production_list',
                                                  'runtime', 'overview', 'tagline', 'plot']],
                              left_on='URL', right_on='letterboxd_uri', how='left')

        # Theatre year: if the film is in your diary, bucket it by the date you actually saw it;
        # otherwise fall back to the film's release year.
        df_theatre['theatre_year'] = df_theatre['Year']
        if df_diary is not None and not df_diary.empty:
            saw_year_by_name = (df_diary.dropna(subset=['Watched Date'])
                                        .groupby('Name')['Watched Date'].max().dt.year)
            mapped_year = df_theatre['Name'].map(saw_year_by_name)
            df_theatre['theatre_year'] = mapped_year.fillna(df_theatre['Year'])
        df_theatre['theatre_year'] = pd.to_numeric(df_theatre['theatre_year'], errors='coerce').astype('Int64')

        # One row per trip, not per film. The list holds each film once, so a film
        # seen twice in a cinema (Obsession, Across the Spider-Verse) counted as one
        # visit. Trips = the film's "theatre"-tagged diary entries, or the "Twice" /
        # "Thrice" note in the list description, whichever is larger; each diary
        # entry is bucketed in the year of that trip.
        theatre_diary = pd.DataFrame(columns=['Name', 'Year', 'Watched Date'])
        if df_diary_tags is not None and not df_diary_tags.empty:
            theatre_diary = df_diary_tags[df_diary_tags['Tags'].fillna('').str.contains('theatre', case=False)]
        visit_rows = []
        for _, r in df_theatre.iterrows():
            dates = theatre_diary.loc[(theatre_diary['Name'] == r['Name']) & (theatre_diary['Year'] == r['Year']),
                                      'Watched Date'].dropna().sort_values().tolist()
            note = str(r.get('Description', '')).lower()
            n_note = 3 if 'thrice' in note else 2 if 'twice' in note else 1
            n = max(len(dates), n_note)
            for i in range(n):
                row = r.copy()
                row['visit_no'] = i + 1
                row['visit_date'] = dates[i] if i < len(dates) else pd.NaT
                if i < len(dates):
                    row['theatre_year'] = dates[i].year
                visit_rows.append(row)
        df_theatre_visits = pd.DataFrame(visit_rows).reset_index(drop=True)
        df_theatre_visits['theatre_year'] = pd.to_numeric(df_theatre_visits['theatre_year'], errors='coerce').astype('Int64')
        visits_per_film = df_theatre_visits.groupby('URL').size()
        df_theatre['visits'] = df_theatre['URL'].map(visits_per_film).fillna(1).astype(int)

        # Filter to only include those that have a rating (meaning they've been watched/enriched)
        df_theatre_watched = df_theatre[df_theatre['user_rating'].notna()].copy()
        if not df_theatre_watched.empty:
            df_theatre_watched['vibe_text'] = df_theatre_watched['overview'].fillna('') + " " + df_theatre_watched['tagline'].fillna('') + " " + df_theatre_watched['plot'].fillna('')

    # 13. THEATRE WATCH ANALYSIS
    if df_theatre is not None:
        st.subheader("🍿 Theatre Watch Analysis")

        if not df_theatre_watched.empty:
            tc1, tc2, tc3, tc4, tc5 = st.columns(5)
            n_rewatch = len(df_theatre_visits) - len(df_theatre)
            rewatched = ", ".join(df_theatre.loc[df_theatre['visits'] > 1, 'Name'])
            with tc1: lb_metric("Theatre Visits", len(df_theatre_visits),
                                help_text=f"{len(df_theatre)} films + {n_rewatch} repeat trips ({rewatched})")
            with tc2: lb_metric("Avg Theatre Rating", f"{df_theatre_watched['user_rating'].mean():.2f}")
            with tc3:
                year_mode = df_theatre_visits['theatre_year'].mode()
                lb_metric("Peak Theatre Year", year_mode.iloc[0] if not year_mode.empty else "N/A")
            with tc5:
                # Same Shannon entropy over genres as the "Taste Diversity" tile at the
                # top of the page, restricted to films seen in a cinema.
                t_entropy = calculate_entropy(df_theatre_watched['genre_list'])
                lb_metric("Theatre Entropy", f"{t_entropy:.2f}",
                          help_text=f"Genre diversity of theatre films (Shannon entropy). "
                                    f"All films: {calculate_entropy(df['genre_list']):.2f}. Higher = more varied.")
            with tc4:
                # Extract most common genre from watched theatre movies
                all_theatre_genres = [g for sublist in df_theatre_watched['genre_list'].dropna() for g in sublist]
                top_theatre_genre = Counter(all_theatre_genres).most_common(1)[0][0] if all_theatre_genres else "N/A"
                lb_metric("Top Theatre Genre", top_theatre_genre)

            st.markdown("#### Theatre Rating Distribution")
            t_rating_counts = df_theatre_watched['user_rating'].value_counts().reset_index()
            t_rating_counts.columns = ['Rating', 'Count']
            fig_t_ratings = px.bar(t_rating_counts.sort_values('Rating'), x='Rating', y='Count',
                                 text='Count', color_discrete_sequence=['#ff8000'])
            fig_t_ratings.update_traces(textposition='outside')
            fig_t_ratings.update_layout(xaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5))
            st.plotly_chart(fig_t_ratings, use_container_width=True)

            t_tab1, t_tab2, t_tab3, t_tab_gap, t_tab4, t_tab5, t_tab6, t_tab7 = st.tabs(["Year-wise", "Genres", "Tags", "Release Gap", "Cast & Crew", "Studios & Runtime", "Themes", "All Watches"])

            with t_tab1:
                st.caption("Counts trips, so a film seen twice counts twice. Trips in your diary are counted in the year you actually went; films not in the diary use their release year.")
                df_t_year = df_theatre_visits.dropna(subset=['theatre_year']).groupby('theatre_year').size().reset_index(name='Count')
                df_t_year['theatre_year'] = df_t_year['theatre_year'].astype(int)
                fig_t_year = px.bar(df_t_year, x='theatre_year', y='Count', title="Theatre Watches per Year",
                                    labels={'theatre_year': 'Year'}, color_discrete_sequence=[HUB_PALETTE[2]])
                update_bar_trace(fig_t_year)
                # Fix year axis to show every year
                fig_t_year.update_layout(xaxis=dict(tickmode='linear', dtick=1))
                st.plotly_chart(fig_t_year, use_container_width=True)

            with t_tab2:
                if all_theatre_genres:
                    t_gen_stats = Counter(all_theatre_genres).most_common(15)
                    df_t_gen = pd.DataFrame(t_gen_stats, columns=['Genre', 'Count'])
                    fig_t_gen = px.bar(df_t_gen, x='Count', y='Genre', orientation='h', title="Theatre Watch Genres", color='Count', color_continuous_scale='Oranges')
                    fig_t_gen.update_layout(yaxis={'categoryorder':'total ascending'})
                    update_bar_trace(fig_t_gen)
                    st.plotly_chart(fig_t_gen, use_container_width=True)
                else:
                    st.info("No genre data available for theatre watches.")

            with t_tab3:
                st.markdown("**Theatre Tag Analysis**")
                # The description is a comma-separated bag mixing six dimensions
                # (company, people, venue, city, time, format). Splitting it on
                # whitespace as well as commas — as this used to — broke every
                # multi-word tag apart, so 'harsh gupta' and 'harsh modi' became the
                # same person. app/theatre_tags.py splits on commas only and
                # classifies each tag into its own dimension.
                render_theatre_tag_analytics(df_theatre, HUB_PALETTE,
                                             style_fn=update_bar_trace)

            with t_tab_gap:
                # Days between the India release and the diary date of each trip.
                # Release dates are cached by scripts/fetch_theatre_release_dates.py;
                # the dashboard does not call TMDB.
                render_release_gap(df_theatre, df_diary_tags,
                                   config.CACHE_DIR / "tmdb_release_dates.json",
                                   lb_metric, HUB_PALETTE, style_fn=update_bar_trace)

            with t_tab4:
                st.markdown("**Top Cast & Crew in Theatres**")
                tc_c1, tc_c2 = st.columns(2)
                with tc_c1:
                    st.caption("**Most Watched Directors (Theatre)**")
                    t_dir = get_frequent_items_with_avg(df_theatre_watched, 'director_list', 10)
                    fig_t_dir = px.bar(t_dir, x='Count', y='director_list', orientation='h', color='Count', hover_data={'avg_rating': ':.2f'})
                    fig_t_dir.update_layout(yaxis={'categoryorder':'total ascending'})
                    update_bar_trace(fig_t_dir)
                    st.plotly_chart(fig_t_dir, use_container_width=True)
                with tc_c2:
                    st.caption("**Most Watched Actors (Theatre)**")
                    t_act = get_frequent_items_with_avg(df_theatre_watched, 'actors_list', 10)
                    fig_t_act = px.bar(t_act, x='Count', y='actors_list', orientation='h', color='Count', hover_data={'avg_rating': ':.2f'})
                    fig_t_act.update_layout(yaxis={'categoryorder':'total ascending'})
                    update_bar_trace(fig_t_act)
                    st.plotly_chart(fig_t_act, use_container_width=True)

            with t_tab5:
                st.markdown("**Studios & Runtime in Theatres**")
                ts_c1, ts_c2 = st.columns(2)
                with ts_c1:
                    st.caption("**Top Production Studios (Theatre)**")
                    t_stu = get_frequent_items_with_avg(df_theatre_watched, 'production_list', 10)
                    fig_t_stu = px.bar(t_stu, x='Count', y='production_list', orientation='h', color='Count', hover_data={'avg_rating': ':.2f'})
                    fig_t_stu.update_layout(yaxis={'categoryorder':'total ascending'})
                    update_bar_trace(fig_t_stu)
                    st.plotly_chart(fig_t_stu, use_container_width=True)
                with ts_c2:
                    st.caption("**Runtime Distribution (Theatre)**")
                    t_bins = [0, 60, 90, 120, 150, 180, 999]
                    t_labels = ['<60m', '60-89m', '90-119m', '120-149m', '150-179m', '180m+']
                    df_theatre_watched['rt_bin'] = pd.cut(df_theatre_watched['runtime'], bins=t_bins, labels=t_labels)
                    t_rt_agg = df_theatre_watched.groupby('rt_bin', observed=True).agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
                    fig_t_rt = px.bar(t_rt_agg, x='rt_bin', y='Count', hover_data={'avg_rating': ':.2f'}, color_discrete_sequence=[HUB_PALETTE[1]])
                    update_bar_trace(fig_t_rt)
                    st.plotly_chart(fig_t_rt, use_container_width=True)

            with t_tab6:
                st.markdown("**Dominant Theatre Themes**")
                st.caption("Common keywords from the plots of movies you watched in theatres.")
                if not df_theatre_watched.empty:
                    t_word_ratings = {}
                    for _, row in df_theatre_watched.dropna(subset=['vibe_text', 'user_rating']).iterrows():
                        text = str(row['vibe_text'])
                        words = set([w.lower() for w in re.findall(r'\w+', text) if len(w) > 3 and w.lower() not in stops])
                        for w in words:
                            if w not in t_word_ratings: t_word_ratings[w] = []
                            t_word_ratings[w].append(row['user_rating'])

                    t_word_stats = []
                    for w, ratings in t_word_ratings.items():
                        t_word_stats.append({'Word': w, 'Count': len(ratings), 'Avg_Rating': np.mean(ratings)})

                    if t_word_stats:
                        twdf = pd.DataFrame(t_word_stats).sort_values('Count', ascending=False).head(40)
                        fig_t_themes = px.treemap(twdf, path=['Word'], values='Count', color='Avg_Rating',
                                                 color_continuous_scale='RdYlGn', range_color=[1, 5],
                                                 title="Theatre Watch Themes (Color = Your Avg Rating)")
                        st.plotly_chart(fig_t_themes, use_container_width=True)
                    else:
                        st.info("Not enough plot data for theatre movies.")

            with t_tab7:
                st.markdown("Detailed list of movies watched in theatres:")
                display_cols = ['Name', 'Year', 'visits', 'user_rating', 'Description']
                df_display = df_theatre[display_cols].copy()
                df_display.columns = ['Movie', 'Year', 'Visits', 'Your Rating', 'Theatre Note']
                # Format rating as stars for display in dataframe
                df_display['Your Rating'] = df_display['Your Rating'].apply(star_text)
                st.dataframe(df_display, use_container_width=True, hide_index=True)

                # Show posters for top rated theatre movies
                st.markdown("#### Top Rated Theatre Experiences")
                top_t = df_theatre_watched.sort_values('user_rating', ascending=False).head(6)
                p_cols = st.columns(6)
                for i, (_, row) in enumerate(top_t.iterrows()):
                    with p_cols[i]:
                        if pd.notna(row['poster']):
                            st.image(f"https://image.tmdb.org/t/p/w500{row['poster']}", use_container_width=True)
                        st.caption(f"**{row['Name']}**  \n{render_stars(row['user_rating'])}", unsafe_allow_html=True)
        else:
            st.info("No theatre watch data found in your Letterboxd activity. Make sure the movies in your theatre-watch list are also in your ratings.csv.")
    else:
        st.info("Theatre watch list not found at `data/raw/movies/lists/theatre-watch.csv`.")
