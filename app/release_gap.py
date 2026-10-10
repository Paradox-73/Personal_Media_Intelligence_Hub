"""Release gap — how many days after a film opened in India you saw it in a cinema.

Watch dates come from the Letterboxd diary; release dates from
data/cache/tmdb_release_dates.json, written by
scripts/fetch_theatre_release_dates.py (TMDB's per-country release dates, earliest
theatrical or limited release in India, US as fallback). The dashboard never calls
TMDB itself: a film missing from that cache is left out and counted in the caption,
and re-running the script picks it up.

Two diary problems are handled here rather than silently plotted:
- a watch date far before release that lands just after release once one year is
  added is a year typo (Captain Marvel is logged 2018-03-09; it opened 2019-03-08),
  so it is shifted and flagged;
- a small negative gap (watched a few days before TMDB's India date) is kept and
  flagged: either a preview screening or a wrong TMDB date.
"""
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

COMPANY_ORDER = ["Family", "Friends", "Alone"]
BUCKETS = [("Before release", lambda g: g < 0), ("Opening day", lambda g: g == 0),
           ("1–3 days", lambda g: 1 <= g <= 3), ("4–7 days", lambda g: 4 <= g <= 7),
           ("8–14 days", lambda g: 8 <= g <= 14), ("15+ days", lambda g: g >= 15)]


def _company(tags):
    parts = [t.strip().lower() for t in str(tags).split(",")]
    if any("fam" in t for t in parts):
        return "Family"
    if any("friend" in t for t in parts):
        return "Friends"
    if any(t in ("alone", "solo") for t in parts):
        return "Alone"
    return "Unknown"


def build_release_gaps(df_theatre, df_diary, cache_path):
    """One row per theatre trip that has both a diary date and a release date.

    Returns (trips, n_missing_release) where n_missing_release counts dated trips
    whose film is not in the release-date cache.
    """
    if df_diary is None or df_theatre is None or not cache_path.exists():
        return pd.DataFrame(), 0
    cache = json.loads(cache_path.read_text(encoding="utf-8"))

    diary = df_diary.dropna(subset=["Watched Date"])[["Name", "Year", "Watched Date", "Tags"]]
    trips = df_theatre[["Name", "Year"]].drop_duplicates().merge(diary, on=["Name", "Year"])

    rows, missing = [], 0
    for _, r in trips.iterrows():
        entry = cache.get(f"{r['Name']}|{int(r['Year'])}")
        if not entry or not (entry.get("IN") or entry.get("US")):
            missing += 1
            continue
        notes = []
        source = "India" if entry.get("IN") else "US"
        if source == "US":
            notes.append("No India date on TMDB; US date used.")
        released = pd.Timestamp(entry.get("IN") or entry["US"])
        watched = pd.Timestamp(r["Watched Date"])
        gap = (watched - released).days
        if gap < -30 and -7 <= (watched + pd.DateOffset(years=1) - released).days <= 60:
            notes.append(f"Diary says {watched:%Y-%m-%d}, before the film existed; read as one year later.")
            watched = watched + pd.DateOffset(years=1)
            gap = (watched - released).days
        elif gap < 0:
            notes.append("Watched before TMDB's India date: a preview screening or a wrong TMDB date.")
        rows.append({"Movie": r["Name"], "Year": int(r["Year"]), "Watched": watched,
                     "Released": released, "Release Region": source, "Days After Release": gap,
                     "With": _company(r["Tags"]), "Note": " ".join(notes)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Watched").reset_index(drop=True)
    return out, missing


def render_release_gap(df_theatre, df_diary, cache_path, metric_fn, palette, style_fn=None):
    """Streamlit block for the Release Gap tab."""
    trips, missing = build_release_gaps(df_theatre, df_diary, cache_path)
    if trips.empty:
        st.info("No release dates cached. Run `python scripts/fetch_theatre_release_dates.py`.")
        return

    colors = {"Family": palette[2], "Friends": palette[0], "Alone": palette[1], "Unknown": palette[3]}
    gaps = trips["Days After Release"]
    longest = trips.loc[gaps.idxmax()]

    st.caption(
        f"Days between each film's India theatrical release (TMDB) and the date in your diary. "
        f"{len(trips)} trips across {trips['Movie'].nunique()} films. "
        f"Theatre-list films with no diary date are not shown"
        + (f"; {missing} dated trips have no cached release date — run "
           f"`python scripts/fetch_theatre_release_dates.py`." if missing else ".")
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_fn("Median Gap", f"{gaps.median():g} days")
    with m2: metric_fn("Opening Day Trips", f"{(gaps <= 0).sum()} of {len(trips)}",
                       "Trips on the India release date, or before it")
    with m3: metric_fn("Within First Week", f"{(gaps <= 7).sum()} of {len(trips)}")
    with m4: metric_fn("Longest Wait", f"{longest['Days After Release']} days", longest["Movie"])

    # --- One row per trip: a stem from opening day (0) out to the gap. ---
    d = trips.iloc[::-1].reset_index(drop=True)  # newest at the top
    d["Row"] = d["Movie"] + "  ·  " + d["Watched"].dt.strftime("%d %b %Y")
    fig = go.Figure()
    for company in COMPANY_ORDER + ["Unknown"]:
        sub = d[d["With"] == company]
        if sub.empty:
            continue
        stem_x, stem_y = [], []
        for _, r in sub.iterrows():
            stem_x += [0, r["Days After Release"], None]
            stem_y += [r["Row"], r["Row"], None]
        fig.add_trace(go.Scatter(x=stem_x, y=stem_y, mode="lines", hoverinfo="skip",
                                 line=dict(color=colors[company], width=2), showlegend=False))
        fig.add_trace(go.Scatter(
            x=sub["Days After Release"], y=sub["Row"], mode="markers", name=company,
            marker=dict(color=colors[company], size=10, line=dict(color="#14171C", width=2)),
            customdata=sub[["Released", "Watched", "Release Region", "Note"]].astype(str).values,
            hovertemplate="<b>%{y}</b><br>Released (%{customdata[2]}): %{customdata[0]}"
                          "<br>Watched: %{customdata[1]}<br><b>%{x} days after</b>"
                          "<br>%{customdata[3]}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#99AABB", width=1.5, dash="dot"),
                  annotation_text="Opening day (India)", annotation_position="top")
    fig.update_layout(title="Every Theatre Trip — Days After Release",
                      height=max(420, 22 * len(d) + 120),
                      xaxis=dict(title="Days after India release", dtick=5, zeroline=False),
                      yaxis=dict(title=None, categoryorder="array", categoryarray=d["Row"][::-1].tolist()),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", title=None),
                      margin=dict(l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        hist = pd.DataFrame([{"Window": name, "Trips": int(gaps.map(test).sum())} for name, test in BUCKETS])
        fig_h = px.bar(hist, x="Window", y="Trips", text="Trips", title="How Soon You Went",
                       color_discrete_sequence=[palette[2]])
        fig_h.update_traces(textposition="outside")
        fig_h.update_layout(xaxis_title=None)
        st.plotly_chart(style_fn(fig_h) if style_fn else fig_h, use_container_width=True)
    with c2:
        by_year = (trips.assign(Year=trips["Watched"].dt.year)
                   .groupby("Year")["Days After Release"].agg(["median", "size"]).reset_index())
        by_year.columns = ["Year", "Median Gap", "Trips"]
        fig_y = px.bar(by_year, x="Year", y="Median Gap", text="Median Gap", hover_data={"Trips": True},
                       title="Median Gap by Year Watched", color_discrete_sequence=[palette[0]])
        fig_y.update_traces(textposition="outside")
        fig_y.update_layout(xaxis=dict(tickmode="linear", dtick=1), yaxis_title="Days")
        st.plotly_chart(style_fn(fig_y) if style_fn else fig_y, use_container_width=True)

    by_co = (trips.groupby("With")["Days After Release"].agg(["size", "median", "mean"])
             .reindex([c for c in COMPANY_ORDER + ["Unknown"] if c in set(trips["With"])]))
    st.caption("  ·  ".join(f"**{c}**: {int(r['size'])} trips, median {r['median']:g} days"
                            for c, r in by_co.iterrows()))

    table = trips.iloc[::-1].copy()
    table["Watched"] = table["Watched"].dt.strftime("%Y-%m-%d")
    table["Released"] = table["Released"].dt.strftime("%Y-%m-%d")
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_viewing_gap(df, df_diary_tags, cache_path, metric_fn, palette, style_fn):
    """Release-to-watch gap for every diary entry, theatre and home (the Release Gap idea, widened)."""
    st.subheader("⏱️ Viewing Gap: Release to Watch")
    if df_diary_tags is None or df_diary_tags.empty:
        st.info("Diary not found at `data/raw/movies/diary.csv`.")
    else:
        gap = df_diary_tags.dropna(subset=['Watched Date']).copy()
        gap['where'] = np.where(gap['Tags'].fillna('').str.contains('theatre', case=False), "Theatre", "Home")
        gap['rewatch'] = gap['Rewatch'].fillna('').eq('Yes')
        # Release date: enriched_data 'released' (TMDB/OMDb, mixed formats). A parsed
        # date more than a year away from the film's year is a misparse (17-Aug-79 ->
        # 2079), so those fall back to 1 July of the film's year and are flagged.
        rel = df[['letterboxd_name', 'year', 'released']].dropna(subset=['letterboxd_name']).copy()
        rel['rel_date'] = pd.to_datetime(rel['released'], errors='coerce', format='mixed', dayfirst=True)
        bad = rel['rel_date'].isna() | ((rel['rel_date'].dt.year - rel['year']).abs() > 1)
        rel['approx'] = bad
        rel.loc[bad, 'rel_date'] = pd.to_datetime(rel.loc[bad, 'year'].astype('Int64').astype(str) + "-07-01", errors='coerce')
        rel = rel.drop_duplicates(['letterboxd_name', 'year'])
        gap = gap.merge(rel[['letterboxd_name', 'year', 'rel_date', 'approx']],
                        left_on=['Name', 'Year'], right_on=['letterboxd_name', 'year'], how='left')
        # Cinema trips: the India release date from the theatre cache beats TMDB's
        # earliest-anywhere date (same source as the Theatre "Release Gap" tab).
        if cache_path.exists():
            rd_cache = json.loads(cache_path.read_text(encoding='utf-8'))
            india = gap.apply(lambda r: (rd_cache.get(f"{r['Name']}|{int(r['Year'])}") or {}).get('IN')
                              if r['where'] == "Theatre" else None, axis=1)
            has_in = india.notna()
            gap.loc[has_in, 'rel_date'] = pd.to_datetime(india[has_in])
            gap.loc[has_in, 'approx'] = False
        gap = gap.dropna(subset=['rel_date'])
        gap['gap_days'] = (gap['Watched Date'] - gap['rel_date']).dt.days.clip(lower=0)
        gap_bins = [-1, 7, 30, 182, 365, 365 * 5, 365 * 10, 365 * 20, 10 ** 6]
        gap_labels = ["Within a week", "Within a month", "1–6 months", "6–12 months",
                      "1–5 years", "5–10 years", "10–20 years", "20+ years"]
        gap['bucket'] = pd.cut(gap['gap_days'], bins=gap_bins, labels=gap_labels)

        # Repeat cinema trips are always kept (Obsession and Across the Spider-Verse
        # are logged as rewatches the second time, and the theatre view counts both
        # trips); the toggle only drops rewatches at home.
        first_only = st.toggle("Leave out rewatches at home (repeat cinema trips are always counted)", value=True,
                               key="gap_first_only")
        g = gap[~(gap['rewatch'] & (gap['where'] == "Home"))] if first_only else gap

        gm1, gm2, gm3, gm4 = st.columns(4)
        with gm1: metric_fn("Median Gap", f"{g['gap_days'].median() / 365.25:.1f} yrs",
                            help_text=f"{g['gap_days'].median():.0f} days over {len(g)} diary entries")
        with gm2: metric_fn("Within a Month", f"{(g['gap_days'] <= 30).sum()} of {len(g)}")
        with gm3:
            home = g[g['where'] == "Home"]
            metric_fn("Median Home Gap", f"{home['gap_days'].median() / 365.25:.1f} yrs" if not home.empty else "N/A")
        with gm4:
            oldest = g.loc[g['gap_days'].idxmax()] if not g.empty else None
            metric_fn("Longest Wait", f"{oldest['gap_days'] / 365.25:.0f} yrs" if oldest is not None else "N/A",
                      help_text=f"{oldest['Name']} ({int(oldest['Year'])})" if oldest is not None else None)

        tg_b, tg_s, tg_t = st.tabs(["Gap Buckets", "Gap vs Rating", "Quickest & Longest"])
        with tg_b:
            bk = (g.groupby(['bucket', 'where'], observed=False)
                   .agg(n=('Name', 'size'), avg=('Rating', 'mean')).reset_index())
            fig = px.bar(bk, x='bucket', y='n', color='where', barmode='group',
                         color_discrete_map={"Home": palette[2], "Theatre": palette[0]},
                         custom_data=['avg'], labels={'bucket': "Time after release", 'n': "Diary entries", 'where': ""},
                         title="How long after release you watched")
            fig.update_traces(hovertemplate="%{x}<br>%{y} entries · avg rating %{customdata[0]:.2f}<extra></extra>")
            st.plotly_chart(style_fn(fig), use_container_width=True)
            bk_all = g.groupby('bucket', observed=True)['Rating'].agg(['size', 'mean', 'median']).reset_index()
            st.dataframe(bk_all.rename(columns={'bucket': "Gap", 'size': "Entries", 'mean': "Avg ★", 'median': "Median ★"})
                         .round(2), hide_index=True, use_container_width=True)
        with tg_s:
            gs = g.dropna(subset=['Rating']).copy()
            gs['gap_years'] = gs['gap_days'] / 365.25
            fig = px.scatter(gs, x='gap_years', y='Rating', color='where', hover_name='Name',
                             color_discrete_map={"Home": palette[2], "Theatre": palette[0]},
                             custom_data=['gap_days', 'Watched Date'],
                             labels={'gap_years': "Years after release", 'Rating': "Your rating", 'where': ""},
                             title="Do you rate films higher when you catch them early?")
            fig.update_traces(marker=dict(size=8, opacity=0.75, line=dict(width=1, color="#14171C")),
                              hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]} days after release"
                                            "<br>Watched %{customdata[1]|%d %b %Y} · %{y}★<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)
            if len(gs) >= 10:
                r = gs['gap_days'].corr(gs['Rating'], method='spearman')
                st.caption(f"Spearman correlation between gap and rating: {r:+.2f} "
                           "(negative = films watched sooner get more stars).")
        with tg_t:
            cols_show = {'Name': "Film", 'Year': "Year", 'where': "Where", 'Watched Date': "Watched",
                         'gap_days': "Days after release", 'Rating': "Your ★"}
            ga, gb = st.columns(2)
            with ga:
                st.markdown("**Quickest at home**")
                st.dataframe(g[g['where'] == "Home"].nsmallest(10, 'gap_days')[list(cols_show)]
                             .rename(columns=cols_show), hide_index=True, use_container_width=True)
            with gb:
                st.markdown("**Longest waits**")
                st.dataframe(g.nlargest(10, 'gap_days')[list(cols_show)].rename(columns=cols_show),
                             hide_index=True, use_container_width=True)
        n_approx = int(g['approx'].fillna(False).sum())
        st.caption(f"Gap = diary watch date minus release date. Theatre trips use the India release date where "
                   f"cached; other films use TMDB's first release date. {n_approx} entries had an unreadable "
                   f"release date and use 1 July of the release year. Gaps below zero (previews) count as 0.")
