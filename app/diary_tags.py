"""Diary-tag analytics — the only record of *how* a film was watched.

Every other signal in the library describes the film (genre, cast, keywords,
critic scores). Letterboxd tags describe the **viewing**: which cinema, who was
in the seat next to you, whether someone talked you into it. That makes them the
one place the data can answer "does the context change the rating?" instead of
"does the content change the rating?".

Tags live on diary *entries*, not on films, so everything here is entry-level: a
rewatch with different company counts twice, which is the correct unit.

Groups are DERIVED, not hardcoded, so new tags file themselves:
  company      'alone', or anything starting 'with ' (with fam, with friends)
  recommended  anything ending 'recommended' / 'rec'
  format       a small known projection-format vocabulary (imax, 4dx, 70mm, …)
  theatre      the umbrella cinema marker itself
  venue        any other tag whose entries ALL also carry the theatre marker —
               i.e. a place you can only have been while at the cinema
  context      everything left over (travel, …)

Plotly + pandas only, so it drops into any page without new dependencies.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

THEATRE_MARKERS = {"theatre", "theater", "cinema", "cinemas"}
FORMAT_TAGS = {"imax", "4dx", "3d", "2d", "70mm", "35mm", "dolby", "dolby atmos",
               "atmos", "screenx", "4k", "70 mm", "imax 70mm"}

GROUP_COMPANY = "Company"
GROUP_RECOMMENDED = "Recommended by"
GROUP_FORMAT = "Format"
GROUP_THEATRE = "Theatre"
GROUP_VENUE = "Cinema"
GROUP_CONTEXT = "Context"

# Minimum entries before a tag's average rating is worth plotting. Below this the
# mean is one or two films wearing a trend's clothes.
MIN_N = 3

# Diverging scale for "above / below your average". Warm (orange) and cool-green
# poles with the theme's neutral gray at the midpoint, matching the rest of the
# dashboard where green = you rated it higher, orange = lower.
DIVERGING = [[0.0, "#ff8000"], [0.5, "#556678"], [1.0, "#00e054"]]


def split_tags(value):
    """'Theatre, With Friends' -> ['theatre', 'with friends']."""
    if pd.isna(value):
        return []
    return [t.strip().lower() for t in str(value).split(",") if t.strip()]


def classify_tags(long_df):
    """Map every distinct tag to a group. Returns {tag: group}.

    The venue rule is the interesting one: a tag is a cinema if *every* entry
    carrying it also carries the theatre marker. That keeps the group list
    self-maintaining — a new multiplex needs no code change — while refusing to
    call 'travel' a cinema just because it sometimes coincides with one.
    """
    marker_entries = {i for i, t in zip(long_df["entry_id"], long_df["tag"])
                      if t in THEATRE_MARKERS}
    groups = {}
    for tag, grp in long_df.groupby("tag"):
        if tag in THEATRE_MARKERS:
            groups[tag] = GROUP_THEATRE
        elif tag == "alone" or tag.startswith("with "):
            groups[tag] = GROUP_COMPANY
        elif tag.endswith("recommended") or tag.endswith(" rec"):
            groups[tag] = GROUP_RECOMMENDED
        elif tag in FORMAT_TAGS:
            groups[tag] = GROUP_FORMAT
        elif set(grp["entry_id"]).issubset(marker_entries):
            groups[tag] = GROUP_VENUE
        else:
            groups[tag] = GROUP_CONTEXT
    return groups


def tag_frame(diary):
    """Explode a Letterboxd diary into one row per (entry, tag).

    Columns: entry_id, Name, Year, Rating, Watched Date, tag, group.
    Returns an empty frame when the diary carries no tags at all.
    """
    if diary is None or diary.empty or "Tags" not in diary.columns:
        return pd.DataFrame()
    d = diary.copy().reset_index(drop=True)
    d["entry_id"] = d.index
    d["_tags"] = d["Tags"].apply(split_tags)
    d = d[d["_tags"].map(len) > 0]
    if d.empty:
        return pd.DataFrame()
    long_df = d.explode("_tags").rename(columns={"_tags": "tag"})
    keep = [c for c in ("entry_id", "Name", "Year", "Rating", "Watched Date")
            if c in long_df.columns]
    long_df = long_df[keep + ["tag"]].reset_index(drop=True)
    long_df["group"] = long_df["tag"].map(classify_tags(long_df))
    return long_df


def tag_summary(long_df, baseline):
    """Per-tag count, mean rating and delta vs the diary average.

    Sorted by count, then by average rating: with a short tag vocabulary most
    counts tie, and an arbitrary tiebreak makes the bar order jump around between
    reruns for no reason the reader can see.
    """
    agg = (long_df.groupby(["tag", "group"])
           .agg(Entries=("Rating", "size"), AvgRating=("Rating", "mean"))
           .reset_index())
    agg["Delta"] = agg["AvgRating"] - baseline
    agg["AvgRating"] = agg["AvgRating"].round(2)
    agg["Delta"] = agg["Delta"].round(2)
    return (agg.sort_values(["Entries", "AvgRating"], ascending=[False, False])
            .reset_index(drop=True))


def _title(tag):
    """'dt mall' -> 'DT Mall'; keeps known acronyms upper-case."""
    small = {"imax": "IMAX", "4dx": "4DX", "3d": "3D", "dt": "DT", "m5": "M5",
             "70mm": "70mm", "35mm": "35mm", "4k": "4K"}
    return " ".join(small.get(w, w.capitalize()) for w in str(tag).split())


def _deviation_bar(rows, baseline, title, height=None):
    """Horizontal diverging bar: average rating per tag, centred on `baseline`.

    Polarity is the job here — 'do I rate these above or below my usual?' — so the
    bars run from the baseline and the color is diverging, not a value ramp.
    """
    # Delta ties are rating ties, so the sample size breaks them — the tag you
    # have more evidence for sits higher.
    rows = rows.sort_values(["Delta", "Entries"], ascending=[True, True])
    span = max(0.55, float(np.abs(rows["Delta"]).max()) * 1.25)
    fig = go.Figure(go.Bar(
        x=rows["Delta"], y=[_title(t) for t in rows["tag"]], orientation="h",
        marker=dict(color=rows["Delta"], colorscale=DIVERGING,
                    cmin=-span, cmax=span,
                    line=dict(width=2, color="#14171C")),
        customdata=np.stack([rows["AvgRating"], rows["Entries"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>Your average: %{customdata[0]:.2f}★"
                       "<br>vs usual: %{x:+.2f}<br>%{customdata[1]} entries"
                       "<extra></extra>"),
        text=[f"{d:+.2f}" for d in rows["Delta"]], textposition="outside",
        cliponaxis=False, showlegend=False))
    fig.add_vline(x=0, line_color="#99AABB", line_width=1)
    fig.update_layout(
        title=title, height=height or max(260, 42 * len(rows) + 120),
        xaxis_title=f"Rating vs your diary average ({baseline:.2f}★)",
        yaxis_title=None, bargap=0.35,
        xaxis=dict(range=[-span * 1.35, span * 1.35]),
        margin=dict(l=10, r=30, t=60, b=40))
    return fig


def _count_bar(rows, title, color, label):
    """Horizontal count bar. One measure, one series, so one flat hue.

    Ties on count break on average rating, so two cinemas you visited the same
    number of times still land in a meaningful order (better-rated on top).
    """
    rows = rows.sort_values(["Entries", "AvgRating"], ascending=[True, True])
    fig = go.Figure(go.Bar(
        x=rows["Entries"], y=[_title(t) for t in rows["tag"]], orientation="h",
        marker=dict(color=color, line=dict(width=2, color="#14171C")),
        customdata=rows["AvgRating"],
        hovertemplate=("<b>%{y}</b><br>%{x} " + label +
                       "<br>Your average: %{customdata:.2f}★<extra></extra>"),
        text=rows["Entries"], textposition="outside", cliponaxis=False,
        showlegend=False))
    fig.update_layout(title=title, height=max(240, 40 * len(rows) + 120),
                      xaxis_title=label.capitalize(), yaxis_title=None,
                      bargap=0.35, margin=dict(l=10, r=30, t=60, b=40))
    return fig


def _strip_plot(long_df, tags, baseline, palette, title):
    """Every individual rating as a dot, with a mean diamond per tag.

    With six-to-eighteen entries per group a box plot implies quartile precision
    the sample does not have. Showing every rating is both honest and denser.
    """
    fig = go.Figure()
    order = [t for t in tags if (long_df["tag"] == t).any()]
    for i, tag in enumerate(order):
        vals = long_df.loc[long_df["tag"] == tag, "Rating"].dropna()
        names = long_df.loc[long_df["tag"] == tag, "Name"]
        # Deterministic jitter: a rerun must not reshuffle the dots.
        jitter = ((np.arange(len(vals)) * 0.37) % 1.0 - 0.5) * 0.34
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=vals, y=np.full(len(vals), i) + jitter, mode="markers",
            name=_title(tag), legendgroup=tag, text=names,
            marker=dict(size=11, color=color, opacity=0.85,
                        line=dict(width=2, color="#14171C")),
            hovertemplate="<b>%{text}</b><br>%{x}★<extra>" + _title(tag) + "</extra>"))
        fig.add_trace(go.Scatter(
            x=[vals.mean()], y=[i], mode="markers", legendgroup=tag,
            showlegend=False, marker=dict(size=17, color=color, symbol="diamond",
                                          line=dict(width=2, color="#E0E0E0")),
            hovertemplate=(f"<b>{_title(tag)} average</b><br>%{{x:.2f}}★<br>"
                           f"{len(vals)} entries<extra></extra>")))
    fig.add_vline(x=baseline, line_dash="dot", line_color="#99AABB",
                  annotation_text=f"your average {baseline:.2f}★",
                  annotation_position="top")
    fig.update_layout(
        title=title, height=120 + 90 * max(1, len(order)),
        xaxis=dict(title="Your rating", range=[0.2, 5.3], dtick=0.5),
        yaxis=dict(tickmode="array", tickvals=list(range(len(order))),
                   ticktext=[f"{_title(t)}  (n={int((long_df['tag'] == t).sum())})"
                             for t in order],
                   range=[-0.6, len(order) - 0.4]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=10, r=20, t=70, b=40))
    return fig


def _monthly_bar(long_df, tag, color, title):
    """Entries carrying `tag` per month. Counts are small integers -> columns."""
    rows = long_df[(long_df["tag"] == tag) & long_df["Watched Date"].notna()]
    if rows.empty:
        return None
    per = (rows.set_index("Watched Date").resample("MS").size()
           .rename("Entries").reset_index())
    if per.empty:
        return None
    # Fill the calendar gaps so a quiet month reads as zero, not as missing.
    full = pd.date_range(per["Watched Date"].min(), per["Watched Date"].max(), freq="MS")
    per = (per.set_index("Watched Date").reindex(full, fill_value=0)
           .rename_axis("Month").reset_index())
    fig = go.Figure(go.Bar(
        x=per["Month"], y=per["Entries"],
        marker=dict(color=color, line=dict(width=2, color="#14171C")),
        hovertemplate="<b>%{x|%b %Y}</b><br>%{y} trips<extra></extra>",
        showlegend=False))
    fig.update_layout(title=title, height=320, bargap=0.3,
                      xaxis_title=None, yaxis_title="Trips",
                      yaxis=dict(dtick=1), margin=dict(l=10, r=20, t=60, b=40))
    return fig


def render_tag_analytics(diary, palette, metric_fn=None):
    """Render the whole diary-tag section into the current Streamlit page.

    diary     : the Letterboxd diary frame (needs Tags, Rating, Watched Date).
    palette   : the page's categorical hue list, used in fixed order.
    metric_fn : optional fn(label, value, help_text=None) rendering a stat tile.
    """
    long_df = tag_frame(diary)
    st.subheader("🏷️ Viewing Context (Diary Tags)")
    if long_df.empty:
        st.info("No diary tags yet. Tag entries on Letterboxd (cinema, who you "
                "were with, who recommended it) and they will show up here.")
        return

    st.caption("Every other chart on this page describes the *film*. These describe "
               "the *watch* — where you were, who you were with, who talked you "
               "into it — and whether any of that moved your rating.")

    rated = long_df[long_df["Rating"].notna()]
    baseline = float(diary["Rating"].dropna().mean())
    summary = tag_summary(rated, baseline)
    groups = dict(zip(summary["tag"], summary["group"]))
    tagged_entries = long_df["entry_id"].nunique()
    total_entries = len(diary)

    def _in(group):
        return summary[summary["group"] == group]

    theatre_tag = next((t for t, g in groups.items() if g == GROUP_THEATRE), None)
    theatre_trips = int(summary.loc[summary["tag"] == theatre_tag, "Entries"].iloc[0]) \
        if theatre_tag else 0
    venues = _in(GROUP_VENUE)

    # --- headline tiles: single numbers, so tiles rather than one-bar charts ---
    if metric_fn:
        cols = st.columns(5)
        with cols[0]:
            metric_fn("Tagged Entries", f"{tagged_entries}",
                      help_text=f"of {total_entries} diary entries "
                                f"({tagged_entries / max(total_entries, 1):.0%})")
        with cols[1]:
            metric_fn("Distinct Tags", f"{summary['tag'].nunique()}")
        with cols[2]:
            metric_fn("Theatre Trips", f"{theatre_trips}")
        with cols[3]:
            metric_fn("Cinemas Visited", f"{len(venues)}")
        with cols[4]:
            best = summary[summary["Entries"] >= MIN_N].sort_values("AvgRating").tail(1)
            if not best.empty:
                b = best.iloc[0]
                metric_fn("Best Context", _title(b["tag"]),
                          help_text=f"{b['AvgRating']:.2f}★ across "
                                    f"{int(b['Entries'])} entries "
                                    f"({b['Delta']:+.2f} vs your average)")
        st.write("")

    t_who, t_where, t_recs, t_all, t_when = st.tabs(
        ["👥 Who With", "🎦 Where & How", "💬 Recommendations",
         "🏷️ Every Tag", "📅 Theatre Habit"])

    # --- WHO WITH -----------------------------------------------------------
    with t_who:
        company = _in(GROUP_COMPANY).sort_values("Entries", ascending=False)
        if company.empty:
            st.info("No company tags yet (`alone`, `with friends`, `with fam`…).")
        else:
            st.plotly_chart(
                _strip_plot(rated, list(company["tag"]), baseline, palette,
                            "Every rating, split by who you watched with"),
                use_container_width=True)
            top = company.sort_values("AvgRating").iloc[-1]
            bot = company.sort_values("AvgRating").iloc[0]
            st.caption(
                f"Diamond = the average for that company; dotted line = your overall "
                f"diary average ({baseline:.2f}★). You rate highest "
                f"**{_title(top['tag'])}** ({top['AvgRating']:.2f}★, n={int(top['Entries'])}) "
                f"and lowest **{_title(bot['tag'])}** ({bot['AvgRating']:.2f}★, "
                f"n={int(bot['Entries'])}). Small samples — read the gap, not the decimals.")

    # --- WHERE & HOW --------------------------------------------------------
    with t_where:
        c_left, c_right = st.columns(2)
        with c_left:
            if venues.empty:
                st.info("No cinema tags yet.")
            else:
                st.plotly_chart(
                    _count_bar(venues, "Trips per cinema", palette[2], "trips"),
                    use_container_width=True)
                st.caption("A cinema tag is any tag that only ever appears on a "
                           "theatre entry, so new venues need no setup.")
        with c_right:
            fmt = _in(GROUP_FORMAT)
            eligible = pd.concat([v for v in (venues, fmt) if not v.empty]) \
                if not (venues.empty and fmt.empty) else pd.DataFrame()
            eligible = eligible[eligible["Entries"] >= MIN_N] if not eligible.empty \
                else eligible
            if eligible.empty:
                st.info(f"No venue or format tag has {MIN_N}+ entries yet.")
            else:
                st.plotly_chart(
                    _deviation_bar(eligible, baseline,
                                   f"How you rate by venue & format (≥{MIN_N} entries)"),
                    use_container_width=True)
                st.caption("Where you sat is not why a film is good — treat this as "
                           "a record of what you chose to see there.")

        if theatre_tag:
            th = rated[rated["tag"] == theatre_tag]["Rating"]
            # entry_id is positional within the diary as passed in, so compare
            # against a positional index — load_diary() sorts by date, leaving
            # the original labels shuffled and unusable as keys here.
            seen_at_cinema = set(long_df.loc[long_df["tag"] == theatre_tag, "entry_id"])
            home = (diary.reset_index(drop=True)
                    .drop(index=[i for i in seen_at_cinema if i < len(diary)])
                    ["Rating"].dropna())
            m1, m2, m3 = st.columns(3)
            if metric_fn:
                with m1:
                    metric_fn("In a Theatre", f"{th.mean():.2f}★",
                              help_text=f"{len(th)} entries")
                with m2:
                    metric_fn("Everywhere Else", f"{home.mean():.2f}★",
                              help_text=f"{len(home)} entries")
                with m3:
                    metric_fn("Theatre Bump", f"{th.mean() - home.mean():+.2f}",
                              help_text="Difference in your average rating.")

    # --- RECOMMENDATIONS ----------------------------------------------------
    with t_recs:
        recs = _in(GROUP_RECOMMENDED)
        if recs.empty:
            st.info("No `… recommended` tags yet.")
        else:
            st.plotly_chart(
                _deviation_bar(recs, baseline, "Did the recommendation land?",
                               height=max(240, 60 * len(recs) + 130)),
                use_container_width=True)
            st.caption(
                f"Measured against your whole diary average ({baseline:.2f}★), not "
                f"against an 'I picked it myself' group — untagged entries predate "
                f"your tagging habit, so they are unknown rather than self-picked.")
            per_tag = (rated[rated["tag"].isin(recs["tag"])]
                       .sort_values("Rating", ascending=False))
            show = per_tag[["Name", "Year", "Rating", "tag"]].rename(
                columns={"Name": "Film", "Rating": "Your Rating", "tag": "Source"})
            show["Source"] = show["Source"].apply(_title)
            show["Year"] = pd.to_numeric(show["Year"], errors="coerce").astype("Int64")
            st.dataframe(show, use_container_width=True, hide_index=True,
                         height=min(430, 60 + 35 * len(show)))

    # --- EVERY TAG ----------------------------------------------------------
    with t_all:
        plot_rows = summary[summary["Entries"] >= MIN_N]
        if plot_rows.empty:
            st.info(f"No tag has {MIN_N}+ entries yet.")
        else:
            st.plotly_chart(
                _deviation_bar(plot_rows, baseline,
                               f"Every tag with {MIN_N}+ entries, vs your average"),
                use_container_width=True)
        fig = px.scatter(summary, x="Entries", y="AvgRating", text=summary["tag"].apply(_title),
                         color="group", hover_data={"Delta": ":+.2f"},
                         color_discrete_sequence=palette,
                         title="How often you use a tag vs how you rate it")
        fig.add_hline(y=baseline, line_dash="dot", line_color="#99AABB")
        fig.update_traces(textposition="top center",
                          marker=dict(size=13, line=dict(width=2, color="#14171C")))
        fig.update_layout(height=520, xaxis_title="Entries tagged",
                          yaxis_title="Your average rating",
                          legend_title_text="Group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Dotted line = your diary average ({baseline:.2f}★). "
                   f"Tags below {MIN_N} entries are shown here and in the table, "
                   f"but are excluded from the deviation chart above.")
        table = summary.rename(columns={"tag": "Tag", "group": "Group",
                                        "AvgRating": "Avg Rating",
                                        "Delta": "vs Average"})
        table["Tag"] = table["Tag"].apply(_title)
        st.dataframe(table[["Tag", "Group", "Entries", "Avg Rating", "vs Average"]],
                     use_container_width=True, hide_index=True,
                     height=min(560, 60 + 35 * len(table)))

    # --- THEATRE HABIT ------------------------------------------------------
    with t_when:
        if not theatre_tag:
            st.info("No theatre tag yet.")
        else:
            fig = _monthly_bar(long_df, theatre_tag, palette[0],
                               "Theatre trips per month")
            if fig is None:
                st.info("No dated theatre entries yet.")
            else:
                st.plotly_chart(fig, use_container_width=True)
                th_rows = long_df[(long_df["tag"] == theatre_tag)
                                  & long_df["Watched Date"].notna()]
                span = th_rows["Watched Date"].max() - th_rows["Watched Date"].min()
                months = max(1, span.days / 30.44)
                st.caption(
                    f"{len(th_rows)} tagged theatre trips between "
                    f"{th_rows['Watched Date'].min():%b %Y} and "
                    f"{th_rows['Watched Date'].max():%b %Y} — about "
                    f"{len(th_rows) / months:.1f} a month. Only tagged entries count, "
                    f"so trips you never tagged are missing.")
