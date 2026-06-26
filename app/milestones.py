"""Shared helpers for 'milestone' (Nth-logged) sections across dashboards.

A milestone is the Nth item you logged, for N in 1, 50, 100, then every 250th
(250, 500, 750, 1000, …) up to the total. Used by the Movies and TV dashboards
to celebrate the 1st / 50th / 100th / … movie, theatre watch, and show logged.
"""
import pandas as pd
import streamlit as st

TMDB_W342 = "https://image.tmdb.org/t/p/w342"


def milestone_positions(n):
    """Return the 1-based milestone positions present in a library of size n:
    1st, 50th, 100th, then every 250th (250, 500, 750, 1000, …) up to n."""
    positions = [m for m in (1, 50, 100) if m <= n]
    k = 250
    while k <= n:
        positions.append(k)
        k += 250
    return positions


def ordinal(n):
    """1 -> '1st', 2 -> '2nd', 50 -> '50th', 1000 -> '1000th'."""
    if 10 <= (n % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def render_milestones(ordered_df, title_col, poster_col, header,
                      poster_base=TMDB_W342, sub_fn=None, empty_msg=None,
                      per_row=6):
    """Render milestone cards for an already logged-order DataFrame.

    ordered_df : rows sorted into the order they were logged (oldest first).
    title_col  : column holding the display title.
    poster_col : column holding a TMDB poster path ('/abc.jpg') or full URL.
    sub_fn     : optional fn(row) -> short caption (e.g. the date logged).
    """
    n = len(ordered_df)
    positions = milestone_positions(n)
    st.subheader(header)
    if not positions:
        st.info(empty_msg or "Not enough logged items for a milestone yet.")
        return
    st.caption(f"{n:,} logged · milestones: " + ", ".join(ordinal(p) for p in positions))

    for i in range(0, len(positions), per_row):
        chunk = positions[i:i + per_row]
        cols = st.columns(per_row)
        for col, pos in zip(cols, chunk):
            row = ordered_df.iloc[pos - 1]
            with col:
                poster = row.get(poster_col)
                if isinstance(poster, str) and poster.strip():
                    url = poster if poster.startswith("http") else poster_base + poster
                    st.image(url, use_container_width=True)
                st.markdown(f"### {ordinal(pos)}")
                title = row.get(title_col)
                st.markdown(f"**{title}**" if pd.notna(title) else "**(unknown)**")
                if sub_fn is not None:
                    sub = sub_fn(row)
                    if sub:
                        st.caption(sub)
