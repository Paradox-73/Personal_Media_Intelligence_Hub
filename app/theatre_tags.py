"""Theatre-list tag parsing — the who/where/when of every cinema trip.

The Letterboxd list `theatre-watch.csv` carries a free-text Description per film.
It is not prose: it is a comma-separated bag of tags recorded by hand over years,
mixing six different dimensions in one field, e.g.

    "Friends, m5, 3d, bangalore, evening, gathik, suryans, bhavya jain"

The dashboard used to split that on ``[,\\s]+`` (comma OR whitespace), which
shattered every multi-word tag: 'bhavya jain' became 'Bhavya' + 'Jain', 'dt mall'
became 'Dt' + 'Mall', and the two distinct friends 'harsh gupta' / 'harsh modi'
both collapsed to 'Harsh'. **The comma is the only real delimiter.**

So: split on commas, then classify each tag into its dimension.

Classification is closed-vocabulary for the four dimensions that have a finite,
knowable set (city, venue, format, time-of-day) plus the relationship words, and
OPEN for people — anything the vocabulary does not recognise is treated as a
person's name. That is the right default here because the vocabularies are
complete and stable while the guest list is not: a new friend added to a future
description files itself correctly with no code change, whereas a new mall would
be a one-line addition.

Aliases exist because the same person is written both ways across the years
('sodhi' early on, 'ishaan sodhi' later). They are merged so the counts are
truthful.
"""
import re

import pandas as pd
import plotly.express as px
import streamlit as st

# --- GROUPS -----------------------------------------------------------------
GROUP_COMPANY = "Company"
GROUP_PERSON = "People"
GROUP_VENUE = "Venue"
GROUP_CITY = "City"
GROUP_TIME = "Time of Day"
GROUP_FORMAT = "Format"
GROUP_OCCASION = "Occasion"

GROUP_ORDER = [GROUP_COMPANY, GROUP_PERSON, GROUP_VENUE, GROUP_CITY,
               GROUP_TIME, GROUP_FORMAT, GROUP_OCCASION]

# --- VOCABULARIES -----------------------------------------------------------
# Keys are the normalised (lowercase, single-spaced) forms seen in the CSV;
# values are the canonical labels to display. Typos map onto the correct spelling
# rather than becoming their own category ('chandigrah' -> Chandigarh).

CITIES = {
    "chandigarh": "Chandigarh",
    "chandigrah": "Chandigarh",     # typo in the 2018 Infinity War row
    "chandighar": "Chandigarh",
    "bangalore": "Bangalore",
    "banglore": "Bangalore",
    "bengaluru": "Bangalore",
    "panchkula": "Panchkula",
    "mohali": "Mohali",
    "delhi": "Delhi",
}

VENUES = {
    "elante": "Elante Mall",
    "dt mall": "DT Mall",
    "dt": "DT Mall",
    "fun republic": "Fun Republic Mall",
    "rajhans": "Rajhans Cinemas",
    "nexus": "Nexus Mall",
    "vega city": "Vega City Mall",
    "vega": "Vega City Mall",
    "m5": "M5 Mall",
    "forum": "Forum Mall",
    "orion": "Orion Mall",
    "phoenix": "Phoenix Mall",
}

FORMATS = {
    "imax": "IMAX",
    "4dx": "4DX",
    "3d": "3D",
    "2d": "2D",
    "70mm": "70mm",
    "70 mm": "70mm",
    "35mm": "35mm",
    "dolby": "Dolby",
    "atmos": "Dolby Atmos",
    "dolby atmos": "Dolby Atmos",
    "screenx": "ScreenX",
    "ice": "ICE",
    "pxl": "PXL",
    "recliner": "Recliner",
}

TIMES = {
    "morning": "Morning",
    "afternoon": "Afternoon",
    "matinee": "Afternoon",
    "evening": "Evening",
    "night": "Night",
    "late night": "Late Night",
    "midnight": "Late Night",
}

# Relationship words — *who* you went with, as a role rather than a name.
COMPANY = {
    "alone": "Alone",
    "solo": "Alone",
    "friend": "Friends",
    "friends": "Friends",
    "fam": "Family",
    "family": "Family",
    "mom": "Mom",
    "mum": "Mom",
    "dad": "Dad",
    "papa": "Dad",
    "didi": "Didi",
    "sister": "Didi",
    "bhai": "Bhai",
    "bua": "Bua",
    "dadu": "Dadu",
    "dadi": "Dadi",
    "nana": "Nana",
    "nani": "Nani",
    "cousin": "Cousins",
    "cousins": "Cousins",
    "school": "School",
    "college": "College",
    "office": "Office",
}

OCCASIONS = {
    "twice": "Watched Twice",
    "thrice": "Watched Thrice",
    "double feature": "Double Feature",
    "triple feature": "Triple Feature",
    "rewatch": "Rewatch",
    "first day first show": "First Day First Show",
    "fdfs": "First Day First Show",
    "premiere": "Premiere",
    "re release": "Re-release",
    "re-release": "Re-release",
}

# Surname- or first-name-only mentions that refer to someone written in full
# elsewhere in the same list. Without this the same friend is counted twice.
PERSON_ALIASES = {
    "sodhi": "Ishaan Sodhi",
    "ishaan": "Ishaan Sodhi",
    "tyagi": "Akshat Tyagi",
    "akshat": "Akshat Tyagi",
    "kunal": "Kunal Mittal",
    "bhavya": "Bhavya Jain",
}

# Every closed vocabulary, in the order they are tested. Order matters only for
# tokens that could appear in two maps; none currently do, but venue is checked
# before city so a mall named after a city would resolve as a venue.
_LOOKUPS = [
    (COMPANY, GROUP_COMPANY),
    (OCCASIONS, GROUP_OCCASION),
    (FORMATS, GROUP_FORMAT),
    (VENUES, GROUP_VENUE),
    (CITIES, GROUP_CITY),
    (TIMES, GROUP_TIME),
]

# Junk that survives hand-editing in Excel: stray separators, empty cells.
_DROP = {"", "-", "--", "n/a", "na", "nan", "none", "?"}


def normalise(tag):
    """One raw comma-field -> its comparison key.

    Collapses the CRLFs that Excel leaves mid-cell, strips surrounding
    punctuation, and squeezes runs of whitespace. Case is folded because the same
    tag appears as 'Friends', 'friends' and 'FRIENDS' across the years.
    """
    if tag is None:
        return ""
    t = str(tag).replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.strip(" .;:•-–—/|")          # trailing separators from hand edits
    return t.lower()


def split_description(value):
    """'Friends, m5, 3d\\r\\n' -> ['friends', 'm5', '3d'].

    Commas only. Whitespace is *inside* tags ('dt mall', 'bhavya jain'), never
    between them, which is precisely what the old space-splitting regex got wrong.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    out, seen = [], set()
    for chunk in str(value).split(","):
        key = normalise(chunk)
        if key in _DROP or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def classify(tag):
    """One normalised tag -> (group, canonical label).

    Closed vocabularies win; anything left over is a person. The fallback is
    deliberate — the guest list grows, the malls do not.
    """
    for table, group in _LOOKUPS:
        if tag in table:
            return group, table[tag]
    if tag in PERSON_ALIASES:
        return GROUP_PERSON, PERSON_ALIASES[tag]
    return GROUP_PERSON, tag.title()


def parse_description(value):
    """One Description cell -> {group: [canonical labels]} in GROUP_ORDER."""
    parsed = {g: [] for g in GROUP_ORDER}
    for tag in split_description(value):
        group, label = classify(tag)
        if label not in parsed[group]:
            parsed[group].append(label)
    return parsed


def explode(df_theatre, desc_col="Description", name_col="Name",
            rating_col="user_rating"):
    """Theatre list -> long frame of one row per (film, tag).

    Columns: Name, Rating, Group, Tag. Films with no description drop out; films
    with no rating are kept (an unrated trip is still a trip) and their Rating is
    NaN, so counts and averages can be taken off the same frame.
    """
    rows = []
    for _, r in df_theatre.iterrows():
        parsed = parse_description(r.get(desc_col))
        rating = r.get(rating_col)
        for group, labels in parsed.items():
            for label in labels:
                rows.append({"Name": r.get(name_col), "Rating": rating,
                             "Group": group, "Tag": label})
    return pd.DataFrame(rows, columns=["Name", "Rating", "Group", "Tag"])


def summarise(long_df, group=None, min_n=1):
    """Per-tag Count and Avg_Rating, optionally for one group only."""
    d = long_df if group is None else long_df[long_df["Group"] == group]
    if d.empty:
        return pd.DataFrame(columns=["Tag", "Count", "Avg_Rating"])
    stats = (d.groupby("Tag")
              .agg(Count=("Tag", "count"), Avg_Rating=("Rating", "mean"))
              .reset_index())
    return stats[stats["Count"] >= min_n].sort_values("Count", ascending=False)


def render_theatre_tag_analytics(df_theatre, palette, min_rated_n=2,
                                 style_fn=None):
    """Streamlit block: one section per dimension of the cinema trip.

    style_fn is the page's bar-styling hook (update_bar_trace), passed in so
    these charts keep the same outline as every other bar on the dashboard.
    """
    def _style(fig):
        return style_fn(fig) if style_fn else fig

    long_df = explode(df_theatre)
    if long_df.empty:
        st.info("No tags found in theatre descriptions.")
        return

    st.caption(
        "Parsed from the list description. Commas separate tags, so multi-word "
        "tags stay whole — 'bhavya jain' is one friend, 'dt mall' is one venue."
    )

    present = [g for g in GROUP_ORDER if not long_df[long_df["Group"] == g].empty]
    tabs = st.tabs(present)
    for tab, group in zip(tabs, present):
        with tab:
            stats = summarise(long_df, group)
            c1, c2 = st.columns(2)
            with c1:
                top = stats.head(15)
                fig = px.bar(top, x="Count", y="Tag", orientation="h",
                             title=f"Most Frequent — {group}",
                             color="Count", color_continuous_scale="Bluered",
                             hover_data={"Count": True, "Avg_Rating": ":.2f"})
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(_style(fig), use_container_width=True)
            with c2:
                rated = (stats[stats["Count"] >= min_rated_n]
                         .dropna(subset=["Avg_Rating"])
                         .sort_values("Avg_Rating", ascending=False).head(15))
                if rated.empty:
                    st.info(f"No {group.lower()} tag has {min_rated_n}+ rated "
                            "watches yet.")
                else:
                    fig = px.bar(rated, x="Avg_Rating", y="Tag", orientation="h",
                                 title=f"Highest Rated — {group} "
                                       f"(min {min_rated_n} watches)",
                                 color="Avg_Rating", color_continuous_scale="Viridis",
                                 range_x=[0, 5],
                                 hover_data={"Count": True, "Avg_Rating": ":.2f"})
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(_style(fig), use_container_width=True)


# --- AUDIT ------------------------------------------------------------------
# `python app/theatre_tags.py` — read the list, show how every row parses, and
# flag what is missing or suspicious. The CSV is a Letterboxd export and is
# overwritten by the next one, so fixes belong on Letterboxd (or in the
# vocabularies above), never in the exported file.

def load_theatre_csv(path):
    """Read the list past its metadata preamble (the header row moves between
    exports, so find 'Position,' instead of trusting a fixed skiprows)."""
    with open(path, encoding="utf-8") as fh:
        lines = fh.readlines()
    header = next((i for i, ln in enumerate(lines) if ln.startswith("Position,")), 3)
    return pd.read_csv(path, skiprows=header)


def _near_misses(tag, cutoff=0.82):
    """Closed-vocabulary entries a person-classified tag nearly matches — i.e.
    likely typos that silently became a new 'person'."""
    import difflib
    vocab = {}
    for table, group in _LOOKUPS:
        for k, v in table.items():
            vocab[k] = f"{v} ({group})"
    hits = difflib.get_close_matches(tag.lower(), list(vocab), n=2, cutoff=cutoff)
    return [vocab[h] for h in hits]


def audit(path):
    df = load_theatre_csv(path)
    print(f"{len(df)} films in {path}\n")

    long_df = explode(df, rating_col="__none__")
    counts = long_df.groupby("Group")["Tag"].nunique().to_dict()
    print("distinct tags per dimension:")
    for g in GROUP_ORDER:
        print(f"   {g:12} {counts.get(g, 0)}")

    # Rows missing a dimension they could plausibly carry.
    print("\nrows with only a company tag (no venue/city/time recorded):")
    thin = 0
    for _, r in df.iterrows():
        p = parse_description(r.get("Description"))
        if p[GROUP_COMPANY] and not (p[GROUP_VENUE] or p[GROUP_CITY] or p[GROUP_TIME]):
            print(f"   {str(r['Name'])[:44]:44} {str(r.get('Description')).strip()}")
            thin += 1
    print(f"   ({thin} rows)")

    # People-classified tags that look like a misspelt vocabulary word.
    print("\npossible typos (classified as a person, but close to a known tag):")
    suspects = 0
    for tag in sorted({t for t in long_df[long_df["Group"] == GROUP_PERSON]["Tag"]}):
        near = _near_misses(tag)
        if near:
            print(f"   {tag:22} -> did you mean {', '.join(near)}?")
            suspects += 1
    if not suspects:
        print("   none")

    # Names appearing once are usually fine, but a single occurrence is also how
    # a typo of a frequent friend shows up.
    singles = summarise(long_df, GROUP_PERSON)
    singles = singles[singles["Count"] == 1]["Tag"].tolist()
    print(f"\npeople seen exactly once ({len(singles)}): {', '.join(singles)}")


if __name__ == "__main__":
    import sys
    from pathlib import Path

    default = (Path(__file__).resolve().parent.parent
               / "data" / "raw" / "movies" / "lists" / "theatre-watch.csv")
    audit(sys.argv[1] if len(sys.argv) > 1 else default)
