"""Shared geography helpers for the dashboards.

Turns the free-text ``country`` field (TMDB-style English names, plus a few
abbreviations like ``USA`` / ``JP`` / ``FR``) into ISO-3166 alpha-3 codes so
they can be plotted on a Plotly choropleth without any external dependency
(no ``pycountry`` required).
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px

# Canonical ISO-3 -> display name. Drives both the reverse lookup used for the
# map tooltips and (lower-cased) the forward name -> ISO-3 resolution below.
ISO3_TO_NAME = {
    "USA": "United States", "GBR": "United Kingdom", "IND": "India",
    "CAN": "Canada", "JPN": "Japan", "FRA": "France", "AUS": "Australia",
    "DEU": "Germany", "CHN": "China", "NZL": "New Zealand", "HKG": "Hong Kong",
    "ITA": "Italy", "MEX": "Mexico", "KOR": "South Korea", "PRK": "North Korea",
    "ARE": "United Arab Emirates", "HUN": "Hungary", "CZE": "Czech Republic",
    "THA": "Thailand", "IRL": "Ireland", "ESP": "Spain", "ISL": "Iceland",
    "FIN": "Finland", "BEL": "Belgium", "SWE": "Sweden", "NOR": "Norway",
    "CHE": "Switzerland", "AUT": "Austria", "JOR": "Jordan", "RUS": "Russia",
    "MAR": "Morocco", "TWN": "Taiwan", "TUR": "Turkey", "ZAF": "South Africa",
    "DNK": "Denmark", "SGP": "Singapore", "NLD": "Netherlands", "POL": "Poland",
    "BRA": "Brazil", "KHM": "Cambodia", "MLT": "Malta", "VNM": "Vietnam",
    "GRC": "Greece", "GMB": "Gambia", "KAZ": "Kazakhstan", "ARM": "Armenia",
    "NPL": "Nepal", "JAM": "Jamaica", "GEO": "Georgia", "UKR": "Ukraine",
    "SRB": "Serbia", "PAK": "Pakistan", "PER": "Peru", "SVN": "Slovenia",
    "EGY": "Egypt", "MCO": "Monaco", "BGD": "Bangladesh", "BGR": "Bulgaria",
    "PHL": "Philippines", "CHL": "Chile", "ARG": "Argentina",
    "COL": "Colombia", "PRT": "Portugal", "ROU": "Romania", "HRV": "Croatia",
    "ISR": "Israel", "IRN": "Iran", "IRQ": "Iraq", "SAU": "Saudi Arabia",
    "QAT": "Qatar", "LBN": "Lebanon", "IDN": "Indonesia", "MYS": "Malaysia",
    "LKA": "Sri Lanka", "NGA": "Nigeria", "KEN": "Kenya", "GHA": "Ghana",
    "ETH": "Ethiopia", "TZA": "Tanzania", "UGA": "Uganda", "DZA": "Algeria",
    "TUN": "Tunisia", "CUB": "Cuba", "VEN": "Venezuela", "URY": "Uruguay",
    "BOL": "Bolivia", "ECU": "Ecuador", "PRY": "Paraguay", "CRI": "Costa Rica",
    "PAN": "Panama", "GTM": "Guatemala", "LUX": "Luxembourg", "LTU": "Lithuania",
    "LVA": "Latvia", "EST": "Estonia", "SVK": "Slovakia", "BIH": "Bosnia and Herzegovina",
    "MKD": "North Macedonia", "ALB": "Albania", "CYP": "Cyprus", "BLR": "Belarus",
    "AZE": "Azerbaijan", "UZB": "Uzbekistan", "MNG": "Mongolia", "MMR": "Myanmar",
    "BTN": "Bhutan", "AFG": "Afghanistan", "ZWE": "Zimbabwe", "SEN": "Senegal",
    "CMR": "Cameroon", "COD": "Democratic Republic of the Congo", "AGO": "Angola",
}

# name (lower-case) -> ISO-3. Built from the canonical names above, then
# extended with abbreviations / historical / spelling variants seen in the data.
COUNTRY_TO_ISO3 = {name.lower(): iso for iso, name in ISO3_TO_NAME.items()}
COUNTRY_TO_ISO3.update({
    "united states of america": "USA", "us": "USA", "u.s.": "USA",
    "u.s.a.": "USA", "usa": "USA", "america": "USA",
    "uk": "GBR", "u.k.": "GBR", "great britain": "GBR", "england": "GBR",
    "scotland": "GBR", "wales": "GBR", "northern ireland": "GBR",
    "russian federation": "RUS", "soviet union": "RUS", "ussr": "RUS",
    "south korea": "KOR", "korea, republic of": "KOR", "republic of korea": "KOR",
    "north korea": "PRK",
    "czechia": "CZE", "czechoslovakia": "CZE",
    "jp": "JPN", "fr": "FRA", "de": "DEU", "it": "ITA", "es": "ESP",
    "in": "IND", "cn": "CHN", "ca": "CAN", "au": "AUS",
    "hong kong sar china": "HKG", "hong kong sar": "HKG",
    "taiwan, province of china": "TWN", "republic of china": "TWN",
    "uae": "ARE", "the netherlands": "NLD", "holland": "NLD",
    "viet nam": "VNM", "republic of serbia": "SRB",
    "macedonia": "MKD", "burma": "MMR", "ivory coast": "CIV",
    "côte d'ivoire": "CIV", "cote d'ivoire": "CIV",
    "congo": "COD", "dr congo": "COD",
})


# ISO 3166-1 alpha-2 -> alpha-3, so values that arrive as 2-letter codes
# (e.g. MusicBrainz artist ``country`` = "GB") resolve too. Derived for every
# country we know a name for, plus a couple of common extras.
ISO2_TO_ISO3 = {
    "US": "USA", "GB": "GBR", "IN": "IND", "CA": "CAN", "JP": "JPN", "FR": "FRA",
    "AU": "AUS", "DE": "DEU", "CN": "CHN", "NZ": "NZL", "HK": "HKG", "IT": "ITA",
    "MX": "MEX", "KR": "KOR", "KP": "PRK", "AE": "ARE", "HU": "HUN", "CZ": "CZE",
    "TH": "THA", "IE": "IRL", "ES": "ESP", "IS": "ISL", "FI": "FIN", "BE": "BEL",
    "SE": "SWE", "NO": "NOR", "CH": "CHE", "AT": "AUT", "JO": "JOR", "RU": "RUS",
    "MA": "MAR", "TW": "TWN", "TR": "TUR", "ZA": "ZAF", "DK": "DNK", "SG": "SGP",
    "NL": "NLD", "PL": "POL", "BR": "BRA", "KH": "KHM", "MT": "MLT", "VN": "VNM",
    "GR": "GRC", "GM": "GMB", "KZ": "KAZ", "AM": "ARM", "NP": "NPL", "JM": "JAM",
    "GE": "GEO", "UA": "UKR", "RS": "SRB", "PK": "PAK", "PE": "PER", "SI": "SVN",
    "EG": "EGY", "MC": "MCO", "BD": "BGD", "BG": "BGR", "PH": "PHL", "CL": "CHL",
    "AR": "ARG", "CO": "COL", "PT": "PRT", "RO": "ROU", "HR": "HRV", "IL": "ISR",
    "IR": "IRN", "IQ": "IRQ", "SA": "SAU", "QA": "QAT", "LB": "LBN", "ID": "IDN",
    "MY": "MYS", "LK": "LKA", "NG": "NGA", "KE": "KEN", "GH": "GHA", "ET": "ETH",
    "TZ": "TZA", "UG": "UGA", "DZ": "DZA", "TN": "TUN", "CU": "CUB", "VE": "VEN",
    "UY": "URY", "BO": "BOL", "EC": "ECU", "PY": "PRY", "CR": "CRI", "PA": "PAN",
    "GT": "GTM", "LU": "LUX", "LT": "LTU", "LV": "LVA", "EE": "EST", "SK": "SVK",
    "BA": "BIH", "MK": "MKD", "AL": "ALB", "CY": "CYP", "BY": "BLR", "AZ": "AZE",
    "UZ": "UZB", "MN": "MNG", "MM": "MMR", "BT": "BTN", "AF": "AFG", "ZW": "ZWE",
    "SN": "SEN", "CM": "CMR", "CD": "COD", "AO": "AGO", "CI": "CIV",
}


def _to_iso3(name):
    """Resolve a country name OR an ISO alpha-2/alpha-3 code to alpha-3.

    Returns ``None`` if unknown. Names win over codes, so aliases like
    ``"us"``/``"uk"`` keep working.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    raw = str(name).strip()
    key = raw.lower()
    if key in ("", "nan", "none", "unknown", "n/a", "na"):
        return None
    if key in COUNTRY_TO_ISO3:
        return COUNTRY_TO_ISO3[key]
    up = raw.upper()
    if len(up) == 2 and up in ISO2_TO_ISO3:
        return ISO2_TO_ISO3[up]
    if len(up) == 3 and up in ISO3_TO_NAME:
        return up
    return None


def iso2_to_name(code):
    """Best-effort full country name for an ISO alpha-2 code (else the code)."""
    if not code:
        return None
    iso3 = ISO2_TO_ISO3.get(str(code).strip().upper())
    return ISO3_TO_NAME.get(iso3, str(code).strip().upper())


def aggregate_country_counts(df, country_col, rating_col="user_rating"):
    """Explode ``country_col`` (a list-per-row column) and count items per country.

    Returns ``(agg_df, unmapped)`` where ``agg_df`` has columns
    ``iso3 / country / Count / avg_rating`` and ``unmapped`` is the sorted list
    of country names that could not be matched to an ISO-3 code (so the caller
    can surface them instead of silently dropping them).
    """
    if country_col not in df.columns:
        return pd.DataFrame(columns=["iso3", "country", "Count", "avg_rating"]), []

    exp = df.explode(country_col)
    exp = exp.dropna(subset=[country_col])
    exp["iso3"] = exp[country_col].map(_to_iso3)

    unmapped = sorted({
        str(n).strip() for n in exp.loc[exp["iso3"].isna(), country_col]
        if str(n).strip().lower() not in ("", "nan", "none", "unknown", "n/a", "na")
    })

    exp = exp.dropna(subset=["iso3"])
    if exp.empty:
        return pd.DataFrame(columns=["iso3", "country", "Count", "avg_rating"]), unmapped

    has_rating = rating_col in exp.columns
    agg = exp.groupby("iso3").agg(
        Count=("iso3", "size"),
        avg_rating=(rating_col, "mean") if has_rating else ("iso3", "size"),
    ).reset_index()
    if not has_rating:
        agg["avg_rating"] = np.nan
    agg["avg_rating"] = agg["avg_rating"].round(2)
    agg["country"] = agg["iso3"].map(ISO3_TO_NAME).fillna(agg["iso3"])
    return agg.sort_values("Count", ascending=False).reset_index(drop=True), unmapped


def make_world_map(agg, *, low_color, high_color, value_col="Count", title=None,
                   land_color="#222831", ocean_color="#14171C", height=480,
                   log_scale=True):
    """Build a dark-themed choropleth coloured by ``value_col``.

    Colour uses a log scale by default so smaller countries stay visible despite
    the heavy skew toward a dominant country; the colour bar ticks and the hover
    tooltip both report the real item counts.
    """
    plot = agg.copy()
    color_col = value_col
    if log_scale:
        plot["_intensity"] = np.log1p(plot[value_col])
        color_col = "_intensity"

    fig = px.choropleth(
        plot, locations="iso3", color=color_col, hover_name="country",
        custom_data=["Count", "avg_rating"],
        color_continuous_scale=[[0, low_color], [1, high_color]],
    )
    fig.update_traces(
        marker_line_color="#14171C", marker_line_width=0.4,
        hovertemplate=(
            "<b>%{hovertext}</b><br>Items: %{customdata[0]}"
            "<br>Avg rating: %{customdata[1]:.2f}<extra></extra>"
        ),
    )

    if log_scale:
        mx = int(plot[value_col].max())
        ticks = [t for t in (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000) if t <= mx]
        if not ticks or ticks[-1] != mx:
            ticks.append(mx)
        fig.update_coloraxes(colorbar=dict(
            title="Items",
            tickvals=[math.log1p(t) for t in ticks],
            ticktext=[str(t) for t in ticks],
        ))
    else:
        fig.update_coloraxes(colorbar=dict(title="Items"))

    fig.update_geos(
        projection_type="natural earth", showframe=False,
        showcoastlines=True, coastlinecolor="#556678",
        showland=True, landcolor=land_color,
        showocean=True, oceancolor=ocean_color,
        showlakes=True, lakecolor=ocean_color,
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40 if title else 0, b=0), height=height,
        paper_bgcolor="rgba(0,0,0,0)", geo_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"), title=title,
    )
    return fig
