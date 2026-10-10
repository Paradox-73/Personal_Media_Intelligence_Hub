"""Hit or flop — each film's worldwide gross as a multiple of its production budget.

Budget and worldwide gross are TMDB's `budget` and `revenue` fields, written into
enriched_data.csv by src/movies/enrich_extended.py. OMDb's `box_office` is NOT used
here: it is the US domestic gross only (Mission: Impossible - Fallout: $220M
domestic vs $824M worldwide), so dividing it by a global budget makes nearly every
international hit look like a flop.

Why a film needs well over 1x its budget to break even:
- cinemas keep roughly half the ticket money (about 75% in China), so the studio
  sees ~40-50% of the gross;
- marketing and prints are not in the production budget and often cost 50-100% of it.
The usual rule of thumb is therefore 2-2.5x. The dashboard defaults to 2.5x and
lets the viewer move it; every tier is defined relative to that number.

Rows left out, and counted in the caption rather than silently dropped:
- budget or gross missing, or below $1,000 (TMDB placeholder entries such as "1");
- gross under 5% of budget: a streaming or TV release with a token theatrical run
  (The Killer, Mank, Frankenstein 2025), which is not a box-office flop.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

VERDICTS = ["Flop", "Underperformer", "Hit", "Blockbuster"]
# Validated with the dataviz palette checker on the #14171C dashboard surface:
# adjacent pairs pass colour-blind and normal-vision separation.
VERDICT_COLORS = {"Flop": "#ff5a4e", "Underperformer": "#e8c547",
                  "Hit": "#40bcf4", "Blockbuster": "#00e054"}
VERDICT_SYMBOLS = {"Flop": "x", "Underperformer": "diamond", "Hit": "circle",
                   "Blockbuster": "star"}
MIN_AMOUNT = 1_000
NO_THEATRICAL_RATIO = 0.05


def build_financials(df):
    """Return (films, excluded): films with a usable budget and worldwide gross,
    and the rows left out with the reason."""
    out = pd.DataFrame({
        "title": df["title"],
        "year": pd.to_numeric(df.get("year"), errors="coerce"),
        "user_rating": pd.to_numeric(df.get("user_rating"), errors="coerce"),
        "budget": pd.to_numeric(df.get("budget"), errors="coerce"),
        "gross": pd.to_numeric(df.get("revenue"), errors="coerce"),
    })
    has_both = (out["budget"] > 0) & (out["gross"] > 0)
    out = out[has_both].copy()
    out["multiple"] = out["gross"] / out["budget"]

    reason = pd.Series("", index=out.index)
    reason[(out["budget"] < MIN_AMOUNT) | (out["gross"] < MIN_AMOUNT)] = "Placeholder figures (< $1,000)"
    reason[(reason == "") & (out["multiple"] < NO_THEATRICAL_RATIO)] = "Streaming / token theatrical run"
    excluded = out[reason != ""].assign(reason=reason[reason != ""])
    return out[reason == ""].copy(), excluded


def classify(multiple, breakeven):
    """Flop: under 1x (did not gross its own budget). Underperformer: 1x to
    break-even. Hit: break-even to 2x break-even. Blockbuster: 2x break-even+."""
    return pd.cut(multiple, bins=[0, 1.0, breakeven, 2 * breakeven, np.inf],
                  labels=VERDICTS, right=False)


def _money(x):
    if x >= 1e9:
        return f"${x / 1e9:.2f}B"
    if x >= 1e6:
        return f"${x / 1e6:.1f}M"
    if x >= 1e3:
        return f"${x / 1e3:.0f}K"
    return f"${x:,.0f}"


def render_box_office_verdict(df, key="bo_verdict"):
    films, excluded = build_financials(df)
    if films.empty:
        st.info("No TMDB budget/revenue data. Run `python -m src.movies.enrich_extended --apply`.")
        return

    st.markdown("#### 🎟️ Hit or Flop?")
    breakeven = st.slider(
        "Break-even multiple (worldwide gross ÷ production budget)", 1.5, 4.0, 2.5, 0.25,
        key=f"{key}_be",
        help="Cinemas keep ~half of ticket sales and marketing is not in the budget, "
             "so a film usually needs 2–2.5× its budget worldwide to make money.")
    films["verdict"] = classify(films["multiple"], breakeven)

    st.caption(
        f"**Flop** < 1× budget · **Underperformer** 1×–{breakeven:g}× · "
        f"**Hit** {breakeven:g}×–{2 * breakeven:g}× · **Blockbuster** ≥ {2 * breakeven:g}×. "
        f"Figures are TMDB worldwide gross and production budget, nominal dollars "
        f"(the ratio needs no inflation adjustment). {len(films)} of {len(df)} films have both; "
        f"{len(excluded)} more left out (see below).")

    counts = films["verdict"].value_counts()
    cols = st.columns(4)
    for col, v in zip(cols, VERDICTS):
        sub = films[films["verdict"] == v]
        avg = sub["user_rating"].mean()
        col.metric(v, f"{int(counts.get(v, 0))}",
                   f"your avg {avg:.2f}★" if pd.notna(avg) else "no ratings",
                   delta_color="off")

    # --- Budget vs gross, log-log, with the tier boundaries as diagonals ---
    fig = go.Figure()
    lo = max(films["budget"].min(), 1e4) / 2
    hi = films["budget"].max() * 2
    for mult, label in [(1.0, "1× (gross = budget)"), (breakeven, f"{breakeven:g}× break-even"),
                        (2 * breakeven, f"{2 * breakeven:g}× blockbuster")]:
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo * mult, hi * mult], mode="lines", hoverinfo="skip",
            line=dict(color="#556678", width=1, dash="dot" if mult == 1.0 else "dash"),
            showlegend=False))
        fig.add_annotation(x=np.log10(hi), y=np.log10(hi * mult), text=label, showarrow=False,
                           xanchor="right", yanchor="bottom", font=dict(color="#9aa5b1", size=11))
    for v in VERDICTS:
        sub = films[films["verdict"] == v]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["budget"], y=sub["gross"], mode="markers", name=f"{v} ({len(sub)})",
            marker=dict(color=VERDICT_COLORS[v], symbol=VERDICT_SYMBOLS[v], size=9,
                        line=dict(width=1, color="#14171C"), opacity=0.85),
            customdata=np.stack([
                sub["title"], sub["year"].fillna(0).astype(int),
                sub["budget"].map(_money), sub["gross"].map(_money),
                sub["multiple"], sub["user_rating"].fillna(-1)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                          "Budget %{customdata[2]} · Worldwide %{customdata[3]}<br>"
                          "%{customdata[4]:.1f}× budget<br>Your rating %{customdata[5]}★"
                          "<extra></extra>"))
    fig.update_layout(
        title="Production budget vs worldwide gross (each dot is a film you logged)",
        xaxis=dict(type="log", title="Production budget (USD, log scale)", range=[np.log10(lo), np.log10(hi)]),
        yaxis=dict(type="log", title="Worldwide gross (USD, log scale)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        height=560, margin=dict(t=90))
    st.plotly_chart(fig, use_container_width=True, key=f"{key}_scatter")

    # --- Do you rate commercial success differently? ---
    c1, c2 = st.columns(2)
    with c1:
        agg = (films.dropna(subset=["user_rating"]).groupby("verdict", observed=False)["user_rating"]
               .agg(["mean", "count"]).reindex(VERDICTS))
        bar = go.Figure(go.Bar(
            x=VERDICTS, y=agg["mean"], marker_color=[VERDICT_COLORS[v] for v in VERDICTS],
            marker_line_width=2, marker_line_color="#14171C",
            text=[f"{m:.2f}★ · n={int(n)}" if pd.notna(m) else "" for m, n in zip(agg["mean"], agg["count"].fillna(0))],
            textposition="outside", textfont=dict(color="#E0E0E0"),
            hovertemplate="%{x}: your average %{y:.2f}★<extra></extra>"))
        bar.update_layout(title="Your average rating by box-office verdict",
                          yaxis=dict(range=[0, 5.4], title="Your rating"), showlegend=False)
        st.plotly_chart(bar, use_container_width=True, key=f"{key}_bar")
    with c2:
        rated = films.dropna(subset=["user_rating"])
        if len(rated) > 2:
            rho = rated["user_rating"].corr(np.log(rated["multiple"]), method="spearman")
            st.metric("Rank correlation: your rating vs gross ÷ budget", f"{rho:+.2f}",
                      help="Spearman correlation. +1 means the bigger the return, the higher you rate it; "
                           "0 means box-office return says nothing about your taste.")
        top = films.sort_values("multiple")
        show = lambda d: d.assign(Budget=d["budget"].map(_money), Worldwide=d["gross"].map(_money),
                                  Multiple=d["multiple"].map(lambda m: f"{m:.2f}×"),
                                  Rating=d["user_rating"])[["title", "year", "Budget", "Worldwide", "Multiple", "Rating"]]
        st.markdown("**Biggest flops you've seen**")
        st.dataframe(show(top.head(5)), hide_index=True, use_container_width=True)
        st.markdown("**Biggest returns you've seen**")
        st.dataframe(show(top.tail(5).iloc[::-1]), hide_index=True, use_container_width=True)

    loved_flops = films[films["verdict"].isin(["Flop", "Underperformer"]) & (films["user_rating"] >= 4)]
    disliked_hits = films[films["verdict"].eq("Blockbuster") & (films["user_rating"] <= 2.5)]
    c3, c4 = st.columns(2)
    with c3:
        st.markdown(f"**Flops & underperformers you rated 4★+** ({len(loved_flops)})")
        st.dataframe(show(loved_flops.sort_values("user_rating", ascending=False)),
                     hide_index=True, use_container_width=True)
    with c4:
        st.markdown(f"**Blockbusters you rated 2.5★ or less** ({len(disliked_hits)})")
        st.dataframe(show(disliked_hits.sort_values("user_rating")),
                     hide_index=True, use_container_width=True)

    with st.expander(f"Left out of the verdict ({len(excluded)} films)"):
        st.caption("Streaming releases with a token theatrical run would otherwise show up as "
                   "catastrophic flops; placeholder entries are TMDB data-entry errors.")
        st.dataframe(excluded.assign(Budget=excluded["budget"].map(_money), Worldwide=excluded["gross"].map(_money))
                     [["title", "year", "Budget", "Worldwide", "reason"]],
                     hide_index=True, use_container_width=True)
