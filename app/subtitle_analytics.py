"""
subtitle_analytics.py — the "What the dialogue says" section of the Movies Dashboard.

Reads data/processed/movies/subtitle_features.csv, built by
src/movies/subtitle_features.py; src/movies/subtitles.py refreshes it after every
daily download run, so this section grows as subtitles arrive.

Curves run on STORY time: 0% = first spoken line, 100% = last, so opening titles
and end credits (no dialogue) don't flatten the ends.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config
from src.common.parsing import canon_genres
from src.movies import subtitles as subs
from src.movies import subtitle_features as sf

# Categorical colours in fixed order (validated for colour-blind separation);
# groups beyond 7 are not drawn rather than given a recycled colour.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767"]
MAX_GROUPS = len(PALETTE)
X = (np.arange(sf.CURVE_BINS) + 0.5) * 100 / sf.CURVE_BINS  # % of story time, bin centres
MOOD_COLS = [f"mood_{k:02d}" for k in range(sf.CURVE_BINS)]
TALK_COLS = [f"talk_{k:02d}" for k in range(sf.CURVE_BINS)]
LANG = {"en": "English", "hi": "Hindi", "ja": "Japanese", "ko": "Korean", "fr": "French",
        "es": "Spanish", "cn": "Cantonese", "zh": "Mandarin", "de": "German", "it": "Italian"}
BANDS = ["0.5–2 ★", "2.5–3 ★", "3.5–4 ★", "4.5–5 ★"]


@st.cache_data(ttl=600)
def load():
    if not sf.FEATURES_PATH.exists():
        return None, 0
    f = pd.read_csv(sf.FEATURES_PATH, dtype={"imdb_id": str})
    m = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH, dtype={"imdb_id": str})
    m = m[m["user_rating"].notna()]
    # same key the subtitle files use: IMDb id, else tmdb<id> for the few without one
    m = m.assign(imdb_id=subs.movie_keys(m)).dropna(subset=["imdb_id"]).drop_duplicates("imdb_id")
    total = len(m)
    m = m[["imdb_id", "title", "year", "user_rating", "imdb_rating", "genre",
           "original_language", "director"]]
    m["genres"] = m["genre"].apply(canon_genres)
    m["year"] = pd.to_numeric(m["year"], errors="coerce")
    m["imdb_rating"] = pd.to_numeric(m["imdb_rating"], errors="coerce")
    m["decade"] = (m["year"] // 10 * 10).map(lambda y: f"{int(y)}s" if pd.notna(y) else "Unknown")
    m["language"] = m["original_language"].map(lambda c: LANG.get(c, str(c).upper()))
    m["rating_band"] = pd.cut(m["user_rating"], [0, 2, 3, 4, 5], labels=BANDS).astype(str)
    m["label"] = m["title"] + " (" + m["year"].map(lambda y: f"{int(y)}" if pd.notna(y) else "?") + ")"
    # derived columns go on the small movie table: adding them to the 130-column
    # feature table one by one fragments it
    return f.merge(m, on="imdb_id", how="left"), total


# Subtitle filler on top of sklearn's English stop words, so the word lists show
# what films talk about rather than "yeah" and "gonna".
FILLER = {"yeah", "okay", "hey", "gonna", "wanna", "gotta", "uh", "um", "huh", "oh", "ah", "hmm",
          "yes", "just", "know", "like", "right", "got", "get", "let", "come", "going", "think",
          "want", "really", "tell", "say", "look", "mean", "well", "didn", "don", "doesn",
          "isn", "wasn", "aren", "won", "ain", "mr", "mrs", "did", "does", "doing", "said",
          "gets", "getting", "thing", "things", "maybe", "way"}


@st.cache_data(persist="disk", show_spinner="Counting words in every subtitle file…")
def film_word_counts(keys, filler):
    """{imdb_id: (total words, content-word counts, swear counts)}.

    `keys` is a tuple of (imdb_id, file_sig) so the disk cache is rebuilt only
    when a subtitle file is added or changed (filler is an argument for the
    same reason: editing FILLER rebuilds the cache). Content words = 3+ letters, no
    apostrophe, not a stop word or filler. Swears use the same pattern as the
    swear_per_1k feature (src/movies/subtitle_features.py)."""
    out = {}
    for imdb_id, _sig in keys:
        d = subs.load_subtitles(imdb_id)
        if d is None:
            continue
        toks = sf.tokens(d.loc[d["dialogue"] != "", "dialogue"])
        words = Counter(w for w in toks if len(w) > 2 and "'" not in w
                        and w not in ENGLISH_STOP_WORDS and w not in filler)
        swears = Counter(w for w in toks if sf._SWEAR.fullmatch(w))
        out[imdb_id] = (len(toks), dict(words), dict(swears))
    return out


def sum_counts(counts, ids, part):
    """Add up one of the per-film counts (1 = words, 2 = swears) over `ids`.
    Returns (total occurrences, number of films each word appears in)."""
    total, films = Counter(), Counter()
    for i in ids:
        if i in counts:
            total.update(counts[i][part])
            films.update(counts[i][part].keys())
    return total, films


def group_members(frame, by):
    """(group name, rows) pairs; a film with 3 genres sits in all 3 genre groups."""
    if by == "Genre":
        ex = frame.explode("genres").dropna(subset=["genres"])
        return {g: frame.loc[ex.index[ex["genres"] == g].unique()] for g in ex["genres"].unique()}
    col = {"Rating band": "rating_band", "Decade": "decade", "Language": "language"}[by]
    return {g: rows for g, rows in frame.groupby(col) if g != "nan"}


def smooth(y, window):
    return pd.Series(y).rolling(window, center=True, min_periods=1).mean().to_numpy() \
        if window > 1 else np.asarray(y)


def layout(fig, x_title, y_title, height=460):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=x_title, yaxis_title=y_title, hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig




def render_subtitle_analytics():
    """Whole section: filters, 5 tabs. Every widget key starts with 'subs_' so it
    cannot collide with the rest of the dashboard."""
    df, total = load()
    st.subheader("💬 What the Dialogue Says (Subtitles)")
    if df is None or df.empty:
        st.info("No subtitle features yet. Run `python -m src.movies.subtitles` to download; "
                "this section fills in as films arrive.")
        return

    st.caption(
        "Every number here comes from the film's English subtitle file. Mood = VADER sentiment "
        "of each spoken line (−1 very negative … +1 very positive). Curves run on **story time**: "
        "0% = first spoken line, 100% = last, so opening titles and end credits don't flatten the "
        "ends. Sound cues ([door slams], (laughs)) exist only in hearing-impaired (SDH) files.")

    f1, f2, f3, f4 = st.columns(4)
    r_lo, r_hi = f1.slider("Your rating", 0.5, 5.0, (0.5, 5.0), 0.5, key="subs_rating")
    all_genres = sorted({g for gl in df["genres"] for g in gl})
    pick_genres = f2.multiselect("Genres (any of)", all_genres, key="subs_genres")
    pick_decades = f3.multiselect("Decades", sorted(df["decade"].unique()), key="subs_decades")
    pick_langs = f4.multiselect("Original language", df["language"].value_counts().index.tolist(),
                                key="subs_langs")

    view = df[df["user_rating"].between(r_lo, r_hi)]
    if pick_genres:
        view = view[view["genres"].apply(lambda gl: any(g in gl for g in pick_genres))]
    if pick_decades:
        view = view[view["decade"].isin(pick_decades)]
    if pick_langs:
        view = view[view["language"].isin(pick_langs)]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Films with subtitles", f"{len(df)} / {total}")
    c2.metric("In current filter", len(view))
    c3.metric("With sound cues (SDH)", int(view["has_cues"].sum()))
    c4.metric("Median words / min", f"{view['words_per_min'].median():.0f}" if len(view) else "–")
    c5.metric("Average mood", f"{view['sent_mean'].mean():+.3f}" if len(view) else "–")
    if len(view) < 5:
        st.warning("Fewer than 5 films match the filters.")
        return

    tab_curve, tab_rate, tab_group, tab_swear, tab_film, tab_words = st.tabs(
        ["📈 Curves", "⭐ Style vs your rating", "🎭 By genre / decade / language",
         "🤬 Swears", "🎬 One film", "🔤 Words"])

    # ---------------- 1. curves ----------------
    with tab_curve:
        a, b, c, d = st.columns([1.2, 1.4, 1, 1])
        curve = a.radio("Curve", ["Mood", "Talk (words per minute)"], horizontal=True, key="subs_curve")
        by = b.selectbox("Overlay by", ["Rating band", "Genre", "Decade", "Language", "Individual films"],
                         key="subs_by")
        window = c.slider("Smoothing (points of 2%)", 1, 15, 5,
                          help="Rolling average over this many 2% points. 5 = a 10% window.",
                          key="subs_window")
        show_band = d.checkbox("Show ±1 standard error", value=True, key="subs_band")
        cols = MOOD_COLS if curve == "Mood" else TALK_COLS

        fig = go.Figure()
        if by == "Individual films":
            titles = st.multiselect(f"Films (up to {MAX_GROUPS})", view.sort_values("user_rating",
                                    ascending=False)["label"].tolist(), max_selections=MAX_GROUPS,
                                    key="subs_titles")
            base = smooth(view[cols].mean().to_numpy(), window)
            fig.add_trace(go.Scatter(x=X, y=base, name=f"Average of filtered films (n={len(view)})",
                                     line=dict(color="#8a8a86", width=2, dash="dash"),
                                     hovertemplate="%{x:.0f}%: %{y:.3f}<extra>average</extra>"))
            for i, t in enumerate(titles):
                row = view[view["label"] == t].iloc[0]
                fig.add_trace(go.Scatter(
                    x=X, y=smooth(row[cols].astype(float).to_numpy(), window),
                    name=f"{t} · {row['user_rating']}★", line=dict(color=PALETTE[i], width=2),
                    hovertemplate="%{x:.0f}%: %{y:.3f}<extra>" + t + "</extra>"))
        else:
            groups = {g: rows for g, rows in group_members(view, by).items() if len(rows) >= 5}
            order = BANDS if by == "Rating band" else sorted(groups, key=lambda g: -len(groups[g]))
            options = [g for g in order if g in groups]
            chosen = st.multiselect(f"Groups (up to {MAX_GROUPS}; groups need 5+ films)", options,
                                    default=options[:4 if by != "Rating band" else MAX_GROUPS],
                                    max_selections=MAX_GROUPS, key=f"subs_groups_{by}")
            for i, g in enumerate(chosen):
                rows = groups[g]
                m = smooth(rows[cols].mean().to_numpy(), window)
                se = smooth((rows[cols].std() / np.sqrt(rows[cols].notna().sum())).to_numpy(), window)
                color = PALETTE[i]
                if show_band:
                    fig.add_trace(go.Scatter(
                        x=np.r_[X, X[::-1]], y=np.r_[m + se, (m - se)[::-1]], fill="toself",
                        fillcolor=color, opacity=0.15, line=dict(width=0), hoverinfo="skip",
                        showlegend=False))
                fig.add_trace(go.Scatter(x=X, y=m, name=f"{g} (n={len(rows)})",
                                         line=dict(color=color, width=2),
                                         hovertemplate="%{x:.0f}%: %{y:.3f}<extra>" + str(g) + "</extra>"))
        y_title = "Mood (VADER, −1…+1)" if curve == "Mood" else "Words per minute"
        st.plotly_chart(layout(fig, "% through the story (first to last spoken line)", y_title),
                        width="stretch")
        st.caption("Each point = the average over films in the group for that 2% slice of the story. "
                   "Shaded band = ±1 standard error: where two bands overlap, the difference is "
                   "within noise.")

    # ---------------- 2. style vs rating ----------------
    with tab_rate:
        cor = sf.correlations(view)
        if cor.empty:
            st.info("Need 30+ rated films with IMDb scores in the filter.")
        else:
            c = cor.iloc[::-1]
            fig = go.Figure()
            for col, name, color in (("r_raw", "With your rating", PALETTE[0]),
                                     ("r_beyond_crowd", "With your rating beyond IMDb score", PALETTE[1])):
                fig.add_trace(go.Scatter(
                    x=c[col], y=c["label"], mode="markers", name=name,
                    marker=dict(size=10, color=color, line=dict(width=1.5, color="white")),
                    customdata=np.c_[c["n"], c["q_raw" if col == "r_raw" else "q_beyond_crowd"]],
                    hovertemplate="%{y}<br>r = %{x:.3f}<br>films = %{customdata[0]}"
                                  "<br>q = %{customdata[1]:.3f}<extra>" + name + "</extra>"))
            fig.add_vline(x=0, line_width=1, line_color="#8a8a86")
            st.plotly_chart(layout(fig, "Spearman correlation (−1…+1)", "", height=30 * len(c) + 120),
                            width="stretch")
            st.caption(
                "**Beyond IMDb score** = correlation with the part of your rating the IMDb score "
                "doesn't already predict — the only part a model could gain from. "
                "**q** = p-value corrected for testing all these features at once; q < 0.1 is "
                "the bar for 'probably real'.")
            tbl = cor[["label", "n", "r_raw", "q_raw", "r_beyond_crowd", "q_beyond_crowd"]].rename(columns={
                "label": "Feature", "n": "Films", "r_raw": "r (your rating)", "q_raw": "q",
                "r_beyond_crowd": "r (beyond IMDb)", "q_beyond_crowd": "q (beyond IMDb)"})
            st.dataframe(tbl.round(3), hide_index=True, width="stretch")

        st.subheader("One feature against your rating")
        feat = st.selectbox("Feature", list(sf.LABELS), format_func=sf.LABELS.get, key="subs_scatter_feat")
        pts = view[["label", "user_rating", feat, "imdb_rating"]].dropna(subset=[feat])
        jit = np.random.default_rng(0).uniform(-0.15, 0.15, len(pts))
        fig = go.Figure(go.Scatter(
            x=pts["user_rating"] + jit, y=pts[feat], mode="markers", name="Film",
            marker=dict(size=8, color=PALETTE[0], opacity=0.55),
            customdata=np.c_[pts["label"], pts["user_rating"], pts["imdb_rating"]],
            hovertemplate="%{customdata[0]}<br>your rating %{customdata[1]}★ · IMDb %{customdata[2]}"
                          "<br>" + sf.LABELS[feat] + ": %{y:.3f}<extra></extra>"))
        med = pts.groupby("user_rating")[feat].median()
        fig.add_trace(go.Scatter(x=med.index, y=med.values, mode="lines+markers", name="Median",
                                 line=dict(color=PALETTE[1], width=2), marker=dict(size=8)))
        st.plotly_chart(layout(fig, "Your rating", sf.LABELS[feat]), width="stretch")

    # ---------------- 3. by group ----------------
    with tab_group:
        a, b = st.columns(2)
        gby = a.selectbox("Group by", ["Genre", "Decade", "Language", "Rating band"], key="subs_gby")
        gfeat = b.selectbox("Feature", list(sf.LABELS), format_func=sf.LABELS.get, key="subs_gfeat")
        rows = []
        for g, part in group_members(view, gby).items():
            v = part[gfeat].dropna()
            if len(v) >= 5:
                rows.append({"group": g, "n": len(v), "min": v.min(), "q1": v.quantile(.25),
                             "med": v.median(), "mean": v.mean(), "q3": v.quantile(.75),
                             "max": v.max(), "min_film": part.loc[v.idxmin(), "label"],
                             "max_film": part.loc[v.idxmax(), "label"],
                             "rating": part["user_rating"].mean()})
        if not rows:
            st.info("No group has 5+ films with this feature in the current filter.")
        else:
            gt = pd.DataFrame(rows).sort_values("med")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=np.r_[gt["q1"], gt["q3"]], y=np.r_[gt["group"], gt["group"]], mode="markers",
                marker=dict(size=1, color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"))
            for _, r in gt.iterrows():
                fig.add_shape(type="line", x0=r["q1"], x1=r["q3"], y0=r["group"], y1=r["group"],
                              line=dict(color=PALETTE[0], width=4), opacity=0.35)
            for col, name, sym in (("min", "Min", "line-ns-open"), ("max", "Max", "line-ns-open"),
                                   ("mean", "Average", "diamond-open")):
                fig.add_trace(go.Scatter(
                    x=gt[col], y=gt["group"], mode="markers", name=name,
                    marker=dict(size=12 if col != "mean" else 10, symbol=sym,
                                color="#8a8a86" if col != "mean" else PALETTE[1],
                                line=dict(width=2, color="#8a8a86" if col != "mean" else PALETTE[1])),
                    customdata=np.c_[gt[f"{col}_film"] if col != "mean" else gt["n"]],
                    hovertemplate="%{y}<br>" + name + " %{x:.3f}"
                                  + ("<br>%{customdata[0]}" if col != "mean" else "") + "<extra></extra>"))
            fig.add_trace(go.Scatter(
                x=gt["med"], y=gt["group"], mode="markers", name="Median",
                marker=dict(size=11, color=gt["rating"], colorscale="Blues", cmin=1, cmax=5,
                            line=dict(width=1.5, color="white"),
                            colorbar=dict(title="Your avg ★", thickness=12)),
                customdata=np.c_[gt["n"], gt["rating"], gt["q1"], gt["q3"]],
                hovertemplate="%{y}<br>median %{x:.3f} (middle 50%: %{customdata[2]:.3f}–"
                              "%{customdata[3]:.3f})<br>films %{customdata[0]} · your avg "
                              "%{customdata[1]:.2f}★<extra></extra>"))
            st.plotly_chart(layout(fig, sf.LABELS[gfeat], "", height=34 * len(gt) + 120),
                            width="stretch")
            st.caption("Filled dot = median, open diamond = average, ticks = min and max, bar = middle "
                       "50% of films. Dot shade = your average rating for that group (darker = higher). "
                       "Groups need 5+ films.")
            tbl = gt.iloc[::-1][["group", "n", "min", "med", "mean", "max", "min_film", "max_film", "rating"]]
            st.dataframe(tbl.rename(columns={
                "group": gby, "n": "Films", "min": "Min", "med": "Median", "mean": "Average",
                "max": "Max", "min_film": "Lowest film", "max_film": "Highest film",
                "rating": "Your avg ★"}).round(3), hide_index=True, width="stretch")

        st.markdown("**Most used words** (stop words removed)")
        members = group_members(view, gby)
        wa, wb = st.columns([2, 1])
        word_pick = wa.selectbox(gby, ["All films"] + sorted(members, key=lambda g: -len(members[g])),
                                 key="subs_words_group")
        word_n = wb.slider("Words shown", 10, 40, 20, 5, key="subs_words_n")
        ids = view["imdb_id"] if word_pick == "All films" else members[word_pick]["imdb_id"]
        counts = film_word_counts(tuple(zip(view["imdb_id"], view["file_sig"].astype(str))), tuple(sorted(FILLER)))
        totals, in_films = sum_counts(counts, ids, 1)
        if not totals:
            st.info("No subtitle words for this group.")
        else:
            wt = pd.DataFrame(totals.most_common(word_n), columns=["Word", "Count"])
            wt["Films"] = wt["Word"].map(in_films)
            fig = go.Figure(go.Bar(x=wt["Count"][::-1], y=wt["Word"][::-1], orientation="h",
                                   marker=dict(color=wt["Count"][::-1],
                                               colorscale=[[0, "#c9dcf5"], [1, PALETTE[0]]]),
                                   customdata=wt["Films"][::-1],
                                   hovertemplate="%{y}: %{x:,} times in %{customdata} films<extra></extra>"))
            n_films = sum(1 for i in ids if i in counts)
            st.plotly_chart(layout(fig, "Total times said", "", height=22 * len(wt) + 120)
                            .update_layout(title=f"Top {len(wt)} words · {word_pick} ({n_films} films)"),
                            width="stretch")

    # ---------------- swears ----------------
    with tab_swear:
        counts = film_word_counts(tuple(zip(view["imdb_id"], view["file_sig"].astype(str))), tuple(sorted(FILLER)))
        sw = view[["imdb_id", "label", "user_rating", "imdb_rating"]].copy()
        sw = sw[sw["imdb_id"].isin(counts)]
        sw["words"] = sw["imdb_id"].map(lambda i: counts[i][0])
        sw["swears"] = sw["imdb_id"].map(lambda i: sum(counts[i][2].values()))
        # Share of all spoken words that are swears (swears / total words).
        sw["swear_share"] = 100 * sw["swears"] / sw["words"].clip(lower=1)
        a, b = st.columns([2, 1])
        rank_by = a.radio("Rank by", ["Total swears", "Swears ÷ total words"], horizontal=True,
                          key="subs_swear_by")
        top_n = b.slider("Films shown", 10, 50, 20, 5, key="subs_swear_n")
        col = "swears" if rank_by == "Total swears" else "swear_share"
        top = sw.nlargest(top_n, col).iloc[::-1]
        fig = go.Figure(go.Bar(
            x=top[col], y=top["label"], orientation="h", marker_color=PALETTE[1],
            text=top[col].map(lambda v: f"{v:.0f}" if col == "swears" else f"{v:.2f}%"), textposition="outside",
            customdata=np.c_[top["swears"], top["words"], top["swear_share"], top["user_rating"]],
            hovertemplate="%{y}<br>%{customdata[0]} swears in %{customdata[1]:,} words "
                          "(%{customdata[2]:.2f}%)<br>your rating %{customdata[3]}★<extra></extra>"))
        st.plotly_chart(layout(fig, "Swears" if col == "swears" else "Swears as % of all words", "",
                               height=24 * len(top) + 120), width="stretch")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Swears across filtered films", f"{sw['swears'].sum():,}")
        c2.metric("Swears ÷ total words", f"{100 * sw['swears'].sum() / max(sw['words'].sum(), 1):.2f}%")
        c3.metric("Films with zero swears", int((sw["swears"] == 0).sum()))
        c4.metric("Your avg ★, top 20 sweariest", f"{sw.nlargest(20, 'swears')['user_rating'].mean():.2f}",
                  f"{sw.nlargest(20, 'swears')['user_rating'].mean() - sw['user_rating'].mean():+.2f} vs all",
                  delta_color="off")

        st.markdown("**Most used swears**")
        sw_words, sw_films = sum_counts(counts, sw["imdb_id"], 2)
        if sw_words:
            swt = pd.DataFrame(sw_words.most_common(20), columns=["Swear", "Times said"])
            swt["Films"] = swt["Swear"].map(sw_films)
            a, b = st.columns([3, 2])
            fig = go.Figure(go.Bar(x=swt["Times said"][::-1], y=swt["Swear"][::-1], orientation="h",
                                   marker_color=PALETTE[1], customdata=swt["Films"][::-1],
                                   hovertemplate="%{y}: said %{x:,} times in %{customdata} films<extra></extra>"))
            a.plotly_chart(layout(fig, "Times said", "", height=24 * len(swt) + 100), width="stretch")
            b.dataframe(swt, hide_index=True, width="stretch")
        st.caption("Counts words matching fuck*, shit*, bitch*, asshole*, bastard*, damn*, dick, piss*, "
                   "cunt*, crap in the subtitle file, so bleeped or non-English swearing is not counted.")
        st.dataframe(sw.sort_values(col, ascending=False)[["label", "swears", "words", "swear_share",
                                                           "user_rating", "imdb_rating"]]
                     .rename(columns={"label": "Film", "swears": "Swears", "words": "Total words",
                                      "swear_share": "Swears ÷ words (%)", "user_rating": "Your ★",
                                      "imdb_rating": "IMDb"}).round(2),
                     hide_index=True, width="stretch")

    # ---------------- 4. one film ----------------
    with tab_film:
        pick = st.selectbox("Film", view.sort_values("label")["label"].tolist(), key="subs_film")
        row = view[view["label"] == pick].iloc[0]
        d = subs.load_subtitles(row["imdb_id"])
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Your rating", f"{row['user_rating']}★", f"IMDb {row['imdb_rating']}", delta_color="off")
        m2.metric("Words / min", f"{row['words_per_min']:.0f}",
                  f"{row['words_per_min'] - view['words_per_min'].median():+.0f} vs filter median",
                  delta_color="off")
        m3.metric("Average mood", f"{row['sent_mean']:+.3f}")
        # swear_per_1k / 10 = swears as a % of all words in the film
        m4.metric("Swears ÷ total words", f"{row['swear_per_1k'] / 10:.2f}%")
        m5.metric("Sound cues / min", f"{row['sound_cues_per_min']:.1f}" if row["has_cues"] else "no SDH file")

        fwin = st.slider("Smoothing (points of 2%)", 1, 15, 3, key="subs_fwin")
        for cols, title in ((MOOD_COLS, "Mood (VADER, −1…+1)"), (TALK_COLS, "Words per minute")):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=X, y=smooth(view[cols].mean().to_numpy(), fwin),
                                     name=f"Average of filtered films (n={len(view)})",
                                     line=dict(color="#8a8a86", width=2, dash="dash")))
            fig.add_trace(go.Scatter(x=X, y=smooth(row[cols].astype(float).to_numpy(), fwin),
                                     name=pick, line=dict(color=PALETTE[0], width=2)))
            st.plotly_chart(layout(fig, "% through the story", title, height=320),
                            width="stretch")

        if d is not None:
            a, b, c = st.columns(3)
            talk = d[d["dialogue"] != ""]
            words = [w for w in sf.tokens(talk["dialogue"])
                     if len(w) > 2 and w not in ENGLISH_STOP_WORDS and "'" not in w]
            a.markdown("**Most-used words**")
            a.dataframe(pd.DataFrame(Counter(words).most_common(20), columns=["word", "count"]),
                        hide_index=True, width="stretch")
            cues = d.loc[d["sounds"] != "", "sounds"].str.lower().str.split("|").explode().str.strip()
            b.markdown("**Most common sound cues**")
            b.dataframe(cues.value_counts().head(20).rename_axis("cue").reset_index(),
                        hide_index=True, width="stretch")
            lyr = d.loc[d["lyrics"] != "", "lyrics"].str.split("|").explode()
            c.markdown(f"**Sung lines** ({len(lyr)})")
            c.dataframe(lyr.head(30).rename("lyric").to_frame(), hide_index=True,
                        width="stretch")

    # ---------------- 5. words ----------------
    with tab_words:
        st.markdown("Words used by more of the films in group A than group B (each word counts "
                    "once per film). Uses the sidebar filters too.")
        a, b = st.columns(2)
        ra = a.slider("Group A: your rating", 0.5, 5.0, (4.5, 5.0), 0.5, key="subs_wa")
        rb = b.slider("Group B: your rating", 0.5, 5.0, (0.5, 2.5), 0.5, key="subs_wb")
        ids_a = tuple(view.loc[view["user_rating"].between(*ra), "imdb_id"])
        ids_b = tuple(view.loc[view["user_rating"].between(*rb), "imdb_id"])
        if len(ids_a) < 10 or len(ids_b) < 10:
            st.info(f"Each group needs 10+ films (A has {len(ids_a)}, B has {len(ids_b)}).")
        else:
            @st.cache_data(ttl=600)
            def words_for(ia, ib):
                return sf.distinctive_words(list(ia), list(ib))
            wa, wb, chance = words_for(ids_a, ids_b)
            st.caption(f"Strength is a z-score. Across this many words, noise alone typically "
                       f"reaches **z ≈ {chance:.1f}** — bars shorter than that are exploratory, "
                       f"not evidence.")
            a, b = st.columns(2)
            for col, w, color, name, n in ((a, wa, PALETTE[0], "A", len(ids_a)),
                                           (b, wb, PALETTE[1], "B", len(ids_b))):
                w = w.iloc[::-1].abs()
                fig = go.Figure(go.Bar(x=w.values, y=w.index, orientation="h", marker_color=color,
                                       hovertemplate="%{y}: z = %{x:.2f}<extra></extra>"))
                fig.add_vline(x=chance, line_dash="dot", line_color="#8a8a86",
                              annotation_text="noise level", annotation_position="top")
                col.plotly_chart(layout(fig, "Strength (z-score)", "", height=560)
                                 .update_layout(title=f"More common in group {name} (n={n})"),
                                 width="stretch")
