import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import html
import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from src import config
# Rows enriched from OMDb carry IMDb's genre taxonomy ("Sci-Fi"), rows from TMDB
# carry TMDB's ("Science Fiction"). Same genre, two bars, split averages.
from src.common.parsing import canon_genre, parse_list, norm_language
from geo_utils import aggregate_country_counts, make_world_map
from ridgeline import genre_rating_ridgeline
from genre_spread import genre_spread_figure, SPREAD_CAPTION
from rating_distribution import render_rating_distribution, parse_percent
from critic_delta import render_critic_delta
from diary_tags import render_tag_analytics
from subtitle_analytics import render_subtitle_analytics
from box_office_verdict import render_box_office_verdict
from release_gap import render_viewing_gap

from dashboard_common import (HUB_PALETTE, apply_hub_theme, update_bar_trace, lb_metric, render_stars,
                              calculate_entropy, get_frequent_items_with_avg)
from movie_people import render_movie_people
from movie_lists import render_ranked_lists
from standup import render_standup
from theatre_watch import render_theatre_watch
from mobile import apply_mobile_css
st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
apply_mobile_css()

apply_hub_theme(bargap=0.05)

def clean_percentage(x):
    if pd.isna(x): return None
    x = str(x).replace('%', '').strip()
    try: return float(x)
    except: return None

def clean_currency(x):
    if pd.isna(x): return None
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return None


# --- LOAD DATA ---
MAX_ACTORS_PER_MOVIE = 10  # Threshold to prevent minor/uncredited actors from skewing stats

@st.cache_data
def load_diary_raw():
    """The diary exactly as exported, dates parsed, nothing dropped."""
    try:
        path = config.DATA_DIR / "raw" / "movies" / "diary.csv"
        if not path.exists():
            return None
        d = pd.read_csv(path)
        # Ensure column names are correct and handle the space in 'Watched Date'
        d['Watched Date'] = pd.to_datetime(d['Watched Date'], errors='coerce')
        return d.sort_values('Watched Date').reset_index(drop=True)
    except Exception as e:
        print(f"Error loading diary: {e}")
        return None

@st.cache_data
def load_diary():
    d = load_diary_raw()
    if d is None:
        return None
    # Drop rows where we don't have a date or a rating
    return d.dropna(subset=['Watched Date', 'Rating'])

@st.cache_data
def load_data():
    try:
        # 1. Load Main Enriched Data (Now consolidated)
        df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH)
        
        # 2. Clean and process for display
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_percentage)
        df['box_office'] = df['box_office'].apply(clean_currency)
        for c in ['runtime', 'year', 'vote_average', 'imdb_rating', 'metascore']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce').fillna(0)
        
        for c in ['genre', 'actors', 'writer', 'director', 'production', 'country', 'language']:
            if c in df.columns:
                df[f'{c}_list'] = df[c].apply(parse_list)
                # Merge IMDb/TMDB genre spellings (Sci-Fi == Science Fiction)
                if c == 'genre':
                    df['genre_list'] = df['genre_list'].apply(
                        lambda lst: list(dict.fromkeys(canon_genre(i) for i in lst)) if isinstance(lst, list) else lst)
                # Apply actor threshold
                if c == 'actors':
                    df[f'{c}_list'] = df[f'{c}_list'].apply(lambda x: x[:MAX_ACTORS_PER_MOVIE] if isinstance(x, list) else x)
                # Collapse equivalent language labels (en == English, etc.)
                if c == 'language':
                    df['language_list'] = df['language_list'].apply(
                        lambda lst: list(dict.fromkeys(norm_language(i) for i in lst)) if isinstance(lst, list) else lst)
            
        df['decade'] = (df['year'] // 10) * 10
        return df
    except Exception as e:
        print(f"Error loading dashboard data: {e}")
        return None

df = load_data()
df_diary = load_diary()
# Tag charts want every diary entry, not only the rated-and-dated ones: an
# untagged-rating entry still records a trip to the cinema.
df_diary_tags = load_diary_raw()

if df is None:
    st.error("❌ Data not found. Run ingestion first.")
    st.stop()

# Diary drives the "Ratings over Time" charts (no date-range filter; full history is shown).
df_diary_filtered = df_diary

st.title("📊 Personal Movie Intelligence")

# 1. METRICS & RATINGS HISTOGRAM
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1: lb_metric("Total Movies", len(df))
with c2: lb_metric("Avg User Rating", f"{df['user_rating'].mean():.2f}")
with c3: lb_metric("Avg Runtime", f"{int(df['runtime'].mean())} min")
with c4: lb_metric("Liked Movies", f"{df['is_liked'].sum()}")

# Calculate Watch Time Tooltip
total_hours = df['runtime'].sum() / 60
total_days = total_hours / 24
total_months = total_days / 30.44
watch_time_tooltip = f"{total_days:.1f} days"
if total_days >= 30:
    watch_time_tooltip += f" (~{total_months:.1f} months)"

with c5: lb_metric("Total Watch Time", f"{int(total_hours)} hrs", help_text=watch_time_tooltip)
with c6:
    if 'genre_list' in df.columns:
        gen_entropy = calculate_entropy(df['genre_list'])
        lb_metric("Taste Diversity", f"{gen_entropy:.2f}", 
                  help_text="Shannon Entropy. Max ~4.5.")

st.markdown("#### Rating Distribution")
# Same axis, one view at a time: your own stars, or the same films as the critics
# scored them (every source rescaled to 0.5-5).
render_rating_distribution(
    df['user_rating'],
    critic_sources=[
        ('IMDb', df.get('imdb_rating'), 10),
        ('TMDB', df.get('vote_average'), 10),
        ('Metacritic', df.get('metascore'), 100),
        ('Rotten Tomatoes',
         df['rotten_tomatoes_rating'].apply(parse_percent) if 'rotten_tomatoes_rating' in df.columns else None, 100),
    ],
    key="movies_dist", user_color='#00e054', critic_color='#40bcf4',
    style_fn=update_bar_trace)

st.divider()

# 1b. VIEWING CONTEXT — what your Letterboxd diary tags say about the watch
render_tag_analytics(df_diary_tags, HUB_PALETTE, metric_fn=lb_metric)

st.divider()

# 2. TIME, LENGTH & SEASONALITY
st.subheader("⏳ Time & Duration")
t_dec, t_run, t_sea, t_time = st.tabs(["Over Time", "Runtime", "Release Season", "Ratings over Time"])

with t_dec:
    # Continuous by release year rather than decade buckets — decade bars hide
    # where inside a decade your taste actually clusters.
    src = df.dropna(subset=['year']).copy()
    src['_year'] = src['year'].astype(int)

    per_year = (src.groupby('_year')
                .agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean'))
                .reindex(range(int(src['_year'].min()), int(src['_year'].max()) + 1),
                         fill_value=0)
                .reset_index().rename(columns={'index': '_year'}))
    per_year['avg_rating'] = per_year['avg_rating'].round(2)

    dec_ent = calculate_entropy(df['decade'])
    st.caption(f"**Decade Diversity (Entropy):** {dec_ent:.2f} — "
               f"{len(src):,} films across "
               f"{per_year['_year'].max() - per_year['_year'].min() + 1} years.")

    fig = px.area(per_year, x='_year', y='Count', title="Films per release year",
                  hover_data={'avg_rating': ':.2f'},
                  labels={'_year': 'Year', 'Count': 'Films'},
                  color_discrete_sequence=[HUB_PALETTE[0]])
    fig.update_traces(mode='lines', line_shape='spline', line_width=2,
                      fillcolor='rgba(0,224,84,0.16)')
    rated = per_year[per_year['Count'] > 0]
    fig.add_scatter(x=rated['_year'], y=rated['avg_rating'], name='Avg rating',
                    yaxis='y2', mode='lines+markers', line=dict(width=1.5, dash='dot'),
                    marker=dict(size=4), line_color=HUB_PALETTE[2 % len(HUB_PALETTE)])
    fig.update_layout(
        yaxis2=dict(title='Avg rating', overlaying='y', side='right',
                    range=[0, 5], showgrid=False),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0),
        hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

with t_run:
    bins = [0, 60, 80, 100, 120, 140, 160, 180, 999]
    labels = ['<60m', '60-79m', '80-99m', '100-119m', '120-139m', '140-159m', '160-179m', '180m+']
    df['runtime_bin'] = pd.cut(df['runtime'], bins=bins, labels=labels, right=False)
    rt_agg = df.groupby('runtime_bin', observed=True).agg(Count=('user_rating', 'size'), avg_rating=('user_rating', 'mean')).reset_index()
    rt_agg['avg_rating'] = rt_agg['avg_rating'].round(3)
    
    fig = px.bar(rt_agg, x='runtime_bin', y='Count', hover_data={'avg_rating': ':.3f'}, 
                 title="Runtime Distribution", color_discrete_sequence=[HUB_PALETTE[1]])
    update_bar_trace(fig)
    st.plotly_chart(fig, use_container_width=True)

with t_sea:
    if 'released' in df.columns:
        df['release_date_dt'] = pd.to_datetime(df['released'], errors='coerce')
        df['month'] = df['release_date_dt'].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                       'July', 'August', 'September', 'October', 'November', 'December']
        season_agg = df.groupby('month')['user_rating'].mean().reindex(month_order).reset_index()
        season_agg['user_rating'] = season_agg['user_rating'].round(3)
        fig = px.bar(season_agg, x='month', y='user_rating',
                      title="Average Rating by Release Month", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[2]])
        update_bar_trace(fig)
        st.plotly_chart(fig, use_container_width=True)

with t_time:
    if df_diary_filtered is not None and not df_diary_filtered.empty:
        st.markdown("#### Ratings Over Time (Direct from Diary)")
        st.caption("Each point represents a watch entry. Use the range slider below the chart to zoom into a period.")
        
        # Scatter plot with trendline
        fig = px.scatter(df_diary_filtered, x='Watched Date', y='Rating', 
                         hover_data=['Name', 'Year'],
                         color='Rating',
                         color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]],
                         title="Your Ratings over Time (Watched Date)",
                         labels={'Watched Date': 'Date Watched', 'Rating': 'Rating'},
                         trendline="lowess",
                         trendline_color_override="#ff8000")
        
        # Add a line connecting all dots in chronological order
        fig.add_trace(go.Scatter(
            x=df_diary_filtered['Watched Date'],
            y=df_diary_filtered['Rating'],
            mode='lines',
            line=dict(color=HUB_PALETTE[2], width=1, dash='dot'), # Blue dotted line
            name='Watch Path',
            hoverinfo='skip'
        ))
        
        # Add range slider and range selectors
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ]),
                    bgcolor="#2C343F",
                    activecolor="#ff8000"
                ),
                rangeslider=dict(visible=True, thickness=0.05),
                type="date"
            ),
            yaxis=dict(range=[0, 5.5], dtick=0.5),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # ONE watch-volume chart, not two. This block previously drew a fixed
        # "Movies Watched per Month" bar chart AND a second granularity-selectable
        # one directly underneath it -- the same 381 watches plotted twice, once
        # locked to months and once not. Merged: the granularity selector covers
        # the monthly case, so the fixed chart had nothing left to add.
        st.markdown("#### Watch Volume Over Time")

        # freq alias, and the noun used in the title/caption.
        watch_grains = {
            "Daily":   ("D", "Day"),
            "Weekly":  ("W", "Week"),
            "Monthly": ("M", "Month"),
            "Yearly":  ("Y", "Year"),
        }
        grain = st.radio("Bucket size", list(watch_grains), index=2, horizontal=True,
                         key="movies_watch_volume_grain")
        freq, unit = watch_grains[grain]

        dated = df_diary_filtered.dropna(subset=['Watched Date'])
        if dated.empty:
            st.info("No dated diary entries to count.")
        else:
            counts = dated['Watched Date'].dt.to_period(freq).value_counts().sort_index()
            # Reindex onto the COMPLETE period range before plotting: groupby only
            # emits periods that actually occur, so a month you watched nothing
            # would simply be absent, and an absent bucket is indistinguishable
            # from one that was never in range. A month off the films is signal.
            counts = counts.reindex(
                pd.period_range(counts.index.min(), counts.index.max(), freq=freq),
                fill_value=0)
            vol = pd.DataFrame({
                'Period': counts.index.to_timestamp(),
                'Films': counts.to_numpy(),
            })

            nonzero = vol[vol['Films'] > 0]
            busiest = vol.loc[vol['Films'].idxmax()]
            st.caption(
                f"**{int(vol['Films'].sum()):,} films** across {len(vol):,} "
                f"{unit.lower()}s — {len(nonzero):,} with at least one watch, "
                f"{int((vol['Films'] == 0).sum()):,} with none. Busiest "
                f"{unit.lower()}: **{busiest['Period'].strftime('%Y-%m-%d')}** "
                f"with {int(busiest['Films'])}. Drag the slider under the chart to zoom.")

            fig_vol = px.bar(vol, x='Period', y='Films',
                             title=f"Movies Watched per {unit}",
                             labels={'Period': unit, 'Films': 'Movies watched'},
                             color_discrete_sequence=[HUB_PALETTE[1]])
            # A DATE axis, not a categorical one: it keeps the 2020 gap the right
            # width instead of squeezing every bucket to equal spacing, and it is
            # what makes the range slider below work. The zero-filled buckets stay
            # in the data so the bar widths and the caption counts stay honest.
            fig_vol.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=6, label="6m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all")
                        ]),
                        bgcolor="#2C343F",
                        activecolor="#ff8000"
                    ),
                    rangeslider=dict(visible=True, thickness=0.05),
                    type="date"
                ),
                bargap=0.15, height=520,
            )
            update_bar_trace(fig_vol)
            st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.info("No watch date data available for the selected period.")

st.divider()

# 2b. RELEASE-TO-WATCH GAP — the theatre "Release Gap" idea, for every diary entry
render_viewing_gap(df, df_diary_tags, config.CACHE_DIR / "tmdb_release_dates.json",
                   lb_metric, HUB_PALETTE, style_fn=update_bar_trace)

st.divider()

render_movie_people(df)

st.divider()

# 3b. RANKED LISTS — your Letterboxd lists tagged director / franchise / studio
render_ranked_lists(df)

st.divider()

# 4. SCORES & FINANCIALS
st.subheader("💰 Scores & Consensus Alignment")
t1, t2, t_hf, t3 = st.tabs(["Critic Alignment", "Box Office Impact", "Hit or Flop", "Global Stats"])

with t1:
    st.markdown("#### How often do you agree with each critic?")
    st.caption("Each source is rescaled to your 0–5 scale, then compared movie-by-movie. "
               "**Same** = within 0.5★ · **Close** = within 1★ · **Far** = more than 1★ apart.")

    # (label, critic score normalised to the user's 0-5 scale)
    critic_specs = []
    if 'rotten_tomatoes_rating' in df.columns:
        critic_specs.append(('Rotten Tomatoes', df['rotten_tomatoes_rating'] / 20.0))   # 0-100 -> 0-5
    if 'metascore' in df.columns:
        critic_specs.append(('Metascore', df['metascore'] / 20.0))                       # 0-100 -> 0-5
    if 'imdb_rating' in df.columns:
        critic_specs.append(('IMDb', df['imdb_rating'] / 2.0))                            # 0-10 -> 0-5
    if 'vote_average' in df.columns:
        critic_specs.append(('TMDB', df['vote_average'] / 2.0))                           # 0-10 -> 0-5

    rows, summary = [], []
    for label, norm in critic_specs:
        comp = pd.DataFrame({'user': df['user_rating'], 'critic': pd.to_numeric(norm, errors='coerce')})
        comp = comp[(comp['critic'] > 0)].dropna()   # drop missing/zero critic scores
        if comp.empty:
            continue
        d = (comp['user'] - comp['critic']).abs()
        same = int((d <= 0.5).sum())
        close = int(((d > 0.5) & (d <= 1.0)).sum())
        far = int((d > 1.0).sum())
        rows += [{'Critic': label, 'Agreement': 'Same', 'Count': same},
                 {'Critic': label, 'Agreement': 'Close', 'Count': close},
                 {'Critic': label, 'Agreement': 'Far', 'Count': far}]
        mode_bucket = max([('Same', same), ('Close', close), ('Far', far)], key=lambda x: x[1])[0]
        summary.append({'Critic': label, '% Same': round(100 * same / len(comp), 1),
                        'Avg |diff|': round(float(d.mean()), 2), 'Most common': mode_bucket, 'Movies': len(comp)})

    if rows:
        agg_df = pd.DataFrame(rows)
        # Order critics on the x-axis by agreement (most agreed-with first)
        order = [s['Critic'] for s in sorted(summary, key=lambda s: (-s['% Same'], s['Avg |diff|']))]
        fig = px.bar(agg_df, x='Critic', y='Count', color='Agreement', barmode='group',
                     title="Agreement Breakdown by Source",
                     category_orders={'Critic': order, 'Agreement': ['Same', 'Close', 'Far']},
                     color_discrete_map={'Same': HUB_PALETTE[1], 'Close': HUB_PALETTE[2], 'Far': HUB_PALETTE[0]})
        fig.update_traces(marker_line_width=1, marker_line_color="#14171C")
        st.plotly_chart(fig, use_container_width=True)

        sdf = pd.DataFrame(summary).sort_values(['% Same', 'Avg |diff|'], ascending=[False, True]).reset_index(drop=True)
        sdf.index += 1
        sdf.index.name = 'Rank'
        st.markdown("**Who you agree with most** — ranked by % Same, then smallest average difference:")
        st.dataframe(sdf, use_container_width=True)
        leader = sdf.iloc[0]
        st.caption(f"🎯 You align most closely with **{leader['Critic']}** "
                   f"({leader['% Same']}% within half a star, avg gap {leader['Avg |diff|']}★).")
    else:
        st.info("No critic scores available to compare.")

    st.divider()
    # --- Added: the same comparison at TITLE level, not just in aggregate. The
    # bars above say how OFTEN you agree; these two say WHERE you disagree.
    render_critic_delta(
        df=df,
        title_col='title',
        user_col='user_rating',
        critic_sources=[
            ('IMDb', df.get('imdb_rating'), 10),
            ('TMDB', df.get('vote_average'), 10),
            ('Metacritic', df.get('metascore'), 100),
            # OMDb stores RT as a percent STRING ("92%"); to_five strips the
            # sign, so this is a real source with 949 of 1062 films scored.
            ('Rotten Tomatoes', df.get('rotten_tomatoes_rating'), 100),
        ],
        key='movies_cd',
    )

with t2:
    # TMDB `revenue` is the worldwide gross. OMDb's `box_office` is US domestic
    # only, which put international hits (most Indian films) in the Indie tier.
    if 'revenue' in df.columns:
        worldwide = pd.to_numeric(df['revenue'], errors='coerce')
        worldwide = worldwide.where(worldwide >= 1000)   # TMDB placeholder entries ("30")
        bins = [0, 1e6, 50e6, 100e6, 500e6, 1e9, 10e9]
        labels = ['Indie (<$1M)', 'Moderate ($1M-$50M)', 'Hit ($50M-$100M)', 'Blockbuster ($100M-$500M)', 'Mega-Blockbuster ($500M-$1B)', 'Historical (>$1B)']
        df['bo_tier'] = pd.cut(worldwide, bins=bins, labels=labels)
        st.caption(f"Worldwide gross from TMDB, nominal dollars. {int(worldwide.notna().sum())} of {len(df)} films have a figure.")
        bo_agg = df.groupby('bo_tier', observed=True)['user_rating'].mean().reset_index()
        bo_agg['user_rating'] = bo_agg['user_rating'].round(3)
        fig = px.bar(bo_agg, x='bo_tier', y='user_rating', title="Do you prefer Blockbusters or Indies?", color='user_rating', range_y=[0, 5], color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[1]]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Box Office data missing.")

with t_hf:
    render_box_office_verdict(df)

with t3:
    st.markdown("#### 🌍 Your Cinematic World Map")
    st.caption("Countries shaded by how many films you've watched from there (log-scaled so smaller countries stay visible). Hover for the exact count and your average rating.")
    geo_agg, geo_unmapped = aggregate_country_counts(df, 'country_list', 'user_rating')
    if not geo_agg.empty:
        map_fig = make_world_map(geo_agg, low_color="#2C343F", high_color=HUB_PALETTE[0])
        st.plotly_chart(map_fig, use_container_width=True)
        if geo_unmapped:
            st.caption(f"Not placed on the map (unrecognized region): {', '.join(geo_unmapped)}")
    else:
        st.info("No country data available to map.")

    c_ctry, c_lang = st.columns(2)
    with c_ctry:
        ctry_ent = calculate_entropy(df['country_list'])
        st.caption(f"**Country Diversity (Entropy):** {ctry_ent:.2f}")
        cnt = get_frequent_items_with_avg(df, 'country_list', 8)
        fig = px.bar(cnt, x='country_list', y='Count', hover_data={'avg_rating': ':.3f'}, title="Top Countries", color_discrete_sequence=[HUB_PALETTE[2]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    with c_lang:
        lang_ent = calculate_entropy(df['language_list'])
        st.caption(f"**Language Diversity (Entropy):** {lang_ent:.2f}")
        lng = get_frequent_items_with_avg(df, 'language_list', 8)
        fig = px.pie(lng, values='Count', names='language_list', hover_data={'avg_rating': ':.3f'}, title="Languages", color_discrete_sequence=HUB_PALETTE)
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# 5. VIBE CHECK
st.subheader("🧠 Plot Vibe Check")
st.caption("Dominant plot themes extracted from Plot + Overview + Tagline. Hover to see your average rating for movies with these themes.")

if 'overview' in df.columns and 'plot' in df.columns:
    df['vibe_text'] = df['overview'].fillna('') + " " + df['tagline'].fillna('') + " " + df['plot'].fillna('')

    # Optional genre filter for the theme treemap (default: all genres).
    vibe_df = df
    if 'genre_list' in df.columns:
        _vibe_genres = sorted({g for gl in df['genre_list'] if isinstance(gl, list)
                               for g in gl if g and str(g).lower() not in ('unknown', 'nan', '')})
        _vibe_pick = st.selectbox("Filter themes by genre", ["All"] + _vibe_genres, key="vibe_genre")
        if _vibe_pick != "All":
            vibe_df = df[df['genre_list'].apply(lambda gl: isinstance(gl, list) and _vibe_pick in gl)]

    stops = set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'where', 'when', 'how', 'who', 'which', 'this', 'that', 'these', 'those', 
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
        'at', 'by', 'for', 'from', 'in', 'into', 'of', 'off', 'on', 'onto', 'out', 'over', 'up', 'down', 'to', 'with', 'within', 'without',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs',
        'i', 'me', 'my', 'mine', 'you', 'your', 'yours', 'we', 'us', 'our', 'ours',
        'after', 'before', 'while', 'during', 'since', 'until', 'through', 'about', 'against', 'between',
        'movie', 'film', 'story', 'plot', 'character', 'life', 'world', 'find', 'finds', 'one', 'two',
        'into', 'back', 'new', 'must', 'will', 'only', 'other', 'city', 'years', 'year', 'himself', 'high', 'most', 'just', 'more', 'there', 'soon',
    ])
    
    word_ratings = {}
    for _, row in vibe_df.dropna(subset=['vibe_text', 'user_rating']).iterrows():
        text = str(row['vibe_text'])
        words = set([w.lower() for w in re.findall(r'\w+', text) if len(w) > 3 and w.lower() not in stops])
        for w in words:
            if w not in word_ratings: word_ratings[w] = []
            word_ratings[w].append(row['user_rating'])
            
    word_stats = []
    for w, ratings in word_ratings.items():
        word_stats.append({'Word': w, 'Count': len(ratings), 'Avg_Rating': np.mean(ratings)})
        
    if word_stats:
        wdf = pd.DataFrame(word_stats).sort_values('Count', ascending=False).head(50)
        wdf['Avg_Rating'] = wdf['Avg_Rating'].round(3)

        fig = px.treemap(wdf, path=['Word'], values='Count', color='Count', hover_data=['Avg_Rating'], title="Dominant Themes in Your Movies", color_continuous_scale=[[0, HUB_PALETTE[2]], [1, HUB_PALETTE[1]]])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No plot text available for the selected genre.")
    
st.divider()

# 9. OSCAR ANALYSIS
if 'awards' in df.columns:
    st.subheader("🏆 The Oscar Effect")
    df['is_oscar_winner'] = df['awards'].fillna('').str.contains(r'Won\s.*Oscar', flags=re.IGNORECASE, regex=True)
    oscar_agg = df.groupby('is_oscar_winner')['user_rating'].mean().reset_index()
    oscar_agg['user_rating'] = oscar_agg['user_rating'].round(3)
    oscar_agg['Label'] = oscar_agg['is_oscar_winner'].map({True: 'Oscar Winner', False: 'Not a Winner'})
    
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(oscar_agg, x='Label', y='user_rating', color='Label', title="Do you prefer Actual Oscar Winners?", range_y=[0, 5], color_discrete_sequence=[HUB_PALETTE[1], HUB_PALETTE[2]])
        fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        oscar_movies = df[df['is_oscar_winner']].sort_values('user_rating', ascending=False).head(5)
        st.write("**Your Top Oscar-Winning Movies:**")
        for _, m in oscar_movies.iterrows():
            st.markdown(f"{render_stars(m['user_rating'])} **{m['title']}**", unsafe_allow_html=True)
            
st.divider()

# 10. HOT TAKES
st.subheader("🌶️ Your Hot Takes (vs General Consensus)")
st.caption("Where your opinion wildly splits from the average critic & audience consensus (Sorted by Magnitude).")

# Calculate unified Critic Average, explicitly ignoring zeroes
df['imdb_100'] = df['imdb_rating'] * 10
df['va_100'] = df['vote_average'] * 10
critic_cols = ['imdb_100', 'metascore', 'rotten_tomatoes_rating', 'va_100']

# Replace 0s with NaNs so they don't drag down the average
df[critic_cols] = df[critic_cols].replace(0, np.nan)
df['critic_avg_100'] = df[critic_cols].mean(axis=1, skipna=True) 
df['critic_norm'] = (df['critic_avg_100'] / 100) * 5

# Drop movies with absolutely no critic scores
ht_df = df.dropna(subset=['user_rating', 'critic_norm']).copy()
ht_df['controversy'] = ht_df['user_rating'] - ht_df['critic_norm']
ht_df['magnitude'] = ht_df['controversy'].abs()

# Filters: Only defend movies critics rated <= 3.5. Only roast movies critics rated >= 2.5
hot_takes_good = ht_df[(ht_df['controversy'] > 0) & (ht_df['critic_norm'] <= 2.9)].sort_values('magnitude', ascending=False).head(10)
hot_takes_bad = ht_df[(ht_df['controversy'] < 0) & (ht_df['critic_norm'] >= 3.5)].sort_values('magnitude', ascending=False).head(10)

c1, c2 = st.columns(2)
with c1:
    st.write("**Movies you defended (You > Critics):**")
    for _, row in hot_takes_good.iterrows():
        st.markdown(f"📈 **{row['title']}** (Diff: +{row['magnitude']:.2f})  \n<span style='font-size:0.95em; color:#99AABB;'>{render_stars(row['user_rating'])} | Critic: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)
with c2:
    st.write("**Movies you roasted (You < Critics):**")
    for _, row in hot_takes_bad.iterrows():
        st.markdown(f"📉 **{row['title']}** (Diff: -{row['magnitude']:.2f})  \n<span style='font-size:0.95em; color:#99AABB;'>{render_stars(row['user_rating'])} | Critic: {row['critic_norm']:.1f}</span>", unsafe_allow_html=True)

st.divider()

# 11. GENRE & MATURITY ANALYSIS
st.subheader("🏷️ Theme Keywords")
st.caption("TMDB keywords are far finer-grained than genre — 'time loop', 'heist', "
           "'found family'. Genre says what shelf a film sits on; keywords say what "
           "it is actually about.")

if 'keywords' in df.columns and df['keywords'].notna().any():
    kw_rows = []
    for _, r in df.iterrows():
        if pd.isna(r.get('keywords')) or pd.isna(r.get('user_rating')):
            continue
        for k in str(r['keywords']).split(','):
            k = k.strip()
            if k:
                kw_rows.append({'keyword': k, 'rating': r['user_rating'],
                                'title': r.get('title')})
    kw_df = pd.DataFrame(kw_rows)
    if not kw_df.empty:
        MIN_N = 5
        agg = (kw_df.groupby('keyword')
               .agg(Films=('rating', 'size'), AvgRating=('rating', 'mean'))
               .query('Films >= @MIN_N')
               .sort_values('AvgRating', ascending=False))
        agg['AvgRating'] = agg['AvgRating'].round(2)
        overall = df['user_rating'].mean()

        t_kw_top, t_kw_common, t_kw_search, t_kw_all, t_lbth_top, t_lbth_heat = st.tabs(
            ["Best & Worst Themes", "Most Common", "🔎 Search by Keyword", "All Keywords",
             "Letterboxd Themes", "Letterboxd Themes × Your Stars"])
        with t_kw_top:
            c_hi, c_lo = st.columns(2)
            with c_hi:
                top = agg.head(12).reset_index()
                fig = px.bar(top.sort_values('AvgRating'), x='AvgRating', y='keyword',
                             orientation='h', hover_data=['Films'],
                             title=f"Themes you rate highest (≥{MIN_N} films)",
                             color_discrete_sequence=[HUB_PALETTE[1]])
                fig.add_vline(x=overall, line_dash="dot", line_color="#99AABB")
                fig.update_layout(yaxis_title=None, height=420)
                st.plotly_chart(fig, use_container_width=True)
            with c_lo:
                bot = agg.tail(12).reset_index()
                fig = px.bar(bot.sort_values('AvgRating'), x='AvgRating', y='keyword',
                             orientation='h', hover_data=['Films'],
                             title=f"Themes you rate lowest (≥{MIN_N} films)",
                             color_discrete_sequence=[HUB_PALETTE[0]])
                fig.add_vline(x=overall, line_dash="dot", line_color="#99AABB")
                fig.update_layout(yaxis_title=None, height=420)
                st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Dotted line = your overall average ({overall:.2f}★). "
                       f"{len(agg)} keywords appear on at least {MIN_N} films.")
        with t_kw_common:
            common = agg.sort_values('Films', ascending=False).head(20).reset_index()
            fig = px.scatter(common, x='Films', y='AvgRating', text='keyword',
                             title="Your most frequent themes vs how you rate them",
                             color='AvgRating', color_continuous_scale='RdYlGn')
            fig.add_hline(y=overall, line_dash="dot", line_color="#99AABB")
            fig.update_traces(textposition='top center', marker=dict(size=12))
            fig.update_layout(height=520)
            st.plotly_chart(fig, use_container_width=True)
        with t_kw_search:
            # Every keyword is searchable here, not just the ones that clear the
            # ≥5-film bar used for the aggregate charts.
            all_kws = sorted(kw_df['keyword'].unique(), key=str.lower)
            st.caption(f"Search any of your {len(all_kws):,} keywords and see every "
                       f"film carrying it, with your rating.")
            q = st.text_input("Keyword", placeholder="e.g. time loop, heist, dystopia",
                              key="kw_search_box").strip().lower()
            if q:
                matches = [k for k in all_kws if q in k.lower()]
                if not matches:
                    st.warning(f"No keyword matches “{q}”.")
                else:
                    pick = matches[0] if len(matches) == 1 else st.selectbox(
                        f"{len(matches)} matching keywords", matches, key="kw_pick")
                    hits = (kw_df[kw_df['keyword'] == pick]
                            .merge(df[['title', 'year', 'user_rating', 'poster',
                                       'director_list', 'genre_list']],
                                   left_on='title', right_on='title', how='left')
                            .drop_duplicates('title')
                            .sort_values('rating', ascending=False))
                    avg = hits['rating'].mean()
                    delta = avg - df['user_rating'].mean()
                    m1, m2, m3 = st.columns(3)
                    with m1: lb_metric("Films", len(hits))
                    with m2: lb_metric("Your Avg", f"{avg:.2f}")
                    with m3: lb_metric("vs Overall", f"{delta:+.2f}",
                                       help_text=f"Your library average is "
                                                 f"{df['user_rating'].mean():.2f}.")
                    show = hits[['title', 'year', 'rating']].rename(
                        columns={'title': 'Film', 'year': 'Year', 'rating': 'Your Rating'})
                    show['Year'] = show['Year'].astype('Int64')
                    st.dataframe(show, use_container_width=True, hide_index=True,
                                 height=min(520, 60 + 35 * len(show)))
                    posters = hits[hits['poster'].notna()].head(10)
                    if not posters.empty:
                        st.caption("Highest rated first")
                        for col, (_, r) in zip(st.columns(len(posters)), posters.iterrows()):
                            with col:
                                st.image(f"https://image.tmdb.org/t/p/w185{r['poster']}")
                                st.caption(f"{r['rating']:.1f}★")
            else:
                st.info("Type a keyword above to see every film that carries it.")

        with t_kw_all:
            st.dataframe(agg.reset_index().rename(
                columns={'keyword': 'Keyword', 'AvgRating': 'Avg Rating'}),
                use_container_width=True, hide_index=True, height=420)

        # Letterboxd themes are hand-curated nanogenres, coarser than TMDB keywords.
        # Theme names contain commas ("Humanity and the world around us"), so split on ';' only.
        th = (df.dropna(subset=['user_rating', 'lb_themes'])
              .assign(theme=lambda d: d['lb_themes'].str.split(';'))
              .explode('theme')) if 'lb_themes' in df.columns else pd.DataFrame()
        if not th.empty:
            th['theme'] = th['theme'].str.strip()
            th = th[th['theme'] != '']
            th_agg = (th.groupby('theme')
                      .agg(Films=('user_rating', 'size'), AvgRating=('user_rating', 'mean'))
                      .query('Films >= @MIN_N')
                      .sort_values('AvgRating', ascending=False))
            th_agg['AvgRating'] = th_agg['AvgRating'].round(2)
        if th.empty:
            for tab in (t_lbth_top, t_lbth_heat):
                with tab:
                    st.info("Letterboxd themes not merged yet — run "
                            "`python -m src.movies.letterboxd_details --apply`.")
        else:
            with t_lbth_top:
                c_hi, c_lo = st.columns(2)
                for col, part, title, color in [
                        (c_hi, th_agg.head(12), "Letterboxd themes you rate highest", HUB_PALETTE[1]),
                        (c_lo, th_agg.tail(12), "Letterboxd themes you rate lowest", HUB_PALETTE[0])]:
                    with col:
                        fig = px.bar(part.reset_index().sort_values('AvgRating'), x='AvgRating',
                                     y='theme', orientation='h', hover_data=['Films'],
                                     title=f"{title} (≥{MIN_N} films)",
                                     color_discrete_sequence=[color])
                        fig.add_vline(x=overall, line_dash="dot", line_color="#99AABB")
                        fig.update_layout(yaxis_title=None, height=420)
                        st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Dotted line = your overall average ({overall:.2f}★). "
                           f"{len(th_agg)} Letterboxd themes appear on at least {MIN_N} films.")
            with t_lbth_heat:
                top_th = th['theme'].value_counts().head(25).index
                th_top = th[th['theme'].isin(top_th)]
                heat = pd.crosstab(th_top['theme'].to_numpy(), th_top['user_rating'].to_numpy(),
                                   normalize='index').reindex(top_th)
                fig = px.imshow(heat * 100, aspect='auto', color_continuous_scale='Greens',
                                labels=dict(x="Your rating", y=None, color="% of theme's films"),
                                title="Your 25 most common Letterboxd themes: how you rate them")
                fig.update_xaxes(type='category')
                fig.update_layout(height=640)
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Each row sums to 100%: the share of that theme's films you gave "
                           "each star value. Rows ordered by how many films carry the theme.")
    else:
        st.info("No keywords parsed yet.")
else:
    st.info("Keywords not ingested yet — run `python -m src.movies.enrich_extended --apply`.")

st.divider()

# 11a. LETTERBOXD CROWD
LB_STARS = np.arange(1, 11) / 2
LB_SHARE_COLS = [f"lb_share_{int(s * 10):02d}" for s in LB_STARS]


@st.cache_data
def lb_crowd_curve(frame):
    """Per film: share of the Letterboxd crowd below, at and above your star, and whether yours is their modal star."""
    c = frame.dropna(subset=['user_rating'] + LB_SHARE_COLS)
    shares = c[LB_SHARE_COLS].to_numpy(dtype=float)
    keep = shares.sum(1) > 0
    c, shares = c[keep].copy(), shares[keep] / shares[keep].sum(1, keepdims=True)
    at_mine = np.isclose(LB_STARS, c['user_rating'].to_numpy()[:, None])
    c['At'] = (shares * at_mine).sum(1)
    c['Below'] = (shares * (LB_STARS < c['user_rating'].to_numpy()[:, None])).sum(1)
    c['Above'] = 1 - c['Below'] - c['At']
    c['Crowd Percentile'] = 100 * (c['Below'] + c['At'] / 2)
    c['Majority'] = shares.argmax(1) == at_mine.argmax(1)
    c['Minority'] = c['At'] < 0.10
    return c


@st.cache_data
def lb_similar_scores(frame):
    """Your rating of each film vs your mean rating of its Letterboxd 'similar films' that you have watched."""
    mine = (frame.dropna(subset=['lb_film_url', 'user_rating'])
            .drop_duplicates('lb_film_url').set_index('lb_film_url')['user_rating'])
    rows = []
    for _, r in frame.dropna(subset=['lb_similar_urls', 'user_rating']).iterrows():
        got = mine.reindex([u.strip() for u in r['lb_similar_urls'].split(';')]).dropna()
        if len(got):
            rows.append({'Film': r['title'], 'Year': r['year'], 'Your Rating': r['user_rating'],
                         'Similar Avg': got.mean(), 'Similar Seen': len(got)})
    return pd.DataFrame(rows)


st.subheader("🟢 You vs the Letterboxd Crowd")
st.caption("Your stars against the Letterboxd crowd: their average, the full spread of "
           "their star ratings per film, and how often your star is the one most people gave.")

if 'lb_avg_rating' in df.columns and df['lb_avg_rating'].notna().any():
    lbc = df.dropna(subset=['user_rating', 'lb_avg_rating'])
    lb_gap = lbc['user_rating'] - lbc['lb_avg_rating']
    has_curve = all(c in df.columns for c in LB_SHARE_COLS)
    curve = lb_crowd_curve(df) if has_curve else None
    m1, m2, m3 = st.columns(3)
    with m1: lb_metric("Films matched", len(lbc))
    with m2: lb_metric("Correlation", f"{lbc['user_rating'].corr(lbc['lb_avg_rating']):.2f}",
                       help_text="Pearson correlation between your rating and the Letterboxd average.")
    with m3: lb_metric("You − Letterboxd", f"{lb_gap.mean():+.2f}★",
                       help_text="Your mean rating minus the Letterboxd mean, on the same films.")
    if curve is not None and not curve.empty:
        m4, m5, m6 = st.columns(3)
        with m4: lb_metric("With the majority", f"{curve['Majority'].sum()} ({curve['Majority'].mean():.0%})",
                           help_text="Films where your star is the one the most Letterboxd users gave.")
        with m5: lb_metric("In the minority", f"{curve['Minority'].sum()} ({curve['Minority'].mean():.0%})",
                           help_text="Films where under 10% of Letterboxd users gave your star.")
        with m6: lb_metric("Median crowd percentile", f"{curve['Crowd Percentile'].median():.0f}",
                           help_text="50 = you sit mid-crowd. Above 50 = you rate higher than most.")

    t_lb_vs, t_lb_pct, t_lb_lonely, t_lb_agree, t_lb_div, t_lb_sim = st.tabs(
        ["You vs the Crowd", "Crowd Percentile", "Lonely Ratings", "Agreement by Genre & Year",
         "Divisive Films", "Similar Films"])
    with t_lb_vs:
        fig = px.scatter(lbc, x='lb_avg_rating', y='user_rating', hover_name='title',
                         hover_data=['year'], opacity=0.6,
                         labels={'lb_avg_rating': 'Letterboxd average', 'user_rating': 'Your rating'},
                         color_discrete_sequence=[HUB_PALETTE[2]])
        fig.add_shape(type='line', x0=0.5, y0=0.5, x1=5, y1=5,
                      line=dict(dash='dot', color='#99AABB'))
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Dotted line = you agree with the crowd exactly. Points above it you liked more.")
    with t_lb_div:
        div = lbc.dropna(subset=['lb_rating_std']).nlargest(15, 'lb_rating_std')
        st.caption("Films whose Letterboxd star ratings are most spread out: the crowd "
                   "is split. Spread is the standard deviation of the crowd's stars.")
        st.dataframe(pd.DataFrame({
            'Film': div['title'], 'Year': div['year'].astype('Int64'),
            'Letterboxd Avg': div['lb_avg_rating'].round(2),
            'Crowd Spread': div['lb_rating_std'].round(2),
            'Your Rating': div['user_rating'],
            'You − Crowd': (div['user_rating'] - div['lb_avg_rating']).round(2)}),
            use_container_width=True, hide_index=True)
    if curve is None or curve.empty:
        for tab in (t_lb_pct, t_lb_lonely, t_lb_agree):
            with tab:
                st.info("Letterboxd rating curves not merged yet — run "
                        "`python -m src.movies.letterboxd_details --apply`.")
    else:
        with t_lb_pct:
            c_l, c_r = st.columns(2)
            with c_l:
                fig = px.histogram(curve, x='Crowd Percentile', nbins=20,
                                   title="Where your star sits in each film's crowd",
                                   color_discrete_sequence=[HUB_PALETTE[1]])
                fig.add_vline(x=50, line_dash="dot", line_color="#99AABB")
                fig.update_layout(height=400, yaxis_title="Films")
                st.plotly_chart(fig, use_container_width=True)
            with c_r:
                split = curve[['Below', 'At', 'Above']].mean().mul(100).round(1)
                fig = px.bar(x=split.index, y=split.values, text=split.values,
                             labels={'x': '', 'y': '% of the crowd'},
                             title="Average share of the crowd rating below / at / above you",
                             color=split.index, color_discrete_sequence=HUB_PALETTE)
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            st.caption("Crowd percentile = % of Letterboxd users who rated the film lower than "
                       "you, plus half of those who gave your exact star.")
        with t_lb_lonely:
            lonely = curve.nsmallest(15, 'At')
            st.caption("Films where the fewest Letterboxd users gave the star you gave.")
            st.dataframe(pd.DataFrame({
                'Film': lonely['title'], 'Year': lonely['year'].astype('Int64'),
                'Your Rating': lonely['user_rating'],
                '% of Crowd at Your Star': (lonely['At'] * 100).round(1),
                'Letterboxd Avg': lonely['lb_avg_rating'].round(2),
                'Crowd Percentile': lonely['Crowd Percentile'].round(0)}),
                use_container_width=True, hide_index=True)
        with t_lb_agree:
            c_g, c_y = st.columns(2)
            with c_g:
                by_genre = (curve.explode('genre_list').groupby('genre_list')
                            .agg(Films=('Majority', 'size'), Pct=('Majority', lambda m: 100 * m.mean()))
                            .query('Films >= 15').sort_values('Pct'))
                fig = px.bar(by_genre.reset_index(), x='Pct',
                             y='genre_list', orientation='h', hover_data=['Films'],
                             labels={'Pct': '% of films with the majority', 'genre_list': ''},
                             title="Agreement with the majority by genre (≥15 films)",
                             color_discrete_sequence=[HUB_PALETTE[2]])
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)
            with c_y:
                by_year = (curve.assign(Year=pd.to_datetime(curve['rating_date'], errors='coerce').dt.year)
                           .groupby('Year').agg(Films=('Majority', 'size'), Pct=('Majority', lambda m: 100 * m.mean()))
                           .query('Films >= 10').reset_index())
                fig = px.line(by_year, x='Year', y='Pct', markers=True,
                              hover_data=['Films'], labels={'Pct': '% of films with the majority'},
                              title="Agreement with the majority by year rated (≥10 films)",
                              color_discrete_sequence=[HUB_PALETTE[1]])
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True)
    with t_lb_sim:
        sim = lb_similar_scores(df) if 'lb_similar_urls' in df.columns else pd.DataFrame()
        if sim.empty:
            st.info("Letterboxd similar-film links not merged yet — run "
                    "`python -m src.movies.letterboxd_details --apply`.")
        else:
            sim['Gap'] = sim['Your Rating'] - sim['Similar Avg']
            st.caption(f"For {len(sim)} films, your rating vs your average rating of the "
                       f"Letterboxd 'similar films' you've also watched. Correlation "
                       f"{sim['Your Rating'].corr(sim['Similar Avg']):.2f}: how well a film's "
                       f"neighbours predict your rating of it.")
            fig = px.scatter(sim, x='Similar Avg', y='Your Rating', size='Similar Seen',
                             hover_name='Film', opacity=0.6,
                             color_discrete_sequence=[HUB_PALETTE[2]])
            fig.add_shape(type='line', x0=0.5, y0=0.5, x1=5, y1=5,
                          line=dict(dash='dot', color='#99AABB'))
            fig.update_layout(height=440)
            st.plotly_chart(fig, use_container_width=True)
            st.write("**Films you rated furthest from their neighbours:**")
            st.dataframe(sim.reindex(sim['Gap'].abs().sort_values(ascending=False).index).head(15)
                         .assign(Year=lambda d: d['Year'].astype('Int64')).round(2),
                         use_container_width=True, hide_index=True)
else:
    st.info("Letterboxd details not merged yet — run `python -m src.movies.letterboxd_details --apply`.")

st.divider()

st.subheader("🎭 Genre & Maturity Analysis")
st.caption("Identify which specific categories drive your ratings up or down.")

t_genre, t_rated = st.tabs(["Genre Influence", "Maturity (MPAA) Ratings"])

with t_genre:
    if 'genre_list' in df.columns:
        gen_ent = calculate_entropy(df['genre_list'])
        st.caption(f"**Genre Diversity (Entropy):** {gen_ent:.2f}")
        
        df_genre_exp = df.explode('genre_list')
        df_genre_exp = df_genre_exp[~df_genre_exp['genre_list'].isin(['Unknown', 'nan', np.nan, ''])]
        
        genre_stats = df_genre_exp.groupby('genre_list').agg(
            Count=('user_rating', 'size'),
            Avg_Rating=('user_rating', 'mean'),
            Mode_Rating=('user_rating', lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        ).reset_index()
        genre_stats['Avg_Rating'] = genre_stats['Avg_Rating'].round(3)
        genre_stats = genre_stats.sort_values('Count', ascending=False)
        
        cg1, cg2 = st.columns([1, 1.2])
        with cg1:
            st.write("**Genre Breakdown**")
            
            # 1. Count Chart (Popularity)
            fig_count = px.bar(
                genre_stats.head(15), 
                x='Count', 
                y='genre_list', 
                orientation='h', 
                title="Most Watched Genres",
                color='Count', 
                color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[1]]],
                text='Count'
            )
            fig_count.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig_count.update_traces(textposition='outside')
            st.plotly_chart(fig_count, use_container_width=True)

            # 2. Avg Rating Chart (Quality)
            top_rated_genres = genre_stats[genre_stats['Count'] >= 25].sort_values('Avg_Rating', ascending=False).head(15)
            fig_avg = px.bar(
                top_rated_genres, 
                x='Avg_Rating', 
                y='genre_list', 
                orientation='h', 
                title="Highest Rated Genres (Min 25 Movies)",
                color='Avg_Rating', 
                color_continuous_scale=[[0, HUB_PALETTE[4]], [1, HUB_PALETTE[0]]],
                range_x=[0, 5],
                text='Avg_Rating'
            )
            fig_avg.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=40, b=0))
            fig_avg.update_traces(textposition='inside', texttemplate='%{text:.2f}')
            st.plotly_chart(fig_avg, use_container_width=True)
            
        with cg2:
            st.write("**Rating Spread per Genre (Influence)**")
            st.caption(SPREAD_CAPTION.format(noun="film"))
            fig = genre_spread_figure(
                df_genre_exp, 'genre_list', 'user_rating', 'title', HUB_PALETTE,
                min_count=5, x_title="Genre", y_title="Your Rating",
                year_col='year', height=740, noun="film(s)")
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough genre-tagged ratings to draw the spread.")

with t_rated:
    if 'rated' in df.columns:
        df['rated'] = df['rated'].fillna('Unrated')
        rated_ent = calculate_entropy(df['rated'])
        st.caption(f"**Maturity Rating Diversity (Entropy):** {rated_ent:.2f}")
        
        rated_stats = df.groupby('rated').agg(
            Count=('user_rating', 'size'),
            Avg_Rating=('user_rating', 'mean')
        ).reset_index()
        rated_stats['Avg_Rating'] = rated_stats['Avg_Rating'].round(3)
        rated_stats = rated_stats.sort_values('Count', ascending=False)
        
        cr1, cr2 = st.columns(2)
        with cr1:
            fig = px.bar(rated_stats, x='rated', y='Count', hover_data={'Avg_Rating': ':.3f'}, title="Maturity Rating Breakdown", color_discrete_sequence=[HUB_PALETTE[2]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
        with cr2:
            fig = px.box(df, x="rated", y="user_rating", color="rated", title="Rating Spread per Maturity Level", color_discrete_sequence=HUB_PALETTE)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# 11b. RATING DISTRIBUTION BY GENRE (RIDGELINE)
st.subheader("🎚️ Rating Distribution by Genre")
st.caption("A ridgeline of how your scores are spread *within* each genre. Each curve is a "
           "smoothed distribution of your ratings; warmer curves are genres you rate higher on "
           "average, and the number on the right is how many movies fed that curve. Genres are "
           "stacked from your lowest- to highest-rated.")
if 'genre_list' in df.columns:
    _ridge_pairs = (df[['genre_list', 'user_rating']].explode('genre_list')
                    .rename(columns={'genre_list': 'genre', 'user_rating': 'rating'}))
    _fig_ridge = genre_rating_ridgeline(_ridge_pairs, rating_label="Your Rating",
                                        count_label="movies", min_count=10)
    if _fig_ridge is not None:
        st.plotly_chart(_fig_ridge, use_container_width=True)
    else:
        st.info("Not enough genre-tagged ratings to draw the ridgeline.")

st.divider()

# 12. SENTIMENT ANALYSIS
st.subheader("🧠 Emotional Sentiment Analysis")
st.caption("Upgraded to use VADER NLP Engine. Contextualizes genre and plot to score movies from -1.0 (Tragic/Dark) to +1.0 (Happy/Wholesome).")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    
    analyzer = SentimentIntensityAnalyzer()
    
    df['sentiment_text'] = "Genres: " + df['genre'].fillna('') + ". Plot: " + df['overview'].fillna('') + " " + df['tagline'].fillna('') + " " + df.get('plot', pd.Series(dtype=str)).fillna('')
    df['sentiment'] = df['sentiment_text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if len(str(x)) > 5 else 0)
    
    t_sent, t_corr = st.tabs(["High Conflict vs. Wholesome", "Does Mood Affect Rating?"])
    
    with t_sent:
        bins = [-1, -0.25, 0.25, 1]
        labels = ['High Conflict/Tragic 🌑', 'Neutral/Balanced 😐', 'Wholesome/Triumphant ☀️']
        df['mood'] = pd.cut(df['sentiment'], bins=bins, labels=labels)
        mood_counts = df['mood'].value_counts().reset_index()
        mood_counts.columns = ['Mood', 'Count']
        
        c_pie, c_list = st.columns([1, 2])
        with c_pie:
            fig = px.pie(mood_counts, values='Count', names='Mood', title="VADER Sentiment Split", hole=0.4, color_discrete_sequence=[HUB_PALETTE[3], HUB_PALETTE[2], HUB_PALETTE[1]])
            fig.update_traces(marker_line_width=1, marker_line_color="#2C343F")
            st.plotly_chart(fig, use_container_width=True)
            
        with c_list:
            st.write("**Most 'Negative/Conflict' Plots:**")
            darkest = df.sort_values('sentiment').head(3)
            for _, r in darkest.iterrows():
                st.write(f"🌑 **{r['title']}** ({r['sentiment']:.2f})")
                
            st.write("**Most 'Positive/Wholesome' Plots:**")
            happiest = df.sort_values('sentiment', ascending=False).head(3)
            for _, r in happiest.iterrows():
                st.write(f"☀️ **{r['title']}** (+{r['sentiment']:.2f})")

    with t_corr:
        fig = px.scatter(
            df, x="sentiment", y="user_rating",
            color="user_rating", hover_data=['title'],
            title="VADER Lexicon Score vs. Your Rating",
            labels={'sentiment': 'Sentiment (-1 Conflict to +1 Wholesome)'},
            range_x=[-1, 1], trendline="ols",
            color_continuous_scale=[[0, HUB_PALETTE[3]], [1, HUB_PALETTE[0]]]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If the line goes UP, you prefer 'wholesome' plots. If DOWN, you prefer plots with high conflict/action keywords.")

except ImportError:
    st.warning("⚠️ vaderSentiment library not found. Run `pip install vaderSentiment` to see Sentiment Analysis.")

st.divider()

# 12b. SUBTITLES — dialogue talk, mood curves, sound cues (see app/subtitle_analytics.py)
render_subtitle_analytics()

st.divider()

# 12c. STANDUP SPECIALS
# Specials are ordinary rows in enriched_data.csv too, so they already count in
# every chart above; this section looks at the Letterboxd list as a group.
render_standup()

st.divider()

# 12d. YOUR LETTERBOXD NETWORK
# Network data is dashboard-only: it never goes into enriched_data.csv or any feature file.
LB_RATING, LB_LINK = 'Rating (out of 5)', 'Letterboxd link'
MIN_RHO_N = 20  # agreement on fewer shared rated films is mostly noise


@st.cache_data
def load_lb_network():
    paths = [config.MOVIES_PROCESSED_DIR / f"letterboxd_network_{n}.csv" for n in ("members", "films")]
    if not all(p.exists() for p in paths):
        return None, None
    return pd.read_csv(paths[0]), pd.read_csv(paths[1])


@st.cache_data
def lb_network_analysis(mine_df, members, films):
    """mine_df: your films (lb_film_url, user_rating, is_liked). Returns member stats, picks,
    per-film you-vs-network, most watched, and how many members saw each of your films."""
    mine_df = mine_df.dropna(subset=['lb_film_url']).drop_duplicates('lb_film_url').set_index('lb_film_url')
    mine = mine_df['user_rating']
    seen, my_rated = set(mine.index), mine.dropna()
    my_fives = set(my_rated[my_rated == 5].index)
    my_liked = set(mine_df.index[mine_df['is_liked'] == 1])
    rated = films.dropna(subset=[LB_RATING])
    rows = []
    for user, g in films.groupby('Username'):
        links = set(g[LB_LINK])
        theirs = rated[rated['Username'] == user].set_index(LB_LINK)[LB_RATING]
        both = theirs.index.intersection(my_rated.index)
        a, b = my_rated[both], theirs[both]
        enough = len(both) >= MIN_RHO_N
        rows.append({'Username': user, 'Shared': len(links & seen),
                     'Jaccard %': 100 * len(links & seen) / len(links | seen),
                     'Shared Rated': len(both),
                     # Spearman = Pearson on ranks; avoids a scipy dependency
                     'Spearman': a.rank().corr(b.rank()) if enough else np.nan,
                     'Mean |Δ★|': (a - b).abs().mean() if enough else np.nan,
                     'Their − Your ★': (b - a).mean() if enough else np.nan,
                     'Your 5★ Seen': len(links & my_fives),
                     'Both Liked': len(set(g.loc[g['Liked'] == 'Yes', LB_LINK]) & my_liked)})
    stats = members.merge(pd.DataFrame(rows), on='Username', how='left')

    weight = stats.set_index('Username')['Spearman'].clip(lower=0).fillna(0)
    unseen = rated[~rated[LB_LINK].isin(seen)].assign(w=lambda d: d['Username'].map(weight))
    unseen = unseen[unseen['w'] > 0].assign(wr=lambda d: d['w'] * d[LB_RATING])
    picks = (unseen.groupby(LB_LINK)
             .agg(Film=('Film', 'first'), Year=('Year', 'first'), Raters=('Username', 'size'),
                  wr=('wr', 'sum'), w=('w', 'sum'), NetAvg=(LB_RATING, 'mean'))
             .query('Raters >= 3'))
    # Pull small samples toward the overall weighted mean, so three 5★ ratings
    # don't outrank thirty 4.8★ ones. PRIOR_W ~ two typical raters' weight.
    PRIOR_W = 1.0
    prior = unseen['wr'].sum() / unseen['w'].sum()
    picks['Score'] = (picks['wr'] + PRIOR_W * prior) / (picks['w'] + PRIOR_W)
    picks = picks.sort_values(['Score', 'Raters'], ascending=False).head(20).reset_index()

    net = (rated[rated[LB_LINK].isin(my_rated.index)].groupby(LB_LINK)
           .agg(Film=('Film', 'first'), Year=('Year', 'first'),
                Raters=('Username', 'size'), NetAvg=(LB_RATING, 'mean'))
           .query('Raters >= 3'))
    net['You'] = my_rated.reindex(net.index)
    net['You − Network'] = net['You'] - net['NetAvg']

    watched = (films.groupby(LB_LINK)
               .agg(Film=('Film', 'first'), Year=('Year', 'first'), Members=('Username', 'size'),
                    NetAvg=(LB_RATING, 'mean'))
               .sort_values('Members', ascending=False).head(20).reset_index())
    watched['You'] = watched[LB_LINK].map(mine)
    watched['Seen'] = watched[LB_LINK].isin(seen)
    seen_by = films[LB_LINK].value_counts().reindex(list(seen), fill_value=0)
    return stats, picks, net.reset_index(), watched, seen_by


def lb_leaderboard(frame, col, ascending, cols=('Member', 'Relation to you', 'Shared Rated')):
    """Top 5 members by one column, as a compact table."""
    st.dataframe(frame.dropna(subset=[col]).sort_values(col, ascending=ascending).head(5)
                 [list(dict.fromkeys([*cols, col]))].round(2),
                 use_container_width=True, hide_index=True)


st.subheader("👥 Your Letterboxd Network")
st.caption("The people you follow and who follow you on Letterboxd: whose taste overlaps "
           "yours, what they rate highly that you haven't seen, and where you part ways.")

net_members, net_films = load_lb_network()
if net_members is None or 'lb_film_url' not in df.columns:
    st.info("Network data not built yet — run `python -m src.movies.letterboxd_details --apply`.")
else:
    nstats, npicks, ndis, nwatched, nseen_by = lb_network_analysis(
        df[['lb_film_url', 'user_rating', 'is_liked']], net_members, net_films)
    twins = nstats.dropna(subset=['Spearman']).nlargest(3, 'Spearman')
    if not twins.empty:
        st.write(f"**Taste twins** (highest rank agreement, ≥{MIN_RHO_N} shared rated films):")
        for col, (_, r) in zip(st.columns(3), twins.iterrows()):
            with col:
                lb_metric(r['Member'], f"ρ {r['Spearman']:.2f}",
                          help_text=f"{r['Shared Rated']} shared rated films, "
                                    f"mean gap {r['Mean |Δ★|']:.2f}★, {r['Relation to you']}")

    (t_n_members, t_n_agree, t_n_overlap, t_n_harsh, t_n_cover,
     t_n_picks, t_n_dis, t_n_pop) = st.tabs(
        ["Members", "Agree & Disagree", "Overlap", "Harshness", "Your Films in the Network",
         "Network Picks", "You vs Your Network", "Most Watched"])
    with t_n_members:
        fig = px.scatter(nstats.dropna(subset=['Spearman']), x='Jaccard %', y='Spearman',
                         size='Shared Rated', color='Relation to you', hover_name='Member',
                         hover_data=['Shared', 'Mean |Δ★|'],
                         title="Overlap (how much you watch the same films) vs agreement",
                         color_discrete_sequence=HUB_PALETTE)
        fig.add_hline(y=0, line_dash="dot", line_color="#99AABB")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Jaccard % = shared films / all films either of you watched. Spearman "
                   f"(rank agreement, −1 to 1), mean star gap and harshness need "
                   f"≥{MIN_RHO_N} shared rated films.")
        st.dataframe(nstats[['Member', 'Relation to you', 'Shared', 'Jaccard %', 'Shared Rated',
                             'Spearman', 'Mean |Δ★|', 'Their − Your ★', 'Your 5★ Seen',
                             'Both Liked', 'Films watched', 'Average rating (out of 5)']]
                     .sort_values('Spearman', ascending=False).round(2),
                     use_container_width=True, hide_index=True, height=420)
    with t_n_agree:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Rank your films most alike (highest Spearman)**")
            lb_leaderboard(nstats, 'Spearman', False)
            st.write("**Closest stars (lowest mean gap)**")
            lb_leaderboard(nstats, 'Mean |Δ★|', True)
        with c2:
            st.write("**Rank your films least alike (lowest Spearman)**")
            lb_leaderboard(nstats, 'Spearman', True)
            st.write("**Furthest stars (highest mean gap)**")
            lb_leaderboard(nstats, 'Mean |Δ★|', False)
        st.write("**Agreement by relation**")
        st.dataframe(nstats.dropna(subset=['Spearman']).groupby('Relation to you')
                     .agg(Members=('Member', 'size'), **{'Mean Spearman': ('Spearman', 'mean'),
                          'Mean |Δ★|': ('Mean |Δ★|', 'mean'), 'Mean Jaccard %': ('Jaccard %', 'mean')})
                     .round(2).reset_index(), use_container_width=True, hide_index=True)
    with t_n_overlap:
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Most shared films**")
            lb_leaderboard(nstats, 'Shared', False, ('Member', 'Relation to you', 'Jaccard %'))
            st.write("**Highest Jaccard (overlap relative to both libraries)**")
            lb_leaderboard(nstats, 'Jaccard %', False, ('Member', 'Relation to you', 'Shared'))
            st.write("**Seen the most of your 5★ films**")
            lb_leaderboard(nstats, 'Your 5★ Seen', False, ('Member', 'Relation to you', 'Shared'))
        with c2:
            st.write("**Fewest shared films**")
            lb_leaderboard(nstats, 'Shared', True, ('Member', 'Relation to you', 'Jaccard %'))
            st.write("**Lowest Jaccard**")
            lb_leaderboard(nstats, 'Jaccard %', True, ('Member', 'Relation to you', 'Shared'))
            st.write("**Most films you both liked (❤️)**")
            lb_leaderboard(nstats, 'Both Liked', False, ('Member', 'Relation to you', 'Shared'))
    with t_n_harsh:
        harsh = nstats.dropna(subset=['Their − Your ★']).sort_values('Their − Your ★')
        fig = px.bar(harsh, x='Their − Your ★', y='Member', orientation='h',
                     color='Relation to you', hover_data=['Shared Rated'],
                     title="Each member's mean rating minus yours, on films you both rated",
                     color_discrete_sequence=HUB_PALETTE)
        fig.add_vline(x=0, line_dash="dot", line_color="#99AABB")
        fig.update_layout(height=max(400, 18 * len(harsh)), yaxis_title=None)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Left of zero = harsher than you on the same films; right = more generous.")
    with t_n_cover:
        unique = df[df['lb_film_url'].isin(nseen_by.index[nseen_by == 0])]
        m1, m2 = st.columns(2)
        with m1: lb_metric("Network coverage", f"{(nseen_by > 0).mean():.0%}",
                           help_text="Share of your films that at least one network member has watched.")
        with m2: lb_metric("Films only you watched", len(unique))
        st.write("**Your films nobody in your network has watched:**")
        st.dataframe(unique[['title', 'year', 'user_rating']].sort_values('user_rating', ascending=False)
                     .rename(columns={'title': 'Film', 'year': 'Year', 'user_rating': 'Your Rating'})
                     .assign(Year=lambda d: d['Year'].astype('Int64')),
                     use_container_width=True, hide_index=True, height=360)
    with t_n_picks:
        st.caption("Films you haven't logged, scored by the average rating of members "
                   "weighted by how well their ranking agrees with yours (positive Spearman "
                   "only). Needs ≥3 such members to have rated it.")
        st.dataframe(npicks[['Film', 'Year', 'Score', 'Raters', 'NetAvg', LB_LINK]]
                     .rename(columns={'NetAvg': 'Plain Avg', LB_LINK: 'Link'})
                     .assign(Year=lambda d: d['Year'].astype('Int64')).round(2),
                     use_container_width=True, hide_index=True,
                     column_config={'Link': st.column_config.LinkColumn()})
    with t_n_dis:
        c_up, c_down = st.columns(2)
        for col, part, label in [
                (c_up, ndis.nlargest(10, 'You − Network'), "You liked more than your network"),
                (c_down, ndis.nsmallest(10, 'You − Network'), "You liked less than your network")]:
            with col:
                st.write(f"**{label}:**")
                for _, r in part.iterrows():
                    st.markdown(f"**{html.escape(str(r['Film']))}** ({r['You − Network']:+.2f})  \n"
                                f"<span style='font-size:0.95em; color:#99AABB;'>{render_stars(r['You'])}"
                                f" | Network: {r['NetAvg']:.2f} from {r['Raters']} members</span>",
                                unsafe_allow_html=True)
    with t_n_pop:
        st.dataframe(nwatched[['Film', 'Year', 'Members', 'NetAvg', 'Seen', 'You']]
                     .rename(columns={'NetAvg': 'Network Avg', 'You': 'Your Rating'})
                     .assign(Year=lambda d: d['Year'].astype('Int64')).round(2),
                     use_container_width=True, hide_index=True)

st.divider()

# 13. THEATRE WATCH ANALYSIS
render_theatre_watch(df, df_diary, df_diary_tags, stops)
