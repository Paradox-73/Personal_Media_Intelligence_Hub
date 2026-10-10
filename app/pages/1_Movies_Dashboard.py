import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
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

        t_kw_top, t_kw_common, t_kw_search, t_kw_all = st.tabs(
            ["Best & Worst Themes", "Most Common", "🔎 Search by Keyword", "All Keywords"])
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
    else:
        st.info("No keywords parsed yet.")
else:
    st.info("Keywords not ingested yet — run `python -m src.movies.enrich_extended --apply`.")

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

# 13. THEATRE WATCH ANALYSIS
render_theatre_watch(df, df_diary, df_diary_tags, stops)
