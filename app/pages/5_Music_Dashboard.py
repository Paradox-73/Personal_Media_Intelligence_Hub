import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import sys
import numpy as np
from pathlib import Path
from collections import Counter

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from src import config
from geo_utils import aggregate_country_counts, make_world_map

st.set_page_config(page_title="Music Intelligence", page_icon="🎵", layout="wide")

# Spotify Brand Colors
SPOTIFY_GREEN = "#1DB954"
SPOTIFY_BLACK = "#191414"
SPOTIFY_WHITE = "#FFFFFF"

# Custom CSS for Spotify vibe
st.markdown(f"""
    <style>
    .main {{
        background-color: {SPOTIFY_BLACK};
        color: {SPOTIFY_WHITE};
    }}
    .stMetric {{
        background-color: #282828;
        padding: 15px;
        border-radius: 10px;
    }}
    div[data-testid="stExpander"] {{
        background-color: #282828;
        border: none;
    }}
    </style>
    """, unsafe_allow_html=True)

DARK_LAYOUT = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color=SPOTIFY_WHITE))

# --- HELPER FUNCTIONS ---

def parse_genres(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s or s.lower() in ('nan', 'none', 'unknown'):
        return []
    if s.startswith('[') and s.endswith(']'):
        try:
            return [str(g).strip() for g in ast.literal_eval(s)]
        except Exception:
            pass
    # Genres are usually comma/semicolon separated
    return [g.strip() for g in s.replace(';', ',').split(',') if g.strip() and g.strip().lower() != 'unknown']

def primary_artist(x):
    # Keep the full credit intact ("Tyler, The Creator" must not be split on the comma).
    if pd.isna(x):
        return None
    s = str(x).strip()
    return s if s and s.lower() not in ('nan', 'none', 'unknown') else None

def get_frequent_items(series, top_n=10):
    all_items = [i for i in series if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=['Item', 'Count'])

def get_frequent_genres(series, top_n=12):
    all_items = []
    for row in series:
        if isinstance(row, list):
            all_items.extend(row)
    all_items = [i for i in all_items if i and str(i).lower() not in ['unknown', 'nan', 'none']]
    return pd.DataFrame(Counter(all_items).most_common(top_n), columns=['Genre', 'Count'])

# --- LYRIC N-GRAMS ---
# Stopwords + lyric filler (oh/yeah/la…) so phrases surface real hooks, not "i the".
LYRIC_STOPWORDS = set("""a about above after again against all am an and any are aren't as at be because been
before being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most
mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she
she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd
we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's will with
won't would wouldn't you you'd you'll you're you've your yours yourself yourselves got get gonna gotta wanna cause
'cause em ll ve re oh yeah yea la na uh ooh ah ay ayy aye hey woah whoa mmm hmm da do nah de just scp
ha doh du mm ba doo hoo hol wassup ya""".split())

# Words/phrases to drop outright from every view — junk and mis-matched-lyric leakage
# (Genius/LRCLIB returned book/wiki text for some tracks: Proust's Guermantes, the
# "Zau al-Makan" tale, SCP wiki, plus artist-name bleed). Any n-gram that contains one
# of these as a contiguous run is removed. Use the drill-down below to find the source tracks.
LYRIC_BLOCKLIST = [
    "scp", "zau", "mme", "guermantes", "villeparisis",
    "tyler the creator", "zau al makan", "mme de guermantes", "mme de villeparisis",
]
_BLOCKED_GRAMS = [tuple(p.split()) for p in LYRIC_BLOCKLIST]

def _gram_blocked(tokens):
    for bl in _BLOCKED_GRAMS:
        L = len(bl)
        if L <= len(tokens) and any(tuple(tokens[i:i + L]) == bl for i in range(len(tokens) - L + 1)):
            return True
    return False

def _is_repetition(tokens):
    """True if the n-gram is a short pattern repeated for at least two cycles —
    pure filler/ad-libs like 'ha ha ha', 'low low low low' (period 1),
    'up hol up hol up' or 'she said she said she' (period 2). Uses the string
    'period' definition (tokens[i] == tokens[i+p]) so phase-shifted, non-integer
    repeats are caught too. p ≤ n//2 ensures ≥2 cycles, so genuine phrases
    ('bitch don't kill my vibe') are kept. Never adds real words to the stopwords."""
    n = len(tokens)
    for p in range(1, n // 2 + 1):
        if all(tokens[i] == tokens[i + p] for i in range(n - p)):
            return True
    return False

_WORD_RE = re.compile(r"[a-z']+")
# Lyric cache files are "Artist - Song [track_id].txt" (legacy: "<track_id>.txt").
# The trailing [track_id] is the lookup key; build a track_id -> Path index once.
_ID_IN_NAME = re.compile(r"\[([A-Za-z0-9]+)\]\.txt$")

def _extract_track_id(filename):
    m = _ID_IN_NAME.search(filename)
    if m:
        return m.group(1)
    stem = filename[:-4] if filename.lower().endswith(".txt") else filename
    return stem if re.fullmatch(r"[A-Za-z0-9]+", stem) else None

@st.cache_data(show_spinner="Indexing lyric cache…")
def index_lyric_cache():
    lyrics_dir = config.CACHE_DIR / "music" / "lyrics"
    idx = {}
    if lyrics_dir.exists():
        for f in lyrics_dir.iterdir():
            if f.suffix.lower() == ".txt":
                tid = _extract_track_id(f.name)
                if tid:
                    idx[tid] = str(f)
    return idx

def _filter_signature():
    """Stable signature of the stopword/blocklist config, passed into the cached
    n-gram function so editing the filters busts Streamlit's cache (st.cache_data
    keys on args + the function body, NOT on module globals like LYRIC_STOPWORDS)."""
    return (tuple(sorted(LYRIC_STOPWORDS)), tuple(LYRIC_BLOCKLIST))

@st.cache_data(show_spinner="Mining lyric phrases…")
def compute_lyric_ngrams(track_ids, filter_sig, top_n=25):
    """Count word/phrase frequencies across the local lyric cache (n = 1–5).

    Reads data/cache/music/lyrics/<track_id>.txt for each track. Returns, per n,
    both a raw most-common list and a 'content' list that additionally drops grams
    made entirely of stopwords/filler. Pure repetitions ('ha ha ha', 'low low low')
    are dropped from BOTH lists — they're never useful. `filter_sig` only keys the
    cache (see _filter_signature); files_read reports how many cached lyrics we hit.
    """
    cache_index = index_lyric_cache()
    counters = {n: Counter() for n in range(1, 6)}
    files_read = 0
    for tid in track_ids:
        path = cache_index.get(tid)
        if not path:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8").lower()
        except Exception:
            continue
        files_read += 1
        tokens = [t.strip("'") for t in _WORD_RE.findall(text)]
        tokens = [t for t in tokens if len(t) > 1 or t == 'i']  # drop stray single letters
        for n in range(1, 6):
            for i in range(len(tokens) - n + 1):
                counters[n][" ".join(tokens[i:i + n])] += 1

    def take_top(counter, content_only):
        out = []
        for gram, c in counter.most_common():
            toks = gram.split()
            if _gram_blocked(toks) or _is_repetition(toks):   # junk/repeats: drop from both views
                continue
            if content_only and all(w in LYRIC_STOPWORDS for w in toks):
                continue
            out.append((gram, c))
            if len(out) >= top_n:
                break
        return out

    result = {n: {'all': take_top(counters[n], False),
                  'content': take_top(counters[n], True)} for n in range(1, 6)}
    return result, files_read

@st.cache_data(show_spinner="Finding songs…")
def songs_for_phrase(track_ids, phrase):
    """Per-track occurrence count of an exact word/phrase across the lyric cache.

    Returns [(track_id, count), …] sorted desc — the drill-down for spotting which
    tracks (often mis-matched lyrics) are inflating a given word/phrase.
    """
    cache_index = index_lyric_cache()
    target = phrase.split()
    L = len(target)
    counts = {}
    for tid in track_ids:
        path = cache_index.get(tid)
        if not path:
            continue
        try:
            text = Path(path).read_text(encoding="utf-8").lower()
        except Exception:
            continue
        toks = [t.strip("'") for t in _WORD_RE.findall(text)]
        toks = [t for t in toks if len(t) > 1 or t == 'i']
        c = sum(1 for i in range(len(toks) - L + 1) if toks[i:i + L] == target)
        if c:
            counts[tid] = c
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:15]

# --- LOAD DATA ---
@st.cache_data
def load_data():
    paths_to_try = [config.MUSIC_ENRICHED_DATA_PATH, config.MUSIC_FULL_VIEW_PATH]
    df = None
    for path in paths_to_try:
        if path.exists():
            try:
                df = pd.read_csv(path)
                break
            except Exception:
                continue

    if df is None:
        return None

    try:
        # Clean Numerics
        df['release_year'] = pd.to_numeric(df.get('release_year'), errors='coerce').fillna(0).astype(int)
        df['rating'] = pd.to_numeric(df.get('rating'), errors='coerce')
        if 'popularity' in df.columns:
            df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')

        # Track length now comes from MusicBrainz (mb_length_ms); fall back to legacy duration_ms
        length_col = 'mb_length_ms' if 'mb_length_ms' in df.columns else ('duration_ms' if 'duration_ms' in df.columns else None)
        if length_col:
            df['duration_min'] = pd.to_numeric(df[length_col], errors='coerce') / 60000
        else:
            df['duration_min'] = np.nan

        # Lyric-derived numeric features
        for c in ['lyric_word_count', 'lyric_unique_ratio', 'lyric_line_count',
                  'lyric_sentiment', 'lyric_pos', 'lyric_neu', 'lyric_neg']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Filter out Year 0 (unknown release)
        df = df[df['release_year'] > 0]

        # Primary artist for grouping (avoid splitting "Tyler, The Creator")
        artist_src = 'artists' if 'artists' in df.columns else 'mb_canonical_artist'
        df['primary_artist'] = df[artist_src].apply(primary_artist)

        # Genres: prefer Spotify artist_genres, fall back to MusicBrainz genres/tags
        genre_src = None
        for cand in ['artist_genres', 'mb_genres', 'mb_tags', 'genre']:
            if cand in df.columns:
                genre_src = cand
                break
        df['genre_list'] = df[genre_src].apply(parse_genres) if genre_src else [[] for _ in range(len(df))]

        df['decade'] = (df['release_year'] // 10) * 10

        # Coalesce artist origin country from every source we have, best first
        # (MusicBrainz alpha-2 code -> Wikidata -> TheAudioDB -> MB area name).
        # geo_utils resolves both ISO codes and full names, so a mixed column is fine.
        country_sources = [c for c in ['mb_artist_country_code', 'wd_country_of_origin',
                                       'audiodb_country', 'mb_artist_country']
                           if c in df.columns]
        if country_sources:
            geo = df[country_sources[0]]
            for c in country_sources[1:]:
                geo = geo.fillna(df[c])
            df['geo_country'] = geo

        # Deezer tempo (BPM) — a current substitute for Spotify's deprecated tempo.
        if 'deezer_bpm' in df.columns:
            df['deezer_bpm'] = pd.to_numeric(df['deezer_bpm'], errors='coerce')

        # Last.fm global popularity (arrives as strings) -> numeric.
        for c in ['lastfm_artist_listeners', 'lastfm_artist_playcount',
                  'lastfm_track_playcount', 'lastfm_track_listeners']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        import traceback
        st.error(f"Error processing music data: {e}")
        st.code(traceback.format_exc())
        return None

df = load_data()

if df is None:
    st.error("❌ Music Data not found. Run the music pipeline first.")
    st.stop()

# --- HEADER ---
st.title("🎵 Music Intelligence")
st.markdown(f"**Sonic Profile Analysis** | {len(df):,} tracks in library")

# 1. TOP METRICS
has_lyrics = 'lyric_word_count' in df.columns
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Tracks", f"{len(df):,}")
with c2:
    total_hrs = df['duration_min'].sum() / 60
    st.metric("Library Duration", f"{int(total_hrs):,} hrs" if total_hrs else "N/A")
with c3:
    if 'lyrics_found' in df.columns:
        cov = df['lyrics_found'].astype(str).str.lower().isin(['true', '1', '1.0']).mean() * 100
        st.metric("Lyrics Coverage", f"{cov:.0f}%")
    else:
        st.metric("Lyrics Coverage", "N/A")
with c4:
    if has_lyrics:
        st.metric("Total Lyric Words", f"{int(df['lyric_word_count'].fillna(0).sum()):,}")
    else:
        st.metric("Total Lyric Words", "N/A")

st.divider()

# 2. LYRICAL LANDSCAPE (replaces Spotify audio features, which are no longer ingested)
if has_lyrics:
    st.subheader("📝 The Lyrical Landscape")
    with st.expander("ℹ️ What do these lyric features mean?"):
        st.markdown("""
        *   **Sentiment**: VADER compound score of the lyrics, from -1.0 (dark/aggressive) to +1.0 (positive/uplifting).
        *   **Word Count**: Total words in the track's lyrics — a proxy for lyrical density.
        *   **Lexical Richness**: Ratio of unique words to total words. Higher = more varied vocabulary.
        """)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        # Which genres carry the most positive vs darkest lyrics. (VADER's raw pos/neu/neg
        # split is ~78% neutral and uninformative, so we lean on genres instead.)
        if 'lyric_sentiment' in df.columns and df['lyric_sentiment'].notna().any():
            gexp = df.explode('genre_list')
            gexp = gexp[gexp['genre_list'].astype(str).str.len().gt(0) & gexp['lyric_sentiment'].notna()]
            gstats = gexp.groupby('genre_list').agg(
                Tracks=('lyric_sentiment', 'size'), Sentiment=('lyric_sentiment', 'mean')
            ).reset_index()
            gstats = gstats[gstats['Tracks'] >= 10].sort_values('Sentiment')
            if not gstats.empty:
                show = pd.concat([gstats.head(8), gstats.tail(8)]).drop_duplicates('genre_list')
                show['Sentiment'] = show['Sentiment'].round(3)
                fig_g = px.bar(show, x='Sentiment', y='genre_list', orientation='h', color='Sentiment',
                               title="Genres by Avg Lyric Sentiment", hover_data={'Tracks': True},
                               color_continuous_scale=['#E22134', '#535353', SPOTIFY_GREEN])
                fig_g.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None},
                                    coloraxis_showscale=False, **DARK_LAYOUT)
                st.plotly_chart(fig_g, use_container_width=True)
                st.caption("Green = genres whose lyrics skew positive; red = genres that skew darker (min 10 tracks).")
            else:
                st.info("Not enough genre-tagged tracks for a sentiment breakdown.")
    with col_b:
        if 'lyric_sentiment' in df.columns and df['lyric_sentiment'].notna().any():
            fig_dist = px.histogram(df.dropna(subset=['lyric_sentiment']), x='lyric_sentiment', nbins=40,
                                    title="Lyric Sentiment Distribution",
                                    labels={'lyric_sentiment': 'Lyric Sentiment (-1 dark → +1 positive)'},
                                    color_discrete_sequence=[SPOTIFY_GREEN])
            fig_dist.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_dist, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        if 'lyric_word_count' in df.columns:
            fig_wc = px.histogram(df.dropna(subset=['lyric_word_count']), x='lyric_word_count', nbins=40,
                                  title="Lyrical Density (Words per Track)",
                                  color_discrete_sequence=[SPOTIFY_GREEN])
            fig_wc.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_wc, use_container_width=True)
    with col_d:
        if 'lyric_unique_ratio' in df.columns:
            fig_lex = px.histogram(df.dropna(subset=['lyric_unique_ratio']), x='lyric_unique_ratio', nbins=40,
                                   title="Lexical Richness (Unique-word Ratio)",
                                   color_discrete_sequence=['#1ED760'])
            fig_lex.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_lex, use_container_width=True)

    st.divider()

# 2b. MOST-REPEATED WORDS & PHRASES (mined from the local lyric cache)
if 'track_id' in df.columns:
    ngrams, n_lyrics = compute_lyric_ngrams(tuple(df['track_id'].dropna().astype(str)), _filter_signature())
    if n_lyrics > 0:
        st.subheader("🔤 Most-Repeated Words & Phrases")
        st.caption(f"What recurs most across the lyrics of {n_lyrics:,} tracks with cached lyrics. "
                   "Pick a phrase length — counts are total occurrences across the whole library.")
        hide_filler = st.checkbox(
            "Hide filler-only phrases (the, you, oh, yeah …)", value=True,
            help="When on, words/phrases made up entirely of common stopwords and lyric filler are dropped, "
                 "so real hooks surface instead of 'i' or 'in the'.")

        tabs = st.tabs(["1 word", "2 words", "3 words", "4 words", "5 words"])
        for n, tab in zip(range(1, 6), tabs):
            with tab:
                pairs = ngrams[n]['content' if hide_filler else 'all'][:15]
                if not pairs:
                    st.info("No phrases of this length found.")
                    continue
                ng_df = pd.DataFrame(pairs, columns=['Phrase', 'Count'])
                label = 'word' if n == 1 else 'phrase'
                fig_ng = px.bar(ng_df, x='Count', y='Phrase', orientation='h',
                                title=f"Top {n}-{label}s by total occurrences",
                                color='Count', color_continuous_scale=['#1d3a2a', SPOTIFY_GREEN])
                fig_ng.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None},
                                     coloraxis_showscale=False, height=460, **DARK_LAYOUT)
                st.plotly_chart(fig_ng, use_container_width=True)

        # Drill-down: which songs are repeating a given word/phrase? Handy for spotting
        # mis-matched lyrics that inflate a count.
        key = 'content' if hide_filler else 'all'
        phrase_choices = []
        for n in range(1, 6):
            phrase_choices.extend(g for g, _ in ngrams[n][key][:15])
        phrase_choices = sorted(set(phrase_choices))
        if phrase_choices:
            st.markdown("**🔎 Which songs repeat a word/phrase?**")
            pick = st.selectbox("Inspect a word or phrase", phrase_choices,
                                help="See the tracks where this appears most — useful to catch "
                                     "wrong-lyric matches.")
            id2name = dict(zip(df['track_id'].astype(str), df['name'].astype(str))) \
                if 'name' in df.columns else {}
            id2artist = dict(zip(df['track_id'].astype(str), df['primary_artist'].astype(str))) \
                if 'primary_artist' in df.columns else {}
            hits = songs_for_phrase(tuple(df['track_id'].dropna().astype(str)), pick)
            if hits:
                hit_df = pd.DataFrame([
                    {'Song': id2name.get(tid, tid),
                     'Artist': id2artist.get(tid, ''),
                     'Times repeated': c} for tid, c in hits
                ])
                st.dataframe(hit_df, use_container_width=True, hide_index=True)
            else:
                st.info("No tracks found for that phrase.")

    st.divider()

# 3. ARTISTS & GENRES
col_art, col_gen = st.columns(2)

with col_art:
    st.subheader("🎤 Top Artists")
    freq_art = get_frequent_items(df['primary_artist'], 12)
    fig_art = px.bar(freq_art, x='Count', y='Item', orientation='h',
                     title="Most Frequent Artists",
                     color='Count', color_continuous_scale=['#1DB954', '#1ED760'])
    fig_art.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None}, **DARK_LAYOUT)
    st.plotly_chart(fig_art, use_container_width=True)

with col_gen:
    st.subheader("🎼 Genre Distribution")
    freq_gen = get_frequent_genres(df['genre_list'], 10)
    if not freq_gen.empty:
        fig_gen = px.pie(freq_gen, values='Count', names='Genre', hole=0.5,
                         title="Top Genres",
                         color_discrete_sequence=px.colors.sequential.Greens_r)
        fig_gen.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_gen, use_container_width=True)
    else:
        st.info("No genre metadata available.")

st.divider()

# 3b. GEOGRAPHY & LANGUAGE  (MusicBrainz/Wikidata/AudioDB origin + work language)
has_country = 'geo_country' in df.columns and df['geo_country'].notna().any()
has_language = 'mb_language' in df.columns and df['mb_language'].notna().any()

if has_country or has_language:
    st.subheader("🌍 Sonic Geography")

    if has_country:
        st.caption("Countries shaded by how many tracks you have from artists who originate there "
                   "(log-scaled so smaller scenes stay visible). Hover for the exact count and your average rating.")
        geo_agg, geo_unmapped = aggregate_country_counts(df, 'geo_country', rating_col='rating')
        if not geo_agg.empty:
            map_fig = make_world_map(geo_agg, low_color="#1d3a2a", high_color=SPOTIFY_GREEN,
                                     land_color="#282828", ocean_color=SPOTIFY_BLACK)
            st.plotly_chart(map_fig, use_container_width=True)
            if geo_unmapped:
                st.caption(f"Not placed on the map (unrecognized region): {', '.join(geo_unmapped)}")

    col_geo, col_lang = st.columns(2)
    with col_geo:
        if has_country and not geo_agg.empty:
            st.markdown("**Top Artist Origins**")
            top_ctry = geo_agg.head(10)
            fig_ctry = px.bar(top_ctry, x='Count', y='country', orientation='h',
                              color='Count', color_continuous_scale=['#1d3a2a', SPOTIFY_GREEN])
            fig_ctry.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None}, **DARK_LAYOUT)
            st.plotly_chart(fig_ctry, use_container_width=True)
    with col_lang:
        if has_language:
            st.markdown("**Lyrics Language**")
            lang_dist = (df.dropna(subset=['mb_language'])
                           .groupby('mb_language').size()
                           .sort_values(ascending=False).head(10)
                           .reset_index(name='Count'))
            fig_lang = px.pie(lang_dist, values='Count', names='mb_language', hole=0.5,
                              color_discrete_sequence=px.colors.sequential.Greens_r)
            fig_lang.update_layout(**DARK_LAYOUT)
            st.plotly_chart(fig_lang, use_container_width=True)
        else:
            st.info("No lyrics-language data yet. Re-run the MusicBrainz stage to populate it.")

    st.divider()

# 3c. TEMPO  (Deezer BPM — replaces Spotify's deprecated tempo feature)
if 'deezer_bpm' in df.columns and df['deezer_bpm'].notna().any():
    bpm = df[df['deezer_bpm'].between(40, 220)]
    if not bpm.empty:
        st.subheader("🥁 Tempo")
        st.caption(f"Beats-per-minute from Deezer for {len(bpm):,} tracks. "
                   f"Median tempo: **{bpm['deezer_bpm'].median():.0f} BPM**.")
        fig_bpm = px.histogram(bpm, x='deezer_bpm', nbins=36,
                               labels={'deezer_bpm': 'BPM'},
                               color_discrete_sequence=[SPOTIFY_GREEN])
        fig_bpm.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_bpm, use_container_width=True)
        st.divider()

# 3d. MAINSTREAM vs NICHE + CROWD STYLES  (Last.fm global stats + Discogs styles)
has_lastfm = 'lastfm_artist_listeners' in df.columns and df['lastfm_artist_listeners'].notna().any()
has_discogs = 'discogs_styles' in df.columns and df['discogs_styles'].fillna('').str.strip().ne('').any()

if has_lastfm or has_discogs:
    st.subheader("📡 Taste Position")
    col_main, col_style = st.columns(2)

    with col_main:
        if has_lastfm:
            st.markdown("**Mainstream ↔ Niche**")
            st.caption("Each artist by global Last.fm listeners. Lower = more obscure (deeper-cut taste).")
            lf = (df.dropna(subset=['lastfm_artist_listeners', 'primary_artist'])
                    .groupby('primary_artist')['lastfm_artist_listeners'].max()
                    .sort_values().head(12).reset_index())
            fig_lf = px.bar(lf, x='lastfm_artist_listeners', y='primary_artist', orientation='h',
                            labels={'lastfm_artist_listeners': 'Global listeners', 'primary_artist': ''},
                            title="Your 12 most niche artists",
                            color_discrete_sequence=[SPOTIFY_GREEN])
            fig_lf.update_layout(yaxis={'categoryorder': 'total descending'}, **DARK_LAYOUT)
            st.plotly_chart(fig_lf, use_container_width=True)

    with col_style:
        if has_discogs:
            st.markdown("**Discogs Styles**")
            st.caption("Finer-grained crowd styles than Spotify genres.")
            styles = Counter()
            for s in df['discogs_styles'].dropna():
                for tok in str(s).split(','):
                    tok = tok.strip()
                    if tok:
                        styles[tok] += 1
            if styles:
                sdf = pd.DataFrame(styles.most_common(12), columns=['Style', 'Count'])
                fig_sty = px.bar(sdf, x='Count', y='Style', orientation='h',
                                 color='Count', color_continuous_scale=['#1d3a2a', SPOTIFY_GREEN])
                fig_sty.update_layout(yaxis={'categoryorder': 'total ascending', 'title': None}, **DARK_LAYOUT)
                st.plotly_chart(fig_sty, use_container_width=True)
    st.divider()

# 4. TIME TRAVEL
st.subheader("📅 Release Era")
c_time1, c_time2 = st.columns(2)

with c_time1:
    df_year = df.groupby('release_year')['rating'].count().reset_index(name='Count')
    fig_year = px.area(df_year, x='release_year', y='Count', title="Tracks by Release Year",
                       color_discrete_sequence=[SPOTIFY_GREEN])
    fig_year.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig_year, use_container_width=True)

with c_time2:
    if 'popularity' in df.columns and df['popularity'].notna().any():
        fig_pop = px.histogram(df, x="popularity", nbins=20, title="Popularity Distribution (Spotify Score)",
                               color_discrete_sequence=[SPOTIFY_GREEN])
        fig_pop.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_pop, use_container_width=True)
    else:
        dec = df.groupby('decade')['rating'].count().reset_index(name='Count')
        fig_dec = px.bar(dec, x='decade', y='Count', title="Tracks by Decade",
                         color_discrete_sequence=[SPOTIFY_GREEN])
        fig_dec.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_dec, use_container_width=True)

# 5. DATA EXPLORER
with st.expander("📂 View Library Data"):
    explorer_cols = [c for c in ['name', 'artists', 'mb_artist_country', 'mb_language',
                                 'discogs_styles', 'lastfm_tags', 'lastfm_similar',
                                 'lastfm_artist_listeners', 'deezer_bpm', 'release_year', 'rating',
                                 'popularity', 'duration_min', 'lyric_word_count', 'lyric_sentiment']
                     if c in df.columns]
    st.dataframe(df[explorer_cols].sort_values('rating', ascending=False), use_container_width=True)
