"""
subtitle_features.py — per-movie style numbers and curves from subtitles.

ADDITIVE
    Features are cached per movie in data/processed/movies/subtitle_features.csv,
    keyed by the .srt file's size + modified time. Each run computes only movies
    whose file is new or changed (or was computed by an older FEATURE_VERSION).
    src/movies/subtitles.py calls update() after each download run, so the table
    grows as subtitles arrive. Run by hand any time:
        python -m src.movies.subtitle_features

FEATURES (one row per movie; runtime = OMDb runtime, else last subtitle time)
    talk        words_per_min, dialogue_time_share, words_per_line, longest_gap_min,
                gaps_over_2min
    language    vocab_richness (moving-window type/token ratio, 500 words),
                avg_word_len, question_share, exclaim_share, swear_per_1k,
                i_per_1k, you_per_1k
    mood        VADER sentiment per dialogue line: sent_mean, sent_std,
                neg_line_share, sent_start / sent_end (first / last 20% of story time),
                sent_shift (end - start), low_point_pos (0 = start, 1 = end)
    curves      mood_00..mood_49 (mean sentiment per 2% of story time: first to last line)
                and talk_00..talk_49 (words per minute per 2% of story time)
    music       lyric_share (share of subtitle blocks that are sung)
    sound (SDH) sound_cues_per_min, music_cue_share, laugh_cues_per_min,
                violence_cues_per_min, speakers_per_min — blank for movies whose
                file has no sound cues (plain subtitles), NOT zero
    checks      runtime_ratio (last subtitle time / runtime; far from 1 suggests
                the wrong film or a partial file), has_cues

DASHBOARD
    app/subtitle_analytics.py (a section of the Movies Dashboard) reads the table;
    correlations() and distinctive_words() below are its analysis helpers.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.movies import subtitles as subs

FEATURES_PATH = config.MOVIES_PROCESSED_DIR / "subtitle_features.csv"

_WORD = re.compile(r"[a-z']+")
_SWEAR = re.compile(r"\b(?:motherfuck\w*|fuck\w*|shit\w*|bullshit|bitch\w*|asshole\w*|"
                    r"bastard\w*|goddamn\w*|damn\w*|dick|dicks|piss\w*|cunt\w*|crap)\b")
_I = {"i", "i'm", "i've", "i'll", "i'd", "me", "my", "mine", "myself"}
_YOU = {"you", "you're", "you've", "you'll", "you'd", "your", "yours", "yourself"}
_MUSIC_CUE = re.compile(r"music|song|singing|sings|humming|hums|playing|plays|band|orchestra")
_LAUGH_CUE = re.compile(r"laugh|chuckl|giggl|snicker")
_VIOLENCE_CUE = re.compile(r"gun|shot|explo|scream|punch|stab|groan|grunt|crash|fight|blast")
_MIN_CUES = 20  # fewer sound-cue blocks than this = a plain (non-SDH) file
CURVE_BINS = 50  # curve resolution: one point per 2% of story time
# Bump when movie_features changes, so cached rows are recomputed.
FEATURE_VERSION = 4

_vader = None


def _sentiment(lines):
    global _vader
    _vader = _vader or SentimentIntensityAnalyzer()
    return np.array([_vader.polarity_scores(t)["compound"] for t in lines])


def _mattr(tokens, window=500):
    """Moving-average type/token ratio: vocabulary richness that does not fall
    just because a film is longer (plain unique/total does)."""
    if len(tokens) < window:
        return len(set(tokens)) / len(tokens) if tokens else np.nan
    step = window // 2
    return float(np.mean([len(set(tokens[i:i + window])) / window
                          for i in range(0, len(tokens) - window + 1, step)]))


def tokens(dialogue):
    return _WORD.findall(" ".join(dialogue).lower().replace("’", "'"))


def movie_features(d, runtime_min):
    """d = parse_srt output for one movie."""
    last_min = d["end_s"].max() / 60
    rt = runtime_min if runtime_min and runtime_min > 0 else last_min
    talk = d[d["dialogue"] != ""]
    toks = tokens(talk["dialogue"])
    n_words = len(toks)
    per_1k = lambda n: 1000 * n / n_words if n_words else np.nan

    f = {
        "runtime_min": rt, "runtime_ratio": last_min / rt, "n_blocks": len(d),
        "words_per_min": n_words / rt,
        "dialogue_time_share": (talk["end_s"] - talk["start_s"]).clip(lower=0).sum() / (rt * 60),
        "words_per_line": n_words / max(len(talk), 1),
    }
    gaps = (d["start_s"].shift(-1) - d["end_s"]).dropna() / 60
    f["longest_gap_min"] = gaps.max() if len(gaps) else np.nan
    f["gaps_over_2min"] = int((gaps > 2).sum())

    f["vocab_richness"] = _mattr(toks)
    f["avg_word_len"] = np.mean([len(t) for t in toks]) if toks else np.nan
    f["question_share"] = talk["dialogue"].str.contains(r"\?").mean()
    f["exclaim_share"] = talk["dialogue"].str.contains("!").mean()
    f["swear_per_1k"] = per_1k(sum(1 for t in toks if _SWEAR.fullmatch(t)))
    f["i_per_1k"] = per_1k(sum(t in _I for t in toks))
    f["you_per_1k"] = per_1k(sum(t in _YOU for t in toks))

    s = _sentiment(talk["dialogue"].tolist())
    f["sent_mean"] = s.mean() if len(s) else np.nan
    f["sent_std"] = s.std() if len(s) else np.nan
    f["neg_line_share"] = (s < -0.05).mean() if len(s) else np.nan
    # Curves, one point per 2% of STORY time: first spoken line = 0%, last = 100%.
    # Official runtime would put opening titles and end credits (no dialogue) in
    # the first/last few points and leave them blank for most films.
    # mood = mean sentiment of the lines in that slice (blank if nobody talks);
    # talk = words per minute in that slice.
    t = talk["start_s"].to_numpy()
    span_min = max((t.max() - t.min()) / 60, 1e-6) if len(t) else 1.0
    story = (t - t.min()) / (span_min * 60) if len(t) else t
    line_words = np.array([len(tokens([x])) for x in talk["dialogue"]])
    bin_of = np.clip((story * CURVE_BINS).astype(int), 0, CURVE_BINS - 1)
    for k in range(CURVE_BINS):
        in_bin = bin_of == k
        f[f"mood_{k:02d}"] = s[in_bin].mean() if in_bin.any() else np.nan
        f[f"talk_{k:02d}"] = line_words[in_bin].sum() / (span_min / CURVE_BINS)
    f["sent_start"] = s[story < 0.2].mean() if (story < 0.2).any() else np.nan
    f["sent_end"] = s[story >= 0.8].mean() if (story >= 0.8).any() else np.nan
    f["sent_shift"] = f["sent_end"] - f["sent_start"]
    # low point on a 10%-smoothed curve, so one grim line cannot set it
    mood = pd.Series([f[f"mood_{k:02d}"] for k in range(CURVE_BINS)], dtype=float)
    smooth = mood.rolling(5, center=True, min_periods=2).mean()
    f["low_point_pos"] = smooth.idxmin() / (CURVE_BINS - 1) if smooth.notna().any() else np.nan

    f["lyric_share"] = (d["lyrics"] != "").mean()

    cues = d.loc[d["sounds"] != "", "sounds"].str.lower()
    f["has_cues"] = len(cues) >= _MIN_CUES
    if f["has_cues"]:
        all_cues = cues.str.split("|").explode()
        f["sound_cues_per_min"] = len(all_cues) / rt
        f["music_cue_share"] = all_cues.str.contains(_MUSIC_CUE).mean()
        f["laugh_cues_per_min"] = all_cues.str.contains(_LAUGH_CUE).sum() / rt
        f["violence_cues_per_min"] = all_cues.str.contains(_VIOLENCE_CUE).sum() / rt
        f["speakers_per_min"] = (d["speakers"] != "").sum() / rt
    else:
        for k in ("sound_cues_per_min", "music_cue_share", "laugh_cues_per_min",
                  "violence_cues_per_min", "speakers_per_min"):
            f[k] = np.nan
    return f


# --- cache ---

def _movies():
    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH, dtype={"imdb_id": str})
    df["runtime_num"] = pd.to_numeric(df["runtime"].astype(str).str.extract(r"(\d+)")[0],
                                      errors="coerce")
    # index by the same movie key the subtitle files use (IMDb id, else tmdb<id>)
    df["imdb_id"] = subs.movie_keys(df)
    return df.dropna(subset=["imdb_id"]).drop_duplicates("imdb_id").set_index("imdb_id")


def update(verbose=True):
    """Compute features for new/changed subtitle files."""
    movies = _movies()
    cache = pd.read_csv(FEATURES_PATH, dtype={"imdb_id": str}) if FEATURES_PATH.exists() \
        else pd.DataFrame(columns=["imdb_id", "file_sig"])
    have = dict(zip(cache["imdb_id"], cache["file_sig"]))

    files = {p.stem: p for p in subs.SUBTITLES_DIR.glob("*.srt")}
    new_rows = []
    for imdb_id, p in files.items():
        st = p.stat()
        sig = f"v{FEATURE_VERSION}-{st.st_size}-{int(st.st_mtime)}"
        if have.get(imdb_id) == sig:
            continue
        d = subs.load_subtitles(imdb_id)
        rt = movies["runtime_num"].get(imdb_id) if imdb_id in movies.index else None
        row = movie_features(d, rt if pd.notna(rt) else None)
        row.update(imdb_id=imdb_id, file_sig=sig)
        new_rows.append(row)

    keep = cache[cache["imdb_id"].isin(files) &
                 ~cache["imdb_id"].isin([r["imdb_id"] for r in new_rows])]
    out = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.drop(columns=[c for c in out.columns if c.startswith("arc_")])  # v1 curve
    cols = ["imdb_id", "file_sig"] + [c for c in out.columns if c not in ("imdb_id", "file_sig")]
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(FEATURES_PATH, index=False)
    if verbose:
        print(f"Subtitle features: {len(new_rows)} new/changed, {len(out)} movies total "
              f"-> {FEATURES_PATH.relative_to(config.BASE_DIR)}", flush=True)
    return out


# --- analysis helpers (used by app/subtitle_analytics.py) ---

LABELS = {
    "words_per_min": "Words per minute", "dialogue_time_share": "Share of runtime with dialogue",
    "words_per_line": "Words per line", "longest_gap_min": "Longest silence (min)",
    "gaps_over_2min": "Silences over 2 min", "vocab_richness": "Vocabulary richness",
    "avg_word_len": "Average word length", "question_share": "Lines with a question",
    "exclaim_share": "Lines with an exclamation", "swear_per_1k": "Swears per 1k words",
    "i_per_1k": "'I/me/my' per 1k words", "you_per_1k": "'You/your' per 1k words",
    "sent_mean": "Mood, average", "sent_std": "Mood swings (std)",
    "neg_line_share": "Negative lines share", "sent_start": "Mood, first 20%",
    "sent_end": "Mood, last 20%", "sent_shift": "Mood shift, end minus start",
    "low_point_pos": "Where the low point falls", "lyric_share": "Sung-lyric share",
    "sound_cues_per_min": "Sound cues per min (SDH)", "music_cue_share": "Music share of cues (SDH)",
    "laugh_cues_per_min": "Laugh cues per min (SDH)",
    "violence_cues_per_min": "Violence cues per min (SDH)",
    "speakers_per_min": "Speaker labels per min (SDH)",
}


def _bh(p):
    """Benjamini-Hochberg q-values: p-values corrected for testing ~25 features."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / (np.arange(len(p)) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def correlations(df):
    """Spearman r of each feature with your rating, raw and beyond crowd.
    'Beyond crowd' = correlation with the part of your rating the IMDb score does
    not explain (rank residual), i.e. what the model could not already get."""
    rows = []
    for feat in LABELS:
        sub = df[[feat, "user_rating", "imdb_rating"]].dropna()
        if len(sub) < 30:
            continue
        r, p = stats.spearmanr(sub[feat], sub["user_rating"])
        ry, rc = stats.rankdata(sub["user_rating"]), stats.rankdata(sub["imdb_rating"])
        resid = ry - np.polyval(np.polyfit(rc, ry, 1), rc)
        r2, p2 = stats.spearmanr(sub[feat], resid)
        rows.append({"feature": feat, "label": LABELS[feat], "n": len(sub),
                     "r_raw": r, "p_raw": p, "r_beyond_crowd": r2, "p_beyond_crowd": p2})
    out = pd.DataFrame(rows)
    if len(out):
        out["q_raw"] = _bh(out["p_raw"])
        out["q_beyond_crowd"] = _bh(out["p_beyond_crowd"])
    return out.sort_values("r_raw", key=np.abs, ascending=False)


def distinctive_words(ids_a, ids_b, top=20, min_films=8):
    """Words used by a larger share of films in group A than group B, and vice
    versa. Each word counts once per film, so one talky film cannot dominate.
    Score = smoothed log-odds ratio / its standard error (a z-score).
    Returns (a_words, b_words, chance_z): chance_z is the largest |z| that pure
    noise typically reaches across this many words; scores below it are not
    evidence."""
    def film_counts(ids):
        c = {}
        for i in ids:
            d = subs.load_subtitles(i)
            if d is None:
                continue
            for t in set(tokens(d.loc[d["dialogue"] != "", "dialogue"])):
                c[t] = c.get(t, 0) + 1
        return pd.Series(c, dtype=float)
    a, b = film_counts(ids_a), film_counts(ids_b)
    vocab = a.add(b, fill_value=0)
    vocab = vocab[vocab >= min_films].index
    a, b = a.reindex(vocab, fill_value=0), b.reindex(vocab, fill_value=0)
    na, nb = len(ids_a), len(ids_b)
    lor = np.log((a + 1) / (na - a + 1)) - np.log((b + 1) / (nb - b + 1))
    z = lor / np.sqrt(1 / (a + 1) + 1 / (na - a + 1) + 1 / (b + 1) + 1 / (nb - b + 1))
    chance = float(np.sqrt(2 * np.log(max(len(z), 2))))
    return z.nlargest(top), z.nsmallest(top), chance


if __name__ == "__main__":
    update()
