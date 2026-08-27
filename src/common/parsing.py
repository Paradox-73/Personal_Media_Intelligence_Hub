"""Canonical parsers shared by every domain's feature, serve and app code.

Why this module exists
----------------------
An audit on 2026-08-16 found these helpers copy-pasted across the repo, and the
copies had DRIFTED:

    parse_list        11 copies,  6 distinct implementations
    clean_year         9 copies,  7 distinct implementations
    sanitize_col       9 copies,  identical
    clean_money        5 copies,  identical
    categorize_rating  5 copies,  2 distinct implementations
    is_effectively_empty 3 copies, 3 distinct implementations

`parse_list` is not a cosmetic duplicate: it decides the KEYS used for target
encoding, and the CSVs store the same column in two shapes ("['A', 'B']" and
"A, B") depending on which script wrote the row. A train copy and a serve copy that
disagree produce keys that never match, which silently flattens the encoding to the
global mean. That exact failure has hit this repo twice (shows creator/actor
encodings, and the movie target-encoding train/serve skew).

Only the provably-identical helpers were migrated wholesale. The divergent ones
(clean_year especially, whose variants parse genuinely different source formats -
RAWG "2011-05-15", Hardcover publishedDate, Letterboxd year) are left in place
deliberately; unifying them blind would change behaviour. They are listed here so
the next person knows the debt exists and can retire them one at a time behind a
measurement.
"""
import ast
import re

import numpy as np
import pandas as pd


def parse_list(x):
    """Parse a multi-value cell into a list of strings.

    Handles both storage shapes the CSVs actually contain:
      "['Denis Villeneuve', 'Jon Spaihts']"  (list repr, written by ingestion.py)
      "Denis Villeneuve, Jon Spaihts"        (comma string, written by fill_missing.py)

    Never raises: a cell it cannot parse comes back as a single-element list, which
    is the behaviour every caller already relied on.
    """
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    try:
        if pd.isna(x):
            return []
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    if not s or s.upper() in ("N/A", "NAN", "NONE", "[]"):
        return []
    try:
        if s.startswith("[") and s.endswith("]"):
            return [str(i).strip() for i in ast.literal_eval(s)]
        return [item.strip() for item in s.split(",") if item.strip()]
    except (ValueError, SyntaxError):
        return [s]


# TMDB and OMDb/IMDb name the same genre differently, and a row carries whichever
# source answered first (ingestion falls back to OMDb when TMDB has no genres), so
# ~9% of movies store the IMDb spelling. Left unmerged, "Sci-Fi" and "Science
# Fiction" split one genre into two everywhere: two bars in the dashboard, two
# boxes in the spread plot, and two near-orthogonal gen_* columns in the model.
# TMDB's vocabulary is canonical here - it covers the other ~91% of rows, and the
# trained feature names already follow it.
GENRE_SYNONYMS = {
    "sci-fi": "Science Fiction",
    "sci fi": "Science Fiction",
    "scifi": "Science Fiction",
    "science-fiction": "Science Fiction",
}
# Note: TV's "Sci-Fi & Fantasy" / "Action & Adventure" are NOT here. They expand to
# two genres each, which is a different operation; that mapping stays in
# unified_feature_engineering.py where the expansion already happens.


def canon_genre(g):
    """Map one genre label onto the canonical (TMDB) spelling."""
    s = str(g).strip()
    return GENRE_SYNONYMS.get(s.lower(), s)


def canon_genres(x):
    """parse_list for a genre cell, with synonyms merged and duplicates dropped.

    Order is preserved because the first element becomes `primary_genre`, which is
    half of the director x genre target-encoding key.
    """
    out = []
    for g in parse_list(x):
        c = canon_genre(g)
        if c and c not in out:
            out.append(c)
    return out


def people_key(x):
    """Canonical display string for a multi-value cell, independent of storage shape.

    Anything that becomes a dictionary key or embedding text must go through here,
    or the same item keys differently at training and inference time.
    """
    return ", ".join(str(i).strip() for i in parse_list(x))


def sanitize_col(col_name):
    """Strip characters XGBoost/LightGBM reject in feature names."""
    return re.sub(r"[\[\]<']", "", str(col_name))


def clean_money(x):
    """Parse a currency cell ("$1,234,567") to float; 0 when absent or unparseable."""
    if pd.isna(x):
        return 0
    if isinstance(x, (int, float)):
        return x
    x = str(x).replace("$", "").replace(",", "").strip()
    try:
        return float(x)
    except ValueError:
        return 0


def categorize_rating(r):
    """Map an MPAA / TV / regional content rating onto {Adult, Teen, General}.

    THE CANONICAL IMPLEMENTATION. Three copies of this had drifted apart, and the
    naive `if "R" in r` test that two of them used is wrong on real data because
    plenty of non-adult certificates contain the letter R:

        "Not Rated" (123 films)  "Unrated" (5)  "Approved" (4)   -> all read as Adult

    That mislabelled 132 of the 453 films in `rated_Adult` -- 12.4% of the movie
    library -- and "no certificate on file" is emphatically not "adult". Books carry
    no certificate at all, so the single-item path's "NR" default sent every book to
    Adult at inference while the batch builder called it General.

    Matching is on whole tokens, not substrings, and the unknown set is explicit.
    Numeric certificates ("15", "16+", "U/A 13+", "MA 15+") are read as an age.
    """
    r = str(r).upper().strip()
    if r in ('', 'NAN', 'NONE', 'N/A', 'NR', 'NOT RATED', 'NOT-RATED', 'UNRATED',
             'APPROVED', 'PASSED', 'UNKNOWN'):
        return 'General'

    # Explicit adult certificates.
    if any(t in r for t in ('NC-17', 'TV-MA', 'MATURE', 'ADULT', 'X-RATED')):
        return 'Adult'
    # Teen certificates, checked BEFORE the bare R so "PG-13" never falls through.
    if any(t in r for t in ('PG-13', 'PG13', 'TV-14', 'TV-PG', 'PG', 'TEEN')):
        return 'Teen'
    # Young/general certificates.
    if any(t in r for t in ('TV-Y', 'TV-G', 'TV-Y7')):
        return 'General'

    # Numeric age certificates: 17+ is adult, 13-16 teen, below that general.
    ages = [int(m) for m in re.findall(r'\d{1,2}', r)]
    if ages:
        a = max(ages)
        return 'Adult' if a >= 17 else ('Teen' if a >= 13 else 'General')

    # A bare "R" as its own token (never a substring of APPROVED / NOT RATED).
    if re.fullmatch(r'R|R\s*/.*|RATED\s*R', r):
        return 'Adult'
    if r in ('G', 'U', 'E', 'TV-Y', 'ALL'):
        return 'General'
    return 'General'

def round_half(x, lo=0.5, hi=5.0):
    """Clip to the valid star range and snap to the 0.5 grid.

    The floor is 0.5, not 0: there is no 0-star rating on Letterboxd, so clipping to
    0 lets a low prediction round to an impossible 0.0.
    """
    return np.round(np.clip(np.asarray(x, dtype=float), lo, hi) * 2) / 2


# TMDB returns an ISO-639-1 code, OMDb returns an English name, and the movie
# `language` column is populated from whichever is present. Left alone that makes
# 'en' and 'English' two orthogonal one-hot features for the same language, and
# splits Japanese 25/6 so it never reaches the top-3 cut. Fold codes onto names.
_LANG_CODES = {
    'en': 'English', 'hi': 'Hindi', 'ja': 'Japanese', 'ko': 'Korean',
    'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian',
    'zh': 'Chinese', 'cn': 'Chinese', 'ru': 'Russian', 'pt': 'Portuguese',
    'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam', 'kn': 'Kannada',
    'mr': 'Marathi', 'bn': 'Bengali', 'pa': 'Punjabi', 'ur': 'Urdu',
    'sv': 'Swedish', 'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish',
    'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'th': 'Thai',
    'ar': 'Arabic', 'he': 'Hebrew', 'fa': 'Persian', 'id': 'Indonesian',
    'vi': 'Vietnamese', 'cs': 'Czech', 'el': 'Greek', 'hu': 'Hungarian',
    'ro': 'Romanian', 'uk': 'Ukrainian', 'la': 'Latin',
}


def norm_language(val):
    """'en' -> 'English'; 'en, ja' -> 'English, Japanese'. Unknown values pass through
    title-cased so a name that is already a name is not mangled."""
    if val is None or (isinstance(val, float) and val != val):
        return ''
    parts = [p.strip() for p in str(val).split(',') if p.strip()]
    out = []
    for p in parts:
        low = p.lower()
        name = _LANG_CODES.get(low, p if len(p) > 3 else p.upper())
        if name not in out:
            out.append(name)
    return ', '.join(out)


_RESIDUE_A = re.compile('Â(?=[^0-9A-Za-zÀ-ɏ])')
_RESIDUE_BULLET = re.compile('â—Â?')


def fix_mojibake(val):
    """Repair double-encoded UTF-8 ("God of War: RagnarÃƒÂ¶k" -> "God of War: Ragnarök").

    The games pipeline reads its enriched CSV with a bare `except: -> encoding='latin1'`
    fallback and writes it back as UTF-8, so every run that hits the fallback re-encodes
    the text one level deeper. 40 of 72 games carry it, 579 corrupt sequences in
    `description_raw` alone -- and description_raw IS the entire embedding body for
    games, so this is corrupting the text the model reads, not just display strings.

    Applied at the text-building functions rather than only at load, so the embeddings
    are clean no matter which reader produced the frame.
    """
    if val is None or (isinstance(val, float) and val != val):
        return val
    text = str(val)
    try:
        import ftfy
    except ImportError:
        return text
    text = ftfy.fix_text(text)
    # ftfy repairs 41 of the 44 corrupt cells; the stragglers are a stray U+00C2
    # left in front of punctuation ("manÃ's") and a mangled bullet glyph.
    # Strip only those two shapes -- anything more aggressive would eat real
    # accented characters, and a Cyrillic tag mangled past recovery is left alone
    # rather than guessed at.
    text = _RESIDUE_A.sub('', text)
    text = _RESIDUE_BULLET.sub('•', text)
    return text


def read_media_csv(path, **kw):
    """Read a media CSV as UTF-8; on failure fall back to latin-1 AND REPAIR.

    The bare `except: -> encoding='latin1'` pattern this replaces is what created the
    games mojibake. Reading UTF-8 bytes as latin-1 turns 'ö' into 'Ã¶'; the frame is
    then written back as UTF-8, so the next run reads 'ÃƒÂ¶', and so on -- the
    corruption compounds once per run and is never repaired, because enrichment only
    refills cells that are EMPTY, and a mojibaked cell is not empty.

    Falling back is still the right behaviour (better than crashing on one bad byte),
    but the fallback must undo its own damage on the way in.
    """
    import pandas as _pd
    try:
        return _pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        df = _pd.read_csv(path, encoding="latin1", **kw)
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].map(lambda v: fix_mojibake(v) if isinstance(v, str) else v)
        print(f"   NOTE: {getattr(path, 'name', path)} was not valid UTF-8; "
              f"read as latin-1 and mojibake-repaired on load.")
        return df
