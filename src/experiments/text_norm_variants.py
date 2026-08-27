"""
text_norm_variants.py — is first-40-words the right way to normalize the shared text?

The unified builder equalizes cross-domain text by truncating every description to
its first 40 words. That is a POSITIONAL filter: it caps length but keeps whatever
the domain happens to put first. RAWG game blurbs open with franchise/platform
boilerplate, Hardcover blurbs open with review copy — so the prefix is exactly where
each domain's register lives.

This compares that scheme against alternatives on the two things the shared space is
supposed to deliver:

  1. DOMAIN INVARIANCE — mean pairwise accuracy of a logistic probe predicting
     media_type from the pca_ block. 0.50 = domains indistinguishable (the goal);
     1.00 = the "shared" space is really a domain label.
  2. TASTE SIGNAL — 5-fold CV skill predicting the rating from pca_ alone, per domain,
     plus a strict CROSS-DOMAIN transfer test (fit on movies, score books/games).
     A variant that wins on (1) by destroying (2) is useless.

Run:  python -m src.experiments.text_norm_variants
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_utils import DomainAligner

SEED = 42
# Averaged over these seeds: one 5-fold split on 72 games is far too noisy to
# rank variants on. Embeddings are computed once per variant; only the PCA fit and
# the CV splits are repeated, so this costs almost nothing.
SEEDS = (42, 7, 13, 2024, 99)
N_WORDS = 40
PCA_K = 10

# Register words that name the MEDIUM rather than the content. These are what a probe
# actually latches onto — "gameplay", "readers", "episode" — and they are content
# words, so stopword removal leaves every one of them untouched.
DOMAIN_MARKERS = {
    'game', 'games', 'gameplay', 'player', 'players', 'playable', 'console', 'pc',
    'playstation', 'xbox', 'nintendo', 'steam', 'multiplayer', 'singleplayer', 'dlc',
    'level', 'levels', 'franchise', 'installment', 'sequel', 'remaster', 'edition',
    'film', 'films', 'movie', 'movies', 'cinema', 'director', 'directed', 'starring',
    'season', 'seasons', 'episode', 'episodes', 'series', 'sitcom', 'showrunner',
    'book', 'books', 'novel', 'novels', 'reader', 'readers', 'author', 'chapter',
    'bestseller', 'bestselling', 'paperback', 'hardcover', 'publisher', 'published',
    'story', 'stories', 'tale', 'saga',
}

DOMAINS = {
    'movie': (config.MOVIES_ENRICHED_DATA_PATH, 'title', 'genre', 'plot', 'user_rating'),
    'tv': (config.TV_SHOWS_ENRICHED_DATA_PATH, 'name', 'genres', 'overview', 'user_rating'),
    'game': (config.GAMES_ENRICHED_DATA_PATH, 'name', 'genres', 'description_raw', 'my_rating'),
    'book': (config.BOOKS_ENRICHED_DATA_PATH, 'title', 'categories', 'description', 'my_rating'),
}


def _load():
    frames = []
    for mt, (path, tcol, gcol, dcol, rcol) in DOMAINS.items():
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding='latin1')
        df = df[df[rcol].notna()].copy()
        frames.append(pd.DataFrame({
            'media_type': mt,
            'title': df[tcol].fillna('').astype(str),
            'genre': df[gcol].fillna('').astype(str).str.replace(r"[\[\]']", '', regex=True),
            'desc': df[dcol].fillna('').astype(str),
            'y': pd.to_numeric(df[rcol], errors='coerce'),
        }))
    out = pd.concat(frames, ignore_index=True)
    return out[out['y'].notna()].reset_index(drop=True)


def _words(s):
    return re.findall(r"[A-Za-z']+", str(s))


# --------------------------- the variants ---------------------------------- #

def v_prefix40(df):
    """CURRENT SCHEME — first 40 words of the description."""
    return [" ".join(_words(d)[:N_WORDS]) for d in df['desc']]


def v_full(df):
    """No truncation — the control that shows what the 40-word cap actually buys."""
    return [str(d) for d in df['desc']]


def v_stopword(df):
    """The proposal: drop stopwords, keep the first 40 CONTENT words."""
    out = []
    for d in df['desc']:
        ws = [w for w in _words(d) if w.lower() not in ENGLISH_STOP_WORDS]
        out.append(" ".join(ws[:N_WORDS]))
    return out


def v_domain_scrub(df):
    """First 40 words with medium-naming words removed, sentences left intact."""
    out = []
    for d in df['desc']:
        ws = [w for w in _words(d) if w.lower() not in DOMAIN_MARKERS]
        out.append(" ".join(ws[:N_WORDS]))
    return out


def v_tfidf_terms(df):
    """Topic/theme extraction: the most distinctive terms, per item, as a bag.

    TF-IDF is fitted ACROSS domains, so a term that is merely common inside one
    medium is down-weighted automatically — which is the effect the stopword idea
    was reaching for, but aimed at the words that actually mark a domain.
    """
    vec = TfidfVectorizer(stop_words='english', max_features=20000,
                          ngram_range=(1, 2), sublinear_tf=True)
    M = vec.fit_transform(df['desc'].tolist())
    vocab = np.array(vec.get_feature_names_out())
    out = []
    for i in range(M.shape[0]):
        row = M.getrow(i)
        if row.nnz == 0:
            out.append("")
            continue
        k = min(N_WORDS, row.nnz)
        top = row.indices[np.argsort(-row.data)[:k]]
        out.append(" ".join(vocab[top]))
    return out


VARIANTS = {
    'prefix40 (current)': v_prefix40,
    'full (no truncation)': v_full,
    'stopwords removed': v_stopword,
    'domain-word scrub': v_domain_scrub,
    'tfidf theme terms': v_tfidf_terms,
}


# --------------------------------------------------------------------------- #
# Chunk-and-pool: read the WHOLE description, whatever its length, and still end
# up with one comparable vector.
#
# Truncation exists because MiniLM only reads 256 tokens. Chunking removes that
# constraint instead of paying for it: split the text on sentence boundaries into
# windows that fit, embed each window, then average the (unit-length) window
# vectors and re-normalize. Because every window vector is unit length, the mean
# of 1 window and the mean of 8 windows are both unit vectors in the same space —
# length-invariance comes from the geometry, not from cutting text off.
#
# Each window is prefixed with the title/genre header so it stands on its own and
# the pooled vector cannot drift off-item on a long blurb.
# --------------------------------------------------------------------------- #

def _sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', str(text).strip())
    return [p for p in parts if p]


def _windows(text, tok, max_tokens):
    """Greedily pack sentences into <=max_tokens windows."""
    out, cur, cur_n = [], [], 0
    for s in _sentences(text):
        n = len(tok.encode(s, add_special_tokens=False))
        if n > max_tokens:                      # one runaway sentence: hard-split
            w = s.split()
            step = max(1, int(max_tokens * 0.6))
            for i in range(0, len(w), step):
                out.append(" ".join(w[i:i + step]))
            continue
        if cur and cur_n + n > max_tokens:
            out.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return out or [""]


def embed_chunk_pooled(df, model, max_tokens=180, pool='mean', salience_temp=8.0):
    """Return one pooled, L2-normalized vector per item, reading the full text.

    pool='salience' additionally down-weights boilerplate. Marketing copy, release
    blurbs and review furniture are the parts every item in a domain shares, so they
    sit close to the corpus centroid; the windows that carry what an item is ABOUT
    sit further out. Weighting each window by its distance from the centroid keeps
    the whole text in play while stopping a long promotional blurb from washing out
    the two sentences that describe the actual story.
    """
    tok = model.tokenizer
    header = [f"Title: {t[:120]}. Genre: {g[:80]}." for t, g in zip(df['title'], df['genre'])]
    flat, owner = [], []
    for i, (h, d) in enumerate(zip(header, df['desc'])):
        for w in _windows(d, tok, max_tokens):
            flat.append(f"{h} {w}".strip())
            owner.append(i)
    E = model.encode(flat, normalize_embeddings=True, batch_size=64,
                     show_progress_bar=False)
    owner = np.asarray(owner)

    centroid = E.mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)
    novelty = 1.0 - (E @ centroid)          # high = distinctive, low = boilerplate

    out = np.zeros((len(df), E.shape[1]), dtype=np.float32)
    for i in range(len(df)):
        m = owner == i
        vs = E[m]
        if pool == 'max':
            out[i] = vs.max(axis=0)
        elif pool == 'salience' and len(vs) > 1:
            w = np.exp(salience_temp * (novelty[m] - novelty[m].max()))
            out[i] = (vs * w[:, None]).sum(axis=0) / w.sum()
        else:
            out[i] = vs.mean(axis=0)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9), owner


# Variants that produce embeddings directly rather than a text string.
EMBED_VARIANTS = {
    'chunk-pool mean (full text)': lambda df, m: embed_chunk_pooled(df, m, pool='mean')[0],
    'chunk-pool max (full text)': lambda df, m: embed_chunk_pooled(df, m, pool='max')[0],
}


# --------------------------- the measurements ------------------------------ #

def domain_separability(X, mt, seed=SEED):
    """Mean pairwise BALANCED probe accuracy. 0.5 = domains indistinguishable.

    Plain accuracy is useless here: the library is 1,062 movies against 72 games, so
    a probe that always answers "movie" already scores 0.94 on that pair and the
    metric reports near-perfect separability for a space carrying no domain signal
    at all. Balanced accuracy with a balanced class weight removes the prior, so the
    number reflects the geometry instead of the row counts.
    """
    scores = []
    names = sorted(set(mt))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = np.isin(mt, [a, b])
            if m.sum() < 30:
                continue
            s = cross_val_score(
                LogisticRegression(max_iter=3000, class_weight='balanced'),
                X[m], mt[m], scoring='balanced_accuracy',
                cv=StratifiedKFold(5, shuffle=True, random_state=seed)).mean()
            scores.append(s)
    return float(np.mean(scores))


def taste_skill(X, y, seed=SEED):
    """1 - MAE/MAE_mean for a ridge on the text block alone. >0 means text helps."""
    pred = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = Ridge(alpha=1.0).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    mae = mean_absolute_error(y, np.clip(pred, 0.5, 5.0))
    base = mean_absolute_error(y, np.full_like(y, y.mean()))
    return 1 - mae / base


def transfer_skill(X, y, mt, src='movie', dst=('book', 'game')):
    """Fit on the source domain only, score the target. The point of a shared space."""
    out = {}
    s = mt == src
    model = Ridge(alpha=1.0).fit(X[s], y[s])
    for d in dst:
        t = mt == d
        if t.sum() < 10:
            continue
        p = np.clip(model.predict(X[t]), 0.5, 5.0)
        base = mean_absolute_error(y[t], np.full(t.sum(), y[s].mean()))
        out[d] = 1 - mean_absolute_error(y[t], p) / base
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:  # noqa: BLE001
        pass

    df = _load()
    mt = df['media_type'].values
    y = df['y'].astype(float).values
    print(f"Averaging over {len(SEEDS)} seeds: {SEEDS}")
    print(f"Loaded {len(df)} rated items: "
          + ", ".join(f"{k}={int((mt == k).sum())}" for k in DOMAINS))

    print("\ndescription length (words) by domain:")
    wl = df['desc'].apply(lambda s: len(_words(s)))
    for k in DOMAINS:
        w = wl[mt == k]
        print(f"   {k:6} median={w.median():6.0f}  mean={w.mean():6.0f}  "
              f"{(w > N_WORDS).mean() * 100:5.1f}% hit the {N_WORDS}-word cap")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    def _emb_for(name):
        if name in EMBED_VARIANTS:
            return EMBED_VARIANTS[name](df, model)
        body = VARIANTS[name](df)
        txt = [f"Title: {t[:120]}. Genre: {g[:80]}. {b}"
               for t, g, b in zip(df['title'], df['genre'], body)]
        return model.encode(txt, normalize_embeddings=True, batch_size=64,
                            show_progress_bar=False)

    rows = []
    for name in list(VARIANTS) + list(EMBED_VARIANTS):
        emb = _emb_for(name)          # embed ONCE, then re-score across seeds
        acc = []
        for seed in SEEDS:
            X = PCA(n_components=PCA_K, random_state=seed).fit_transform(emb)

            # Measure with the SAME centroid aligner the unified trainer applies, so
            # this reflects the space the model actually sees.
            al = DomainAligner(method='centroid')
            al.fit(X, mt)
            Xa = al.transform(X, mt)

            tr = transfer_skill(Xa, y, mt)
            acc.append({
                # Before and after the centroid aligner, so it is clear how much of
                # the invariance comes from the text scheme and how much from the aligner.
                'sep_raw': domain_separability(X, mt, seed),
                'sep_aligned': domain_separability(Xa, mt, seed),
                'skill_book': taste_skill(Xa[mt == 'book'], y[mt == 'book'], seed),
                'skill_game': taste_skill(Xa[mt == 'game'], y[mt == 'game'], seed),
                'skill_movie': taste_skill(Xa[mt == 'movie'], y[mt == 'movie'], seed),
                'xfer_mv->book': tr.get('book', float('nan')),
                'xfer_mv->game': tr.get('game', float('nan')),
            })
        row = {'variant': name}
        for k in acc[0]:
            vals = [a[k] for a in acc]
            row[k] = round(float(np.mean(vals)), 3)
            # Spread across seeds, so a difference smaller than this is not a result.
            row[k + '_sd'] = round(float(np.std(vals)), 3)
        rows.append(row)
        print(f"   ✓ {name}")

    res = pd.DataFrame(rows)
    print("\n" + "=" * 96)
    print("domain_sep: 0.50 = domains indistinguishable (GOOD), 1.00 = pure domain label (BAD)")
    print("skill_*   : 1 - MAE/MAE_mean from the text block alone; higher is better, <0 is harmful")
    print("=" * 96)
    print(res.to_string(index=False))

    out = Path(config.BASE_DIR) / "reports" / "text_norm_variants.csv"
    res.to_csv(out, index=False)
    print(f"\n saved -> {out}")


if __name__ == "__main__":
    main()
