"""
text_embedding.py — one description in, one comparable vector out, nothing dropped.

THE PROBLEM
    MiniLM reads at most 256 tokens. Descriptions do not respect that: movie plots
    run ~28 words, game blurbs ~168, book blurbs ~104. The pipeline handled this by
    truncating every description to its first 40 words. That is a positional cut,
    so it threw away 77% of the average game description and 62% of the average book
    description — and it kept whichever part the source happens to put first, which
    is exactly where each source's boilerplate lives.

THE FIX
    Split the description on sentence boundaries into windows that fit the model,
    embed every window, then average the window vectors and re-normalize. The whole
    text is read. And because each window vector is unit length, the mean of one
    window and the mean of eight are both unit vectors in the same space — items stay
    comparable across domains by construction rather than by cutting them to a
    common length.

    Each window carries the title/genre header so it is self-contained and a long
    blurb cannot drift off-item.

MEASURED — 5 seeds, mean +- sd, text block alone (1 - MAE/MAE_mean); see
src/experiments/text_norm_variants.py and reports/text_norm_variants.csv:

    variant                       book_skill     movie->book    movie->game
    prefix40 (the old cut)       0.045 +-.006   0.060 +-.001   0.002 +-.001
    stopwords removed            0.057 +-.007   0.035 +-.002   0.005 +-.000
    tfidf theme terms            0.035 +-.004   0.012 +-.001  -0.007 +-.001
    chunk-pool mean (this)       0.098 +-.008   0.070 +-.002   0.007 +-.001

    Books more than double and BOTH cross-domain transfers improve. Games move
    0.043 -> 0.035 with a +-0.012 seed spread, i.e. no measurable change.

    Keyword extraction was tried and is worse, not better: stripping a description
    to its most distinctive terms destroys the word order a sentence transformer
    encodes, and cross-domain transfer collapses (movie->game goes NEGATIVE).
    Reading the whole text is what makes domains comparable, not compressing it.

    chunk-pool max scores marginally higher on books alone (0.110) but loses
    movie->game transfer, so the plain mean ships.
"""
import re

import numpy as np

# Leave headroom under the 256-token limit for the header and special tokens.
DEFAULT_WINDOW_TOKENS = 180


def _sentences(text):
    parts = re.split(r'(?<=[.!?])\s+', str(text or '').strip())
    return [p for p in parts if p]


def split_windows(text, tokenizer, max_tokens=DEFAULT_WINDOW_TOKENS):
    """Greedily pack whole sentences into windows of at most `max_tokens` tokens."""
    out, cur, cur_n = [], [], 0
    for s in _sentences(text):
        n = len(tokenizer.encode(s, add_special_tokens=False))
        if n > max_tokens:
            # A single sentence longer than the window (scraped text with no
            # punctuation). Fall back to a word-count split so it is still read.
            words = s.split()
            step = max(1, int(max_tokens * 0.6))
            for i in range(0, len(words), step):
                out.append(" ".join(words[i:i + step]))
            continue
        if cur and cur_n + n > max_tokens:
            out.append(" ".join(cur))
            cur, cur_n = [], 0
        cur.append(s)
        cur_n += n
    if cur:
        out.append(" ".join(cur))
    return out or [""]


def encode_pooled(headers, descriptions, model, max_tokens=DEFAULT_WINDOW_TOKENS,
                  batch_size=64, show_progress_bar=False):
    """Embed `len(headers)` items, reading each description in full.

    headers      : per-item prefix, e.g. "Title: Dune. Genre: Sci-Fi."
    descriptions : per-item body text of any length
    returns      : (n_items, dim) float32, L2-normalized

    One flat `model.encode` call covers every window in the corpus, so this costs
    about the same as the old single-shot encode plus the extra windows themselves.
    """
    headers = list(headers)
    descriptions = list(descriptions)
    if len(headers) != len(descriptions):
        raise ValueError("headers and descriptions must be the same length")

    tokenizer = model.tokenizer
    flat, owner = [], []
    for i, (h, d) in enumerate(zip(headers, descriptions)):
        for w in split_windows(d, tokenizer, max_tokens):
            flat.append(f"{h} {w}".strip())
            owner.append(i)

    emb = model.encode(flat, normalize_embeddings=True, batch_size=batch_size,
                       show_progress_bar=show_progress_bar)
    emb = np.asarray(emb, dtype=np.float32)
    owner = np.asarray(owner)

    out = np.zeros((len(headers), emb.shape[1]), dtype=np.float32)
    # np.add.at accumulates per owner in one pass; counts turn the sum into a mean.
    np.add.at(out, owner, emb)
    counts = np.bincount(owner, minlength=len(headers)).astype(np.float32)
    out /= np.maximum(counts, 1.0)[:, None]

    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def encode_pooled_single(header, description, model, **kw):
    """Single-item form, so the Oracle path and the batch builder share one function
    and cannot drift apart (see scripts/audit_inference_parity.py)."""
    return encode_pooled([header], [description], model, **kw)
