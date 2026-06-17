"""
Latent-space & transfer wiring diagnostics (err.txt follow-up).

Runs six pass/fail checks to decide whether the cross-domain transfer NULL is a
*real* finding or an artifact of a broken pipeline (empty embeddings, mismatched
input text, un-applied CORAL, leaked domain-identity features, un-unified genre
vocab, or a per-domain rating-scale mismatch). Plus a music-lyrics check.

Run:  python -m src.experiments.latent_space_diagnostics
Prints a PASS / FAIL / INFO verdict per check and a final summary. Read-only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.unified_model.unified_utils import DomainAligner
from src.experiments.transfer_study import select_features

RATED = ["movie", "tv", "game", "book"]
results = {}


def banner(n, title):
    print(f"\n{'='*70}\nCHECK {n}: {title}\n{'='*70}")


def verdict(n, status, msg):
    results[n] = status
    print(f"  -> [{status}] {msg}")


# ----------------------------------------------------------------------------
def check1_embeddings_populated(uni):
    """Are the vibe (pca_) embeddings real, or silently zeroed for games/books?"""
    banner(1, "Embeddings populated, or silently zeroed?")
    pca = [c for c in uni.columns if c.startswith("pca_")]
    worst = None
    for d in RATED + ["music"]:
        sl = uni[uni.media_type == d][pca]
        if sl.empty:
            continue
        norms = np.linalg.norm(sl.values, axis=1)
        frac_zero = float((norms < 1e-9).mean())
        # fraction of rows that are exact duplicates of another row (collapse)
        dup = float(sl.duplicated().mean())
        print(f"  {d:6s} N={len(sl):4d}  mean|vec|={norms.mean():.4f}  "
              f"std|vec|={norms.std():.4f}  frac_zero={frac_zero:.2%}  dup_rows={dup:.2%}")
        if frac_zero > 0.5 or norms.mean() < 1e-6:
            worst = d
    if worst:
        verdict(1, "FAIL", f"{worst} embeddings look empty/zeroed (collapsed cluster).")
    else:
        verdict(1, "PASS", "every domain has non-zero, varied embeddings (not silently zeroed).")


def check2_text_shape(uni):
    """Is each domain's embedded text the same *kind* of text (prose vs keyword soup)?"""
    banner(2, "Embedded input text — comparable shape across domains?")
    # Reconstruct the unified embed text exactly as unified_feature_engineering does:
    #   txt = "Title: <title>. Lead: <director|developers|authors>. <overview|description>"
    sources = {
        "movie": (config.MOVIES_ENRICHED_DATA_PATH, "title", "director", "plot"),
        "tv": (config.TV_SHOWS_ENRICHED_DATA_PATH, "name", "created_by", "overview"),
        "game": (config.GAMES_ENRICHED_DATA_PATH, "name", "developers", "description_raw"),
        "book": (config.BOOKS_ENRICHED_DATA_PATH, "title", "authors", "description"),
    }
    stats = {}
    for d, (path, tcol, lcol, ocol) in sources.items():
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, encoding="latin1")
        def g(col):
            return df[col].fillna("").astype(str) if col in df.columns else pd.Series([""] * len(df))
        txt = ("Title: " + g(tcol) + ". Lead: " + g(lcol) + ". " + g(ocol))
        lens = txt.str.len()
        commas = txt.str.count(",")
        # words-per-sentence heuristic: prose has long sentences; tag soup has many short comma chunks
        stats[d] = {"chars": lens.median(), "commas": commas.median(),
                    "empty_overview": float((g(ocol).str.len() < 3).mean())}
        print(f"  {d:6s} median_chars={lens.median():6.0f}  median_commas={commas.median():4.0f}  "
              f"empty_description={stats[d]['empty_overview']:.1%}")
        ex = txt.iloc[0][:160].replace("\n", " ")
        print(f"         e.g. “{ex}…”")
    # FAIL if some domains are mostly empty descriptions, or char length differs by >5x
    chars = [s["chars"] for s in stats.values()]
    empties = [d for d, s in stats.items() if s["empty_overview"] > 0.5]
    if empties:
        verdict(2, "FAIL", f"{empties} have mostly-empty descriptions -> embeddings near title-only.")
    elif max(chars) > 5 * max(min(chars), 1):
        verdict(2, "WARN", "text length differs >5x across domains (prose vs short) — "
                           "MiniLM may place them in different regions regardless of genre.")
    else:
        verdict(2, "PASS", "comparable text length/shape; descriptions populated across domains.")


def check3_coral(uni):
    """Does CORAL/centroid alignment actually pull the domain islands together?"""
    banner(3, "CORAL alignment — does it reduce domain separability?")
    pca = [c for c in uni.columns if c.startswith("pca_")]
    X = uni[pca].values
    mt = uni.media_type.values

    def separability(Xv):
        # ratio of mean inter-domain centroid distance to mean within-domain radius.
        cents = {d: Xv[mt == d].mean(0) for d in np.unique(mt)}
        inter = np.mean([np.linalg.norm(cents[a] - cents[b])
                         for a in cents for b in cents if a < b])
        intra = np.mean([np.linalg.norm(Xv[mt == d] - cents[d], axis=1).mean() for d in cents])
        return inter / (intra + 1e-9)

    raw = separability(X)
    al = DomainAligner(method="centroid"); al.fit(X, pd.Series(mt))
    Xc = al.transform(X, pd.Series(mt))
    aligned = separability(Xc)
    try:
        alc = DomainAligner(method="coral"); alc.fit(X, pd.Series(mt))
        coral = separability(alc.transform(X, pd.Series(mt)))
    except Exception as e:
        coral = float("nan"); print("   (coral path error:", e, ")")
    print(f"  domain-separability ratio  raw={raw:.3f}  centroid-aligned={aligned:.3f}  coral={coral:.3f}")
    print("  (higher = domains sit in separate islands; alignment should LOWER it)")
    if aligned < raw * 0.6:
        verdict(3, "PASS", f"alignment cuts separability {raw:.2f}->{aligned:.2f} (islands pulled together).")
    else:
        verdict(3, "WARN", f"alignment barely moves separability ({raw:.2f}->{aligned:.2f}); "
                           "the explorer/grid may be showing the un-aligned space.")


def check4_identity_leak(uni):
    """Does the transfer-grid feature set leak domain-identity columns?"""
    banner(4, "Domain-identity leak into the transfer-grid features?")
    feat = select_features(uni, max_genres=40)
    ident = [c for c in feat if c.startswith("has_") or c.startswith("is_") or c == "media_type"]
    print(f"  transfer-grid features: {len(feat)}  | identity columns present: {ident}")
    if ident:
        verdict(4, "FAIL", f"{ident} let the model see the domain label — a transfer grid must be "
                           "domain-blind. In augmented (Protocol C) these masks vary and the model "
                           "can split on domain identity instead of transferring content.")
    else:
        verdict(4, "PASS", "no domain-identity columns in the grid feature set.")


def check5_genre_vocab(uni):
    """Is genre a single shared vocabulary (gen_Action fires across domains)?"""
    banner(5, "Unified genre vocabulary?")
    gen = [c for c in uni.columns if c.startswith("gen_")]
    shared = []
    for g in gen:
        fires = uni[uni[g] == 1].media_type.nunique()
        if fires >= 2:
            shared.append(g)
    print(f"  total gen_ columns: {len(gen)}  | columns firing in >=2 domains: {len(shared)}")
    for g in ["gen_Action", "gen_Adventure", "gen_Science Fiction", "gen_Fantasy"]:
        if g in uni.columns:
            print(f"    {g}: {uni[uni[g]==1].media_type.value_counts().to_dict()}")
    if shared:
        verdict(5, "PASS", f"{len(shared)} genre columns fire across >=2 domains — genre vocab IS unified "
                           "(movie-action and game-action are the SAME feature).")
    else:
        verdict(5, "FAIL", "no genre column fires across domains — genre is NOT unified.")


def check6_taste_consistency(uni):
    """Do rating scale/behaviour differ by domain (a ceiling on transfer independent of embeddings)?"""
    banner(6, "Per-domain rating behaviour (taste-consistency ceiling)")
    sub = uni[uni.media_type.isin(RATED)]
    g = sub.groupby("media_type")["target_reg"].agg(["mean", "std", "count"])
    print(g.round(3).to_string())
    spread = g["mean"].max() - g["mean"].min()
    print(f"  mean-rating spread across domains: {spread:.2f} stars")
    if spread > 0.5:
        verdict(6, "INFO", f"rating *level* differs by {spread:.2f}★ across domains — even a perfect shared "
                           "space transfers poorly if you rate domains on different scales. Real ceiling, not a bug.")
    else:
        verdict(6, "INFO", f"rating levels are similar across domains ({spread:.2f}★); scale mismatch is NOT "
                           "the main ceiling — look to embeddings/wiring.")


def check_music(uni):
    """Music: is the embedded text actual lyrics, or just title/artist? (PU-label caveat noted.)"""
    banner("M", "Music — lyrics actually embedded, or title/tags only?")
    try:
        mp = config.MUSIC_ENRICHED_DATA_PATH
        df = pd.read_csv(mp)
        has_lyric_col = "lyric_embed_text" in df.columns
        if has_lyric_col:
            nonempty = float((df["lyric_embed_text"].fillna("").str.len() > 20).mean())
            print(f"  lyric_embed_text column present; non-trivial in {nonempty:.1%} of tracks")
            ex = str(df["lyric_embed_text"].dropna().iloc[0])[:160] if nonempty else "(none)"
            print(f"    e.g. “{ex}…”")
            if nonempty > 0.3:
                verdict("M", "PASS", "lyrics are populated and reach the music text vector.")
            else:
                verdict("M", "FAIL", "lyric_embed_text is mostly empty — music embeds title/tags, not lyrics.")
        else:
            verdict("M", "FAIL", "no lyric_embed_text column — music 'description' is title/artist/tags only.")
    except Exception as e:
        verdict("M", "SKIP", f"music enriched data unavailable: {e}")


def main():
    uni = pd.read_csv(config.UNIFIED_TRAINING_DATA_PATH)
    print(f"Loaded unified matrix: {uni.shape}  domains={uni.media_type.value_counts().to_dict()}")
    check1_embeddings_populated(uni)
    check2_text_shape(uni)
    check3_coral(uni)
    check4_identity_leak(uni)
    check5_genre_vocab(uni)
    check6_taste_consistency(uni)
    check_music(uni)
    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    for k, v in results.items():
        print(f"  Check {k}: {v}")
    fails = [k for k, v in results.items() if v == "FAIL"]
    print(f"\n  {'⚠️  FAILS: ' + str(fails) if fails else '✅ no hard FAILs'} "
          f"— investigate FAIL/WARN before trusting the transfer verdict (either direction).")


if __name__ == "__main__":
    main()
