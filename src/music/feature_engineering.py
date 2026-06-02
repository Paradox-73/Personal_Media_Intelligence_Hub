"""
feature_engineering.py — Stage 4 of the Music pipeline.

Merges the three enriched sources (Spotify+ReccoBeats, MusicBrainz, Genius) into
one model-ready table and turns it into a numeric feature matrix, exactly the way
the Books / Movies pipelines do it:

  * numeric features   : duration, year, popularity, the 9 audio features, lyric stats
  * multi-hot encoding  : genres/tags (Spotify + MusicBrainz) and top artists
  * semantic embeddings : SentenceTransformers over (title + genres + lyric snippet),
                          reduced with PCA

Outputs:
  data/processed/music_processed.csv   merged human-readable table
  data/processed/music_features.npz    X, y, feature_names
  models/music/preprocessors.joblib    fitted scaler / pca / encoders (for inference)

Run:
    python src/music/feature_engineering.py
    python src/music/feature_engineering.py --ratings data/raw/my_ratings.csv
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_GENRES = 60
TOP_ARTISTS = 40
PCA_COMPONENTS = 32


def load_and_merge(ratings_path=None) -> pd.DataFrame:
    df = pd.read_csv(config.LIBRARY_CSV)
    for path in (config.MUSICBRAINZ_CSV, config.LYRICS_CSV):
        if path.exists():
            df = df.merge(pd.read_csv(path), on="track_id", how="left")

    # Optional: real user ratings override the synthesised implicit rating.
    if ratings_path:
        r = pd.read_csv(ratings_path)
        cols = {c.lower(): c for c in r.columns}
        id_col = cols.get("track_id") or cols.get("id")
        rate_col = cols.get("rating") or cols.get("stars") or cols.get("score")
        if id_col and rate_col:
            r = r[[id_col, rate_col]].rename(columns={id_col: "track_id", rate_col: "rating_user"})
            df = df.merge(r, on="track_id", how="left")
            df["rating"] = df["rating_user"].fillna(df["rating"])
            df.drop(columns=["rating_user"], inplace=True)
    return df


def split_terms(series, sep_pattern=r",|;"):
    return (series.fillna("").astype(str)
            .str.lower().str.split(sep_pattern)
            .apply(lambda xs: [x.strip() for x in xs if x.strip()]))


def multi_hot(term_lists, vocabulary, prefix):
    cols = {f"{prefix}_{g}".replace(" ", "_"): np.zeros(len(term_lists)) for g in vocabulary}
    vocab = set(vocabulary)
    for i, terms in enumerate(term_lists):
        for t in terms:
            if t in vocab:
                cols[f"{prefix}_{t}".replace(" ", "_")][i] = 1.0
    return pd.DataFrame(cols)


def build_numeric(df) -> pd.DataFrame:
    num = pd.DataFrame(index=df.index)
    num["duration_min"] = df["duration_ms"].fillna(df["duration_ms"].median()) / 60000
    num["release_year"] = df["release_year"].fillna(df["release_year"].median())
    num["track_age"] = 2026 - num["release_year"]
    num["popularity"] = df["popularity"].fillna(df["popularity"].median())
    num["artist_popularity"] = df.get("artist_popularity", 0).fillna(0)
    num["artist_followers_log"] = np.log1p(df.get("artist_followers", 0).fillna(0))
    num["album_total_tracks"] = df.get("album_total_tracks", 0).fillna(0)
    num["track_number"] = df.get("track_number", 0).fillna(0)
    num["explicit"] = df["explicit"].fillna(False).astype(int)

    # Audio features (ReccoBeats). Impute median + carry a "missing" flag so the
    # model can tell a real 0 from an absent value.
    for col in config.AUDIO_FEATURE_COLS:
        present = df[col].notna() if col in df else pd.Series(False, index=df.index)
        vals = df[col] if col in df else pd.Series(np.nan, index=df.index)
        num[col] = vals.fillna(vals.median() if present.any() else 0.0)
        num[f"{col}_missing"] = (~present).astype(int)

    # Lyric features (Genius/VADER)
    for col in ["lyric_word_count", "lyric_unique_ratio", "lyric_line_count",
                "lyric_sentiment", "lyric_pos", "lyric_neu", "lyric_neg"]:
        num[col] = df[col].fillna(0) if col in df else 0.0
    num["lyrics_found"] = df["lyrics_found"].fillna(False).astype(int) if "lyrics_found" in df else 0
    return num


def build_genre_vocab(df):
    parts = []
    for c in ("artist_genres", "mb_genres", "mb_tags"):
        if c in df:
            parts.append(split_terms(df[c]))
    if not parts:
        return [], pd.Series([[]] * len(df))
    combined = parts[0]
    for p in parts[1:]:
        combined = combined.combine(p, lambda a, b: list(set(a) | set(b)))
    counts = pd.Series([g for row in combined for g in row]).value_counts()
    return counts.head(TOP_GENRES).index.tolist(), combined


def build_embeddings(df):
    from sentence_transformers import SentenceTransformer
    text = (
        df["name"].fillna("") + ". "
        + df.get("artist_genres", "").fillna("") + ". "
        + df.get("mb_tags", "").fillna("") + ". "
        + df.get("lyric_embed_text", "").fillna("")
    ).tolist()
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(text, show_progress_bar=True, batch_size=64, normalize_embeddings=True)
    n_comp = min(PCA_COMPONENTS, emb.shape[0] - 1, emb.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    reduced = pca.fit_transform(emb)
    cols = [f"emb_{i}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=cols, index=df.index), pca


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratings", default=None, help="optional CSV of real star ratings")
    args = ap.parse_args()

    df = load_and_merge(args.ratings)
    df.to_csv(config.MASTER_CSV, index=False)
    print(f"Merged table -> {config.MASTER_CSV}  ({len(df)} tracks)")

    numeric = build_numeric(df)

    genre_vocab, genre_lists = build_genre_vocab(df)
    genre_mh = multi_hot(genre_lists, genre_vocab, "g")

    artist_lists = split_terms(df["primary_artist"].fillna(""))
    artist_counts = pd.Series([a for row in artist_lists for a in row]).value_counts()
    artist_vocab = artist_counts.head(TOP_ARTISTS).index.tolist()
    artist_mh = multi_hot(artist_lists, artist_vocab, "a")

    print("Building semantic embeddings (SentenceTransformers + PCA)...")
    emb_df, pca = build_embeddings(df)

    feature_df = pd.concat([numeric.reset_index(drop=True),
                            genre_mh.reset_index(drop=True),
                            artist_mh.reset_index(drop=True),
                            emb_df.reset_index(drop=True)], axis=1)

    scaler = StandardScaler()
    scaled_numeric = pd.DataFrame(scaler.fit_transform(numeric),
                                  columns=numeric.columns, index=numeric.index)
    X = pd.concat([scaled_numeric.reset_index(drop=True),
                   genre_mh.reset_index(drop=True),
                   artist_mh.reset_index(drop=True),
                   emb_df.reset_index(drop=True)], axis=1)
    y = df["rating"].astype(float).values

    np.savez_compressed(config.FEATURES_NPZ,
                        X=X.values, y=y, feature_names=np.array(X.columns))
    joblib.dump({
        "scaler": scaler, "pca": pca,
        "numeric_cols": list(numeric.columns),
        "genre_vocab": genre_vocab, "artist_vocab": artist_vocab,
        "embed_model": EMBED_MODEL, "feature_names": list(X.columns),
    }, config.MODELS_DIR / "preprocessors.joblib")

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Saved -> {config.FEATURES_NPZ} and models/music/preprocessors.joblib")


if __name__ == "__main__":
    main()