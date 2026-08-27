"""Season-level TV data: the single source of truth for what you actually watched.

`data/processed/shows/season_features.csv` carries one row per season, with a
hand-entered `season_rating`. A season counts as WATCHED when it has a rating --
that is the rule the totals, the dashboard toggle and the season model all use.

Why this module exists: `enriched_data.csv` stores `number_of_episodes` /
`number_of_seasons` straight from TMDB, i.e. every season that exists, including
ones that aired after you stopped watching and ones not yet released. Summing
those overstates episode counts and watch time. Everything that reports "how much
have I watched" must go through `watched_totals()` instead.

Show-level rating from seasons:
  * anthology shows  -> no average (each season is a different story; keep them
                        separate, `show_season_avg` is NaN)
  * everything else  -> mean of the seasons you rated, which also covers shows you
                        genuinely left partway (only the watched seasons count)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

SEASON_FEATURES_PATH = config.TV_SHOWS_PROCESSED_DIR / "season_features.csv"
DEFAULT_EPISODE_RUNTIME = 45  # minutes, when a show has no runtime


def load_seasons(path=None):
    """One row per season. Returns an empty frame if the sheet is absent."""
    path = Path(path or SEASON_FEATURES_PATH)
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        # A spreadsheet round-trip re-saves this as cp1252; read it either way.
        df = pd.read_csv(path, encoding="cp1252")

    for c in ("tmdb_id", "season_number", "episode_count", "air_year",
              "season_rating", "show_user_rating", "is_anthology",
              "imdb_season_mean", "imdb_season_min", "imdb_season_max",
              "imdb_episode_spread", "imdb_delta_vs_prev", "cast_overlap_prev",
              "cast_turnover", "cast_new_count", "crew_change", "tmdb_season_vote"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["episode_count"] = df["episode_count"].fillna(0)
    # "if i gave rating its watched"
    df["watched"] = df["season_rating"].notna()
    # Unreleased placeholders can never be watched, whatever else is set.
    df.loc[df["episode_count"] <= 0, "watched"] = False

    anth = df.groupby("tmdb_id")["is_anthology"].transform("max").fillna(0)
    df["is_anthology_show"] = anth == 1

    avg = (df[df["watched"]].groupby("tmdb_id")["season_rating"].mean().round(2))
    df["show_season_avg"] = df["tmdb_id"].map(avg)
    df.loc[df["is_anthology_show"], "show_season_avg"] = np.nan
    return df


def watched_totals(seasons=None):
    """Per-show watched counts, indexed by tmdb_id.

    Columns: seasons_watched, seasons_total, episodes_watched, episodes_total,
             show_season_avg, is_anthology_show.
    """
    s = load_seasons() if seasons is None else seasons
    if s.empty:
        return pd.DataFrame()
    w = s[s["watched"]]
    out = pd.DataFrame({
        "seasons_total": s.groupby("tmdb_id").size(),
        "episodes_total": s.groupby("tmdb_id")["episode_count"].sum(),
        "seasons_watched": w.groupby("tmdb_id").size(),
        "episodes_watched": w.groupby("tmdb_id")["episode_count"].sum(),
        "show_season_avg": s.groupby("tmdb_id")["show_season_avg"].first(),
        "is_anthology_show": s.groupby("tmdb_id")["is_anthology_show"].first(),
    })
    out[["seasons_watched", "episodes_watched"]] = (
        out[["seasons_watched", "episodes_watched"]].fillna(0).astype(int))
    return out


def attach_watched_counts(shows, id_col="tmdb_id"):
    """Add watched-only episode/season counts + watch hours to a show-level frame.

    Adds `episodes_watched`, `seasons_watched`, `hours_watched` and leaves the
    original TMDB totals untouched. Falls back to the TMDB totals for any show the
    season sheet does not cover, so a partial sheet never silently zeroes a show.
    """
    df = shows.copy()
    totals = watched_totals()
    runtime = pd.to_numeric(df.get("runtime"), errors="coerce").fillna(
        DEFAULT_EPISODE_RUNTIME)
    tmdb_eps = pd.to_numeric(df.get("number_of_episodes"), errors="coerce").fillna(0)
    tmdb_seasons = pd.to_numeric(df.get("number_of_seasons"), errors="coerce").fillna(0)

    if totals.empty:
        df["episodes_watched"] = tmdb_eps
        df["seasons_watched"] = tmdb_seasons
        df["covered_by_season_sheet"] = False
    else:
        ids = pd.to_numeric(df[id_col], errors="coerce")
        eps = ids.map(totals["episodes_watched"])
        seas = ids.map(totals["seasons_watched"])
        df["covered_by_season_sheet"] = eps.notna()
        df["episodes_watched"] = eps.fillna(tmdb_eps)
        df["seasons_watched"] = seas.fillna(tmdb_seasons)
        df["show_season_avg"] = ids.map(totals["show_season_avg"])

    df["hours_watched"] = df["episodes_watched"] * runtime / 60
    return df


# --------------------------------------------------------------------------- #
# Season narrative — the one long-form, in-order plot text this project owns
# --------------------------------------------------------------------------- #
#
# TV is the worst-served domain in the text pipeline: a TMDB show `overview` runs
# a MEDIAN OF ONE SENTENCE, so there is no narrative shape in it at all, and the
# arc experiment (embedding thirds of the description) found nothing to read
# because there was nothing there to read.
#
# `season_features.csv` already carries a per-season `season_overview` -- 433 of
# 581 seasons populated, covering 119 of 173 shows, median ~41 words. Concatenated
# in season order that is a genuine multi-act summary of the show: what changes
# between season 1 and season 5, in one consistent register, from one source. It
# is the only beat-ordered plot text in the project that needs no new API and no
# generation step.
#
# Seasons are labelled and joined IN ORDER rather than blended, so the sentence
# windows the encoder builds land on act boundaries instead of straddling them.

SEASON_NARRATIVE_COL = "season_narrative"
MAX_NARRATIVE_SEASONS = 12   # a 20-season procedural would otherwise dwarf every other row


def season_overview_by_show(path=None, max_seasons=MAX_NARRATIVE_SEASONS):
    """-> {tmdb_id (int): "Season 1: ... Season 2: ..."} in season order.

    Shows with no populated season overview are simply absent from the mapping,
    so callers can `.map(...)` and get NaN rather than an empty-string column that
    looks populated.
    """
    seasons = load_seasons(path)
    if seasons.empty or "season_overview" not in seasons.columns:
        return {}

    df = seasons.copy()
    df["_ov"] = df["season_overview"].fillna("").astype(str).str.strip()
    df = df[df["_ov"] != ""]
    if df.empty:
        return {}

    df["_id"] = pd.to_numeric(df.get("tmdb_id"), errors="coerce")
    df["_sn"] = pd.to_numeric(df.get("season_number"), errors="coerce")
    df = df[df["_id"].notna()].sort_values(["_id", "_sn"])

    out = {}
    for sid, grp in df.groupby("_id"):
        parts = []
        for _, r in grp.head(max_seasons).iterrows():
            n = r["_sn"]
            label = f"Season {int(n)}" if pd.notna(n) else "Season"
            parts.append(f"{label}: {r['_ov']}")
        out[int(sid)] = " ".join(parts)
    return out


def attach_season_narrative(df, id_col="tmdb_id", path=None):
    """Add the `season_narrative` column to a SHOW-level frame, in place-safe copy.

    Every code path that builds show text must call this -- the batch builder, the
    scoring pipeline, the Oracle and the active-learning queue all share
    `_show_body`, and a path that skips this would silently embed a different
    string than training did. That is the exact train/serve skew class that
    `scripts/audit_inference_parity.py` exists to catch.
    """
    out = df.copy()
    mapping = season_overview_by_show(path)
    if not mapping:
        out[SEASON_NARRATIVE_COL] = ""
        return out
    ids = pd.to_numeric(out.get(id_col), errors="coerce")
    out[SEASON_NARRATIVE_COL] = ids.map(mapping).fillna("")
    return out
