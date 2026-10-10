"""
subtitles.py — fetch one English subtitle file per movie, resumably.

SOURCES
    subdl          (default) main source, 300 downloads/day.
    opensubtitles  second source, used ONLY for movies SubDL could not supply
                   (last status no_subs / bad_file). Free tier: 5 downloads/day
                   per IP, or 20/day when logged in. Needs in .env:
                     OPENSUBTITLES_API_KEY   (opensubtitles.com -> Profile -> API consumers)
                     OPENSUBTITLES_USERNAME, OPENSUBTITLES_PASSWORD  (optional: 5 -> 20/day)
                   Searching does not use the download quota.

MOVIE KEY
    Files and manifest rows are keyed by IMDb id ("tt0133093"); the few rated
    movies with no IMDb id use their TMDB id instead ("tmdb1631102"), and are
    searched by TMDB id. movie_keys() is the one place that rule lives.

WHY
    Subtitles carry things a plot summary cannot: how much a film talks, how much
    it swears, how its mood moves from start to end, how much of it is music or
    sound. This module only FETCHES and PARSES them; feature extraction and the
    with/without ablation come later.

LIMITS (SubDL free key, read from GET /api/v1/me)
    * search:   2,000 requests/day per account     -> 1-2 requests per movie
    * download: the free key's own download quota is 0, so downloads go out
                unauthenticated and use the anonymous cap of 300 files/day per IP
                -> the real bottleneck
    Both reset at 00:00 UTC (05:30 IST). ~1,100 movies take ~4 daily runs.

RESUMING
    Every movie is written to the manifest the moment it is tried, so a run can be
    stopped at any point (quota, Ctrl-C, crash) and the next run starts at the
    first movie not yet in the manifest. A run keeps downloading until SubDL
    refuses, then exits cleanly — just rerun it the next day.

STORAGE (all under data/, which git ignores — the raw files are copyrighted)
    data/raw/movies/subtitles/<movie key>.srt  UTF-8, the file exactly as downloaded
    data/raw/movies/subtitles/_manifest.csv    one row per attempt (latest wins):
        imdb_id (= movie key), status (ok | no_subs | bad_file), release_name,
        hi, fetched_at, source (subdl | opensubtitles)

USAGE
    python -m src.movies.subtitles                     # until today's quota runs out
    python -m src.movies.subtitles --max-downloads 20
    python -m src.movies.subtitles --retry no_subs     # try the misses again
    python -m src.movies.subtitles --source opensubtitles   # SubDL misses only
"""
import argparse
import io
import os
import re
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

load_dotenv()

SUBDL_KEY = os.getenv("SUBDL_API_KEY")
SEARCH_URL = "https://api.subdl.com/api/v1/subtitles"
DOWNLOAD_BASE = "https://dl.subdl.com"

OS_KEY = os.getenv("OPENSUBTITLES_API_KEY")
OS_USER = os.getenv("OPENSUBTITLES_USERNAME")
OS_PASS = os.getenv("OPENSUBTITLES_PASSWORD")
OS_API = "https://api.opensubtitles.com/api/v1"
OS_UA = "PersonalMediaIntelligenceHub v1.0"  # OpenSubtitles rejects requests without one

SUBTITLES_DIR = config.MOVIES_RAW_DIR / "subtitles"
MANIFEST_PATH = SUBTITLES_DIR / "_manifest.csv"
MANIFEST_COLS = ["imdb_id", "status", "release_name", "hi", "fetched_at", "source"]

PAUSE_S = 1.0
# Files tried per search (SDH, then plain) before giving up on a movie; each
# failed try costs one of the 300 daily downloads.
CANDIDATES_PER_SEARCH = 2
# Several broken files in a row almost always means SubDL has started refusing
# (quota page instead of a zip), not that several movies are genuinely broken.
MAX_BAD_STREAK = 3


class QuotaExhausted(Exception):
    pass


# --- manifest ---

def load_manifest():
    if not MANIFEST_PATH.exists():
        return pd.DataFrame(columns=MANIFEST_COLS)
    man = pd.read_csv(MANIFEST_PATH, dtype=str)
    if "source" not in man.columns:
        # rows written before OpenSubtitles was added all came from SubDL;
        # rewrite once so appended rows line up with the header
        man["source"] = "subdl"
        man[MANIFEST_COLS].to_csv(MANIFEST_PATH, index=False)
    return man


def append_manifest(row):
    pd.DataFrame([row], columns=MANIFEST_COLS).to_csv(
        MANIFEST_PATH, mode="a", header=not MANIFEST_PATH.exists(), index=False)


def forget(imdb_ids):
    """Drop manifest rows so these movies are tried again next run."""
    man = load_manifest()
    man[~man["imdb_id"].isin(imdb_ids)][MANIFEST_COLS].to_csv(MANIFEST_PATH, index=False)


def movie_keys(df):
    """IMDb id where there is one, else 'tmdb<id>'; None if neither."""
    imdb = df["imdb_id"].astype("string").str.strip()
    tmdb = pd.to_numeric(df.get("tmdb_id"), errors="coerce")
    alt = tmdb.map(lambda t: f"tmdb{int(t)}" if pd.notna(t) else None)
    return imdb.where(imdb.str.startswith("tt", na=False), alt).astype(object)


def _id_param(key):
    return {"tmdb_id": key[4:]} if key.startswith("tmdb") else {"imdb_id": key}


# --- SubDL calls ---

def _get(url, **kw):
    r = requests.get(url, timeout=30, **kw)
    if r.status_code in (403, 429):
        raise QuotaExhausted(f"HTTP {r.status_code} from {url.split('?')[0]}")
    r.raise_for_status()
    return r


def search(key, hi_only):
    """Return SubDL's English subtitle list for one movie (may be empty).
    hi=1 is a FILTER (hearing-impaired files only), not an include-the-flag switch."""
    params = {"api_key": SUBDL_KEY, **_id_param(key), "type": "movie",
              "languages": "EN", "subs_per_page": 30, "client": "custom_integration"}
    if hi_only:
        params["hi"] = 1
    r = _get(SEARCH_URL, params=params)
    data = r.json()
    if not data.get("status"):
        err = str(data.get("error", ""))
        if "limit" in err.lower():
            raise QuotaExhausted(err)
        return []  # "not found" comes back as status false
    return data.get("subtitles") or []


def usable(text, runtime_min):
    """Reject empty files and files for the wrong video. SubDL's matches include
    trailers, music videos and other films filed under the IMDb id (The Matrix ->
    a Pitbull song). The subtitle's last line must land at 70-135% of the film's
    runtime; the upper slack lets extended cuts through (Watchmen Ultimate Cut)."""
    if not text:
        return False
    d = parse_srt(text)
    if len(d) < 50:
        return False
    if runtime_min and runtime_min > 0:
        return 0.70 <= d["end_s"].max() / 60 / runtime_min <= 1.35
    return True


def runtimes():
    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH, dtype={"imdb_id": str})
    rt = pd.to_numeric(df["runtime"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    return dict(zip(movie_keys(df), rt))


def fetch(key, runtime_min=None, source="subdl"):
    """Download and save the best subtitle for one movie.
    Returns (status, chosen subtitle dict or None, downloads used).

    Prefer a hearing-impaired (SDH) file: its sound cues, speaker labels and
    lyrics are data parse_srt keeps. Fall back to a plain file when there is no
    SDH one or it is unusable (the SDH pool has mismatches, e.g. Twilight -> a
    music-video subtitle). The manifest's `hi` column records which one we got,
    so cue-based features can treat plain-file movies as missing, not as silent.
    Costs 1-2 searches and 1-2 downloads."""
    find, get = (search, download_srt) if source == "subdl" else (os_search, os_download)
    status, chosen, n = "no_subs", None, 0
    for hi_only in (True, False):
        subs = find(key, hi_only)
        time.sleep(PAUSE_S)
        for chosen in subs[:CANDIDATES_PER_SEARCH]:  # the source's own ranking
            try:
                text = get(chosen)
            except requests.HTTPError as e:
                if e.response is None or e.response.status_code not in (404, 410):
                    raise
                text = None  # dead link on SubDL's side: try the next file
            n += 1
            time.sleep(PAUSE_S)
            if usable(text, runtime_min):
                # newline="" keeps the file's own \r\n; Windows would otherwise
                # rewrite each one to \r\r\n.
                (SUBTITLES_DIR / f"{key}.srt").write_text(text, encoding="utf-8", newline="")
                return "ok", chosen, n
            status = "bad_file"
    return status, chosen, n


# --- OpenSubtitles calls ---

_os_token = None


def _os_headers():
    h = {"Api-Key": OS_KEY, "User-Agent": OS_UA, "Accept": "application/json",
         "Content-Type": "application/json"}
    if _os_token:
        h["Authorization"] = f"Bearer {_os_token}"
    return h


def os_login():
    """Log in when credentials exist (20 downloads/day instead of 5 per IP)."""
    global _os_token
    if not (OS_USER and OS_PASS):
        return False
    r = requests.post(f"{OS_API}/login", json={"username": OS_USER, "password": OS_PASS},
                      headers=_os_headers(), timeout=30)
    r.raise_for_status()
    _os_token = r.json()["token"]
    return True


def os_search(key, hi_only):
    """OpenSubtitles English results for one movie, most-downloaded first, as
    dicts shaped like SubDL's (release_name, hi) plus the file_id to download."""
    ident = {"tmdb_id": key[4:]} if key.startswith("tmdb") else {"imdb_id": int(key[2:])}
    params = {**ident, "languages": "en", "type": "movie", "order_by": "download_count",
              "hearing_impaired": "only" if hi_only else "exclude"}
    r = requests.get(f"{OS_API}/subtitles", params=params, headers=_os_headers(), timeout=30)
    if r.status_code == 429:
        time.sleep(5)  # search rate limit, not the daily quota: back off once
        r = requests.get(f"{OS_API}/subtitles", params=params, headers=_os_headers(), timeout=30)
    r.raise_for_status()
    out = []
    for item in r.json().get("data", []):
        a = item.get("attributes", {})
        files = a.get("files") or []
        if files:
            out.append({"file_id": files[0]["file_id"], "release_name": a.get("release", ""),
                        "hi": bool(a.get("hearing_impaired"))})
    return out


def os_download(sub):
    """Spend one download: ask for a temporary link, then fetch the .srt text."""
    r = requests.post(f"{OS_API}/download", json={"file_id": sub["file_id"]},
                      headers=_os_headers(), timeout=30)
    if r.status_code in (403, 406, 429):
        raise QuotaExhausted(f"OpenSubtitles download refused (HTTP {r.status_code}): "
                             f"{r.text[:120].strip()}")
    r.raise_for_status()
    info = r.json()
    f = requests.get(info["link"], timeout=60)
    f.raise_for_status()
    if info.get("remaining", 1) <= 0:
        sub["_last_one"] = True  # this file arrived; the next request would be refused
    return decode(f.content)


def download_srt(sub):
    """Download the zip and return the text of its largest .srt, or None."""
    # SubDL embeds our api_key in the link; with a free key that routes the
    # download to a 0-download account quota. Strip it to use the anonymous
    # 300/day-per-IP cap instead.
    url = re.sub(r"[?&]api_key=[^&]*", "", sub["url"])
    r = _get(DOWNLOAD_BASE + url)
    try:
        zf = zipfile.ZipFile(io.BytesIO(r.content))
    except zipfile.BadZipFile:
        if b"limit" in r.content[:2000].lower():
            raise QuotaExhausted("download refused: " + r.text[:120].strip())
        return None
    srts = [i for i in zf.infolist() if i.filename.lower().endswith(".srt")]
    if not srts:
        return None
    # Multi-CD releases ship CD1/CD2; the largest file is the safest single pick.
    return decode(zf.read(max(srts, key=lambda i: i.file_size)))


def decode(raw):
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return None


# --- parsing ---

_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")
_TAG = re.compile(r"<[^>]+>|\{[^}]+\}")          # <i>, {\an8}: formatting only
_SOUND = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)")  # [door slams], (whispers)
_LYRIC = re.compile(r"[♪#]+([^♪#]*)[♪#]*")        # ♪ lyrics ♪, # lyrics #
_SPEAKER = re.compile(r"^\s*-?\s*([A-Z][A-Z0-9 .'-]+):\s*")  # PAUL: / - JESSICA:


def _secs(h, m, s, ms):
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def _split_line(line):
    """One subtitle line -> (dialogue, speakers, sounds, lyrics)."""
    speakers, sounds, lyrics = [], [], []
    m = _SPEAKER.match(line)
    if m and len(m.group(1).strip()) > 1:
        speakers.append(m.group(1).strip())
        line = line[m.end():]
    sounds += [a or b for a, b in _SOUND.findall(line)]
    line = _SOUND.sub(" ", line)
    lyrics += [l.strip() for l in _LYRIC.findall(line) if l.strip()]
    line = _LYRIC.sub(" ", line)
    return " ".join(line.split()).strip(" -"), speakers, sounds, lyrics


def parse_srt(text):
    """SRT text -> one row per subtitle block. Nothing is dropped; each block is
    split into what kind of text it is:

        start_s, end_s  timing in seconds
        text            the whole block, formatting tags removed
        dialogue        spoken words only
        speakers        speaker labels ("PAUL"), '|'-joined
        sounds          sound cues ("door slams", "whispers"), '|'-joined
        lyrics          sung lines, '|'-joined
    """
    text = re.sub(r"\r+\n?", "\n", text)  # \r\n, \r\r\n and bare \r all -> \n
    rows = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.strip().split("\n")
        for i, line in enumerate(lines):
            m = _TIME.search(line)
            if not m:
                continue
            body = [_TAG.sub("", l).strip() for l in lines[i + 1:]]
            body = [l for l in body if l]
            if body:
                parts, in_lyric = [], False
                for l in body:
                    if in_lyric:
                        # continuation of a lyric that opened on an earlier line:
                        # "♪ I get up out of bed and" / "wave to my homegirls ♪"
                        parts.append(("", [], [], [l.strip("♪# ").strip()]))
                        in_lyric = not re.search(r"[♪#]", l)
                    else:
                        parts.append(_split_line(l))
                        in_lyric = len(re.findall(r"[♪#]+", l)) % 2 == 1
                g = m.groups()
                rows.append((
                    _secs(*g[:4]), _secs(*g[4:]), " ".join(body),
                    " ".join(p[0] for p in parts if p[0]),
                    "|".join(x for p in parts for x in p[1]),
                    "|".join(x for p in parts for x in p[2]),
                    "|".join(x for p in parts for x in p[3]),
                ))
            break
    return pd.DataFrame(rows, columns=["start_s", "end_s", "text", "dialogue",
                                       "speakers", "sounds", "lyrics"])


def load_subtitles(imdb_id):
    """Parsed subtitles for one movie, or None if not downloaded."""
    p = SUBTITLES_DIR / f"{imdb_id}.srt"
    # read_bytes: text mode would turn a stray \r\r\n into two line breaks.
    return parse_srt(p.read_bytes().decode("utf-8")) if p.exists() else None


# --- run ---

def targets(retry, source="subdl"):
    """Movie keys to try this run.
    subdl:          never tried, plus any whose latest status is in `retry`.
    opensubtitles:  only SubDL's misses (latest status no_subs/bad_file) that
                    OpenSubtitles has not already been asked about."""
    df = pd.read_csv(config.MOVIES_ENRICHED_DATA_PATH, dtype={"imdb_id": str})
    keys = movie_keys(df[df["user_rating"].notna()]).dropna().drop_duplicates()
    last = load_manifest().drop_duplicates("imdb_id", keep="last").set_index("imdb_id")
    if source == "opensubtitles":
        miss = set(last[last["status"].isin(["no_subs", "bad_file"]) &
                        (last["source"] != "opensubtitles")].index)
        return [k for k in keys if k in miss]
    # Latest status wins, so a retried movie that now succeeded counts as done.
    done = set(last.index) - set(last[last["status"].isin(retry)].index)
    return [k for k in keys if k not in done]


def run(max_downloads, retry, source="subdl"):
    if source == "subdl" and not SUBDL_KEY:
        sys.exit("SUBDL_API_KEY missing in .env (free key: subdl.com account panel).")
    if source == "opensubtitles":
        if not OS_KEY:
            sys.exit("OPENSUBTITLES_API_KEY missing in .env: create one at opensubtitles.com "
                     "-> Profile -> API consumers. Optionally add OPENSUBTITLES_USERNAME and "
                     "OPENSUBTITLES_PASSWORD for 20 downloads/day instead of 5.")
        print("OpenSubtitles: " + ("logged in (20/day)" if os_login() else
                                   "not logged in (5/day per IP)"), flush=True)
    SUBTITLES_DIR.mkdir(parents=True, exist_ok=True)
    todo = targets(retry, source)
    rts = runtimes()
    cap = "until the source refuses" if max_downloads is None else f"at most {max_downloads}"
    print(f"{len(todo)} movies to try from {source}; downloading {cap} this run.", flush=True)
    counts = {"ok": 0, "no_subs": 0, "bad_file": 0}
    downloads, bad_streak = 0, []
    try:
        for imdb_id in todo:
            if max_downloads is not None and downloads >= max_downloads:
                break
            rt = rts.get(imdb_id)
            status, sub, n = fetch(imdb_id, rt if pd.notna(rt) else None, source)
            downloads += n
            row = {"imdb_id": imdb_id, "status": status,
                   "release_name": sub.get("release_name", "") if sub else "",
                   "hi": str(bool(sub.get("hi"))) if sub else "",
                   "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "source": source}
            append_manifest(row)
            counts[row["status"]] += 1
            bad_streak = bad_streak + [imdb_id] if row["status"] == "bad_file" else []
            if len(bad_streak) >= MAX_BAD_STREAK:
                forget(bad_streak)  # likely refusals, not broken files: retry them
                counts["bad_file"] -= len(bad_streak)
                raise QuotaExhausted(f"{MAX_BAD_STREAK} unreadable downloads in a row")
            if sum(counts.values()) % 25 == 0:
                print(f"  … {counts}", flush=True)
            if sub and sub.get("_last_one"):
                raise QuotaExhausted("OpenSubtitles says 0 downloads remain today")
    except QuotaExhausted as e:
        print(f"STOPPED: {e}. Progress is saved; rerun after the daily reset.", flush=True)
    except requests.RequestException as e:
        # Network down or the source erroring: the current movie is not recorded, so
        # the next run retries it. Nothing already saved is lost.
        print(f"STOPPED on network/server error: {e}. Progress is saved; rerun.", flush=True)
    left = len(targets(retry=[]))
    print(f"This run: {counts}, {downloads} downloads. Movies never tried: {left}.", flush=True)
    if counts["ok"]:
        # Add the new movies to the style-feature table and redraw the charts.
        from src.movies.subtitle_features import update
        update()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-downloads", type=int, default=None,
                    help="stop after this many downloads (default: until quota runs out)")
    ap.add_argument("--retry", nargs="*", default=[], choices=["no_subs", "bad_file"],
                    help="re-attempt movies whose last status was one of these (subdl)")
    ap.add_argument("--source", choices=["subdl", "opensubtitles"], default="subdl",
                    help="opensubtitles = second source, tries only SubDL's misses")
    args = ap.parse_args()
    run(args.max_downloads, args.retry, args.source)
