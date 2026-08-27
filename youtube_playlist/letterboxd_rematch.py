"""Re-match the SSS shorts against TMDB with candidate scoring, then emit a
'log or not' document for Letterboxd.

Fixes a recall bug in letterboxd_match.py: that version returned the first exact
title match, so a feature-length film shadowed a same-named short. This version
scores every candidate and prefers shorts with a corroborating director.

TMDB is free. Uses cached YouTube titles, so this costs 0 YouTube quota.
"""

import concurrent.futures as futures
import csv
import html
import json
import os
import re
import sys

import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
load_dotenv(os.path.join(ROOT, ".env"))
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB = "https://api.themoviedb.org/3"

SHORT_MAX = 40
OUT_CSV = os.path.join(HERE, "sss_letterboxd_log.csv")
OUT_HTML = os.path.join(HERE, "sss_letterboxd_log.html")

VIDEO_URL_RE = re.compile(
    r"(?:youtu\.be/|youtube\.com/(?:watch\?(?:[^\s&]*&)*v=|shorts/|embed/|live/))"
    r"([A-Za-z0-9_-]{11})"
)
JUNK_LABEL_RE = re.compile(r"^(video\s*\d*|part\s*\d*|)$", re.I)
NOISE_RE = re.compile(
    r"\b(short\s+horror\s+film|horror\s+short\s+film|short\s+film|horror\s+short"
    r"|animated\s+horror\s+(story|short)|scary\s+short\s+film|online\s+premiere"
    r"|award\s+winning|official|indie\s+animation|found\s+footage|full\s+film"
    r"|animated|animation|4k|hd)\b",
    re.I,
)
# Channels whose output is festival/curated film, so a TMDB entry is expected.
FEST_CHANNELS = ("ALTER", "Crypt", "Screamfest", "Omeleto", "Short of the Week",
                 "CalArts", "Bloody Cuts", "Sundance")


def normalize(text):
  text = re.sub(r"[''‘’“”]", "'", text or "")
  text = re.sub(r"[^\w\s]", " ", text.lower())
  text = re.sub(r"^\s*the\s+", "", text)
  return re.sub(r"\s+", " ", text).strip()


def clean_youtube_title(title):
  quoted = re.findall(r'"([^"]{2,})"', title) or re.findall(
      r"[“]([^”]{2,})[”]", title)
  if quoted:
    return quoted[0].strip()
  head = re.split(r"[|]", title)[0]
  head = re.sub(r"\[[^\]]*\]", " ", head)
  head = re.sub(r"\((?:19|20)\d{2}\)", " ", head)
  head = NOISE_RE.sub(" ", head)
  return re.sub(r"\s+", " ", head).strip(" -–—:()") or title


def parse_labels(videos):
  labels = {}
  for video in videos:
    for line in video["description"].split("\n"):
      ids = VIDEO_URL_RE.findall(line)
      if not ids:
        continue
      label = line.split("http")[0].strip().rstrip(":").strip()
      match = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", label)
      title, creator = (
          (match.group(1).strip(), match.group(2).strip()) if match
          else (label, "")
      )
      if JUNK_LABEL_RE.match(title.strip()):
        title = ""
      for vid in ids:
        if vid not in labels or (title and not labels[vid][0]):
          labels[vid] = (title, creator)
  return labels


def tmdb_get(session, path, **params):
  params["api_key"] = TMDB_KEY
  for _ in range(3):
    try:
      response = session.get(f"{TMDB}{path}", params=params, timeout=15)
      if response.status_code == 200:
        return response.json()
      if response.status_code != 429:
        return None
    except requests.RequestException:
      continue
  return None


def score_candidate(details, want, creator, channel):
  """Scores one TMDB film against what we are looking for. None = reject."""
  got = normalize(details.get("title", ""))
  alt = normalize(details.get("original_title", ""))
  exact = want in (got, alt)
  contains = bool(want) and (want in got or got in want) and len(got) > 3
  if not (exact or contains):
    return None

  runtime = details.get("runtime") or 0
  crew = details.get("credits", {}).get("crew", [])
  directors = [c["name"] for c in crew if c.get("job") == "Director"]
  creator_hit = bool(
      creator and any(
          normalize(creator) in normalize(d) or normalize(d) in normalize(creator)
          for d in directors)
  )
  fest = any(k in (channel or "") for k in FEST_CHANNELS)

  score = 0
  score += 50 if exact else 15
  if 0 < runtime <= SHORT_MAX:
    score += 40           # a short is what we are looking for
  elif runtime == 0:
    score += 5            # unknown runtime, mildly plausible
  else:
    score -= 35           # feature length, probably a title collision
  if creator_hit:
    score += 45           # director corroboration is the strongest signal
  if fest and 0 < runtime <= SHORT_MAX:
    score += 15
  score += min(details.get("vote_count", 0), 40) / 20.0

  return {
      "score": score,
      "exact": exact,
      "tmdb_id": details["id"],
      "tmdb_title": details.get("title", ""),
      "year": (details.get("release_date") or "")[:4],
      "runtime": runtime,
      "directors": "; ".join(directors[:3]),
      "creator_match": creator_hit,
      "votes": details.get("vote_count", 0),
      "tmdb_score": details.get("vote_average", 0),
      "letterboxd": f"https://letterboxd.com/tmdb/{details['id']}/",
      "tmdb_url": f"https://www.themoviedb.org/movie/{details['id']}",
  }


# SSS animations are YouTube-era work. A match older than this is almost always
# a different film that happens to share the title.
MIN_PLAUSIBLE_YEAR = 1995


def classify(best, channel):
  """Buckets a winning candidate into a log recommendation."""
  if not best:
    return "not_found"
  runtime, exact = best["runtime"], best["exact"]
  fest = any(k in (channel or "") for k in FEST_CHANNELS)

  if runtime > SHORT_MAX:
    return "feature_collision"

  year = best.get("year") or ""
  if year.isdigit() and int(year) < MIN_PLAUSIBLE_YEAR and not best["creator_match"]:
    return "era_collision"
  if best["creator_match"] and exact:
    return "log_it"
  if exact and fest and 0 < runtime <= SHORT_MAX:
    return "log_it"
  if exact and 0 < runtime <= SHORT_MAX:
    return "probably" if len(best["tmdb_title"].split()) > 2 else "verify"
  if exact and runtime == 0:
    return "verify"
  return "verify"


def match_one(session, target):
  want = normalize(target["query"])
  if not want or len(want) < 2:
    return {**target, "bucket": "no_title"}

  results = []
  search = tmdb_get(session, "/search/movie", query=target["query"])
  if search:
    results = search.get("results", [])[:8]

  # Second pass: many shorts are indexed under "Title (Year)" style variants.
  if not results and target["creator"]:
    alt = tmdb_get(session, "/search/movie",
                   query=f"{target['query']} {target['creator']}")
    if alt:
      results = alt.get("results", [])[:5]

  scored = []
  for result in results:
    details = tmdb_get(session, f"/movie/{result['id']}",
                       append_to_response="credits")
    if not details:
      continue
    candidate = score_candidate(details, want, target["creator"],
                                target["channel"])
    if candidate:
      scored.append(candidate)

  if not scored:
    return {**target, "bucket": "not_found"}

  scored.sort(key=lambda c: c["score"], reverse=True)
  best = scored[0]
  return {**target, "bucket": classify(best, target["channel"]), **best}


BUCKET_META = {
    "log_it": ("Log it", "Confirmed short on TMDB, so it has a Letterboxd page. "
                         "Director or festival channel corroborates the match."),
    "probably": ("Probably — quick check", "Exact title, short runtime, but "
                 "no director confirmation. Open the link before logging."),
    "verify": ("Verify first", "Title matches but it is short or generic enough "
               "to be a different film. Confirm it is the one you watched."),
    "feature_collision": ("Do not log — feature-length", "The only TMDB hit is a "
                          "feature-length film that happens to share the title. "
                          "Not your short."),
    "era_collision": ("Do not log — wrong era", "The title matches a film from "
                      "before 1995. These animations are YouTube-era work, so "
                      "this is a different film with the same name."),
    "not_found": ("Not on Letterboxd", "No TMDB entry, so no Letterboxd page. "
                  "YouTube-native work — nothing to log."),
    "no_title": ("No usable title", "Could not derive a searchable title."),
}
ORDER = ["log_it", "probably", "verify", "feature_collision", "era_collision",
         "not_found", "no_title"]


def build_html(rows, counts):
  loggable = counts.get("log_it", 0)
  checkable = counts.get("probably", 0) + counts.get("verify", 0)
  total = len(rows)

  parts = [f"""<title>SSS Shorts — Letterboxd Log List</title>
<style>
:root {{
  --bg:#f6f7f9; --card:#fff; --tx:#14171a; --mut:#5b6570; --line:#e3e7ec;
  --good:#1a7f4b; --goodbg:#e6f4ec; --warn:#8a6100; --warnbg:#fdf3dd;
  --bad:#8f2f2f; --badbg:#fbeaea; --dead:#4a5560; --deadbg:#eef1f4;
}}
@media (prefers-color-scheme:dark) {{
  :root:not([data-theme=light]) {{
    --bg:#0f1216; --card:#171b21; --tx:#e8edf2; --mut:#95a1ae; --line:#252c35;
    --good:#5fd39b; --goodbg:#12301f; --warn:#e8bd63; --warnbg:#2e2513;
    --bad:#f08a8a; --badbg:#301616; --dead:#9aa6b2; --deadbg:#1b2027;
  }}
}}
:root[data-theme=dark] {{
  --bg:#0f1216; --card:#171b21; --tx:#e8edf2; --mut:#95a1ae; --line:#252c35;
  --good:#5fd39b; --goodbg:#12301f; --warn:#e8bd63; --warnbg:#2e2513;
  --bad:#f08a8a; --badbg:#301616; --dead:#9aa6b2; --deadbg:#1b2027;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--tx);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
 padding:32px 20px 80px}}
.wrap{{max-width:1100px;margin:0 auto}}
h1{{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--mut);margin:0 0 26px;font-size:14px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
 gap:12px;margin-bottom:26px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:10px;
 padding:14px 16px}}
.tile .n{{font-size:26px;font-weight:650;letter-spacing:-.02em}}
.tile .l{{color:var(--mut);font-size:12px;margin-top:2px}}
.controls{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:22px}}
input[type=search]{{flex:1;min-width:220px;padding:9px 12px;border-radius:8px;
 border:1px solid var(--line);background:var(--card);color:var(--tx);font-size:14px}}
section{{margin-bottom:30px}}
.head{{display:flex;align-items:baseline;gap:10px;margin-bottom:4px}}
.head h2{{font-size:17px;margin:0}}
.pill{{font-size:12px;padding:2px 9px;border-radius:20px;font-weight:600}}
.p-log{{background:var(--goodbg);color:var(--good)}}
.p-probably,.p-verify{{background:var(--warnbg);color:var(--warn)}}
.p-feature_collision,.p-era_collision{{background:var(--badbg);color:var(--bad)}}
.p-not_found,.p-no_title{{background:var(--deadbg);color:var(--dead)}}
.desc{{color:var(--mut);font-size:13px;margin:0 0 12px}}
.tblwrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);
 border-radius:10px}}
table{{width:100%;border-collapse:collapse;font-size:13.5px;min-width:720px}}
th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;
 color:var(--mut);padding:10px 12px;border-bottom:1px solid var(--line);
 font-weight:600}}
td{{padding:9px 12px;border-bottom:1px solid var(--line);vertical-align:top}}
tr:last-child td{{border-bottom:none}}
.ttl{{font-weight:600}}
.yt{{color:var(--mut);font-size:12px;display:block;margin-top:1px}}
.meta{{color:var(--mut);white-space:nowrap}}
a{{color:inherit}}
.lb{{display:inline-block;padding:3px 10px;border-radius:6px;font-size:12px;
 font-weight:600;text-decoration:none;background:var(--goodbg);color:var(--good);
 white-space:nowrap}}
.star{{color:var(--good);font-weight:700}}
.hidden{{display:none}}
footer{{color:var(--mut);font-size:12.5px;border-top:1px solid var(--line);
 padding-top:16px;margin-top:34px}}
code{{background:var(--deadbg);padding:1px 5px;border-radius:4px;font-size:12px}}
</style>

<div class="wrap">
<h1>Spooky Scary Sunday — what to log on Letterboxd</h1>
<p class="sub">The {total} animations CoryxKenshin reacted to, checked against TMDB.
Letterboxd builds its database from TMDB, so a TMDB entry means a Letterboxd page exists.</p>

<div class="tiles">
  <div class="tile"><div class="n" style="color:var(--good)">{loggable}</div>
    <div class="l">Ready to log</div></div>
  <div class="tile"><div class="n" style="color:var(--warn)">{checkable}</div>
    <div class="l">Worth a quick check</div></div>
  <div class="tile"><div class="n" style="color:var(--dead)">{counts.get('not_found',0)}</div>
    <div class="l">Not on Letterboxd</div></div>
  <div class="tile"><div class="n">{total}</div>
    <div class="l">Total reviewed</div></div>
</div>

<div class="controls">
  <input type="search" id="q" placeholder="Filter by title, director, or channel…">
</div>
"""]

  for bucket in ORDER:
    group = [r for r in rows if r["bucket"] == bucket]
    if not group:
      continue
    label, blurb = BUCKET_META[bucket]
    group.sort(key=lambda r: (-(r.get("votes") or 0),
                              r.get("tmdb_title", "") or r["yt_title"]))
    parts.append(f"""<section>
<div class="head"><h2>{html.escape(label)}</h2>
<span class="pill p-{bucket}">{len(group)}</span></div>
<p class="desc">{html.escape(blurb)}</p>
<div class="tblwrap"><table>""")

    if bucket in ("not_found", "no_title"):
      parts.append("<tr><th>Video</th><th>Channel</th></tr>")
      for row in group:
        parts.append(
            f'<tr class="r" data-s="{html.escape((row["yt_title"]+" "+row["channel"]).lower())}">'
            f'<td class="ttl">{html.escape(row["yt_title"][:80])}</td>'
            f'<td class="meta">{html.escape(row["channel"])}</td></tr>')
    else:
      parts.append("<tr><th>Film</th><th>Year</th><th>Run</th>"
                   "<th>Director</th><th>Channel</th><th>Letterboxd</th></tr>")
      for row in group:
        star = ' <span class="star" title="Director matches the credited animator">✓</span>' if row.get("creator_match") else ""
        searchable = " ".join([
            row.get("tmdb_title", ""), row["yt_title"], row["channel"],
            row.get("directors", "")]).lower()
        runtime = f'{row.get("runtime", 0)}m' if row.get("runtime") else "—"
        parts.append(
            f'<tr class="r" data-s="{html.escape(searchable)}">'
            f'<td><span class="ttl">{html.escape(row.get("tmdb_title", ""))}</span>{star}'
            f'<span class="yt">{html.escape(row["yt_title"][:70])}</span></td>'
            f'<td class="meta">{html.escape(str(row.get("year", "")))}</td>'
            f'<td class="meta">{runtime}</td>'
            f'<td class="meta">{html.escape((row.get("directors") or "—")[:34])}</td>'
            f'<td class="meta">{html.escape(row["channel"][:22])}</td>'
            f'<td><a class="lb" href="{row.get("letterboxd", "#")}" target="_blank">Open ↗</a></td>'
            "</tr>")
    parts.append("</table></div></section>")

  parts.append("""
<footer>
<p><strong>How to read this.</strong> A <span class="star">✓</span> means the TMDB
director matches the animator Cory credited in his description — the strongest
signal the match is right. Runtime is capped at 40 minutes, the usual festival
definition of a short.</p>
<p><strong>Caveat.</strong> Letterboxd pages are inferred, not verified. Letterboxd
has no open API and blocks automated requests, so each link goes through
<code>letterboxd.com/tmdb/&lt;id&gt;/</code>, which resolves in your browser.</p>
<p>The "Not on Letterboxd" group is mostly YouTube-native work — analog horror,
POV story channels, FNAF animation — which film databases do not catalogue.</p>
</footer>
</div>
<script>
var q = document.getElementById('q');
q.addEventListener('input', function() {
  var v = q.value.toLowerCase().trim();
  document.querySelectorAll('tr.r').forEach(function(tr) {
    tr.classList.toggle('hidden', v && tr.dataset.s.indexOf(v) === -1);
  });
  document.querySelectorAll('section').forEach(function(s) {
    var any = s.querySelectorAll('tr.r:not(.hidden)').length;
    s.classList.toggle('hidden', v && any === 0);
  });
});
</script>""")
  return "\n".join(parts)


def main():
  if not TMDB_KEY:
    sys.exit("TMDB_API_KEY missing from .env")

  videos = json.load(open(os.path.join(HERE, "descriptions.json"),
                          encoding="utf-8"))
  audit = json.load(open(os.path.join(HERE, "audit_report.json"),
                         encoding="utf-8"))
  meta = json.load(open(os.path.join(HERE, "video_meta.json"), encoding="utf-8"))
  labels = parse_labels(videos)

  targets = []
  for vid in audit["universe"]:
    label_title, creator = labels.get(vid, ("", ""))
    info = meta.get(vid, {})
    yt_title = info.get("title", "")
    targets.append({
        "video_id": vid,
        "query": label_title or clean_youtube_title(yt_title),
        "creator": creator,
        "yt_title": yt_title or "(unavailable video)",
        "channel": info.get("channel", ""),
    })

  print(f"Re-matching {len(targets)} shorts with candidate scoring...")
  session = requests.Session()
  rows = []
  with futures.ThreadPoolExecutor(max_workers=6) as pool:
    for i, row in enumerate(
        pool.map(lambda t: match_one(session, t), targets), 1):
      rows.append(row)
      if i % 50 == 0:
        print(f"  {i}/{len(targets)}")

  counts = {}
  for row in rows:
    counts[row["bucket"]] = counts.get(row["bucket"], 0) + 1

  fields = ["bucket", "tmdb_title", "year", "runtime", "directors",
            "creator_match", "letterboxd", "tmdb_url", "votes", "tmdb_score",
            "yt_title", "channel", "query", "video_id"]
  with open(OUT_CSV, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for bucket in ORDER:
      for row in rows:
        if row["bucket"] == bucket:
          writer.writerow(row)

  with open(OUT_HTML, "w", encoding="utf-8") as handle:
    handle.write(build_html(rows, counts))

  print("\n=== LOG RECOMMENDATION ===")
  for bucket in ORDER:
    if bucket in counts:
      print(f"  {BUCKET_META[bucket][0]:22s} {counts[bucket]:4d}")
  print(f"\nWrote {OUT_HTML}")
  print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
  main()
