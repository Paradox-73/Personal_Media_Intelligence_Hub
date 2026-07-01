# Daily Playlist Maker

Auto-builds one Spotify playlist every day, mixing three buckets
(default **200 tracks, 33 / 33 / 34**):

| Bucket | What | Source |
| --- | --- | --- |
| **Heavy rotation** | Your most-played tracks this past week | Last.fm `user.getTopTracks` (7day) |
| **Older liked** | Liked songs you haven't scrobbled in a while | Spotify saved tracks minus recent Last.fm scrobbles |
| **Discovery** | New tracks similar to your taste, not already liked | Last.fm `track.getSimilar`, resolved on Spotify |

The same playlist is replaced in-place each run, so a cron job keeps it fresh.

> Lives inside the Personal Media Intelligence Hub repo and reuses its root
> `.env` and the `content_rec` venv. The GitHub Actions workflow is at
> `.github/workflows/daily-playlist.yml` (repo root).

> **Why Last.fm for discovery?** Spotify deprecated the `/v1/recommendations`
> endpoint (Nov 2024) for apps without prior access, so discovery is seeded from
> your Last.fm loved/top tracks and their similar tracks instead.

## Setup

### 1. Keys
Your root `.env` already has `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, and
`LASTFM_API_KEY`. Add these three lines to the same `.env`:

```
SPOTIPY_REDIRECT_URI=http://127.0.0.1:8080/callback
LASTFM_USER=your_lastfm_username
SPOTIFY_REFRESH_TOKEN=        # filled in by step 2
```
The redirect URI must **exactly** match one registered in your Spotify app
dashboard (https://developer.spotify.com/dashboard).

### 2. Get a Spotify refresh token (one time, local)
Run from the repo root:
```
content_rec\Scripts\python.exe -m pip install -r daily_playlist\requirements.txt
content_rec\Scripts\python.exe daily_playlist\get_refresh_token.py
```
Approve in the browser; paste the printed token into `SPOTIFY_REFRESH_TOKEN`
in your `.env`.

### 3. Test locally
```
content_rec\Scripts\python.exe daily_playlist\daily_playlist.py
```

### 4. Automate on GitHub Actions
This repo already has the workflow. In **Settings -> Secrets and variables ->
Actions**, add these **secrets** (same values as your `.env`):

- `SPOTIPY_CLIENT_ID`
- `SPOTIPY_CLIENT_SECRET`
- `SPOTIPY_REDIRECT_URI`
- `SPOTIFY_REFRESH_TOKEN`
- `LASTFM_API_KEY`
- `LASTFM_USER`

Optionally add repo **variables** `PLAYLIST_NAME` / `PLAYLIST_SIZE`.

The workflow runs daily at 06:30 UTC and can be triggered manually from the
**Actions** tab (**Daily Playlist -> Run workflow**).

## Tuning
All knobs are environment variables: `PLAYLIST_SIZE`, `RATIO_HEAVY` /
`RATIO_OLDER` / `RATIO_DISCOVERY`, `RECENT_WINDOW_DAYS` (how "stale" a liked song
must be to count as *older*), `DISCOVERY_SEEDS`, `PLAYLIST_PUBLIC`, and
`RANDOM_SEED` (reproducible runs).

## Notes & caveats
- **"Haven't heard in a while"** is approximated from Last.fm scrobbles, not
  Spotify play history (Spotify's API only exposes ~50 recently-played tracks).
- Cross-service matching normalizes names but isn't perfect; a few similar
  tracks may fail to resolve on Spotify and are silently skipped.
- The script refuses to wipe the playlist if it assembles zero tracks.
