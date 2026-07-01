#!/usr/bin/env python3
"""
One-time helper: obtain a Spotify refresh token for headless (CI) use.

Run this ONCE on your local machine:

    python get_refresh_token.py

It opens a browser, you approve, and it prints a long-lived refresh token.
Copy that token into your GitHub repo secrets as SPOTIFY_REFRESH_TOKEN.

Requires these env vars (or a .env file) to be set first:
    SPOTIPY_CLIENT_ID
    SPOTIPY_CLIENT_SECRET
    SPOTIPY_REDIRECT_URI   (e.g. http://127.0.0.1:8080/callback -- must match
                            exactly what you registered in the Spotify dashboard)
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from spotipy.oauth2 import SpotifyOAuth

SCOPE = "user-library-read playlist-modify-public playlist-modify-private"

for var in ("SPOTIPY_CLIENT_ID", "SPOTIPY_CLIENT_SECRET", "SPOTIPY_REDIRECT_URI"):
    if not os.environ.get(var):
        raise SystemExit(f"Set {var} before running (see .env.example).")

auth = SpotifyOAuth(scope=SCOPE, open_browser=True, cache_path=".cache-token-helper")
token_info = auth.get_access_token(as_dict=True, check_cache=False)

print("\n" + "=" * 60)
print("SUCCESS. Add this to your GitHub repo secrets:")
print("\n  SPOTIFY_REFRESH_TOKEN =", token_info["refresh_token"])
print("=" * 60)
print("\n(Also add SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, "
      "SPOTIPY_REDIRECT_URI, LASTFM_API_KEY, LASTFM_USER as secrets.)")
