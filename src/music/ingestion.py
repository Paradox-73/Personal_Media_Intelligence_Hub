import os
import sys
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import lyricsgenius
from tqdm import tqdm
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MUSIC_ENRICHED_DATA_PATH

# --- Setup ---
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=os.environ.get('SPOTIPY_CLIENT_ID'),
    client_secret=os.environ.get('SPOTIPY_CLIENT_SECRET'),
    redirect_uri='http://localhost:8888/callback',
    scope="user-library-read"
))

genius = lyricsgenius.Genius(os.environ.get('GENIUS_ACCESS_TOKEN'), timeout=15, retries=3)
genius.remove_section_headers = True

def fetch_spotify_tracks(include_liked=True, playlist_ids=None):
    """Fetches track metadata from Spotify."""
    tracks_data = []
    seen = set()

    if include_liked:
        print("Fetching Liked Songs...")
        results = sp.current_user_saved_tracks(limit=50)
        while results:
            for item in results['items']:
                t = item['track']
                if t['id'] not in seen:
                    tracks_data.append({'track_id': t['id'], 'track_name': t['name'], 'artist_name': t['artists'][0]['name']})
                    seen.add(t['id'])
            results = sp.next(results) if results['next'] else None

    if playlist_ids:
        for p_id in playlist_ids:
            print(f"Fetching Playlist: {p_id}...")
            results = sp.playlist_tracks(p_id, limit=50)
            while results:
                for item in results['items']:
                    t = item['track']
                    if t and t['id'] not in seen:
                        tracks_data.append({'track_id': t['id'], 'track_name': t['name'], 'artist_name': t['artists'][0]['name']})
                        seen.add(t['id'])
                results = sp.next(results) if results['next'] else None
                
    return pd.DataFrame(tracks_data)

def ingest_data():
    # 1. Get Base Tracks
    custom_playlists = [] # Add your specific playlist IDs here
    df = fetch_spotify_tracks(include_liked=True, playlist_ids=custom_playlists)
    
    # 2. Scrape Lyrics (The Encrichment Phase)
    print(f"Scraping lyrics for {len(df)} tracks...")
    df['lyrics'] = None
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Genius API"):
        try:
            song = genius.search_song(row['track_name'], row['artist_name'])
            if song:
                df.at[index, 'lyrics'] = song.lyrics
        except Exception:
            pass
        time.sleep(2) # Prevent IP ban from Genius
        
        # Save checkpoints safely
        if (index + 1) % 50 == 0:
            df.to_csv(MUSIC_ENRICHED_DATA_PATH, index=False)
            
    # Drop instrumentals / failed scrapes to keep the dataset clean
    df = df.dropna(subset=['lyrics']).reset_index(drop=True)
    df.to_csv(MUSIC_ENRICHED_DATA_PATH, index=False)
    print(f"Ingestion complete. Saved to {MUSIC_ENRICHED_DATA_PATH}")

if __name__ == "__main__":
    ingest_data()