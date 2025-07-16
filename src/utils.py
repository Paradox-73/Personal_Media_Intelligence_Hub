import requests
from PIL import Image
from io import BytesIO
import re
import os
import time
import json # For handling JSON responses
from dotenv import load_dotenv

load_dotenv()

# --- API Configuration Placeholders ---
# IMPORTANT: Replace these with your actual API keys.
# You can get them from their respective websites.
RAWG_API_KEY = os.getenv("RAWG_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")
TVMAZE_BASE_URL = "https://api.tvmaze.com"
# Spotify API Keys - Use the provided CLIENT_ID and CLIENT_SECRET for consistency
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:5173/callback' # Not directly used in current app, but kept for context
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

RAWG_BASE_URL = "https://api.rawg.io/api"
OMDB_BASE_URL = "http://www.omdbapi.com/"
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1"
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_URL = "https://api.spotify.com/v1"

# --- Generic Utility Functions ---

def download_image_from_url(url: str) -> Image.Image | None:
    """
    Downloads an image from a given URL and returns it as a PIL Image object.

    Args:
        url (str): The URL of the image.

    Returns:
        PIL.Image.Image | None: The downloaded image as a PIL Image, or None if an error occurs.
    """
    if not url:
        return None
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        with BytesIO(response.content) as buffer:
            image = Image.open(buffer).convert("RGB")
        return image
    except Exception as e:
        print(f"Image download failed from {url}: {e}")
        return None

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning: lowercasing, removing punctuation and extra whitespace.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove punctuation and special characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space
    return text

def ensure_directory_exists(path: str):
    """
    Ensures that a directory exists. If not, it creates it.

    Args:
        path (str): The directory path to check/create.
    """
    os.makedirs(path, exist_ok=True)
    print(f"Ensured directory exists: {path}")

# --- API Specific Functions (Placeholder for now) ---

# RAWG (Games)
def _make_rawg_request(endpoint: str, params: dict = None) -> dict | None:
    """Helper to make requests to RAWG API."""
    if not RAWG_API_KEY or RAWG_API_KEY == "YOUR_RAWG_API_KEY":
        print("RAWG API key not set. Cannot make requests.")
        return None
    full_url = f"{RAWG_BASE_URL}/{endpoint}"
    all_params = {"key": RAWG_API_KEY}
    if params:
        all_params.update(params)
    try:
        response = requests.get(full_url, params=all_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"RAWG API request failed for {endpoint}: {e}")
        return None

def search_game_rawg(query: str) -> list[dict]:
    """Searches for games on RAWG.io."""
    print(f"Searching RAWG for: {query}")
    response_data = _make_rawg_request("games", {"search": query, "page_size": 5})
    if response_data and 'results' in response_data:
        return response_data['results']
    return []

def get_game_details_rawg(game_id: int) -> dict | None:
    """Fetches detailed information for a specific game from RAWG.io."""
    if not game_id:
        return None
    print(f"Fetching details for game ID: {game_id}")
    return _make_rawg_request(f"games/{game_id}")

# OMDB (Movies)
def _make_omdb_request(params: dict = None) -> dict | None:
    """Helper to make requests to OMDB API."""
    if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_OMDB_API_KEY":
        print("OMDB API key not set. Cannot make requests.")
        return None
    all_params = {"apikey": OMDB_API_KEY}
    if params:
        all_params.update(params)
    try:
        response = requests.get(OMDB_BASE_URL, params=all_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"OMDB API request failed: {e}")
        return None

def search_movie_omdb(query: str) -> list[dict]:
    """Searches for movies on OMDB."""
    print(f"Searching OMDB for: {query}")
    response_data = _make_omdb_request({"s": query, "type": "movie"})
    if response_data and response_data.get('Response') == 'True' and 'Search' in response_data:
        return response_data['Search']
    return []

def get_movie_details_omdb(imdb_id: str) -> dict | None:
    """Fetches detailed information for a specific movie from OMDB."""
    if not imdb_id:
        return None
    print(f"Fetching details for movie ID: {imdb_id}")
    response_data = _make_omdb_request({"i": imdb_id, "plot": "full"})
    if response_data and response_data.get('Response') == 'True':
        return response_data
    return None

# TVmaze (Shows)
def _make_tvmaze_request(endpoint: str, params: dict = None) -> dict | None:
    """Helper to make requests to TVmaze API."""
    full_url = f"{TVMAZE_BASE_URL}/{endpoint}"
    try:
        response = requests.get(full_url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"TVmaze API request failed for {endpoint}: {e}")
        return None

def search_show_tvmaze(query: str) -> list[dict]:
    """Searches for shows on TVmaze."""
    print(f"Searching TVmaze for: {query}")
    response_data = _make_tvmaze_request("search/shows", {"q": query})
    if response_data:
        # TVmaze search returns a list of dicts, each with 'score' and 'show'
        return [item['show'] for item in response_data if 'show' in item]
    return []

def get_show_details_tvmaze(tvmaze_id: int) -> dict | None:
    """Fetches detailed information for a specific show from TVmaze."""
    if not tvmaze_id:
        return None
    print(f"Fetching details for show ID: {tvmaze_id}")
    return _make_tvmaze_request(f"shows/{tvmaze_id}")

# Spotify (Music)
def _get_spotify_access_token():
    """Gets a Spotify access token using Client Credentials Flow."""
    if not SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_ID == "YOUR_SPOTIPY_CLIENT_ID" or \
       not SPOTIPY_CLIENT_SECRET or SPOTIPY_CLIENT_SECRET == "YOUR_SPOTIPY_CLIENT_SECRET":
        print("Spotify API client ID or secret not set. Cannot get access token.")
        return None

    auth_response = requests.post(
        SPOTIFY_AUTH_URL,
        data={
            'grant_type': 'client_credentials',
            'client_id': SPOTIPY_CLIENT_ID,
            'client_secret': SPOTIPY_CLIENT_SECRET,
        },
        timeout=10
    )
    auth_response.raise_for_status()
    auth_data = auth_response.json()
    return auth_data['access_token']

def _make_spotify_request(endpoint: str, params: dict = None) -> dict | None:
    """Helper to make requests to Spotify API."""
    token = _get_spotify_access_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    full_url = f"{SPOTIFY_API_URL}/{endpoint}"
    try:
        response = requests.get(full_url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Spotify API request failed for {endpoint}: {e}")
        return None

def search_music_spotify(query: str, type: str = "track") -> list[dict]:
    """
    Searches for music on Spotify (tracks, artists, albums).
    If searching for album/artist, attempts to return associated top tracks for prediction.
    """
    print(f"Searching Spotify for: {query} (type: {type})")
    response_data = _make_spotify_request("search", {"q": query, "type": type, "limit": 5})

    results = []
    if response_data:
        if type == "track" and 'tracks' in response_data and 'items' in response_data['tracks']:
            results = response_data['tracks']['items']
        elif type == "album" and 'albums' in response_data and 'items' in response_data['albums']:
            # If searching for album, get tracks from the first album found
            album_id = response_data['albums']['items'][0].get('id')
            if album_id:
                album_tracks_data = _make_spotify_request(f"albums/{album_id}/tracks", {"limit": 5})
                if album_tracks_data and 'items' in album_tracks_data:
                    # For each track in the album, fetch full track details to get album info etc.
                    track_ids = [t['id'] for t in album_tracks_data['items']]
                    for track_id in track_ids:
                        track_detail = get_track_details_spotify(track_id)
                        if track_detail:
                            results.append(track_detail)
        elif type == "artist" and 'artists' in response_data and 'items' in response_data['artists']:
            # If searching for artist, get top tracks for the first artist found
            artist_id = response_data['artists']['items'][0].get('id')
            if artist_id:
                artist_top_tracks = _make_spotify_request(f"artists/{artist_id}/top-tracks", {"market": "US"}) # market is required
                if artist_top_tracks and 'tracks' in artist_top_tracks:
                    # Spotify top tracks API already returns full track objects
                    results = artist_top_tracks['tracks'][:5] # Limit to top 5
    return results

def get_track_details_spotify(track_id: str) -> dict | None:
    """Fetches detailed information for a specific track from Spotify."""
    if not track_id:
        return None
    print(f"Fetching track details for ID: {track_id}")
    # Also get audio features for the track
    track_details = _make_spotify_request(f"tracks/{track_id}")
    audio_features = _make_spotify_request(f"audio-features/{track_id}")
    if track_details and audio_features:
        # Merge audio features into track details for easier access
        track_details.update(audio_features)
    return track_details

def get_album_details_spotify(album_id: str) -> dict | None:
    """Fetches detailed information for a specific album from Spotify."""
    if not album_id:
        return None
    print(f"Fetching album details for ID: {album_id}")
    return _make_spotify_request(f"albums/{album_id}")

def get_artist_details_spotify(artist_id: str) -> dict | None:
    """Fetches detailed information for a specific artist from Spotify."""
    if not artist_id:
        return None
    print(f"Fetching artist details for ID: {artist_id}")
    return _make_spotify_request(f"artists/{artist_id}")


# Google Books
def _make_google_books_request(endpoint: str = "volumes", params: dict = None) -> dict | None:
    """Helper to make requests to Google Books API."""
    if not GOOGLE_BOOKS_API_KEY or GOOGLE_BOOKS_API_KEY == "YOUR_GOOGLE_BOOKS_API_KEY":
        print("Google Books API key not set. Cannot make requests.")
        return None
    full_url = f"{GOOGLE_BOOKS_BASE_URL}/{endpoint}"
    all_params = {"key": GOOGLE_BOOKS_API_KEY}
    if params:
        all_params.update(params)
    try:
        response = requests.get(full_url, params=all_params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Google Books API request failed for {endpoint}: {e}")
        return None

def search_book_google_books(query: str) -> list[dict]:
    """Searches for books on Google Books."""
    print(f"Searching Google Books for: {query}")
    response_data = _make_google_books_request(params={"q": query, "maxResults": 5})
    if response_data and 'items' in response_data:
        return response_data['items']
    return []

def get_book_details_google_books(volume_id: str) -> dict | None:
    """Fetches detailed information for a specific book from Google Books."""
    if not volume_id:
        return None
    print(f"Fetching book details for ID: {volume_id}")
    return _make_google_books_request(endpoint=f"volumes/{volume_id}")

# --- Generic Search and Detail Fetching ---

def search_content(content_type: str, query: str) -> list[dict]:
    """
    Generic search function that dispatches to the correct API based on content type.
    """
    if content_type == "Game":
        return search_game_rawg(query)
    elif content_type == "Show":
        return search_show_tvmaze(query)
    elif content_type == "Movie":
        return search_movie_omdb(query)
    elif content_type == "Music":
        # For music, we search for tracks, albums, and artists, then combine results
        # and prioritize showing tracks for prediction.
        track_results = search_music_spotify(query, type="track")
        album_results_tracks = search_music_spotify(query, type="album")
        artist_results_tracks = search_music_spotify(query, type="artist")
        
        # Combine all tracks found from various searches
        all_music_results = []
        # Add a 'source_type' to differentiate if needed in UI, though for prediction, they are all tracks
        all_music_results.extend([item for item in track_results])
        all_music_results.extend([item for item in album_results_tracks])
        all_music_results.extend([item for item in artist_results_tracks])
        
        # Remove duplicates based on track ID
        unique_tracks = {track['id']: track for track in all_music_results if 'id' in track}
        return list(unique_tracks.values())

    elif content_type == "Book":
        return search_book_google_books(query)
    else:
        print(f"Unsupported content type for search: {content_type}")
        return []

def get_content_details(content_type: str, content_id: str | int) -> dict | None:
    """
    Generic function to fetch details that dispatches to the correct API based on content type.
    For Music, it always fetches track details as the prediction is track-based.
    """
    if content_type == "Game":
        return get_game_details_rawg(content_id)
    elif content_type == "Show":
        return get_show_details_tvmaze(content_id)
    elif content_type == "Movie":
        return get_movie_details_omdb(content_id)
    elif content_type == "Music":
        # For music, we always get track details, as the model predicts track ratings
        return get_track_details_spotify(content_id)
    elif content_type == "Book":
        return get_book_details_google_books(content_id)
    else:
        print(f"Unsupported content type for details: {content_type}")
        return None
