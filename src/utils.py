import requests                                                                     # Imports the requests library, used for making web requests (e.g., to APIs).
from PIL import Image                                                               # Imports the Image module from Pillow (PIL), used for image processing.
from io import BytesIO                                                              # Imports BytesIO, used to handle binary data in memory, especially for images.
import re                                                                           # Imports re, the regular expression library, used for advanced text pattern matching.
import os                                                                           # Imports the os library, which allows the script to interact with the operating system (e.g., file paths).
import time                                                                         # Imports time, used for time-related functions (not directly used in this snippet, but often for delays).
import json                                                                         # Imports json, used for handling JSON (JavaScript Object Notation) data, common in web APIs.
from dotenv import load_dotenv                                                      # Imports load_dotenv from dotenv, used to load environment variables from a .env file.

load_dotenv()                                                                       # Loads environment variables from a .env file. This is where API keys are stored securely.

# --- API Configuration Placeholders ---
# IMPORTANT: Replace these with your actual API keys.
# You can get them from their respective websites.
RAWG_API_KEY = os.getenv("RAWG_API_KEY")                                            # Retrieves the RAWG API key from environment variables.
OMDB_API_KEY = os.getenv("OMDB_API_KEY")                                            # Retrieves the OMDB API key from environment variables.
TVMAZE_BASE_URL = "https://api.tvmaze.com"                                          # Defines the base URL for the TVmaze API.
# Spotify API Keys - Use the provided CLIENT_ID and CLIENT_SECRET for consistency
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")                                  # Retrieves the Spotify Client ID from environment variables.
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")                          # Retrieves the Spotify Client Secret from environment variables.
REDIRECT_URI = 'http://localhost:5173/callback'                                     # Defines a redirect URI for Spotify (not used in this CLI app, but common for web apps).
GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")                            # Retrieves the Google Books API key from environment variables.

RAWG_BASE_URL = "https://api.rawg.io/api"                                           # Defines the base URL for the RAWG API.
OMDB_BASE_URL = "http://www.omdbapi.com/"                                           # Defines the base URL for the OMDB API.
GOOGLE_BOOKS_BASE_URL = "https://www.googleapis.com/books/v1"                       # Defines the base URL for the Google Books API.
SPOTIFY_AUTH_URL = "https://accounts.spotify.com/api/token"                         # Defines the Spotify authentication URL.
SPOTIFY_API_URL = "https://api.spotify.com/v1"                                      # Defines the base URL for the Spotify API.

# --- Generic Utility Functions ---

def download_image_from_url(url: str) -> Image.Image | None:                        # Defines a function to download an image from a URL.
    """
    Downloads an image from a given URL and returns it as a PIL Image object.

    Args:
        url (str): The URL of the image.

    Returns:
        PIL.Image.Image | None: The downloaded image as a PIL Image, or None if an error occurs.
    """
    if not url:                                                                     # Checks if the URL is empty.
        return None                                                                 # If so, returns None.
    try:                                                                            # Starts a 'try' block to handle potential errors during download.
        response = requests.get(url, stream=True, timeout=10)                       # Makes a web request to the URL, with a 10-second timeout.
        response.raise_for_status()                                                 # Checks if the request was successful (raises an error for bad responses).
        with BytesIO(response.content) as buffer:                                   # Creates an in-memory buffer from the image content.
            image = Image.open(buffer).convert("RGB")                               # Opens the image from the buffer and converts it to RGB format.
        return image                                                                # Returns the processed image.
    except Exception as e:                                                          # If any error occurs during download or processing...
        print(f"Image download failed from {url}: {e}")                            # ...print an error message.
        return None                                                                 # ...and returns None.

def clean_text(text: str) -> str:                                                   # Defines a function to clean up text.
    """
    Performs basic text cleaning: lowercasing, removing punctuation and extra whitespace.

    Args:
        text (str): The input text string.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):                                                   # Checks if the input is actually a string.
        return ""                                                                   # If not, returns an empty string.
    text = text.lower()                                                             # Converts all text to lowercase.
    text = re.sub(r'<.*?>', '', text)                                                # Removes any HTML tags (e.g., <b>, <p>).
    text = re.sub(r'[^a-z0-9\\s]', '', text)                                        # Removes any characters that are not letters, numbers, or spaces.
    text = re.sub(r'\\s+', ' ', text).strip()                                        # Replaces multiple spaces with a single space and removes leading/trailing spaces.
    return text                                                                     # Returns the cleaned text.

def ensure_directory_exists(path: str):                                             # Defines a function to ensure a directory exists.
    """
    Ensures that a directory exists. If not, it creates it.

    Args:
        path (str): The directory path to check/create.
    """
    os.makedirs(path, exist_ok=True)                                                # Creates the directory (and any necessary parent directories) if it doesn't exist.
    print(f"Ensured directory exists: {path}")                                     # Prints a confirmation message.

# --- API Specific Functions ---

# RAWG (Games)
def _make_rawg_request(endpoint: str, params: dict = None) -> dict | None:          # Helper function to make requests to the RAWG API.
    """Helper to make requests to RAWG API."""
    if not RAWG_API_KEY or RAWG_API_KEY == "YOUR_RAWG_API_KEY":                     # Checks if the RAWG API key is set.
        print("RAWG API key not set. Cannot make requests.")                        # If not, prints a warning.
        return None                                                                 # And returns None.
    full_url = f"{RAWG_BASE_URL}/{endpoint}"                                         # Constructs the full URL for the API request.
    all_params = {"key": RAWG_API_KEY}                                             # Starts with the API key as a parameter.
    if params:                                                                      # If additional parameters are provided...
        all_params.update(params)                                                   # ...add them to the main parameters.
    try:                                                                            # Starts a 'try' block to handle request errors.
        response = requests.get(full_url, params=all_params, timeout=10)            # Makes the GET request with parameters and a timeout.
        response.raise_for_status()                                                 # Checks for HTTP errors.
        return response.json()                                                      # Returns the response as a JSON dictionary.
    except requests.exceptions.RequestException as e:                               # Catches any request-related errors.
        print(f"RAWG API request failed for {endpoint}: {e}")                      # Prints an error message.
        return None                                                                 # Returns None.

def search_game_rawg(query: str) -> list[dict]:                                     # Defines a function to search for games on RAWG.io.
    """Searches for games on RAWG.io."""
    print(f"Searching RAWG for: {query}")                                          # Prints the search query.
    response_data = _make_rawg_request("games", {"search": query, "page_size": 5}) # Calls the helper to search for games, limiting to 5 results.
    if response_data and 'results' in response_data:                                # Checks if the response is valid and contains results.
        return response_data['results']                                             # Returns the list of game results.
    return []                                                                       # Returns an empty list if no results or an error.

def get_game_details_rawg(game_id: int) -> dict | None:                             # Defines a function to get detailed information for a specific game.
    """Fetches detailed information for a specific game from RAWG.io."""
    if not game_id:                                                                 # Checks if a game ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching details for game ID: {game_id}")                            # Prints the game ID being fetched.
    return _make_rawg_request(f"games/{game_id}")                                  # Calls the helper to get details for the specific game ID.

# OMDB (Movies)
def _make_omdb_request(params: dict = None) -> dict | None:                         # Helper function to make requests to the OMDB API.
    """Helper to make requests to OMDB API."""
    if not OMDB_API_KEY or OMDB_API_KEY == "YOUR_OMDB_API_KEY":                     # Checks if the OMDB API key is set.
        print("OMDB API key not set. Cannot make requests.")                        # If not, prints a warning.
        return None                                                                 # And returns None.
    all_params = {"apikey": OMDB_API_KEY}                                           # Starts with the API key.
    if params:                                                                      # If additional parameters are provided...
        all_params.update(params)                                                   # ...add them.
    try:                                                                            # Starts a 'try' block.
        response = requests.get(OMDB_BASE_URL, params=all_params, timeout=10)       # Makes the GET request.
        response.raise_for_status()                                                 # Checks for HTTP errors.
        return response.json()                                                      # Returns the JSON response.
    except requests.exceptions.RequestException as e:                               # Catches request errors.
        print(f"OMDB API request failed: {e}")                                     # Prints an error message.
        return None                                                                 # Returns None.

def search_movie_omdb(query: str) -> list[dict]:                                    # Defines a function to search for movies on OMDB.
    """Searches for movies on OMDB."""
    print(f"Searching OMDB for: {query}")                                          # Prints the search query.
    response_data = _make_omdb_request({"s": query, "type": "movie"})              # Calls the helper to search for movies.
    if response_data and response_data.get('Response') == 'True' and 'Search' in response_data: # Checks if the response is successful and contains search results.
        return response_data['Search']                                              # Returns the list of movie results.
    return []                                                                       # Returns an empty list if no results or an error.

def get_movie_details_omdb(imdb_id: str) -> dict | None:                            # Defines a function to get detailed information for a specific movie.
    """Fetches detailed information for a specific movie from OMDB."""
    if not imdb_id:                                                                 # Checks if an IMDb ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching details for movie ID: {imdb_id}")                             # Prints the movie ID being fetched.
    response_data = _make_omdb_request({"i": imdb_id, "plot": "full"})             # Calls the helper to get full plot details for the specific IMDb ID.
    if response_data and response_data.get('Response') == 'True':                 # Checks if the response is successful.
        return response_data                                                        # Returns the movie details.
    return None                                                                     # Returns None if an error.

# TVmaze (Shows)
def _make_tvmaze_request(endpoint: str, params: dict = None) -> dict | None:        # Helper function to make requests to the TVmaze API.
    """Helper to make requests to TVmaze API."""
    full_url = f"{TVMAZE_BASE_URL}/{endpoint}"                                       # Constructs the full URL.
    try:                                                                            # Starts a 'try' block.
        response = requests.get(full_url, params=params, timeout=10)                # Makes the GET request.
        response.raise_for_status()                                                 # Checks for HTTP errors.
        return response.json()                                                      # Returns the JSON response.
    except requests.exceptions.RequestException as e:                               # Catches request errors.
        print(f"TVmaze API request failed for {endpoint}: {e}")                    # Prints an error message.
        return None                                                                 # Returns None.

def search_show_tvmaze(query: str) -> list[dict]:                                   # Defines a function to search for shows on TVmaze.
    """Searches for shows on TVmaze."""
    print(f"Searching TVmaze for: {query}")                                        # Prints the search query.
    response_data = _make_tvmaze_request("search/shows", {"q": query})             # Calls the helper to search for shows.
    if response_data:                                                               # Checks if there's a response.
        return [item['show'] for item in response_data if 'show' in item]           # Extracts the 'show' data from each item in the response.
    return []                                                                       # Returns an empty list if no results.

def get_show_details_tvmaze(tvmaze_id: int) -> dict | None:                         # Defines a function to get detailed information for a specific show.
    """Fetches detailed information for a specific show from TVmaze."""
    if not tvmaze_id:                                                               # Checks if a TVmaze ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching details for show ID: {tvmaze_id}")                            # Prints the show ID being fetched.
    return _make_tvmaze_request(f"shows/{tvmaze_id}")                              # Calls the helper to get details for the specific show ID.

# Spotify (Music)
def _get_spotify_access_token():                                                    # Defines a function to get an access token for the Spotify API.
    """Gets a Spotify access token using Client Credentials Flow."""
    if not SPOTIPY_CLIENT_ID or SPOTIPY_CLIENT_ID == "YOUR_SPOTIPY_CLIENT_ID" or \
       not SPOTIPY_CLIENT_SECRET or SPOTIPY_CLIENT_SECRET == "YOUR_SPOTIPY_CLIENT_SECRET": # Checks if Spotify API credentials are set.
        print("Spotify API client ID or secret not set. Cannot get access token.")  # If not, prints a warning.
        return None                                                                 # Returns None.

    auth_response = requests.post(                                                  # Makes a POST request to the Spotify authentication URL.
        SPOTIFY_AUTH_URL,
        data={
            'grant_type': 'client_credentials',                                     # Specifies the type of authorization flow.
            'client_id': SPOTIPY_CLIENT_ID,
            'client_secret': SPOTIPY_CLIENT_SECRET,
        },
        timeout=10
    )
    auth_response.raise_for_status()                                                # Checks for HTTP errors.
    auth_data = auth_response.json()                                                # Parses the JSON response.
    return auth_data['access_token']                                                # Returns the access token.

def _make_spotify_request(endpoint: str, params: dict = None) -> dict | None:       # Helper function to make requests to the Spotify API.
    """Helper to make requests to Spotify API."""
    token = _get_spotify_access_token()                                             # Gets an access token.
    if not token:                                                                   # If no token is obtained...
        return None                                                                 # ...returns None.
    headers = {"Authorization": f"Bearer {token}"}                                # Sets the authorization header with the token.
    full_url = f"{SPOTIFY_API_URL}/{endpoint}"                                       # Constructs the full URL.
    try:                                                                            # Starts a 'try' block.
        response = requests.get(full_url, headers=headers, params=params, timeout=10) # Makes the GET request.
        response.raise_for_status()                                                 # Checks for HTTP errors.
        return response.json()                                                      # Returns the JSON response.
    except requests.exceptions.RequestException as e:                               # Catches request errors.
        print(f"Spotify API request failed for {endpoint}: {e}")                   # Prints an error message.
        return None                                                                 # Returns None.

def search_music_spotify(query: str, type: str = "track") -> list[dict]:          # Defines a function to search for music on Spotify.
    """
    Searches for music on Spotify (tracks, artists, albums).
    If searching for album/artist, attempts to return associated top tracks for prediction.
    """
    print(f"Searching Spotify for: {query} (type: {type})")                        # Prints the search query and type.
    response_data = _make_spotify_request("search", {"q": query, "type": type, "limit": 5}) # Calls the helper to search Spotify.

    results = []                                                                    # Initializes an empty list to store results.
    if response_data:                                                               # If there's a response...
        if type == "track" and 'tracks' in response_data and 'items' in response_data['tracks']:# If searching for tracks and results are found...
            results = response_data['tracks']['items']                              # ...get the track items.
        elif type == "album" and 'albums' in response_data and 'items' in response_data['albums']:# If searching for albums and results are found...
            album_id = response_data['albums']['items'][0].get('id')               # ...get the ID of the first album.
            if album_id:                                                            # If an album ID is found...
                album_tracks_data = _make_spotify_request(f"albums/{album_id}/tracks", {"limit": 5}) # ...get tracks from that album.
                if album_tracks_data and 'items' in album_tracks_data:              # If tracks are found...
                    track_ids = [t['id'] for t in album_tracks_data['items']]       # ...get their IDs.
                    for track_id in track_ids:                                      # For each track ID...
                        track_detail = get_track_details_spotify(track_id)          # ...get full track details.
                        if track_detail:                                            # If details are found...
                            results.append(track_detail)                            # ...add them to the results list.
        elif type == "artist" and 'artists' in response_data and 'items' in response_data['artists']:# If searching for artists and results are found...
            artist_id = response_data['artists']['items'][0].get('id')             # ...get the ID of the first artist.
            if artist_id:                                                           # If an artist ID is found...
                artist_top_tracks = _make_spotify_request(f"artists/{artist_id}/top-tracks", {"market": "US"}) # ...get their top tracks.
                if artist_top_tracks and 'tracks' in artist_top_tracks:             # If top tracks are found...
                    results = artist_top_tracks['tracks'][:5]                       # ...add them to the results list (limited to 5).
    return results                                                                  # Returns the collected results.

def get_track_details_spotify(track_id: str) -> dict | None:                       # Defines a function to get detailed information for a specific track.
    """
    Fetches detailed information for a specific track from Spotify.
    """
    if not track_id:                                                                # Checks if a track ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching track details for ID: {track_id}")                            # Prints the track ID being fetched.
    track_details = _make_spotify_request(f"tracks/{track_id}")                    # Gets basic track details.
    audio_features = _make_spotify_request(f"audio-features/{track_id}")           # Gets audio features for the track.
    if track_details and audio_features:                                            # If both details are found...
        track_details.update(audio_features)                                        # ...merge the audio features into the track details.
    return track_details                                                            # Returns the combined details.

def get_album_details_spotify(album_id: str) -> dict | None:                       # Defines a function to get detailed information for a specific album.
    """
    Fetches detailed information for a specific album from Spotify.
    """
    if not album_id:                                                                # Checks if an album ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching album details for ID: {album_id}")                            # Prints the album ID being fetched.
    return _make_spotify_request(f"albums/{album_id}")                             # Calls the helper to get album details.

def get_artist_details_spotify(artist_id: str) -> dict | None:                     # Defines a function to get detailed information for a specific artist.
    """
    Fetches detailed information for a specific artist from Spotify.
    """
    if not artist_id:                                                               # Checks if an artist ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching artist details for ID: {artist_id}")                          # Prints the artist ID being fetched.
    return _make_spotify_request(f"artists/{artist_id}")                           # Calls the helper to get artist details.


# Google Books
def _make_google_books_request(endpoint: str = "volumes", params: dict = None) -> dict | None: # Helper function to make requests to the Google Books API.
    """
    Helper to make requests to Google Books API.
    """
    if not GOOGLE_BOOKS_API_KEY or GOOGLE_BOOKS_API_KEY == "YOUR_GOOGLE_BOOKS_API_KEY": # Checks if the Google Books API key is set.
        print("Google Books API key not set. Cannot make requests.")                # If not, prints a warning.
        return None                                                                 # Returns None.
    full_url = f"{GOOGLE_BOOKS_BASE_URL}/{endpoint}"                                 # Constructs the full URL.
    all_params = {"key": GOOGLE_BOOKS_API_KEY}                                     # Starts with the API key.
    if params:                                                                      # If additional parameters are provided...
        all_params.update(params)                                                   # ...add them.
    try:                                                                            # Starts a 'try' block.
        response = requests.get(full_url, params=all_params, timeout=10)            # Makes the GET request.
        response.raise_for_status()                                                 # Checks for HTTP errors.
        return response.json()                                                      # Returns the JSON response.
    except requests.exceptions.RequestException as e:                               # Catches request errors.
        print(f"Google Books API request failed for {endpoint}: {e}")              # Prints an error message.
        return None                                                                 # Returns None.

def search_book_google_books(query: str) -> list[dict]:                             # Defines a function to search for books on Google Books.
    """
    Searches for books on Google Books.
    """
    print(f"Searching Google Books for: {query}")                                  # Prints the search query.
    response_data = _make_google_books_request(params={"q": query, "maxResults": 5}) # Calls the helper to search for books, limiting to 5 results.
    if response_data and 'items' in response_data:                                  # Checks if the response is valid and contains items.
        return response_data['items']                                               # Returns the list of book items.
    return []                                                                       # Returns an empty list if no results.

def get_book_details_google_books(volume_id: str) -> dict | None:                   # Defines a function to get detailed information for a specific book.
    """
    Fetches detailed information for a specific book from Google Books.
    """
    if not volume_id:                                                               # Checks if a volume ID is provided.
        return None                                                                 # If not, returns None.
    print(f"Fetching book details for ID: {volume_id}")                            # Prints the volume ID being fetched.
    return _make_google_books_request(endpoint=f"volumes/{volume_id}")             # Calls the helper to get details for the specific volume ID.

# --- Generic Search and Detail Fetching ---

def search_content(content_type: str, query: str) -> list[dict]:                    # Defines a generic function to search for content across different APIs.
    """
    Generic search function that dispatches to the correct API based on content type.
    """
    if content_type == "Game":                                                      # If the content type is "Game"...
        return search_game_rawg(query)                                              # ...call the RAWG game search function.
    elif content_type == "Show":                                                    # If the content type is "Show"...
        return search_show_tvmaze(query)                                            # ...call the TVmaze show search function.
    elif content_type == "Movie":                                                   # If the content type is "Movie"...
        return search_movie_omdb(query)                                             # ...call the OMDB movie search function.
    elif content_type == "Music":                                                   # If the content type is "Music"...
        # For music, we search for tracks, albums, and artists, then combine results
        # and prioritize showing tracks for prediction.
        track_results = search_music_spotify(query, type="track")                   # Search for tracks.
        album_results_tracks = search_music_spotify(query, type="album")            # Search for albums and get their tracks.
        artist_results_tracks = search_music_spotify(query, type="artist")          # Search for artists and get their top tracks.
        
        # Combine all tracks found from various searches
        all_music_results = []                                                      # Initialize a list to hold all music results.
        all_music_results.extend([item for item in track_results])                  # Add track results.
        all_music_results.extend([item for item in album_results_tracks])           # Add tracks from albums.
        all_music_results.extend([item for item in artist_results_tracks])          # Add tracks from artists.
        
        # Remove duplicates based on track ID
        unique_tracks = {track['id']: track for track in all_music_results if 'id' in track} # Use a dictionary to keep only unique tracks by ID.
        return list(unique_tracks.values())                                         # Return the unique tracks as a list.

    elif content_type == "Book":                                                    # If the content type is "Book"...
        return search_book_google_books(query)                                      # ...call the Google Books search function.
    else:                                                                           # For any unsupported content type...
        print(f"Unsupported content type for search: {content_type}")              # ...print an error message.
        return []                                                                   # ...and return an empty list.

def get_content_details(content_type: str, content_id: str | int) -> dict | None:   # Defines a generic function to get detailed content information.
    """
    Generic function to fetch details that dispatches to the correct API based on content type.
    For Music, it always fetches track details as the prediction is track-based.
    """
    if content_type == "Game":                                                      # If the content type is "Game"...
        return get_game_details_rawg(content_id)                                    # ...call the RAWG game details function.
    elif content_type == "Show":                                                    # If the content type is "Show"...
        return get_show_details_tvmaze(content_id)                                  # ...call the TVmaze show details function.
    elif content_type == "Movie":                                                   # If the content type is "Movie"...
        return get_movie_details_omdb(content_id)                                   # ...call the OMDB movie details function.
    elif content_type == "Music":                                                   # If the content type is "Music"...
        return get_track_details_spotify(content_id)                                # ...always call the Spotify track details function (as prediction is track-based).
    elif content_type == "Book":                                                    # If the content type is "Book"...
        return get_book_details_google_books(content_id)                            # ...call the Google Books details function.
    else:                                                                           # For any unsupported content type...
        print(f"Unsupported content type for details: {content_type}")             # ...print an error message.
        return None                                                                 # ...and return None.