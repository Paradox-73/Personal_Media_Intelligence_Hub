import pandas as pd                                                                 # Imports the pandas library, used for creating and working with data tables (DataFrames).
import os                                                                           # Imports the os library, which allows the script to interact with the operating system (e.g., file paths).
from typing import Dict, Any, cast
from pandas import Series

# This large dictionary is the main configuration for the entire data loading process.
# For each content type (like "Game", "Show", etc.), it defines:
# - Where to find the raw data (csv_paths).
# - What the columns are named in the original files.
# - Which columns are numerical, which are text, and which are categories.
# - How to handle missing data.
# - The name of the column that we want to predict ('my_rating').
CONTENT_COLUMN_MAPPING: Dict[str, Dict[str, Any]] = {
    "Game": {                                                                       # Configuration for game data.
        "csv_paths": [os.path.join('data', 'games_data.csv')],                       # Specifies the CSV file containing game data.
        "id": None,                                                                 # No ID column in the CSV; will be found later using an API.
        "title": "name",                                                            # The 'title' in our program corresponds to the 'name' column in the CSV.
        "description": "description_raw",                                           # The 'description' corresponds to the 'description_raw' column.
        "genres": "genres",                                                         # The 'genres' correspond to the 'genres' column.
        "image_url": "cover",                                                       # The 'image_url' corresponds to the 'cover' column.
        "my_rating_source_col": "my_rating",                                        # The column in the CSV that contains the rating we want to predict.
        "my_rating": "my_rating",                                                   # The standard name we'll use in our program for the rating.
        "release_date": "released",                                                 # The 'release_date' corresponds to the 'released' column.
        "numerical_features": ["metacritic", "rating", "ratings_count", "reviews_count"], # Lists all columns that contain numbers.
        "categorical_features": ["platform_from_text", "age_rating", "developers", "publishers"], # Lists all columns that contain categories (text labels).
        "text_features": ["description_raw", "tags", "name"],                       # Lists all columns that contain free-form text.
        "date_features": ["released"],                                              # Lists all columns that contain dates.
        "fill_na_numerical_strategy": "median",                                     # If a number is missing, fill it with the median of that column.
        "fill_na_categorical_strategy": "Unknown",                                  # If a category is missing, fill it with the word "Unknown".
        "my_rating_threshold": 4.0,                                                 # A rating of 4.0 or higher will be considered a "like".
    },
    "Show": {                                                                       # Configuration for TV show data.
        "csv_paths": [os.path.join('data', 'shows_data.csv')],                       # Specifies the CSV file for show data.
        "id": "tvmaze_id",                                                          # The 'id' corresponds to the 'tvmaze_id' column.
        "title": "title",                                                           # The 'title' corresponds to the 'title' column.
        "description": "summary",                                                   # The 'description' corresponds to the 'summary' column.
        "genres": "genres",                                                         # The 'genres' correspond to the 'genres' column.
        "image_url": None,                                                          # No image URL column in this CSV.
        "my_rating_source_col": "my_rating",                                        # The source column for our target rating.
        "my_rating": "my_rating",                                                   # The standard name for our target rating.
        "release_date": "premiered",                                                # The 'release_date' corresponds to the 'premiered' column.
        "numerical_features": ["runtime", "average_runtime", "rating_avg", "popularity", "watch_count", "episode_count"], # Lists numerical columns.
        "categorical_features": ["language", "status", "show_type", "network_country"], # Lists categorical columns.
        "text_features": ["summary", "genres", "title", "cast", "characters", "crew"], # Lists free-form text columns.
        "date_features": ["premiered"],                                             # Lists date columns.
        "fill_na_numerical_strategy": "median",                                     # Use median for missing numbers.
        "fill_na_categorical_strategy": "Unknown",                                  # Use "Unknown" for missing categories.
        "my_rating_threshold": 4.0,                                                 # Ratings >= 4.0 are a "like".
    },
    "Movie": {                                                                      # Configuration for movie data.
        "csv_paths": [os.path.join('data', 'movies_data.csv')],                      # Specifies the CSV file for movie data.
        "id": "imdb_id",                                                            # The 'id' corresponds to the 'imdb_id' column.
        "title": "title",                                                           # 'title' -> 'title'.
        "description": "plot",                                                      # 'description' -> 'plot'.
        "genres": "genre",                                                          # 'genres' -> 'genre'.
        "image_url": "poster",                                                      # 'image_url' -> 'poster'.
        "my_rating_source_col": "my_rating",                                        # Source for the target rating.
        "my_rating": "my_rating",                                                   # Standard name for the target rating.
        "release_date": "released",                                                 # 'release_date' -> 'released'.
        "numerical_features": ["year", "imdb_rating", "metascore"],                 # Lists numerical columns.
        "categorical_features": ["rated", "language", "director", "writer"],        # Lists categorical columns.
        "text_features": ["plot", "genre", "title", "actors", "awards"],            # Lists free-form text columns.
        "date_features": ["released"],                                              # Lists date columns.
        "fill_na_numerical_strategy": "median",                                     # Use median for missing numbers.
        "fill_na_categorical_strategy": "Unknown",                                  # Use "Unknown" for missing categories.
        "my_rating_threshold": 4.0,                                                 # Ratings >= 4.0 are a "like".
    },
    "Music": {                                                                      # Configuration for music data.
        "csv_paths": [                                                              # Specifies multiple CSV files for music data.
            os.path.join('data', 'spotify_all_unique_tracks_combined.csv'),
            os.path.join('data', 'spotify_saved_albums.csv'),
            os.path.join('data', 'spotify_top_tracks.csv'),
        ],
        "id": "Track ID",                                                           # 'id' -> 'Track ID'.
        "title": "Track Name",                                                      # 'title' -> 'Track Name'.
        "description": "Album Name",                                                # 'description' -> 'Album Name' (used as a proxy).
        "genres": "Genre(s)",                                                       # 'genres' -> 'Genre(s)'.
        "image_url": "Album Image URL",                                             # 'image_url' -> 'Album Image URL'.
        "my_rating_source_col": "Popularity",                                       # The rating will be derived from the 'Popularity' column.
        "my_rating": "my_rating",                                                   # Standard name for the derived rating.
        "release_date": "Release Date",                                             # 'release_date' -> 'Release Date'.
        "numerical_features": ["Popularity", "Danceability", "Energy", "Loudness", "Speechiness", # Lists all numerical audio features from Spotify.
                               "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo",
                               "Duration (ms)", "Time Signature"],
        "categorical_features": ["Artist(s)", "Key", "Mode", "Genre(s)"],           # Lists categorical music features.
        "text_features": ["Track Name", "Album Name"],                              # Lists free-form text features.
        "date_features": ["Release Date", "Added At"],
        "fill_na_numerical_strategy": "median",                                     # Use median for missing numbers.
        "fill_na_categorical_strategy": "Unknown",                                  # Use "Unknown" for missing categories.
        "my_rating_threshold": 3.0,                                                 # A rating of 3.0 or higher is a "like" for music.
    },
    "Book": {                                                                       # Configuration for book data.
        "csv_paths": [os.path.join('data', 'books_data.csv')],                       # Specifies the CSV file for book data.
        "id": None,                                                                 # No ID column in the CSV.
        "title": "title",                                                           # 'title' -> 'title'.
        "description": "description",                                               # 'description' -> 'description'.
        "genres": "categories",                                                     # 'genres' -> 'categories'.
        "image_url": "thumbnail",                                                   # 'image_url' -> 'thumbnail'.
        "my_rating_source_col": "my_rating",                                        # Source for the target rating.
        "my_rating": "my_rating",                                                   # Standard name for the target rating.
        "release_date": "publishedDate",                                            # 'release_date' -> 'publishedDate'.
        "numerical_features": ["averageRating", "ratingsCount", "pageCount"],       # Lists numerical book features.
        "categorical_features": ["language", "authors", "publisher"],               # Lists categorical book features.
        "text_features": ["title", "description", "categories"],                    # Lists free-form text features.
        "date_features": ["publishedDate"],
        "fill_na_numerical_strategy": "median",                                     # Use median for missing numbers.
        "fill_na_categorical_strategy": "Unknown",                                  # Use "Unknown" for missing categories.
        "my_rating_threshold": 4.0,                                                 # Ratings >= 4.0 are a "like".
    },
}

def load_content_data(content_type: str) -> pd.DataFrame:                            # Defines a function to load and clean data for a specific content type.
    """
    Loads content data from CSV files and performs initial cleaning based on content type.

    Args:
        content_type (str): The type of content to load (e.g., "Game", "Show", "Movie", "Music", "Book").

    Returns:
        pd.DataFrame: The loaded and initially cleaned DataFrame for the specified content type.
                      Returns an empty DataFrame if the content type is not supported or data loading fails.
    """
    print(f"--- Starting data loading for {content_type} ---")                      # Prints a message to show which content type is being loaded.

    if content_type not in CONTENT_COLUMN_MAPPING:                                  # Checks if the requested content type is defined in our configuration.
        print(f"Error: Content type '{content_type}' is not supported.")            # If not, prints an error.
        return pd.DataFrame()                                                       # Returns an empty data table.

    config: Dict[str, Any] = CONTENT_COLUMN_MAPPING[content_type]                                   # Gets the configuration for the requested content type.
    all_dfs = []                                                                    # Creates an empty list to hold the data from each CSV file.

    # Load CSV files
    for filepath in config["csv_paths"]:
        try:
            df_part = pd.read_csv(filepath)
            all_dfs.append(df_part)
            print(f"Successfully loaded {filepath}")
        except FileNotFoundError:
            print(f"Warning: The file '{filepath}' was not found. Skipping.")
        except Exception as e:
            print(f"An error occurred while loading '{filepath}': {e}")

    if not all_dfs:
        print(f"No data loaded for content type: {content_type}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Initial DataFrame info for {content_type}:")
    df.info()
    print(f"\nInitial DataFrame head for {content_type}:")
    print(df.head())

    # --- Data Cleaning and Preprocessing ---

    # This section renames the columns from their original names in the CSV files
    # to the standard, generic names we use throughout the program (e.g., 'name' becomes 'title').
    rename_mapping = {}
    for generic_col, original_col_val in config.items():
        if isinstance(original_col_val, str) and original_col_val in df.columns:
            original_col = cast(str, original_col_val)
            if generic_col == "my_rating_source_col":
                rename_mapping[original_col] = "my_rating"
            else:
                rename_mapping[original_col] = generic_col

    df = df.rename(columns=rename_mapping).copy()
    print(f"Columns renamed: {rename_mapping}")

    # --- Custom Cleaning for Specific Columns ---
    if 'runtime' in df.columns and content_type == 'Movie':
        df['runtime'] = cast(Series[str], df['runtime']).str.extract(r'(\d+)').astype(float)
        print("Cleaned 'runtime' column for Movie data.")


    # This section handles dates, converting them from text into a proper date format.
    if "release_date" in df.columns:
        if content_type == "Show":
            df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce', dayfirst=True)
        elif content_type == "Movie":
            df["release_date"] = pd.to_datetime(df["release_date"], format='%d-%b-%y', errors='coerce')
        else:
            df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')

        df['release_year'] = df["release_date"].dt.year
        print(f"Converted 'release_date' to datetime and extracted 'release_year'.")

    # This section fills in missing values (empty cells) in the numerical columns.
    for col_original in config["numerical_features"]:
        col_in_df = rename_mapping.get(col_original, col_original)
        if col_in_df in df.columns:
            if df[col_in_df].isnull().any():
                if config["fill_na_numerical_strategy"] == "median":
                    median_val = df[col_in_df].median()
                    df[col_in_df] = df[col_in_df].fillna(median_val)
                    print(f"Imputed missing values in '{col_in_df}' (from original '{col_original}') with median: {median_val}")
                elif config["fill_na_numerical_strategy"] == "mean":
                    mean_val = df[col_in_df].mean()
                    df[col_in_df] = df[col_in_df].fillna(mean_val)
                    print(f"Imputed missing values in '{col_in_df}' (from original '{col_original}') with mean: {mean_val}")
                elif isinstance(config["fill_na_numerical_strategy"], (int, float)):
                    df[col_in_df] = df[col_in_df].fillna(config["fill_na_numerical_strategy"])
                    print(f"Imputed missing values in '{col_in_df}' (from original '{col_original}') with constant: {config['fill_na_numerical_strategy']}")
                else:
                    print(f"Warning: No imputation strategy defined for numerical column '{col_in_df}'. Leaving NaNs.")

    # This section fills in missing values for all text and categorical columns.
    all_source_text_and_categorical_cols = list(set(config["text_features"] + config["categorical_features"])) # Gets a list of all text-based columns.
    for col_original in all_source_text_and_categorical_cols:
        col_in_df = rename_mapping.get(col_original, col_original)
        if col_in_df in df.columns:
            if not isinstance(df[col_in_df].dtype, pd.CategoricalDtype):
                df[col_in_df] = df[col_in_df].astype(str).replace('nan', config["fill_na_categorical_strategy"])
                df[col_in_df] = df[col_in_df].replace('', config["fill_na_categorical_strategy"])
                df[col_in_df] = df[col_in_df].fillna(config["fill_na_categorical_strategy"])
            print(f"Filled missing values in '{col_in_df}' (from original '{col_original}') with '{config['fill_na_categorical_strategy']}'.")

    # This section does a final check to make sure all numerical columns are actually numbers.
    for col_original in config["numerical_features"]:
        col_in_df = rename_mapping.get(col_original, col_original)
        if col_in_df in df.columns:
            df[col_in_df] = pd.to_numeric(df[col_in_df], errors='coerce')
            if df[col_in_df].isnull().any():
                if config["fill_na_numerical_strategy"] == "median":
                    median_val = df[col_in_df].median()
                    df[col_in_df] = df[col_in_df].fillna(median_val)
                elif config["fill_na_numerical_strategy"] == "mean":
                    mean_val = df[col_in_df].mean()
                    df[col_in_df] = df[col_in_df].fillna(mean_val)
                elif isinstance(config["fill_na_numerical_strategy"], (int, float)):
                    df[col_in_df] = df[col_in_df].fillna(config["fill_na_numerical_strategy"])
                print(f"Re-imputed '{col_in_df}' after numeric coercion errors.")


    # This section creates our target variable, 'my_rating', which is what the model will learn to predict.
    # For Music, it's derived from 'Popularity'. For others, it's taken directly.
    if content_type == "Music":
        if "Popularity" in df.columns:
            # This formula scales the Popularity score (0-100) to our desired rating scale (1-5).
            min_popularity = 0
            max_popularity = 100
            new_min_rating = 1
            new_max_rating = 5
            df['my_rating'] = ((df['Popularity'] - min_popularity) / (max_popularity - min_popularity)) * \
                              (new_max_rating - new_min_rating) + new_min_rating
            df['my_rating'] = df['my_rating'].round(1)
            print("Derived 'my_rating' for Music from 'Popularity' (0-100) scaled to 1-5 range.")
        else:
            print("Warning: 'Popularity' column not found for Music. Cannot derive 'my_rating'.")

    if "my_rating" in df.columns:
        df["my_rating"] = pd.to_numeric(df["my_rating"], errors='coerce')
        original_rows = len(df)
        df.dropna(subset=["my_rating"], inplace=True)
        if len(df) < original_rows:
            print(f"Dropped {original_rows - len(df)} rows due to missing 'my_rating'.")

        # This creates a second target variable, 'like_dislike', for classification.
        my_rating_threshold = config.get("my_rating_threshold", 4.0)
        df['like_dislike'] = (df["my_rating"] >= my_rating_threshold).astype(int)
        print(f"Created 'like_dislike' target based on 'my_rating' >= {my_rating_threshold}.")

    else:
        # This warning is shown if, after all processing, there's no 'my_rating' column to use for training.
        print("Warning: Generic target column 'my_rating' not found in DataFrame for {content_type}. 'like_dislike' will not be created. Please check CONTENT_COLUMN_MAPPING and input CSVs for 'my_rating_source_col'.")


    print(f"\nDataFrame info after cleaning for {content_type}:")
    df.info()
    print(f"\nDataFrame head after cleaning for {content_type}:")
    print(df.head())

    return df
