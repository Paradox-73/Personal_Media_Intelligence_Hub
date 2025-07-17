import pandas as pd
import numpy as np
import os
import re

# This dictionary defines the mapping from generic column names to content-specific column names,
# and also specifies numerical, categorical, and text features for each content type.
# 'id' is for external API lookups. 'image_url' is for image feature extraction.
# 'my_rating' is the target variable.
CONTENT_COLUMN_MAPPING = {
    "Game": {
        "csv_paths": [os.path.join('data', 'games_data.csv')],
        "id": None, # rawg_id not directly in CSV, will need API search to get it
        "title": "name",
        "description": "description_raw",
        "genres": "genres",
        "image_url": "cover", # URL to game cover image
        "my_rating_source_col": "my_rating", # ORIGINAL COLUMN NAME for target
        "my_rating": "my_rating",
        "release_date": "released",
        "numerical_features": ["metacritic", "rating", "ratings_count", "reviews_count"],
        "categorical_features": ["platform_from_text", "age_rating"],
        "text_features": ["description_raw", "tags", "name", "developers", "publishers"],
        "date_features": ["released"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0, # For like/dislike classification
    },
    "Show": {
        "csv_paths": [os.path.join('data', 'shows_data.csv')],
        "id": "tvmaze_id",
        "title": "title",
        "description": "summary",
        "genres": "genres",
        "image_url": None, # No direct image URL in CSV, will need API lookup
        "my_rating_source_col": "my_rating", # ORIGINAL COLUMN NAME for target
        "my_rating": "my_rating",
        "release_date": "premiered",
        "numerical_features": ["runtime", "average_runtime", "rating_avg", "popularity", "watch_count", "episode_count"],
        "categorical_features": ["language", "status", "show_type"],
        "text_features": ["summary", "genres", "title", "cast", "characters", "crew"],
        "date_features": ["premiered"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0,
    },
    "Movie": {
        "csv_paths": [os.path.join('data', 'movies_data.csv')],
        "id": "imdb_id",
        "title": "title",
        "description": "plot",
        "genres": "genre",
        "image_url": "poster", # URL to movie poster
        "my_rating_source_col": "user_rating", # ORIGINAL COLUMN NAME for target
        "my_rating": "my_rating",
        "release_date": "released",
        "numerical_features": ["year", "imdb_rating", "metascore"], # imdb_votes, rotten_tomatoes_rating, box_office need careful conversion
        "categorical_features": ["rated", "language"],
        "text_features": ["plot", "genre", "title", "director", "writer", "actors", "awards"],
        "date_features": ["released"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0,
    },
    "Music": { # Assuming main data for training is 'all_tracks.csv' for simplicity, could be adapted
        "csv_paths": [
            os.path.join('data', 'spotify_all_unique_tracks_combined.csv'),
            os.path.join('data', 'spotify_saved_albums.csv'),
            os.path.join('data', 'spotify_top_tracks.csv'),
            # os.path.join('data', 'spotify_top_artists.csv'), # Artist data is different, skipping for track rating
        ],
        "id": "Track ID", # Original column for unique ID
        "title": "Track Name",
        "description": "Album Name", # Using Album Name as a proxy for description for text features
        "genres": "Genre(s)", # Assuming some genre info is available or can be derived
        "image_url": "Album Image URL", # Placeholder, might need to derive from album data if not direct
        "my_rating_source_col": "Popularity", # Using Popularity as the source for my_rating
        "my_rating": "my_rating", # The target column created internally
        "release_date": "Release Date",
        "numerical_features": ["Popularity", "Danceability", "Energy", "Loudness", "Speechiness",
                               "Acousticness", "Instrumentalness", "Liveness", "Valence", "Tempo",
                               "Duration (ms)", "Time Signature"], # Spotify audio features
        "categorical_features": ["Artist(s)", "Key", "Mode"], # Added Key and Mode
        "text_features": ["Track Name", "Artist(s)", "Album Name"], # Simplified, could include Genre(s) if robust
        "date_features": ["Release Date", "Added At"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 3.0, # Adjust threshold for 1-5 scale for music popularity
    },
    "Book": { # Placeholder for future book data
        "csv_paths": [os.path.join('data', 'books_data.csv')], # Assuming 'books_data.csv' exists
        "id": None, # Book IDs not directly in a generic CSV usually, will need API look up
        "title": "title",
        "description": "description",
        "genres": "categories",
        "image_url": "thumbnail", # Assuming a 'thumbnail' URL column for book covers
        "my_rating_source_col": "my_rating", # User needs to provide how 'my_rating' is derived for books for training
        "my_rating": "my_rating",
        "release_date": "publishedDate",
        "numerical_features": ["averageRating", "ratingsCount"],
        "categorical_features": ["language"],
        "text_features": ["title", "description", "authors", "categories", "publisher"],
        "date_features": ["publishedDate"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0, # Placeholder, will not be used if my_rating is None
    },
}

def load_content_data(content_type: str) -> pd.DataFrame:
    """
    Loads content data from CSV files and performs initial cleaning based on content type.

    Args:
        content_type (str): The type of content to load (e.g., "Game", "Show", "Movie", "Music", "Book").

    Returns:
        pd.DataFrame: The loaded and initially cleaned DataFrame for the specified content type.
                      Returns an empty DataFrame if the content type is not supported or data loading fails.
    """
    print(f"--- Starting data loading for {content_type} ---")

    if content_type not in CONTENT_COLUMN_MAPPING:
        print(f"Error: Content type '{content_type}' is not supported.")
        return pd.DataFrame()

    config = CONTENT_COLUMN_MAPPING[content_type]
    all_dfs = []

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

    # Map actual column names from CSVs to generic names used internally (e.g., 'user_rating' to 'my_rating')
    # Create a dictionary for renaming.
    rename_mapping = {}
    for generic_col, original_col in config.items():
        # Only map direct string columns that are actually present in the DataFrame and are not lists (features lists)
        if isinstance(original_col, str) and original_col in df.columns:
            # Handle special case for target column where we want to map its *source* name to 'my_rating'
            if generic_col == "my_rating_source_col":
                rename_mapping[original_col] = "my_rating"
            else:
                rename_mapping[original_col] = generic_col # e.g., 'name' -> 'title', 'summary' -> 'description'
    
    # Perform the renaming. Using .copy() to avoid SettingWithCopyWarning
    df = df.rename(columns=rename_mapping).copy()
    print(f"Columns renamed: {rename_mapping}")


    # Handle release dates - now 'release_date' should be the generic column name
    if "release_date" in df.columns:
        df["release_date"] = pd.to_datetime(df["release_date"], errors='coerce')
        df['release_year'] = df["release_date"].dt.year
        print(f"Converted 'release_date' to datetime and extracted 'release_year'.")

    # Impute missing numerical values
    # These lists now refer to the original column names from CONTENT_COLUMN_MAPPING
    # We need to map them to the potentially renamed columns in the dataframe for imputation.
    for col_original in config["numerical_features"]:
        col_in_df = rename_mapping.get(col_original, col_original) # Get renamed name or keep original if not renamed
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

    # Fill missing text/categorical fields
    all_source_text_and_categorical_cols = list(set(config["text_features"] + config["categorical_features"]))
    for col_original in all_source_text_and_categorical_cols:
        col_in_df = rename_mapping.get(col_original, col_original) # Get renamed name or keep original
        if col_in_df in df.columns:
            if not isinstance(df[col_in_df].dtype, pd.CategoricalDtype):
                df[col_in_df] = df[col_in_df].astype(str).replace('nan', config["fill_na_categorical_strategy"])
                df[col_in_df] = df[col_in_df].replace('', config["fill_na_categorical_strategy"])
                df[col_in_df] = df[col_in_df].fillna(config["fill_na_categorical_strategy"])
            print(f"Filled missing values in '{col_in_df}' (from original '{col_original}') with '{config['fill_na_categorical_strategy']}'.")

    # Ensure all numerical columns are indeed numeric
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


    # Define the 'like' target based on 'my_rating' (which is now the generic column name)
    # Check for the generic 'my_rating' column after the renaming process
    # Special handling for Music to convert 'Popularity' to 'my_rating' (1-5 scale)
    if content_type == "Music":
        if "Popularity" in df.columns:
            # Scale Popularity (0-100) to my_rating (1-5)
            # Using a simple linear scaling: (x - min) / (max - min) * (new_max - new_min) + new_min
            min_popularity = 0
            max_popularity = 100
            new_min_rating = 1
            new_max_rating = 5
            df['my_rating'] = ((df['Popularity'] - min_popularity) / (max_popularity - min_popularity)) * \
                              (new_max_rating - new_min_rating) + new_min_rating
            df['my_rating'] = df['my_rating'].round(1) # Round to one decimal place
            print(f"Derived 'my_rating' for Music from 'Popularity' (0-100) scaled to 1-5 range.")
        else:
            print(f"Warning: 'Popularity' column not found for Music. Cannot derive 'my_rating'.")
            # If popularity isn't there, we can't train for music.
            # The general 'my_rating' check below will handle dropping rows if it's still missing.
    
    if "my_rating" in df.columns:
        # Ensure 'my_rating' is numeric
        df["my_rating"] = pd.to_numeric(df["my_rating"], errors='coerce')
        # Drop rows where 'my_rating' is missing, as it's our target variable
        original_rows = len(df)
        df.dropna(subset=["my_rating"], inplace=True)
        if len(df) < original_rows:
            print(f"Dropped {original_rows - len(df)} rows due to missing 'my_rating'.")

        # Apply the like/dislike classification
        my_rating_threshold = config.get("my_rating_threshold", 4.0)
        df['like_dislike'] = (df["my_rating"] >= my_rating_threshold).astype(int)
        print(f"Created 'like_dislike' target based on 'my_rating' >= {my_rating_threshold}.")

        # FIX: Ensure negative examples from neg_shows.csv are explicitly marked as disliked (like_dislike = 0)
        # if content_type == "Show" and 'label' in df.columns:
        #     # For rows where 'label' is 0 (from neg_shows.csv), explicitly set 'like_dislike' to 0
        #     df.loc[df['label'] == 0, 'like_dislike'] = 0
        #     print("Adjusted 'like_dislike' for 'Show' content based on 'label' column (neg_shows.csv).")

    else:
        # This warning message is now more accurate
        print(f"Warning: Generic target column 'my_rating' not found in DataFrame for {content_type}. 'like_dislike' will not be created. Please check CONTENT_COLUMN_MAPPING and input CSVs for 'my_rating_source_col'.")


    print(f"\nDataFrame info after cleaning for {content_type}:")
    df.info()
    print(f"\nDataFrame head after cleaning for {content_type}:")
    print(df.head())

    return df