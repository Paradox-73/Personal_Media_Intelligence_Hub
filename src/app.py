#python src/app.py
import pandas as pd                                                                 # Imports the pandas library, used for creating and working with data tables (DataFrames).
import numpy as np                                                                  # Imports numpy, a library for numerical operations, especially with arrays.
import joblib                                                                       # Imports joblib, used for saving and loading Python objects, like our trained models.
import os                                                                           # Imports the os library, which allows the script to interact with the operating system (e.g., file paths).
import torch                                                                        # Imports PyTorch, a machine learning library, used here to manage which device (CPU or GPU) to use.
import sys                                                                          # Imports the sys library, which provides access to system-specific parameters and functions.
import traceback                                                                    # Imports traceback, used for printing detailed error information when something goes wrong.

# This block of code checks if the script is being run directly.
# If so, it adds the project's main folder to the list of places Python looks for files.
# This is important so we can import our own custom Python files from the 'src' directory.
if __name__ == "__main__":                                                           # Checks if this script is the main program being run.
    try:                                                                            # Starts a 'try' block to handle potential errors gracefully.
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the parent directory of this file to Python's path.
        from src.utils import (                                                     # Imports specific functions from our 'utils.py' file.
            clean_text,                                                             # A function to clean up text data.
            search_content,                                                         # A function to search for content online (e.g., movies, games).
            get_content_details,                                                    # A function to get detailed information about a piece of content.
            get_artist_details_spotify                                              # A function to get details about a music artist from Spotify.
        )
        from src.feature_extractor import FeatureExtractor                          # Imports the FeatureExtractor class, which turns data into numbers for the model.
        from src.data_loader import CONTENT_COLUMN_MAPPING                          # Imports a configuration dictionary that describes our data.
    except ImportError:                                                             # If any of the imports fail...
        print("Could not import custom modules. Please ensure the script is run from the correct directory structure.") # ...print an error message.
        sys.exit(1)                                                                 # ...and exit the program.


# --- Configuration ---
MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')            # Defines the main folder where all our trained models are stored.

# --- Helper Functions for Loading Models ---
def load_model_and_extractor(content_type: str):                                    # Defines a function to load a specific model (e.g., for movies).
    """
    Loads the trained XGBoost model and FeatureExtractor for a given content type.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, content_type.lower())                  # Sets the directory for the specific content type's model.
    xgboost_model_path = os.path.join(model_dir, f'xgboost_model_{content_type.lower()}.pkl') # Creates the full file path for the XGBoost model file.
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor')           # Creates the full file path for the feature extractor.

    if not os.path.exists(xgboost_model_path):                                      # Checks if the model file actually exists.
        print(f"❌ Error: Model file not found for {content_type}: {xgboost_model_path}. Please train the model first.") # If not, prints an error.
        return None, None                                                           # And returns nothing, indicating failure.
    if not os.path.exists(feature_extractor_path):                                  # Checks if the feature extractor files exist.
        print(f"❌ Error: Feature Extractor not found for {content_type}: {feature_extractor_path}. Please train the model first.") # If not, prints an error.
        return None, None                                                           # And returns nothing.

    try:                                                                            # Starts a 'try' block to handle potential loading errors.
        xgb_model = joblib.load(xgboost_model_path)                                 # Loads the saved XGBoost model from its file.
        feature_extractor = FeatureExtractor.load(feature_extractor_path)           # Loads the saved feature extractor.
        print(f"✅ Successfully loaded model and extractor for {content_type}.")      # Prints a success message.
        return xgb_model, feature_extractor                                         # Returns the loaded model and feature extractor.
    except Exception as e:                                                          # If any error occurs during loading...
        print(f"❌ Error loading model or feature extractor for {content_type}: {e}") # ...print an error message with details.
        return None, None                                                           # ...and return nothing.
    
def load_unified_models():                                                          # Defines a function to load the "unified" models (that work for all content types).
    """
    Loads the trained unified regression and classification models and their preprocessor.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, 'unified')                             # Sets the directory for the unified model.
    reg_model_path = os.path.join(model_dir, 'unified_regression_model.pkl')        # Creates the file path for the regression model (predicts a number).
    cls_model_path = os.path.join(model_dir, 'unified_classification_model.pkl')    # Creates the file path for the classification model (predicts a category).
    preprocessor_path = os.path.join(model_dir, 'unified_preprocessor.pkl')         # Creates the file path for the data preprocessor.

    if not os.path.exists(reg_model_path) or not os.path.exists(cls_model_path) or not os.path.exists(preprocessor_path): # Checks if any of the unified model files are missing.
        print(f"❌ Error: Unified model files not found in {model_dir}. Please train the unified models first.") # If so, prints an error.
        return None, None, None                                                     # And returns nothing.

    try:                                                                            # Starts a 'try' block for loading.
        reg_model = joblib.load(reg_model_path)                                     # Loads the regression model.
        cls_model = joblib.load(cls_model_path)                                     # Loads the classification model.
        preprocessor = joblib.load(preprocessor_path)                               # Loads the preprocessor.
        print("OK: Successfully loaded unified models and preprocessor.")           # Prints a success message.
        return reg_model, cls_model, preprocessor                                   # Returns all the loaded components.
    except Exception as e:                                                          # If an error occurs...
        print(f"ERROR: Error loading unified models: {e}")                          # ...print the error.
        return None, None, None                                                     # ...and return nothing.

def prepare_dataframe_for_prediction(details: dict, content_type: str) -> pd.DataFrame: # Defines a function to get data ready for the model to use.
    """
    Prepares a single content's details (from API response) into a Pandas DataFrame
    suitable for prediction. It standardizes the structure from various APIs
    to match the training data format.
    """
    print("\nGathering features for prediction...")                                 # Prints a status message.
    processed_details = {}                                                          # Creates an empty dictionary to hold the cleaned-up data.
    config = CONTENT_COLUMN_MAPPING[content_type]                                   # Gets the specific configuration for the given content type.

    # This section checks the content type and then pulls the relevant information
    # from the 'details' dictionary, giving it standard names the model understands.
    if content_type == "Movie":                                                     # If the content is a Movie...
        processed_details['title'] = details.get('Title')                           # ...get the title.
        processed_details['description'] = details.get('Plot')                      # ...get the plot description.
        processed_details['genres'] = details.get('Genre')                          # ...get the genres.
        processed_details['image_url'] = details.get('Poster')                      # ...get the URL for the movie poster image.
        processed_details['release_date'] = details.get('Released')                 # ...get the release date.
        processed_details['year'] = pd.to_numeric(details.get('Year'), errors='coerce') # ...get the year and make sure it's a number.
        processed_details['imdb_rating'] = pd.to_numeric(details.get('imdbRating'), errors='coerce') # ...get the IMDb rating and make sure it's a number.
        processed_details['metascore'] = pd.to_numeric(details.get('Metascore'), errors='coerce') # ...get the Metascore and make sure it's a number.
        processed_details['rated'] = details.get('Rated')                           # ...get the age rating (e.g., PG-13).
        processed_details['language'] = details.get('Language')                     # ...get the language.
        processed_details['director'] = details.get('Director')                     # ...get the director's name.
        processed_details['writer'] = details.get('Writer')                         # ...get the writer's name.
        processed_details['actors'] = details.get('Actors')                         # ...get the actors' names.
        processed_details['awards'] = details.get('Awards')                         # ...get any awards the movie has won.

    elif content_type == "Game":                                                    # Or if the content is a Game...
        processed_details['title'] = details.get('name')                            # ...get the title.
        processed_details['description'] = details.get('description_raw')           # ...get the raw text description.
        processed_details['image_url'] = details.get('cover')                       # ...get the URL for the game's cover image.
        processed_details['release_date'] = details.get('released')                 # ...get the release date.
        processed_details['metacritic'] = details.get('metacritic')                 # ...get the Metacritic score.
        processed_details['rating'] = details.get('rating')                         # ...get the user rating.
        processed_details['ratings_count'] = details.get('ratings_count')           # ...get how many ratings it has.
        processed_details['reviews_count'] = details.get('reviews_count')           # ...get how many reviews it has.
        processed_details['platform_from_text'] = ", ".join([p['platform']['name'] for p in details.get('platforms', [])]) # ...get the platforms (e.g., PC, PS5).
        processed_details['age_rating'] = details.get('esrb_rating', {}).get('name') if details.get('esrb_rating') else 'Not Rated' # ...get the ESRB age rating.
        processed_details['developers'] = ", ".join([d['name'] for d in details.get('developers', [])]) # ...get the developers.
        processed_details['publishers'] = ", ".join([p['name'] for p in details.get('publishers', [])]) # ...get the publishers.
        processed_details['tags'] = ", ".join([t['name'] for t in details.get('tags', [])]) # ...get the tags associated with the game.
        processed_details['genres'] = ", ".join([g['name'] for g in details.get('genres', [])]) # ...get the genres.

    elif content_type == "Show":                                                    # Or if the content is a TV Show...
        processed_details['title'] = details.get('name')                            # ...get the title.
        processed_details['description'] = details.get('summary')                   # ...get the summary.
        processed_details['image_url'] = details.get('image', {}).get('medium') if details.get('image') else None # ...get the URL for the show's image.
        processed_details['release_date'] = details.get('premiered')                # ...get the premiere date.
        processed_details['runtime'] = details.get('runtime')                       # ...get the runtime per episode.
        processed_details['average_runtime'] = details.get('averageRuntime')        # ...get the average runtime.
        processed_details['rating_avg'] = details.get('rating', {}).get('average') if details.get('rating') else None # ...get the average rating.
        processed_details['popularity'] = details.get('weight')                     # ...get the popularity score.
        processed_details['language'] = details.get('language')                     # ...get the language.
        processed_details['status'] = details.get('status')                         # ...get the status (e.g., Ended, Running).
        processed_details['show_type'] = details.get('type')                        # ...get the type of show (e.g., Scripted).
        processed_details['genres'] = "|".join(details.get('genres', []))           # ...get the genres.

    elif content_type == "Book":                                                    # Or if the content is a Book...
        vol_info = details.get('volumeInfo', {})                                   # ...get the 'volumeInfo' section which contains most details.
        processed_details['title'] = vol_info.get('title')                          # ...get the title.
        processed_details['description'] = vol_info.get('description')              # ...get the description.
        processed_details['image_url'] = vol_info.get('imageLinks', {}).get('thumbnail') if vol_info.get('imageLinks') else None # ...get the URL for the book cover thumbnail.
        processed_details['release_date'] = vol_info.get('publishedDate')           # ...get the publication date.
        processed_details['averageRating'] = vol_info.get('averageRating')          # ...get the average rating.
        processed_details['ratingsCount'] = vol_info.get('ratingsCount')            # ...get the number of ratings.
        processed_details['authors'] = ", ".join(vol_info.get('authors', []))       # ...get the authors.
        processed_details['publisher'] = vol_info.get('publisher')                  # ...get the publisher.
        processed_details['language'] = vol_info.get('language')                    # ...get the language.
        processed_details['genres'] = ", ".join(vol_info.get('categories', []))     # ...get the categories (genres).

    # Convert the processed dictionary to a single-row DataFrame
    df_pred = pd.DataFrame([processed_details])                                     # Turns the dictionary of details into a data table with one row.

    # This section makes sure that the data table has all the columns the model was trained on.
    # If a column is missing, it adds it and fills it with a default value.
    all_trained_cols = set(config.get('numerical_features', []) + config.get('categorical_features', []) + config.get('text_features', [])) # Gets a list of all feature columns.
    if config.get('image_url'):                                                     # If the model uses an image URL...
        all_trained_cols.add('image_url')                                           # ...add it to the list of columns.

    for col in all_trained_cols:                                                    # Loops through every column the model needs.
        if col not in df_pred.columns:                                              # If the column is missing from our data table...
            if col in config.get('numerical_features', []):                         # ...and it's supposed to be a number...
                df_pred[col] = 0.0                                                  # ...add the column and fill it with 0.
            else:                                                                   # ...otherwise (if it's text)...
                df_pred[col] = "Unknown"                                            # ...add the column and fill it with "Unknown".

    # Final cleanup for consistency
    for col in config.get('numerical_features', []):                                # Loops through all the number columns.
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(0.0)      # Ensures they are numbers and fills any empty spots with 0.
    for col in list(set(config.get('categorical_features', []) + config.get('text_features', []))): # Loops through all the text columns.
        df_pred[col] = df_pred[col].fillna("Unknown")                               # Fills any empty spots with "Unknown".
    if 'image_url' in df_pred.columns:                                              # If there's an image URL column...
        df_pred['image_url'] = df_pred['image_url'].fillna("")                      # ...fill any empty spots with an empty string.


    return df_pred                                                                  # Returns the prepared data table.

def prepare_dataframe_for_unified_prediction(details: dict, content_type: str) -> pd.DataFrame: # Defines a function to prepare data for the unified model.
    """
    Prepares a single content's details for the unified model.
    """
    df = prepare_dataframe_for_prediction(details, content_type)                    # First, runs the standard preparation function.
    df['content_type'] = content_type                                               # Adds a new column to specify the content type (e.g., 'Movie').

    # This section creates a new 'critic_rating_normalized' column.
    # It takes various rating scores (like Metacritic, IMDb) and puts them on a standard 1-10 scale.
    if content_type == 'Game':                                                      # If it's a game...
        df['critic_rating_normalized'] = df.get('metacritic', df.get('rating', 50)) / 10 # ...use Metacritic (0-100) or rating, scaled to 0-10.
    elif content_type == 'Show':                                                    # If it's a show...
        df['critic_rating_normalized'] = df.get('rating_avg', 5)                    # ...use the average rating (already 0-10).
    elif content_type == 'Movie':                                                   # If it's a movie...
        df['critic_rating_normalized'] = df.get('imdb_rating', df.get('metascore', 50) / 10) # ...use IMDb rating (0-10) or Metascore (0-100) scaled to 0-10.
    elif content_type == 'Music':                                                   # If it's music...
        df['critic_rating_normalized'] = df.get('my_rating', 50) / 10               # ...use 'my_rating' scaled to 0-10.
    elif content_type == 'Book':                                                    # If it's a book...
        df['critic_rating_normalized'] = df.get('averageRating', 2.5) * 2           # ...use the average rating (0-5) and multiply by 2 to scale it to 0-10.
    
    if 'critic_rating_normalized' in df.columns:                                    # If the normalized rating column exists...
        df['critic_rating_normalized'] = df['critic_rating_normalized'].fillna(5.0) # ...fill any empty spots with a neutral 5.0.
    else:                                                                           # Otherwise...
        df['critic_rating_normalized'] = 5.0                                        # ...create the column and set the value to 5.0.

    # Define the full set of features the unified model expects
    unified_text_features = ['title', 'description', 'genres']                      # Defines the text features the unified model uses.
    unified_categorical_features = [                                                # Defines the categorical (text category) features.
        'content_type', 'language', 'status', 'show_type', 'network_country', 
        'rated', 'director', 'writer', 'actors', 'awards', 'platform_from_text', 
        'age_rating', 'developers', 'publishers', 'tags', 'authors', 'publisher'
    ]
    unified_numerical_features = [                                                  # Defines the numerical features.
        'critic_rating_normalized', 'runtime', 'average_runtime', 'popularity', 
        'watch_count', 'episode_count', 'year', 'imdb_rating', 'metascore', 
        'ratings_count', 'reviews_count', 'pageCount'
    ]

    all_unified_cols = set(unified_text_features + unified_categorical_features + unified_numerical_features) # Combines all feature names into one set.

    # Ensure all expected columns are present, filling with defaults if necessary
    for col in all_unified_cols:                                                    # Loops through every column the unified model needs.
        if col not in df.columns:                                                   # If a column is missing...
            if col in unified_numerical_features:                                   # ...and it should be a number...
                df[col] = 0.0                                                       # ...add it and fill with 0.
            else:                                                                   # ...otherwise (if it's text)...
                df[col] = "Unknown"                                                 # ...add it and fill with "Unknown".

    # Final cleanup for consistency
    for col in unified_numerical_features:                                          # Loops through all numerical columns.
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)                # Makes sure they are numbers, filling errors/missing values with 0.
    for col in unified_text_features + unified_categorical_features:                # Loops through all text and category columns.
        if col in df.columns:                                                       # If the column exists...
            df[col] = df[col].fillna("Unknown")                                     # ...fill any missing values with "Unknown".

    return df                                                                       # Returns the prepared data table.

def get_user_input():                                                               # Defines a function to get input from the user in the terminal.
    """Gets content type and search query from the user."""
    content_type_options = list(CONTENT_COLUMN_MAPPING.keys())                      # Gets the list of available content types (Movie, Game, etc.).
    print("\n\n--- New Prediction ---")                                              # Prints a header for the user.
    print("1️⃣ Select a Content Type:")                                               # Asks the user to make a selection.
    for i, option in enumerate(content_type_options):                               # Loops through the content types.
        print(f"   {i+1}. {option}")                                                # Prints each one as a numbered option.

    while True:                                                                     # Starts a loop that continues until the user gives valid input.
        try:                                                                        # Starts a 'try' block to handle non-number inputs.
            choice_input = input("Enter the number of your choice: ")               # Asks the user to enter a number.
            if not choice_input: continue                                           # If the user just presses Enter, ask again.
            choice = int(choice_input) - 1                                          # Converts the user's text input to a number and subtracts 1 (because lists start at 0).
            if 0 <= choice < len(content_type_options):                             # Checks if the number is valid.
                selected_content_type = content_type_options[choice]                # Gets the selected content type from the list.
                break                                                               # Exits the loop because we have a valid choice.
            else:                                                                   # If the number is not in the list...
                print("Invalid choice. Please select a number from the list.")      # ...print an error message.
        except ValueError:                                                          # If the user enters text that isn't a number...
            print("Invalid input. Please enter a number.")                          # ...print an error message.

    search_query = input(f"\n2️⃣ Enter the name of the {selected_content_type.lower()} to search for: ") # Asks the user to type what they want to search for.
    return selected_content_type, search_query                                      # Returns the chosen content type and the search text.

def select_content_from_results(found_content, content_type):                       # Defines a function to let the user pick from search results.
    """Displays search results and prompts the user to select one."""
    print("\n3️⃣ Select the correct item from the search results:")                  # Prints a header for the user.
    content_options_display = []                                                    # Creates an empty list (not used, can be removed).
    content_id_map = {}                                                             # Creates a dictionary to link the menu number to the item's ID.

    for i, item in enumerate(found_content):                                        # Loops through each item found in the search.
        display_text = "Unknown Content"                                            # Sets a default display text.
        item_id = None                                                              # Sets a default item ID.

        # This section formats the display text differently for each content type.
        if content_type == "Game":                                                  # If it's a game...
            item_id = item.get('id')                                                # ...get the game's ID.
            year = item.get('released', 'N/A').split('-')[0] if item.get('released') else 'N/A' # ...get the release year.
            display_text = f"{item.get('name', 'N/A')} ({year})"                    # ...format it as "Name (Year)".
        elif content_type == "Show":                                                # If it's a show...
            item_id = item.get('id')                                                # ...get the show's ID.
            year = item.get('premiered', 'N/A').split('-')[0] if item.get('premiered') else 'N/A' # ...get the premiere year.
            display_text = f"{item.get('name', 'N/A')} ({year})"                    # ...format it as "Name (Year)".
        elif content_type == "Movie":                                               # If it's a movie...
            item_id = item.get('imdbID')                                            # ...get the movie's IMDb ID.
            display_text = f"{item.get('Title', 'N/A')} ({item.get('Year', 'N/A')})" # ...format it as "Title (Year)".
        elif content_type == "Music":                                               # If it's music...
            item_id = item.get('id')                                                # ...get the track's ID.
            artists = ", ".join([a['name'] for a in item.get('artists', [])])       # ...get the artist names.
            album = item.get('album',{}).get('name','N/A')                          # ...get the album name.
            year = item.get('album',{}).get('release_date','N/A').split('-')[0] if item.get('album',{}).get('release_date') else 'N/A' # ...get the release year.
            display_text = f"{item.get('name','N/A')} - {artists} (Album: {album}, {year})" # ...format it with all the details.
        elif content_type == "Book":                                                # If it's a book...
            item_id = item.get('id')                                                # ...get the book's ID.
            vol = item.get('volumeInfo',{})                                         # ...get the volume info.
            authors = ", ".join(vol.get('authors',[]))                              # ...get the author names.
            year = vol.get('publishedDate','N/A').split('-')[0] if vol.get('publishedDate') else 'N/A' # ...get the publication year.
            display_text = f"{vol.get('title','N/A')} by {authors} ({year})"        # ...format it as "Title by Authors (Year)".

        if item_id:                                                                 # If we successfully got an ID for the item...
            display_text_with_num = f"   {i + 1}. {display_text}"                    # ...create the numbered menu item text.
            print(display_text_with_num)                                            # ...print it for the user.
            content_id_map[i + 1] = (item_id, display_text)                         # ...store the ID and display text in our map, using the menu number as the key.

    if not content_id_map:                                                          # If the map is empty (no results found)...
        print("No valid results found to select.")                                  # ...tell the user.
        return None, None                                                           # ...and return nothing.

    while True:                                                                     # Starts a loop to get the user's selection.
        try:                                                                        # Starts a 'try' block to handle non-number inputs.
            choice_input = input("\nEnter the number of the content you want to predict: ") # Asks the user to pick a number.
            if not choice_input: continue                                           # If they press Enter, ask again.
            choice = int(choice_input)                                              # Convert the input to a number.
            if choice in content_id_map:                                            # If the number is a valid choice in our map...
                return content_id_map[choice]                                       # ...return the ID and display text for that choice.
            else:                                                                   # If it's not a valid number...
                print("Invalid choice. Please try again.")                          # ...show an error.
        except ValueError:                                                          # If they type something that's not a number...
            print("Invalid input. Please enter a number.")                          # ...show an error.


def main():                                                                         # The main function where the program's execution starts.
    """Main function to run the terminal-based predictor."""
    print("--- Universal Content Rating Predictor ---")                              # Prints a welcome message.
    print("(Press Ctrl+C at any time to exit)")                                     # Gives the user instructions on how to exit.

    unified_reg_model, unified_cls_model, unified_preprocessor = load_unified_models() # Loads the unified models at the start.

    while True:                                                                     # Starts the main loop of the program, which runs forever until exited.
        try:                                                                        # Starts a 'try' block to catch errors and keep the program running.
            selected_content_type, search_query = get_user_input()                  # Gets the content type and search query from the user.

            xgb_model, feature_extractor = load_model_and_extractor(selected_content_type) # Loads the specific model for the chosen content type.
            if not xgb_model or not feature_extractor:                              # If the model failed to load...
                continue                                                            # ...skip the rest of this loop and start over.

            print(f"\nSearching for '{search_query}'...")                           # Tells the user it's searching.
            found_content = search_content(selected_content_type, search_query)     # Performs the search using the function from utils.py.

            if not found_content:                                                   # If the search returned no results...
                print(f"No {selected_content_type.lower()} found for your search.") # ...tell the user.
                continue                                                            # ...and start the loop over.

            selected_content_id, selected_content_name = select_content_from_results(found_content, selected_content_type) # Shows the results and asks the user to pick one.
            if not selected_content_id:                                             # If the user didn't select anything...
                continue                                                            # ...start the loop over.

            print(f"\nFetching details for '{selected_content_name}'...")           # Tells the user it's getting details.
            details = get_content_details(selected_content_type, selected_content_id) # Gets the detailed information for the selected item.
            if not details:                                                         # If it couldn't get details...
                print("Could not fetch details for the selected content.")          # ...tell the user.
                continue                                                            # ...and start the loop over.

            # Content-specific prediction
            df_pred = prepare_dataframe_for_prediction(details, selected_content_type) # Prepares the detailed data for the specific model.
            print("⚙️  Running content-specific prediction...")                      # Prints a status message.
            if not df_pred.empty:                                                   # If the data table is not empty...
                combined_features = feature_extractor.transform(df_pred)            # ...use the feature extractor to turn the data into numbers.
                features_for_xgb = combined_features                                # ...assigns the features to a new variable.
                predicted_rating = xgb_model.predict(features_for_xgb)[0]           # ...use the specific model to predict a rating. The [0] gets the single number from the result.
                predicted_rating = max(1.0, min(5.0, predicted_rating))             # ...make sure the rating is between 1.0 and 5.0.
            else:                                                                   # If the data table was empty...
                predicted_rating = 0.0                                              # ...set the prediction to 0.0.

            # Unified model prediction
            if unified_reg_model and unified_cls_model and unified_preprocessor:    # Checks if the unified models were loaded successfully.
                df_unified_pred = prepare_dataframe_for_unified_prediction(details, selected_content_type) # Prepares the data for the unified model.
                print("⚙️  Running unified model prediction...")                     # Prints a status message.
                if not df_unified_pred.empty:                                       # If the data is not empty...
                    unified_predicted_rating = unified_reg_model.predict(df_unified_pred)[0] # ...predict a rating with the unified regression model.
                    unified_predicted_rating = max(1.0, min(5.0, unified_predicted_rating)) # ...make sure the rating is between 1.0 and 5.0.
                    unified_predicted_sentiment = unified_cls_model.predict(df_unified_pred)[0] # ...predict a sentiment (e.g., 'like it') with the classification model.
                else:                                                               # If the data was empty...
                    unified_predicted_rating = 0.0                                  # ...set the rating to 0.0.
                    unified_predicted_sentiment = "Unknown"                         # ...set the sentiment to "Unknown".

            print("\n" + "="*35)                                                    # Prints a separator line.
            print("        ⭐ PREDICTION RESULT ⭐")                                  # Prints the header for the results.
            print("="*35)                                                           # Prints another separator line.
            print(f"  Content: {selected_content_name}")                            # Prints the name of the content.
            print(f"  Predicted Rating (Specific): {predicted_rating:.2f} / 5.0")   # Prints the prediction from the content-specific model, formatted to 2 decimal places.
            if unified_reg_model and unified_cls_model:                             # If the unified models exist...
                print(f"  Predicted Rating (Unified): {unified_predicted_rating:.2f} / 5.0") # ...print the prediction from the unified model.
                print(f"  Predicted Sentiment (Unified): {unified_predicted_sentiment}") # ...print the predicted sentiment.
            print("="*35)                                                           # Prints a final separator line.

        except KeyboardInterrupt:                                                   # If the user presses Ctrl+C...
            print("\nExiting predictor. Goodbye! 👋")                               # ...print a goodbye message.
            break                                                                   # ...and exit the main loop, ending the program.
        except Exception as e:                                                      # If any other unexpected error occurs...
            print(f"An unexpected error occurred: {e}")                             # ...print a generic error message.
            traceback.print_exc()                                                   # ...print the full technical error details.
            continue                                                                # ...and continue to the next iteration of the loop.

if __name__ == "__main__":                                                          # Checks if this script is the main program being run.
    main()                                                                          # Calls the main function to start the program.