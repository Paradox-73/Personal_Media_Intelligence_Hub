# python src/app.py
import os
import sys
import traceback

import joblib  # type: ignore
import pandas as pd
from typing import Any

# This block of code checks if the script is being run directly.
# If so, it adds the project's main folder to the list of places Python looks for files.
# This is important so we can import our own custom Python files from the 'src' directory.
if __name__ == "__main__":
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    except ImportError:
        print("Could not import custom modules. Please ensure the script is run from the correct directory structure.")
        sys.exit(1)

from src.utils import (
    search_content,
    get_content_details,
)
from src.feature_extractor import FeatureExtractor
from src.data_loader import CONTENT_COLUMN_MAPPING

# --- Configuration ---
MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')            # Defines the main folder where all our trained models are stored.

# --- Helper Functions for Loading Models ---
def load_model_and_extractor(content_type: str):
    """
    Loads the trained XGBoost model and FeatureExtractor for a given content type.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, content_type.lower())                  # Sets the directory for the specific content type's model.
    xgboost_model_path = os.path.join(model_dir, f'xgboost_model_{content_type.lower()}.pkl') # Creates the full file path for the XGBoost model file.
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor')           # Creates the full file path for the feature extractor.

    if not os.path.exists(xgboost_model_path):
        print(f"❌ Error: Model file not found for {content_type}: {xgboost_model_path}. Please train the model first.") # If not, prints an error.
        return None, None                                                           # And returns nothing, indicating failure.
    if not os.path.exists(feature_extractor_path):
        print(f"❌ Error: Feature Extractor not found for {content_type}: {feature_extractor_path}. Please train the model first.") # If not, prints an error.
        return None, None                                                           # And returns nothing.

    try:
        xgb_model = joblib.load(xgboost_model_path)                                 # Loads the saved XGBoost model from its file.
        feature_extractor = FeatureExtractor.load(feature_extractor_path)           # Loads the saved feature extractor.
        print(f"✅ Successfully loaded model and extractor for {content_type}.")      # Prints a success message.
        return xgb_model, feature_extractor                                         # Returns the loaded model and feature extractor.
    except Exception as e:
        print(f"❌ Error loading model or feature extractor for {content_type}: {e}") # ...print an error message with details.
        return None, None                                                           # ...and return nothing.

def load_unified_models():
    """
    Loads the trained unified regression and classification models and their preprocessor.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, 'unified')                             # Sets the directory for the unified model.
    reg_model_path = os.path.join(model_dir, 'unified_regression_model.pkl')        # Creates the file path for the regression model (predicts a number).
    cls_model_path = os.path.join(model_dir, 'unified_classification_model.pkl')    # Creates the file path for the classification model (predicts a category).
    preprocessor_path = os.path.join(model_dir, 'unified_preprocessor.pkl')         # Creates the file path for the data preprocessor.

    if not os.path.exists(reg_model_path) or not os.path.exists(cls_model_path) or not os.path.exists(preprocessor_path):
        print(f"❌ Error: Unified model files not found in {model_dir}. Please train the unified models first.") # If so, prints an error.
        return None, None, None                                                     # And returns nothing.

    try:
        reg_model = joblib.load(reg_model_path)                                     # Loads the regression model.
        cls_model = joblib.load(cls_model_path)                                     # Loads the classification model.
        preprocessor = joblib.load(preprocessor_path)                               # Loads the preprocessor.
        print("OK: Successfully loaded unified models and preprocessor.")           # Prints a success message.
        return reg_model, cls_model, preprocessor                                   # Returns all the loaded components.
    except Exception as e:
        print(f"ERROR: Error loading unified models: {e}")                          # ...print the error.
        return None, None, None                                                     # ...and return nothing.

def prepare_dataframe_for_prediction(details: dict, content_type: str) -> pd.DataFrame:
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
        processed_details['year'] = pd.to_numeric(details.get('Year', ''), errors='coerce')
        processed_details['imdb_rating'] = pd.to_numeric(details.get('imdbRating', ''), errors='coerce')
        processed_details['metascore'] = pd.to_numeric(details.get('Metascore', ''), errors='coerce')
        processed_details['rated'] = details.get('Rated')
        processed_details['language'] = details.get('Language')
        processed_details['director'] = details.get('Director')
        processed_details['writer'] = details.get('Writer')
        processed_details['actors'] = details.get('Actors')
        processed_details['awards'] = details.get('Awards')

    elif content_type == "Game":
        processed_details['title'] = details.get('name')
        processed_details['description'] = details.get('description_raw')
        processed_details['image_url'] = details.get('cover')
        processed_details['release_date'] = details.get('released')
        processed_details['metacritic'] = details.get('metacritic')
        processed_details['rating'] = details.get('rating')
        processed_details['ratings_count'] = details.get('ratings_count')
        processed_details['reviews_count'] = details.get('reviews_count')
        processed_details['platform_from_text'] = ", ".join([p['platform']['name'] for p in details.get('platforms', [])])
        processed_details['age_rating'] = details.get('esrb_rating', {}).get('name') if details.get('esrb_rating') else 'Not Rated'
        processed_details['developers'] = ", ".join([d['name'] for d in details.get('developers', [])])
        processed_details['publishers'] = ", ".join([p['name'] for p in details.get('publishers', [])])
        processed_details['tags'] = ", ".join([t['name'] for t in details.get('tags', [])])
        processed_details['genres'] = ", ".join([g['name'] for g in details.get('genres', [])])

    elif content_type == "Show":
        processed_details['title'] = details.get('name')
        processed_details['description'] = details.get('summary')
        processed_details['image_url'] = details.get('image', {}).get('medium') if details.get('image') else None
        processed_details['release_date'] = details.get('premiered')
        processed_details['runtime'] = details.get('runtime')
        processed_details['average_runtime'] = details.get('averageRuntime')
        processed_details['rating_avg'] = details.get('rating', {}).get('average') if details.get('rating') else None
        processed_details['popularity'] = details.get('weight')
        processed_details['language'] = details.get('language')
        processed_details['status'] = details.get('status')
        processed_details['show_type'] = details.get('type')
        processed_details['genres'] = "|".join(details.get('genres', []))

    elif content_type == "Book":
        vol_info = details.get('volumeInfo', {})
        processed_details['title'] = vol_info.get('title')
        processed_details['description'] = vol_info.get('description')
        processed_details['image_url'] = vol_info.get('imageLinks', {}).get('thumbnail') if vol_info.get('imageLinks') else None
        processed_details['release_date'] = vol_info.get('publishedDate')
        processed_details['averageRating'] = vol_info.get('averageRating')
        processed_details['ratingsCount'] = vol_info.get('ratingsCount')
        processed_details['authors'] = ", ".join(vol_info.get('authors', []))
        processed_details['publisher'] = vol_info.get('publisher')
        processed_details['language'] = vol_info.get('language')
        processed_details['genres'] = ", ".join(vol_info.get('categories', []))

    # Convert the processed dictionary to a single-row DataFrame
    df_pred = pd.DataFrame([processed_details])

    # This section makes sure that the data table has all the columns the model was trained on.
    # If a column is missing, it adds it and fills it with a default value.
    all_trained_cols = set(config.get('numerical_features', []) + config.get('categorical_features', []) + config.get('text_features', []))
    if config.get('image_url'):
        all_trained_cols.add('image_url')

    for col in all_trained_cols:
        if col not in df_pred.columns:
            if col in config.get('numerical_features', []):
                df_pred[col] = 0.0
            else:
                df_pred[col] = "Unknown"

    # Final cleanup for consistency
    for col in config.get('numerical_features', []):
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce').fillna(0.0)
    for col in list(set(config.get('categorical_features', []) + config.get('text_features', []))):
        df_pred[col] = df_pred[col].fillna("Unknown")
    if 'image_url' in df_pred.columns:
        df_pred['image_url'] = df_pred['image_url'].fillna("")


    return df_pred

def prepare_dataframe_for_unified_prediction(details: dict, content_type: str) -> pd.DataFrame:
    """
    Prepares a single content's details for the unified model.
    """
    df = prepare_dataframe_for_prediction(details, content_type)
    df['content_type'] = content_type

    # This section creates a new 'critic_rating_normalized' column.
    # It takes various rating scores (like Metacritic, IMDb) and puts them on a standard 1-10 scale.
    df['critic_rating_normalized'] = _calculate_critic_rating_normalized(df, content_type)
    if 'critic_rating_normalized' in df.columns:
        df['critic_rating_normalized'] = df['critic_rating_normalized'].fillna(5.0)
    else:
        df['critic_rating_normalized'] = 5.0

    # Define the full set of features the unified model expects
    unified_text_features = ['title', 'description', 'genres']
    unified_categorical_features = [
        'content_type', 'language', 'status', 'show_type', 'network_country',
        'rated', 'director', 'writer', 'actors', 'awards', 'platform_from_text',
        'age_rating', 'developers', 'publishers', 'tags', 'authors', 'publisher'
    ]
    unified_numerical_features = [
        'critic_rating_normalized', 'runtime', 'average_runtime', 'popularity',
        'watch_count', 'episode_count', 'year', 'imdb_rating', 'metascore',
        'ratings_count', 'reviews_count', 'pageCount'
    ]

    all_unified_cols = set(unified_text_features + unified_categorical_features + unified_numerical_features)

    # Ensure all expected columns are present, filling with defaults if necessary
    for col in all_unified_cols:
        if col not in df.columns:
            if col in unified_numerical_features:
                df[col] = 0.0
            else:
                df[col] = "Unknown"

    # Final cleanup for consistency
    for col in unified_numerical_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    for col in unified_text_features + unified_categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df

def _calculate_critic_rating_normalized(df: pd.DataFrame, content_type: str) -> "pd.Series[Any]":
    """Calculates a normalized critic rating based on content type."""
    if content_type == 'Game':
        return df.get('metacritic', pd.Series([50])) / 10
    elif content_type == 'Show':
        return df.get('rating_avg', pd.Series([5]))
    elif content_type == 'Movie':
        return df.get('imdb_rating', df.get('metascore', pd.Series([50])) / 10)
    elif content_type == 'Music':
        return df.get('my_rating', pd.Series([50])) / 10
    elif content_type == 'Book':
        return df.get('averageRating', pd.Series([2.5])) * 2
    return pd.Series([5.0])

def get_user_input():
    """Gets content type and search query from the user."""
    content_type_options = list(CONTENT_COLUMN_MAPPING.keys())
    print("\n\n--- New Prediction ---")
    print("1️⃣ Select a Content Type:")
    for i, option in enumerate(content_type_options):
        print(f"   {i + 1}. {option}")

    while True:
        try:
            choice_input = input("Enter the number of your choice: ")
            if not choice_input:
                continue
            choice = int(choice_input) - 1
            if 0 <= choice < len(content_type_options):
                selected_content_type = content_type_options[choice]
                break
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    search_query = input(f"\n2️⃣ Enter the name of the {selected_content_type.lower()} to search for: ")
    return selected_content_type, search_query

def select_content_from_results(found_content, content_type):
    """Displays search results and prompts the user to select one."""
    print("\n3️⃣ Select the correct item from the search results:")
    content_id_map = {}

    for i, item in enumerate(found_content):
        item_id, display_text = _format_content_display(item, content_type)

        if item_id:
            display_text_with_num = f"   {i + 1}. {display_text}"
            print(display_text_with_num)
            content_id_map[i + 1] = (item_id, display_text)

    if not content_id_map:
        print("No valid results found to select.")
        return None, None

    while True:
        try:
            choice_input = input("\nEnter the number of the content you want to predict: ")
            if not choice_input:
                continue
            choice = int(choice_input)
            if choice in content_id_map:
                return content_id_map[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def _format_content_display(item: dict, content_type: str):
    """Helper to format content display text based on content type."""
    display_text = "Unknown Content"
    item_id = None

    if content_type == "Game":
        item_id = item.get('id')
        year = item.get('released', 'N/A').split('-')[0] if item.get('released') else 'N/A'
        display_text = f"{item.get('name', 'N/A')} ({year})"
    elif content_type == "Show":
        item_id = item.get('id')
        year = item.get('premiered', 'N/A').split('-')[0] if item.get('premiered') else 'N/A'
        display_text = f"{item.get('name', 'N/A')} ({year})"
    elif content_type == "Movie":
        item_id = item.get('imdbID')
        display_text = f"{item.get('Title', 'N/A')} ({item.get('Year', 'N/A')})"
    elif content_type == "Music":
        item_id = item.get('id')
        artists = ", ".join([a['name'] for a in item.get('artists', [])])
        album = item.get('album', {}).get('name', 'N/A')
        year = item.get('album', {}).get('release_date', 'N/A').split('-')[0] if item.get('album', {}).get('release_date') else 'N/A'
        display_text = f"{item.get('name', 'N/A')} - {artists} (Album: {album}, {year})"
    elif content_type == "Book":
        item_id = item.get('id')
        vol = item.get('volumeInfo', {})
        authors = ", ".join(vol.get('authors', []))
        year = vol.get('publishedDate', 'N/A').split('-')[0] if vol.get('publishedDate') else 'N/A'
        display_text = f"{vol.get('title', 'N/A')} by {authors} ({year})"
    return item_id, display_text

def main():
    """Main function to run the terminal-based predictor."""
    print("--- Universal Content Rating Predictor ---")
    print("(Press Ctrl+C at any time to exit)")

    unified_reg_model, unified_cls_model, unified_preprocessor = load_unified_models()

    while True:
        try:
            selected_content_type, search_query = get_user_input()

            xgb_model, feature_extractor = load_model_and_extractor(selected_content_type)
            if not xgb_model or not feature_extractor:
                continue

            print(f"\nSearching for '{search_query}'...")
            found_content = search_content(selected_content_type, search_query)

            if not found_content:
                print(f"No {selected_content_type.lower()} found for your search.")
                continue

            selected_content_id, selected_content_name = select_content_from_results(found_content, selected_content_type)
            if not selected_content_id:
                continue

            print(f"\nFetching details for '{selected_content_name}'...")
            details = get_content_details(selected_content_type, selected_content_id)
            if not details:
                print("Could not fetch details for the selected content.")
                continue

            _run_and_display_predictions(
                details,
                selected_content_type,
                selected_content_name,
                xgb_model,
                feature_extractor,
                unified_reg_model,
                unified_cls_model,
                unified_preprocessor
            )

        except KeyboardInterrupt:
            print("\nExiting predictor. Goodbye! 👋")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            continue

def _run_and_display_predictions(
    details: dict,
    selected_content_type: str,
    selected_content_name: str,
    xgb_model,
    feature_extractor,
    unified_reg_model,
    unified_cls_model,
    unified_preprocessor
):
    """Runs content-specific and unified predictions and displays the results."""
    # Content-specific prediction
    df_pred = prepare_dataframe_for_prediction(details, selected_content_type)
    print("⚙️  Running content-specific prediction...")
    if not df_pred.empty:
        combined_features = feature_extractor.transform(df_pred)
        features_for_xgb = combined_features
        predicted_rating = xgb_model.predict(features_for_xgb)[0]
        predicted_rating = max(1.0, min(5.0, predicted_rating))
    else:
        predicted_rating = 0.0

    # Unified model prediction
    unified_predicted_rating = 0.0
    unified_predicted_sentiment = "Unknown"
    if unified_reg_model and unified_cls_model and unified_preprocessor:
        df_unified_pred = prepare_dataframe_for_unified_prediction(details, selected_content_type)
        print("⚙️  Running unified model prediction...")
        if not df_unified_pred.empty:
            unified_predicted_rating = unified_reg_model.predict(unified_preprocessor.transform(df_unified_pred))[0]
            unified_predicted_rating = max(1.0, min(5.0, unified_predicted_rating))
            unified_predicted_sentiment = unified_cls_model.predict(unified_preprocessor.transform(df_unified_pred))[0]

    print("\n" + "=" * 35)
    print("        ⭐ PREDICTION RESULT ⭐")
    print("=" * 35)
    print(f"  Content: {selected_content_name}")
    print(f"  Predicted Rating (Specific): {predicted_rating:.2f} / 5.0")
    if unified_reg_model and unified_cls_model:
        print(f"  Predicted Rating (Unified): {unified_predicted_rating:.2f} / 5.0")
        print(f"  Predicted Sentiment (Unified): {unified_predicted_sentiment}")
    print("=" * 35)

if __name__ == "__main__":
    main()
