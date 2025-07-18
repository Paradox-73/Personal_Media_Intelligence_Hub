#python src/app.py
import pandas as pd
import numpy as np
import joblib
import os
import torch  # For device management
import sys
import traceback  # For printing full error stacks

# Add the parent directory of the current script to sys.path
# This assumes your script is in a 'src' directory and models/utils are structured as before.
if __name__ == "__main__":
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from src.utils import (
            clean_text,
            search_content,
            get_content_details,
            get_artist_details_spotify
        )
        from src.feature_extractor import FeatureExtractor
        from src.data_loader import CONTENT_COLUMN_MAPPING
    except ImportError:
        print("Could not import custom modules. Please ensure the script is run from the correct directory structure.")
        sys.exit(1)


# --- Configuration ---
MODEL_BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

# --- Helper Functions for Loading Models ---
def load_model_and_extractor(content_type: str):
    """
    Loads the trained XGBoost model and FeatureExtractor for a given content type.
    """
    model_dir = os.path.join(MODEL_BASE_DIR, content_type.lower())
    xgboost_model_path = os.path.join(model_dir, f'xgboost_model_{content_type.lower()}.pkl')
    feature_extractor_path = os.path.join(model_dir, 'feature_extractor')

    if not os.path.exists(xgboost_model_path):
        print(f"❌ Error: Model file not found for {content_type}: {xgboost_model_path}. Please train the model first.")
        return None, None
    if not os.path.exists(feature_extractor_path):
        print(f"❌ Error: Feature Extractor not found for {content_type}: {feature_extractor_path}. Please train the model first.")
        return None, None

    try:
        xgb_model = joblib.load(xgboost_model_path)
        feature_extractor = FeatureExtractor.load(feature_extractor_path)
        print(f"✅ Successfully loaded model and extractor for {content_type}.")
        return xgb_model, feature_extractor
    except Exception as e:
        print(f"❌ Error loading model or feature extractor for {content_type}: {e}")
        return None, None
    
def prepare_dataframe_for_prediction(details: dict, content_type: str) -> pd.DataFrame:
    """
    Prepares a single content's details (from API response) into a Pandas DataFrame
    suitable for prediction. It standardizes the structure from various APIs
    to match the training data format.
    """
    print("\nGathering features for prediction...")
    processed_details = {}
    config = CONTENT_COLUMN_MAPPING[content_type]

    # Manually map API fields to our internal standard fields
    if content_type == "Movie":
        processed_details['title'] = details.get('Title')
        processed_details['description'] = details.get('Plot')
        processed_details['genres'] = details.get('Genre')
        processed_details['image_url'] = details.get('Poster')
        processed_details['release_date'] = details.get('Released')
        processed_details['year'] = pd.to_numeric(details.get('Year'), errors='coerce')
        processed_details['imdb_rating'] = pd.to_numeric(details.get('imdbRating'), errors='coerce')
        processed_details['metascore'] = pd.to_numeric(details.get('Metascore'), errors='coerce')
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

    # Ensure all columns the model was trained on are present, filling missing ones with defaults
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

def get_user_input():
    """Gets content type and search query from the user."""
    content_type_options = list(CONTENT_COLUMN_MAPPING.keys())
    print("\n\n--- New Prediction ---")
    print("1️⃣ Select a Content Type:")
    for i, option in enumerate(content_type_options):
        print(f"   {i+1}. {option}")

    while True:
        try:
            choice_input = input("Enter the number of your choice: ")
            if not choice_input: continue
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
    content_options_display = []
    content_id_map = {}

    for i, item in enumerate(found_content):
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
            album = item.get('album',{}).get('name','N/A')
            year = item.get('album',{}).get('release_date','N/A').split('-')[0] if item.get('album',{}).get('release_date') else 'N/A'
            display_text = f"{item.get('name','N/A')} - {artists} (Album: {album}, {year})"
        elif content_type == "Book":
            item_id = item.get('id')
            vol = item.get('volumeInfo',{})
            authors = ", ".join(vol.get('authors',[]))
            year = vol.get('publishedDate','N/A').split('-')[0] if vol.get('publishedDate') else 'N/A'
            display_text = f"{vol.get('title','N/A')} by {authors} ({year})"

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
            if not choice_input: continue
            choice = int(choice_input)
            if choice in content_id_map:
                return content_id_map[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")



def main():
    """Main function to run the terminal-based predictor."""
    print("--- Universal Content Rating Predictor ---")
    print("(Press Ctrl+C at any time to exit)")

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

            df_pred = prepare_dataframe_for_prediction(details, selected_content_type)

            print("⚙️  Running prediction...")
            combined_features = feature_extractor.transform(df_pred)
            # No change needed here, the tensor is already on the correct device
            features_for_xgb = combined_features

            predicted_rating = xgb_model.predict(features_for_xgb)[0]
            predicted_rating = max(1.0, min(5.0, predicted_rating))

            print("\n" + "="*35)
            print("        ⭐ PREDICTION RESULT ⭐")
            print("="*35)
            print(f"  Content: {selected_content_name}")
            print(f"  Predicted Rating: {predicted_rating:.2f} / 5.0")
            print("="*35)

        except KeyboardInterrupt:
            print("\nExiting predictor. Goodbye! 👋")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()