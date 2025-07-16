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
    suitable for prediction. It applies content-type-specific feature engineering
    and ensures all expected columns are present with appropriate default values.
    """
    df = pd.DataFrame([details]) # Convert single detail dictionary to a DataFrame

    # Get the column mappings for the specific content type from data_loader
    config = CONTENT_COLUMN_MAPPING[content_type]

    # --- Handle common features expected by the FeatureExtractor ---

    # Ensure numerical features are present and are numeric, fill NaNs
    numerical_cols = config.get("numerical_features", [])
    # Also include specific engineered numeric features that FeatureExtractor expects
    # These are hardcoded as they come from data_loader_show.py's engineering.
    if content_type == "Show":
        numerical_cols.extend(['genre_count', 'has_ended', 'is_english'])

    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0.0 # Default to 0.0 for missing numericals from API
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0) # Ensure numeric, fill any NaNs

    # Ensure categorical features are present and fill NaNs with 'Unknown'
    categorical_cols = config.get("categorical_features", [])
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = config.get("fill_na_categorical_strategy", "Unknown")
        df[col] = df[col].fillna(config.get("fill_na_categorical_strategy", "Unknown"))

    # Ensure text features are present and fill NaNs with empty strings
    text_cols = config.get("text_features", [])
    for col in text_cols:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('')

    # --- Content-type specific feature engineering ---
    # This recreates the features that were engineered during training
    if content_type == "Show":
        # Ensure base columns for engineering are filled, although largely handled above
        df['genres'] = df['genres'].fillna('')
        df['language'] = df['language'].fillna('Unknown')
        df['status'] = df['status'].fillna('Unknown')

        # Engineered features for 'Show'
        df['genre_count'] = df['genres'].apply(lambda x: len(x.split('|')) if x else 0)
        df['has_ended'] = df['status'].str.lower().str.contains('ended').astype(int)
        df['is_english'] = df['language'].str.lower().str.contains('english').astype(int)

        # Specific handling for show runtime/episode_count if they might be NaN from API
        if 'runtime' in df.columns and pd.isna(df['runtime'].iloc[0]):
            df['runtime'] = 0.0
        if 'episode_count' in df.columns and pd.isna(df['episode_count'].iloc[0]):
            df['episode_count'] = 0.0

    # Add elif blocks here for other content types (Game, Movie, etc.)
    # if they have specific feature engineering steps.
    # elif content_type == "Game":
    #     # Example for Game:
    #     # df['platform_count'] = df['platforms'].apply(lambda x: len(x.split(',')) if x else 0)
    #     pass

    return df

def get_user_input():
    """Gets content type and search query from the user."""
    content_type_options = list(CONTENT_COLUMN_MAPPING.keys())
    print("\n\n--- New Prediction ---")
    print("1️⃣ Select a Content Type:")
    for i, option in enumerate(content_type_options):
        print(f"   {i+1}. {option}")

    while True:
        try:
            choice = int(input("Enter the number of your choice: ")) - 1
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
            choice = int(input("\nEnter the number of the content you want to predict: "))
            if choice in content_id_map:
                return content_id_map[choice]
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def prepare_dataframe_for_prediction(details, content_type):
    """Prepares the pandas DataFrame for the model."""
    prediction_input_data = {}
    cfg = CONTENT_COLUMN_MAPPING[content_type]

    # This function would contain the detailed, content-specific logic
    # from your Streamlit app to extract and map all the necessary features
    # (numerical, categorical, text, etc.) into the prediction_input_data dict.
    # For this example, we'll keep it conceptual.
    print("\nGathering features for prediction...")

    # Title
    prediction_input_data[cfg['title']] = details.get(cfg['title'], 'N/A')

    # Example for one numerical feature
    if content_type == "Movie":
        raw_rating = details.get('imdbRating', 'N/A')
        rating_val = pd.to_numeric(raw_rating, errors='coerce')
        prediction_input_data['imdb_rating'] = rating_val
    # ... and so on for all features defined in CONTENT_COLUMN_MAPPING

    # Build DataFrame
    all_cols = set(cfg.get('numerical_features', []) + cfg.get('categorical_features', []) + cfg.get('text_features', []))
    df_pred = pd.DataFrame([prediction_input_data], columns=list(all_cols))

    # Impute missing values
    for col in cfg.get('numerical_features', []):
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].fillna(0.0)

    for col in cfg.get('categorical_features', []) + cfg.get('text_features', []):
        if col in df_pred.columns:
            fill_value = cfg.get('fill_na_categorical_strategy', 'Unknown')
            df_pred[col] = df_pred[col].astype(str).replace('nan', fill_value).fillna(fill_value)

    return df_pred


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
            if isinstance(combined_features, torch.Tensor):
                features_for_xgb = combined_features.cpu().detach().numpy()
            else:
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