import os                                                                           # Imports the os library, which allows the script to interact with the operating system (e.g., file paths).
import sys                                                                          # Imports the sys library, which provides access to system-specific parameters and functions.
import pandas as pd                                                                 # Imports the pandas library, used for creating and working with data tables (DataFrames).
import joblib  # type: ignore                                                                     # Imports joblib, used for saving and loading Python objects, like our trained models.
from sklearn.model_selection import train_test_split  # type: ignore                               # Imports a function to split data into training and testing sets.
from sklearn.compose import ColumnTransformer  # type: ignore                                    # A tool to apply different preparations to different columns of data.
from sklearn.pipeline import Pipeline  # type: ignore                                            # A tool to chain multiple data preparation steps together.
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore                         # Imports TF-IDF Vectorizer, a method to convert text into numerical features.
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore                     # Imports tools from scikit-learn for preparing data for ML.
from sklearn.linear_model import Ridge, LogisticRegression  # type: ignore                          # Imports Ridge (for regression) and Logistic Regression (for classification) models.
from src.data_loader import load_content_data, CONTENT_COLUMN_MAPPING               # Imports functions and configurations for loading and understanding our data.

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the project's main folder to Python's search path.

def get_unified_feature_config():                                                   # Defines a function to get the configuration for features used by the unified model.
    # Define a unified set of features to be used across all content types
    # These are the "generic" column names after mapping in data_loader
    unified_features = {                                                            # Defines a dictionary to hold the feature categories.
        'text_features': ['title', 'description', 'genres'],                        # Lists text-based features.
        'categorical_features': [                                                   # Lists categorical features (text labels).
            'content_type', 'language', 'status', 'show_type', 'network_country',
            'rated', 'director', 'writer', 'actors', 'awards', 'platform_from_text',
            'age_rating', 'developers', 'publishers', 'tags', 'authors', 'publisher'
        ],
        'numerical_features': [                                                     # Lists numerical features.
            'critic_rating_normalized', 'runtime', 'average_runtime', 'popularity',
            'watch_count', 'episode_count', 'year', 'imdb_rating', 'metascore',
            'ratings_count', 'reviews_count', 'pageCount'
        ]
    }
    return unified_features                                                         # Returns the defined feature configuration.

def prepare_unified_data():                                                         # Defines a function to load and prepare data from all content types for the unified model.
    all_data = []                                                                   # Creates an empty list to store data from all content types.
    for content_type in CONTENT_COLUMN_MAPPING.keys():                              # Loops through each content type defined in our data configuration.
        print(f"Loading data for {content_type}...")                               # Prints a message indicating which content type is being loaded.
        df = load_content_data(content_type)                                        # Loads the data for the current content type using the data_loader.
        if not df.empty and 'my_rating' in df.columns:                              # Checks if data was loaded successfully and if it contains the 'my_rating' column.
            df['content_type'] = content_type                                       # Adds a new column to the DataFrame to indicate the content type.

            # Normalize critic ratings to a 0-10 scale
            # This is a simplified approach; more sophisticated scaling could be used
            if content_type == 'Game':                                              # If the content type is 'Game'...
                df['critic_rating_normalized'] = df.get('metacritic', df.get('rating', 50)) / 10 # ...normalize game ratings (Metacritic or rating) to a 0-10 scale.
            elif content_type == 'Show':                                            # If the content type is 'Show'...
                df['critic_rating_normalized'] = df.get('rating_avg', 5)            # ...use the average rating (already 0-10).
            elif content_type == 'Movie':                                           # If the content type is 'Movie'...
                df['critic_rating_normalized'] = df.get('imdb_rating', df.get('metascore', 50) / 10) # ...normalize movie ratings (IMDb or Metascore) to a 0-10 scale.
            elif content_type == 'Music':                                           # If the content type is 'Music'...
                df['critic_rating_normalized'] = df.get('Popularity', 50) / 10       # ...normalize music popularity to a 0-10 scale.
            elif content_type == 'Book':                                            # If the content type is 'Book'...
                df['critic_rating_normalized'] = df.get('averageRating', 2.5) * 2   # ...normalize book average rating to a 0-10 scale.

            # Fill any remaining NaNs in the normalized critic rating with the median
            if 'critic_rating_normalized' in df.columns:                            # Checks if the normalized critic rating column exists.
                df['critic_rating_normalized'].fillna(df['critic_rating_normalized'].median(), inplace=True) # Fills any missing values in this column with its median.

            all_data.append(df)                                                     # Adds the prepared DataFrame to the list of all data.

    if not all_data:                                                                # Checks if any data was collected.
        print("No data available to train the unified model.")                      # If not, prints a message.
        return None                                                                 # Returns None, indicating no data.

    unified_df = pd.concat(all_data, ignore_index=True)                             # Combines all individual DataFrames into one large DataFrame.

    # Fill remaining NaNs in numerical columns with 0
    numerical_cols = get_unified_feature_config()['numerical_features']             # Gets the list of numerical features for the unified model.
    for col in numerical_cols:                                                      # Loops through each numerical column.
        if col in unified_df.columns:                                               # Checks if the column exists in the combined DataFrame.
            unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce').fillna(0) # Ensures numerical columns are numbers and fills missing values with 0.

    # Define sentiment categories
    bins = [0, 1, 2, 3, 4, 5]                                                       # Defines the boundaries for rating bins.
    labels = ['hate it', 'dislike it', 'meh', 'like it', 'love it']                 # Defines the sentiment labels for each bin.
    unified_df['sentiment'] = pd.cut(unified_df['my_rating'], bins=bins, labels=labels, right=True, include_lowest=True) # Creates a new 'sentiment' column by categorizing 'my_rating'.

    # Drop rows where target is missing
    unified_df.dropna(subset=['my_rating', 'sentiment'], inplace=True)              # Removes any rows where 'my_rating' or 'sentiment' are missing.

    return unified_df                                                               # Returns the final prepared unified DataFrame.

def train_unified_model():                                                          # Defines the main function to train the unified models.
    unified_df = prepare_unified_data()                                             # Calls the function to prepare the unified data.
    if unified_df is None or unified_df.empty:                                      # Checks if data preparation was successful.
        return                                                                      # If not, stops the training process.

    features_config = get_unified_feature_config()                                  # Gets the feature configuration for the unified model.

    # Define features and target
    X = unified_df[features_config['text_features'] + features_config['categorical_features'] + features_config['numerical_features']] # 'X' contains all the input features.
    y_reg = unified_df['my_rating']                                                 # 'y_reg' is the target for the regression model (the actual rating).
    y_cls = unified_df['sentiment']                                                 # 'y_cls' is the target for the classification model (the sentiment category).

    # Create preprocessing pipelines for different feature types
    text_transformer = TfidfVectorizer(stop_words='english', max_features=5000)     # Creates a TF-IDF vectorizer to convert text into numerical features.
    numerical_transformer = StandardScaler()                                        # Creates a StandardScaler to normalize numerical features.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')                # Creates a OneHotEncoder to convert categorical features into numerical 0s and 1s.

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(                                               # Creates a ColumnTransformer to apply different transformations to different columns.
        transformers=[
            ('text_title', text_transformer, 'title'),                             # Applies TF-IDF to the 'title' column.
            ('text_desc', TfidfVectorizer(stop_words='english', max_features=5000), 'description'), # Applies TF-IDF to the 'description' column.
            ('cat', categorical_transformer, features_config['categorical_features']), # Applies OneHotEncoder to categorical features.
            ('num', numerical_transformer, features_config['numerical_features'])   # Applies StandardScaler to numerical features.
        ],
        remainder='drop'                                                            # Drops any columns not specified in the transformers.
    )

    # --- Split Data ---
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(                    # Splits the data into training and testing sets for classification.
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls                    # Uses 80% for training, 20% for testing, and ensures sentiment categories are evenly distributed.
    )
    y_train_reg = y_reg.loc[X_train.index]                                          # Gets the corresponding regression targets for the training set.

    # --- Train Regression Model ---
    reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', Ridge(alpha=1.0))])                # Creates a pipeline for the regression model: preprocess data, then apply Ridge regression.

    print("Training unified regression model...")                                   # Prints a status message.
    reg_pipeline.fit(X_train, y_train_reg)                                          # Trains the regression model using the training data.

    # --- Train Classification Model ---
    cls_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))]) # Creates a pipeline for the classification model: preprocess data, then apply Logistic Regression.

    print("Training unified classification model...")                               # Prints a status message.
    cls_pipeline.fit(X_train, y_train_cls)                                          # Trains the classification model using the training data.

    # Save the models and the preprocessor
    output_dir = os.path.join('models', 'unified')                                  # Defines the directory where the models will be saved.
    os.makedirs(output_dir, exist_ok=True)                                          # Creates the directory if it doesn't exist.

    joblib.dump(reg_pipeline, os.path.join(output_dir, 'unified_regression_model.pkl')) # Saves the trained regression model.
    joblib.dump(cls_pipeline, os.path.join(output_dir, 'unified_classification_model.pkl')) # Saves the trained classification model.
    joblib.dump(preprocessor, os.path.join(output_dir, 'unified_preprocessor.pkl')) # Saves the data preprocessor.

    print(f"Unified models and preprocessor saved to {output_dir}")                # Prints a confirmation message.

if __name__ == '__main__':                                                          # Checks if this script is the main program being run.
    train_unified_model()                                                           # Calls the main training function.
