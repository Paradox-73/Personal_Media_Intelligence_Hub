
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from src.data_loader import load_content_data, CONTENT_COLUMN_MAPPING

def get_unified_feature_config():
    # Define a unified set of features to be used across all content types
    # These are the "generic" column names after mapping in data_loader
    unified_features = {
        'text_features': ['title', 'description', 'genres'],
        'categorical_features': ['content_type', 'language', 'status', 'show_type', 'network_country', 'rated', 'director', 'writer', 'actors', 'awards', 'platform_from_text', 'age_rating', 'developers', 'publishers', 'tags', 'authors', 'publisher'],
        'numerical_features': ['critic_rating_normalized', 'runtime', 'average_runtime', 'popularity', 'watch_count', 'episode_count', 'year', 'imdb_rating', 'metascore', 'ratings_count', 'reviews_count', 'pageCount']
    }
    return unified_features

def prepare_unified_data():
    all_data = []
    for content_type in CONTENT_COLUMN_MAPPING.keys():
        print(f"Loading data for {content_type}...")
        df = load_content_data(content_type)
        if not df.empty and 'my_rating' in df.columns:
            df['content_type'] = content_type
            
            # Normalize critic ratings to a 0-10 scale
            # This is a simplified approach; more sophisticated scaling could be used
            if content_type == 'Game':
                # Metacritic is 0-100, rating is 0-5. We'll prioritize metacritic.
                df['critic_rating_normalized'] = df.get('metacritic', df.get('rating', 50)) / 10
            elif content_type == 'Show':
                # rating_avg is 0-10
                df['critic_rating_normalized'] = df.get('rating_avg', 5)
            elif content_type == 'Movie':
                # imdb_rating is 0-10, metascore is 0-100
                df['critic_rating_normalized'] = df.get('imdb_rating', df.get('metascore', 50) / 10)
            elif content_type == 'Music':
                # Popularity is 0-100
                df['critic_rating_normalized'] = df.get('Popularity', 50) / 10
            elif content_type == 'Book':
                # averageRating is 0-5
                df['critic_rating_normalized'] = df.get('averageRating', 2.5) * 2
            
            # Fill any remaining NaNs in the normalized critic rating with the median
            if 'critic_rating_normalized' in df.columns:
                df['critic_rating_normalized'].fillna(df['critic_rating_normalized'].median(), inplace=True)

            all_data.append(df)

    if not all_data:
        print("No data available to train the unified model.")
        return None

    unified_df = pd.concat(all_data, ignore_index=True)

    # Fill remaining NaNs in numerical columns with 0
    numerical_cols = get_unified_feature_config()['numerical_features']
    for col in numerical_cols:
        if col in unified_df.columns:
            unified_df[col] = pd.to_numeric(unified_df[col], errors='coerce').fillna(0)
    
    # Define sentiment categories
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['hate it', 'dislike it', 'meh', 'like it', 'love it']
    unified_df['sentiment'] = pd.cut(unified_df['my_rating'], bins=bins, labels=labels, right=True, include_lowest=True)
    
    # Drop rows where target is missing
    unified_df.dropna(subset=['my_rating', 'sentiment'], inplace=True)

    return unified_df

def train_unified_model():
    unified_df = prepare_unified_data()
    if unified_df is None or unified_df.empty:
        return

    features_config = get_unified_feature_config()
    
    # Define features and target
    X = unified_df[features_config['text_features'] + features_config['categorical_features'] + features_config['numerical_features']]
    y_reg = unified_df['my_rating']
    y_cls = unified_df['sentiment']

    # Create preprocessing pipelines for different feature types
    text_transformer = TfidfVectorizer(stop_words='english', max_features=5000)
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('text_title', text_transformer, 'title'),
            ('text_desc', TfidfVectorizer(stop_words='english', max_features=5000), 'description'),
            ('cat', categorical_transformer, features_config['categorical_features']),
            ('num', numerical_transformer, features_config['numerical_features'])
        ],
        remainder='drop'
    )

    # --- Split Data ---
    # We use a single stratified split to ensure consistency for the preprocessor
    X_train, X_test, y_train_cls, y_test_cls = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    y_train_reg = y_reg.loc[X_train.index]
    y_test_reg = y_reg.loc[X_test.index]

    # --- Train Regression Model ---
    reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', Ridge(alpha=1.0))])
    
    print("Training unified regression model...")
    reg_pipeline.fit(X_train, y_train_reg)
    
    # --- Train Classification Model ---
    cls_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))])

    print("Training unified classification model...")
    cls_pipeline.fit(X_train, y_train_cls)

    # Save the models and the preprocessor
    output_dir = os.path.join('models', 'unified')
    os.makedirs(output_dir, exist_ok=True)
    
    joblib.dump(reg_pipeline, os.path.join(output_dir, 'unified_regression_model.pkl'))
    joblib.dump(cls_pipeline, os.path.join(output_dir, 'unified_classification_model.pkl'))
    joblib.dump(preprocessor, os.path.join(output_dir, 'unified_preprocessor.pkl'))
    
    print(f"Unified models and preprocessor saved to {output_dir}")

if __name__ == '__main__':
    train_unified_model()
