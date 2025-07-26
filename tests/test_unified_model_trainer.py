import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from src.unified_model_trainer import (
    get_unified_feature_config,
    prepare_unified_data,
    train_unified_model
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# Mock CONTENT_COLUMN_MAPPING for isolated testing
MOCK_CONTENT_COLUMN_MAPPING_UNIFIED = {
    "Game": {
        "csv_paths": ["data/games_data.csv"],
        "my_rating_source_col": "my_rating",
        "my_rating": "my_rating",
        "numerical_features": ["metacritic", "rating"],
        "categorical_features": ["platform_from_text"],
        "text_features": ["name", "description_raw"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0,
    },
    "Movie": {
        "csv_paths": ["data/movies_data.csv"],
        "my_rating_source_col": "my_rating",
        "my_rating": "my_rating",
        "numerical_features": ["imdb_rating", "metascore"],
        "categorical_features": ["rated"],
        "text_features": ["title", "plot"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "Unknown",
        "my_rating_threshold": 4.0,
    }
}

@pytest.fixture
def mock_game_df():
    # Increased number of rows to ensure enough samples for stratification
    num_rows = 20
    ratings = [4.5, 4.8, 3.5, 3.8, 2.5, 2.8, 1.5, 1.8, 0.5, 0.8] * (num_rows // 10)
    platforms = ['PC', 'PS', 'Xbox', 'PC', 'PS', 'Xbox', 'PC', 'PS', 'Xbox', 'PC'] * (num_rows // 10)
    genres = ['Action', 'RPG', 'Strategy', 'Adventure', 'Sports', 'Puzzle', 'Simulation', 'Horror', 'Racing', 'Indie'] * (num_rows // 10)
    age_ratings = ['Everyone', 'Teen', 'Mature', 'Everyone', 'Teen', 'Mature', 'Everyone', 'Teen', 'Mature', 'Everyone'] * (num_rows // 10)

    return pd.DataFrame({
        'my_rating': ratings[:num_rows],
        'metacritic': [np.random.randint(10, 100) for _ in range(num_rows)],
        'rating': [np.random.uniform(0.5, 5.0) for _ in range(num_rows)],
        'platform_from_text': platforms[:num_rows],
        'name': [f'Game {i}' for i in range(num_rows)],
        'description_raw': [f'Desc {i}' for i in range(num_rows)],
        'title': [f'Game {i}' for i in range(num_rows)], # Unified feature
        'description': [f'Desc {i}' for i in range(num_rows)], # Unified feature
        'genres': genres[:num_rows], # Unified feature
        'language': ['English'] * num_rows,
        'status': ['Released'] * num_rows,
        'show_type': [''] * num_rows,
        'network_country': [''] * num_rows,
        'rated': [''] * num_rows,
        'director': [''] * num_rows,
        'writer': [''] * num_rows,
        'actors': [''] * num_rows,
        'awards': [''] * num_rows,
        'age_rating': age_ratings[:num_rows],
        'developers': [f'Dev {i}' for i in range(num_rows)],
        'publishers': [f'Pub {i}' for i in range(num_rows)],
        'tags': [f'Tag {i}' for i in range(num_rows)],
        'authors': [''] * num_rows,
        'publisher': [''] * num_rows,
        'runtime': [0] * num_rows,
        'average_runtime': [0] * num_rows,
        'popularity': [0] * num_rows,
        'watch_count': [0] * num_rows,
        'episode_count': [0] * num_rows,
        'year': [2020] * num_rows,
        'imdb_rating': [0] * num_rows,
        'metascore': [0] * num_rows,
        'ratings_count': [0] * num_rows,
        'reviews_count': [0] * num_rows,
        'pageCount': [0] * num_rows,
    })

@pytest.fixture
def mock_movie_df():
    # Increased number of rows to ensure enough samples for stratification
    num_rows = 15
    ratings = [4.0, 3.0, 2.0, 1.0, 5.0, 0.0] * (num_rows // 6 + 1) # Ensure enough for num_rows
    rated = ['PG-13', 'R', 'PG', 'G', 'NC-17', 'Unrated'] * (num_rows // 6 + 1)
    genres = ['Action', 'Drama', 'Comedy', 'Thriller', 'Sci-Fi', 'Horror'] * (num_rows // 6 + 1)

    return pd.DataFrame({
        'my_rating': ratings[:num_rows],
        'imdb_rating': [np.random.uniform(1.0, 10.0) for _ in range(num_rows)],
        'metascore': [np.random.randint(1, 100) for _ in range(num_rows)],
        'rated': rated[:num_rows],
        'title': [f'Movie {i}' for i in range(num_rows)],
        'plot': [f'Plot {i}' for i in range(num_rows)],
        'genres': genres[:num_rows],
        'description': [f'Plot {i}' for i in range(num_rows)], # Unified feature
        'language': ['English'] * num_rows,
        'status': ['Released'] * num_rows,
        'show_type': [''] * num_rows,
        'network_country': [''] * num_rows,
        'director': [''] * num_rows,
        'writer': [''] * num_rows,
        'actors': [''] * num_rows,
        'awards': [''] * num_rows,
        'platform_from_text': [''] * num_rows,
        'age_rating': [''] * num_rows,
        'developers': [''] * num_rows,
        'publishers': [''] * num_rows,
        'tags': [''] * num_rows,
        'authors': [''] * num_rows,
        'publisher': [''] * num_rows,
        'runtime': [0] * num_rows,
        'average_runtime': [0] * num_rows,
        'popularity': [0] * num_rows,
        'watch_count': [0] * num_rows,
        'episode_count': [0] * num_rows,
        'year': [2010] * num_rows,
        'ratings_count': [0] * num_rows,
        'reviews_count': [0] * num_rows,
        'pageCount': [0] * num_rows,
    })

# --- Tests for get_unified_feature_config ---

def test_get_unified_feature_config():
    config = get_unified_feature_config()
    assert isinstance(config, dict)
    assert 'text_features' in config
    assert 'categorical_features' in config
    assert 'numerical_features' in config
    assert 'title' in config['text_features']
    assert 'content_type' in config['categorical_features']
    assert 'critic_rating_normalized' in config['numerical_features']

# --- Tests for prepare_unified_data ---

@patch('src.unified_model_trainer.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING_UNIFIED)
@patch('src.unified_model_trainer.load_content_data')
def test_prepare_unified_data_success(mock_load_content_data, mock_game_df, mock_movie_df):
    mock_load_content_data.side_effect = [mock_game_df, mock_movie_df]
    unified_df = prepare_unified_data()

    assert unified_df is not None
    assert not unified_df.empty
    assert 'content_type' in unified_df.columns
    assert 'critic_rating_normalized' in unified_df.columns
    assert 'sentiment' in unified_df.columns
    assert len(unified_df) == len(mock_game_df) + len(mock_movie_df)
    assert 'Game' in unified_df['content_type'].unique()
    assert 'Movie' in unified_df['content_type'].unique()
    assert not unified_df['my_rating'].isnull().any()
    assert not unified_df['sentiment'].isnull().any()

@patch('src.unified_model_trainer.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING_UNIFIED)
@patch('src.unified_model_trainer.load_content_data', return_value=pd.DataFrame())
def test_prepare_unified_data_no_data(mock_load_content_data):
    unified_df = prepare_unified_data()
    assert unified_df is None

@patch('src.unified_model_trainer.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING_UNIFIED)
@patch('src.unified_model_trainer.load_content_data')
def test_prepare_unified_data_missing_my_rating(mock_load_content_data, mock_game_df):
    df_no_rating = mock_game_df.drop(columns=['my_rating'])
    mock_load_content_data.return_value = df_no_rating
    unified_df = prepare_unified_data()
    assert unified_df is None # Should return None if my_rating is missing

# --- Tests for train_unified_model ---

@patch('src.unified_model_trainer.prepare_unified_data')
@patch('src.unified_model_trainer.joblib')
@patch('src.unified_model_trainer.ColumnTransformer')
@patch('src.unified_model_trainer.Pipeline')
@patch('src.unified_model_trainer.os.makedirs')
def test_train_unified_model_success(
    mock_makedirs, mock_pipeline, mock_columntransformer, mock_joblib, mock_prepare_unified_data,
    mock_game_df, mock_movie_df
):
    # Combine mock dataframes to simulate unified_df
    combined_df = pd.concat([mock_game_df, mock_movie_df], ignore_index=True)
    combined_df['content_type'] = ['Game'] * len(mock_game_df) + ['Movie'] * len(mock_movie_df)
    combined_df['critic_rating_normalized'] = np.random.rand(len(combined_df)) * 10
    bins = [0, 1, 2, 3, 4, 5]
    labels = ['hate it', 'dislike it', 'meh', 'like it', 'love it']
    combined_df['sentiment'] = pd.cut(combined_df['my_rating'], bins=bins, labels=labels, right=True, include_lowest=True)

    mock_prepare_unified_data.return_value = combined_df

    # Mock the pipeline fit method
    mock_reg_pipeline_instance = MagicMock()
    mock_cls_pipeline_instance = MagicMock()
    mock_pipeline.side_effect = [mock_reg_pipeline_instance, mock_cls_pipeline_instance]

    train_unified_model()

    mock_prepare_unified_data.assert_called_once()
    mock_makedirs.assert_called_once_with(os.path.join('models', 'unified'), exist_ok=True)

    # Assert that pipelines were created and fitted
    assert mock_pipeline.call_count == 2
    mock_reg_pipeline_instance.fit.assert_called_once()
    mock_cls_pipeline_instance.fit.assert_called_once()

    # Assert that models and preprocessor were saved
    assert mock_joblib.dump.call_count == 3
    mock_joblib.dump.assert_any_call(mock_reg_pipeline_instance, os.path.join('models', 'unified', 'unified_regression_model.pkl'))
    mock_joblib.dump.assert_any_call(mock_cls_pipeline_instance, os.path.join('models', 'unified', 'unified_classification_model.pkl'))
    mock_joblib.dump.assert_any_call(mock_columntransformer.return_value, os.path.join('models', 'unified', 'unified_preprocessor.pkl'))

@patch('src.unified_model_trainer.prepare_unified_data', return_value=None)
def test_train_unified_model_no_data(mock_prepare_unified_data):
    train_unified_model()
    mock_prepare_unified_data.assert_called_once()
    # Ensure no further calls are made if no data
    assert not MagicMock(spec=os.makedirs).called
    assert not MagicMock(spec=pd.DataFrame).called
