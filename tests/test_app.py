# tests/test_app.py

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
from src.app import (
    load_model_and_extractor,
    prepare_dataframe_for_prediction,
    main,
    get_user_input,
    select_content_from_results,
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Module-level patch for builtins.input to prevent tests from hanging
# Provides default inputs for any unmocked input() calls
@patch('builtins.input', side_effect=['1', 'test_query', '1'])
# Module-level patch for builtins.input to prevent tests from hanging
# Provides default inputs for any unmocked input() calls
@patch('builtins.input', side_effect=['1', 'test_query', '1'])
# --- Fixtures ---
@pytest.fixture
def mock_details():
    """Provides a sample details dictionary for a movie."""
    return {
        'Title': 'Inception',
        'Plot': 'A thief who steals corporate secrets through the use of dream-sharing technology.',
        'Genre': 'Action, Adventure, Sci-Fi',
        'Poster': 'http://example.com/inception.jpg',
        'Released': '16 Jul 2010',
        'Year': '2010',
        'imdbRating': '8.8',
        'Metascore': '74',
        'Rated': 'PG-13',
        'Language': 'English',
        'Director': 'Christopher Nolan',
        'Writer': 'Christopher Nolan',
        'Actors': 'Leonardo DiCaprio, Joseph Gordon-Levitt, Elliot Page',
        'Awards': 'Won 4 Oscars.'
    }

# --- Tests for Helper Functions ---

@patch('src.app.joblib.load')
@patch('src.feature_extractor.FeatureExtractor.load')
@patch('os.path.exists', return_value=True)
def test_load_model_and_extractor_success(mock_exists, mock_feature_extractor_load, mock_joblib_load):
    """
    Tests that the model and feature extractor are loaded successfully when files exist.
    """
    mock_joblib_load.return_value = MagicMock()
    mock_feature_extractor_load.return_value = MagicMock()

    xgb_model, feature_extractor = load_model_and_extractor('Movie')

    assert xgb_model is not None
    assert feature_extractor is not None
    mock_joblib_load.assert_called_once()
    mock_feature_extractor_load.assert_called_once()

@patch('os.path.exists', return_value=False)
def test_load_model_and_extractor_file_not_found(mock_exists):
    """
    Tests that the function returns None when model files are not found.
    """
    xgb_model, feature_extractor = load_model_and_extractor('Movie')
    assert xgb_model is None
    assert feature_extractor is None

@patch('src.app.CONTENT_COLUMN_MAPPING', {'Movie': {'numerical_features': ['imdb_rating', 'metascore'], 'categorical_features': ['Rated', 'Language'], 'text_features': ['Title', 'Plot', 'Genre', 'Director', 'Writer', 'Actors', 'Awards'], 'image_url': 'Poster'}})
def test_prepare_dataframe_for_prediction(mock_details):
    """
    Tests the preparation of a DataFrame for prediction from a details dictionary.
    """
    df = prepare_dataframe_for_prediction(mock_details, 'Movie')

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'title' in df.columns
    assert df['title'].iloc[0] == 'Inception'
    assert 'imdb_rating' in df.columns
    assert df['imdb_rating'].iloc[0] == 8.8

@patch('builtins.input', side_effect=['1', 'Inception'])
def test_get_user_input(mock_input):
    content_type, search_query = get_user_input()
    assert content_type == 'Game'
    assert search_query == 'Inception'

@patch('builtins.input', side_effect=['1'])
def test_select_content_from_results(mock_input):
    found_content = [
        {'id': '123', 'name': 'Game A', 'released': '2022-01-01'},
        {'id': '456', 'name': 'Game B', 'released': '2021-01-01'}
    ]
    content_id, content_name = select_content_from_results(found_content, 'Game')
    assert content_id == '123'
    assert content_name == 'Game A (2022)'

# --- Tests for Main Function ---

@patch('src.app.get_content_details')
@patch('src.app.search_content')
@patch('src.app.get_user_input')
@patch('src.app.load_model_and_extractor')
@patch('src.app.select_content_from_results')
@patch('src.app.prepare_dataframe_for_prediction')
@patch('src.app.load_unified_models')
def test_main_full_flow(
    mock_load_unified, mock_prepare_df, mock_select_content,
    mock_load_model, mock_get_input, mock_search, mock_get_details
):
    """
    Tests the main function's full workflow from user input to prediction.
    """
    # Mock user input
    mock_get_input.return_value = ('Movie', 'Inception')

    # Mock model loading
    mock_xgb = MagicMock()
    mock_xgb.predict.return_value = [4.5]
    mock_fe = MagicMock()
    mock_fe.transform.return_value = torch.randn(1, 10)
    mock_load_model.return_value = (mock_xgb, mock_fe)

    # Mock unified models
    mock_unified_reg = MagicMock()
    mock_unified_reg.predict.return_value = [4.2]
    mock_unified_cls = MagicMock()
    mock_unified_cls.predict.return_value = ['like']
    mock_load_unified.return_value = (mock_unified_reg, mock_unified_cls, MagicMock())

    # Mock API calls and user selection
    mock_search.return_value = [{'imdbID': 'tt1375666', 'Title': 'Inception', 'Year': '2010'}]
    mock_select_content.return_value = ('tt1375666', 'Inception (2010)')
    mock_get_details.return_value = {'Title': 'Inception', 'Plot': 'A mind-bending thriller.'}

    # Mock DataFrame preparation
    mock_prepare_df.return_value = pd.DataFrame({'title': ['Inception'], 'imdb_rating': [8.8], 'metascore': [74]})

    # Run the main loop once
    with patch('builtins.print'), patch('src.app.main', side_effect=KeyboardInterrupt):
        try:
            main()
        except KeyboardInterrupt:
            pass # Expected exit

    mock_get_input.assert_called_once()
    mock_load_model.assert_called_once_with('Movie')
    mock_search.assert_called_once_with('Movie', 'Inception')
    mock_select_content.assert_called_once()
    mock_get_details.assert_called_once_with('Movie', 'tt1375666')
    mock_prepare_df.assert_called()
    mock_fe.transform.assert_called_once()
    mock_xgb.predict.assert_called_once()
