import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock
from src.model_trainer import train_model, objective

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# Mock CONTENT_COLUMN_MAPPING for isolated testing
MOCK_CONTENT_COLUMN_MAPPING_TRAINER = {
    "TestType": {
        "csv_paths": ["data/test_data.csv"],
        "id": "id_col",
        "title": "name_col",
        "description": "desc_col",
        "genres": "genres_col",
        "image_url": "img_url_col",
        "my_rating_source_col": "rating_src",
        "my_rating": "my_rating",
        "release_date": "release_dt",
        "numerical_features": ["num_feat_1", "num_feat_2"],
        "categorical_features": ["cat_feat_1"],
        "text_features": ["desc_col", "name_col"],
        "date_features": ["release_dt"],
        "fill_na_numerical_strategy": "median",
        "fill_na_categorical_strategy": "N/A",
        "my_rating_threshold": 3.5,
    }
}

@pytest.fixture
def mock_data_frame():
    """Provides a mock DataFrame for testing."""
    return pd.DataFrame({
        "id_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "name_col": [f"Item {chr(65 + i)}" for i in range(10)],
        "desc_col": [f"Desc {chr(65 + i)}" for i in range(10)],
        "genres_col": ["Genre1", "Genre2", "Genre1", "Genre3", "Genre2", "Genre1", "Genre4", "Genre2", "Genre3", "Genre1"],
        "img_url_col": [f"url{i}" for i in range(1, 11)],
        "rating_src": [4.0, 2.0, 4.5, 1.5, 3.8, 3.0, 2.5, 4.2, 1.8, 3.5],
        "release_dt": [f"2020-01-0{i + 1}" for i in range(10)],
        "num_feat_1": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "num_feat_2": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        "cat_feat_1": ["CatA", "CatB", "CatA", "CatC", "CatB", "CatA", "CatD", "CatB", "CatC", "CatA"],
        "my_rating": [4.0, 2.0, 4.5, 1.5, 3.8, 3.0, 2.5, 4.2, 1.8, 3.5],
        "like_dislike": [1, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    })

# --- Tests for train_model functionality (remaining passing tests) ---

@patch('src.model_trainer.load_content_data')
@patch('src.model_trainer.FeatureExtractor')
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING_TRAINER)
def test_train_model_empty_data_skips_training(mock_xgbregressor, mock_feature_extractor_cls, mock_load_content_data):
    """
    Tests that training is skipped if data loading returns an empty DataFrame.
    """
    mock_load_content_data.return_value = pd.DataFrame()
    train_model("TestType", "data", "models")
    mock_load_content_data.assert_called_once_with("TestType")
    mock_feature_extractor_cls.assert_not_called()
    mock_xgbregressor.assert_not_called()

@patch('src.model_trainer.load_content_data')
@patch('src.model_trainer.FeatureExtractor')
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.CONTENT_COLUMN_MAPPING', {
    "TestType": {
        "csv_paths": ["data/test_data.csv"],
        "my_rating_source_col": "rating_src",
        # Missing "my_rating" and "like_dislike" in config
    }
})
def test_train_model_missing_target_columns_skips_training(mock_xgbregressor, mock_feature_extractor_cls, mock_load_content_data, mock_data_frame):
    """
    Tests that training is skipped if target columns are missing from the DataFrame.
    """
    df_no_target = mock_data_frame.drop(columns=['my_rating', 'like_dislike'])
    mock_load_content_data.return_value = df_no_target
    train_model("TestType", "data", "models")
    mock_load_content_data.assert_called_once_with("TestType")
    mock_feature_extractor_cls.assert_not_called()
    mock_xgbregressor.assert_not_called()

@patch('src.model_trainer.load_content_data')
@patch('src.model_trainer.FeatureExtractor')
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING_TRAINER)
def test_train_model_not_enough_data_for_split_skips_training(mock_xgbregressor, mock_feature_extractor_cls, mock_load_content_data, mock_data_frame):
    """
    Tests that training is skipped if there's not enough data for train/test split.
    """
    mock_load_content_data.return_value = mock_data_frame.head(1) # Only 1 row
    train_model("TestType", "data", "models")
    mock_load_content_data.assert_called_once_with("TestType")
    mock_feature_extractor_cls.assert_not_called()
    mock_xgbregressor.assert_not_called()

# New tests for objective function
@pytest.mark.parametrize("device_available, expected_device", [
    (True, 'cuda'),
    (False, 'cpu')
])
@patch('src.model_trainer.torch.cuda.is_available')
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.r2_score', return_value=0.95)
def test_objective_function_device_selection_and_return_value(
    mock_r2_score, mock_xgbregressor, mock_is_available, device_available, expected_device
):
    """
    Tests that the objective function correctly selects 'cuda' or 'cpu' and returns the R2 score.
    """
    mock_is_available.return_value = device_available

    mock_trial = MagicMock()
    mock_trial.suggest_int.side_effect = lambda name, low, high: 100
    mock_trial.suggest_float.side_effect = lambda name, low, high: 0.1

    X_train = np.random.rand(10, 5)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 5)
    y_test = np.random.rand(5)

    mock_xgb_model_instance = MagicMock()
    mock_xgb_model_instance.predict.return_value = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    mock_xgbregressor.return_value = mock_xgb_model_instance

    result = objective(mock_trial, X_train, y_train, X_test, y_test)

    assert mock_xgbregressor.call_args[1]['device'] == expected_device
    mock_xgb_model_instance.fit.assert_called_once_with(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    mock_xgb_model_instance.predict.assert_called_once_with(X_test)
    mock_r2_score.assert_called_once_with(y_test, mock_xgb_model_instance.predict.return_value)
    assert result == 0.95

@patch('src.model_trainer.torch.cuda.is_available', return_value=True)
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.r2_score', return_value=0.9)
def test_objective_with_cuda_available(mock_r2_score, mock_xgbregressor, mock_is_available):
    """
    Tests objective function behavior when CUDA is available.
    """
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 500
    mock_trial.suggest_float.return_value = 0.05

    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)

    mock_xgb_model_instance = MagicMock()
    mock_xgb_model_instance.predict.return_value = np.random.rand(20)
    mock_xgbregressor.return_value = mock_xgb_model_instance

    objective(mock_trial, X_train, y_train, X_test, y_test)
    mock_xgbregressor.assert_called_once()
    assert mock_xgbregressor.call_args[1]['device'] == 'cuda'


@patch('src.model_trainer.torch.cuda.is_available', return_value=False)
@patch('src.model_trainer.XGBRegressor')
@patch('src.model_trainer.r2_score', return_value=0.8)
def test_objective_with_cuda_not_available(mock_r2_score, mock_xgbregressor, mock_is_available):
    """
    Tests objective function behavior when CUDA is not available.
    """
    mock_trial = MagicMock()
    mock_trial.suggest_int.return_value = 700
    mock_trial.suggest_float.return_value = 0.03

    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.rand(20)

    mock_xgb_model_instance = MagicMock()
    mock_xgb_model_instance.predict.return_value = np.random.rand(20)
    mock_xgbregressor.return_value = mock_xgb_model_instance

    objective(mock_trial, X_train, y_train, X_test, y_test)
    mock_xgbregressor.assert_called_once()
    assert mock_xgbregressor.call_args[1]['device'] == 'cpu'
