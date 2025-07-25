import pandas as pd
import numpy as np
import os
from unittest.mock import patch

# Adjust path to import modules from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_content_data
from src.utils import clean_text, ensure_directory_exists

# --- Tests for src/utils.py ---

def test_clean_text():
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  Leading and Trailing Spaces  ") == "leading and trailing spaces"
    assert clean_text("Text with <p>HTML</p> tags.") == "text with html tags"
    assert clean_text("Multiple   spaces") == "multiple spaces"
    assert clean_text("Text with numbers 123 and symbols!@#") == "text with numbers 123 and symbols"
    assert clean_text(None) == ""
    assert clean_text(123) == ""

@patch('os.makedirs')
def test_ensure_directory_exists(mock_makedirs):
    path = "test/path/to/dir"
    ensure_directory_exists(path)
    mock_makedirs.assert_called_once_with(path, exist_ok=True)

# --- Tests for src/data_loader.py ---

# Mock CONTENT_COLUMN_MAPPING for isolated testing
MOCK_CONTENT_COLUMN_MAPPING = {
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

@patch('src.data_loader.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING)
@patch('pandas.read_csv')
@patch('os.path.exists', return_value=True)
def test_load_content_data_success(mock_exists, mock_read_csv):
    # Mock CSV content
    mock_df_content = pd.DataFrame({
        "id_col": [1, 2, 3],
        "name_col": ["Item A", "Item B", "Item C"],
        "desc_col": ["Description A", "Description B", "Description C"],
        "genres_col": ["Genre1", "Genre2", "Genre1"],
        "img_url_col": ["url1", "url2", "url3"],
        "rating_src": [4.0, np.nan, 2.5],
        "release_dt": ["2020-01-01", "2021-05-10", "2019-11-20"],
        "num_feat_1": [10, 20, np.nan],
        "num_feat_2": [100, np.nan, 300],
        "cat_feat_1": ["CatA", np.nan, "CatB"]
    })
    mock_read_csv.return_value = mock_df_content

    df = load_content_data("TestType")

    mock_read_csv.assert_called_once_with("data/test_data.csv")
    assert not df.empty
    assert "my_rating" in df.columns
    assert "like_dislike" in df.columns
    assert "release_year" in df.columns
    assert df["my_rating"].iloc[0] == 4.0
    assert df["like_dislike"].iloc[0] == 1  # 4.0 >= 3.5
    assert df["like_dislike"].iloc[1] == 0  # 2.5 < 3.5
    assert df["num_feat_1"].iloc[1] == 15.0  # Median imputation (10, 20 -> 15) for the remaining row


@patch('src.data_loader.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING)
@patch('pandas.read_csv', side_effect=FileNotFoundError)
@patch('os.path.exists', return_value=False)
def test_load_content_data_file_not_found(mock_exists, mock_read_csv):
    df = load_content_data("TestType")
    assert df.empty
    mock_read_csv.assert_called_once_with("data/test_data.csv")

@patch('src.data_loader.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING)
@patch('pandas.read_csv')
@patch('os.path.exists', return_value=True)
def test_load_content_data_empty_csv(mock_exists, mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame()  # Empty DataFrame
    df = load_content_data("TestType")
    assert df.empty

@patch('src.data_loader.CONTENT_COLUMN_MAPPING', MOCK_CONTENT_COLUMN_MAPPING)
@patch('pandas.read_csv')
@patch('os.path.exists', return_value=True)
def test_load_content_data_missing_my_rating_source_col(mock_exists, mock_read_csv):
    mock_df_content = pd.DataFrame({
        "id_col": [1],
        "name_col": ["Item A"],
        "desc_col": ["Description A"],
        "genres_col": ["Genre1"],
        "img_url_col": ["url1"],
        # "rating_src": [4.0], # Missing this column
        "release_dt": ["2020-01-01"],
        "num_feat_1": [10],
        "num_feat_2": [100],
        "cat_feat_1": ["CatA"]
    })
    mock_read_csv.return_value = mock_df_content

    df = load_content_data("TestType")
    assert "my_rating" not in df.columns
    assert "like_dislike" not in df.columns
    assert not df.empty  # Still returns data, but without target columns

def test_load_content_data_unsupported_type():
    df = load_content_data("UnsupportedType")
    assert df.empty