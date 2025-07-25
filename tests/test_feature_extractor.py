import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import torch
import os
import sys
from src.feature_extractor import FeatureExtractor

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# --- Fixtures ---

@pytest.fixture
def feature_extractor():
    """Provides a FeatureExtractor instance for testing."""
    return FeatureExtractor(
        text_model_name='sentence-transformers/all-MiniLM-L6-v2',
        image_model_name='google/vit-base-patch16-224-in21k'
    )

@pytest.fixture
def sample_data():
    """Provides a sample DataFrame for testing."""
    return pd.DataFrame({
        'text_col': ['Hello world', 'This is a test', 'Another sentence'],
        'image_url': ['http://example.com/img1.jpg', 'http://example.com/img2.jpg', None],
        'num_col': [10, 20, 30],
        'cat_col': ['A', 'B', 'A']
    })

# --- Tests for Core Methods ---

@patch('src.feature_extractor.SentenceTransformer')
@patch('src.feature_extractor.AutoImageProcessor')
@patch('src.feature_extractor.AutoModel')
def test_load_transformer_models(mock_auto_model, mock_image_processor, mock_sentence_transformer, feature_extractor):
    """
    Tests that the transformer models are loaded correctly.
    """
    feature_extractor._load_transformer_models()
    mock_sentence_transformer.assert_called_once_with(
        feature_extractor.text_model_name,
        device=str(feature_extractor.device),
        revision=feature_extractor.text_model_revision
    )
    mock_image_processor.from_pretrained.assert_called_once_with(
        feature_extractor.image_model_name,
        revision=feature_extractor.image_model_revision,
        use_fast=True
    )
    mock_auto_model.from_pretrained.assert_called_once_with(
        feature_extractor.image_model_name,
        revision=feature_extractor.image_model_revision
    )

@patch('src.feature_extractor.SentenceTransformer')
def test_extract_text_features(mock_sentence_transformer, feature_extractor, sample_data):
    """
    Tests the extraction of text features.
    """
    mock_model = MagicMock()
    mock_model.encode.return_value = torch.randn(3, 10)  # (batch_size, feature_size)
    mock_sentence_transformer.return_value = mock_model

    feature_extractor._load_transformer_models()
    embeddings = feature_extractor.extract_text_features(sample_data['text_col'])

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape == (3, 10)
    mock_model.encode.assert_called_once()

@patch('src.feature_extractor.download_image_from_url')
@patch('src.feature_extractor.AutoImageProcessor')
@patch('src.feature_extractor.AutoModel')
def test_extract_image_features(mock_auto_model, mock_image_processor, mock_download, feature_extractor, sample_data):
    """
    Tests the extraction of image features.
    """
    # Mock the image model and processor
    mock_image_model_instance = MagicMock()
    mock_image_model_instance.config.hidden_size = 128
    mock_image_model_instance.return_value.last_hidden_state = torch.randn(1, 1, 128)
    mock_auto_model.from_pretrained.return_value.to.return_value = mock_image_model_instance

    mock_processor_instance = MagicMock()
    mock_processor_instance.return_value = {'pixel_values': torch.randn(1, 3, 224, 224)}
    mock_image_processor.from_pretrained.return_value = mock_processor_instance

    # Mock the download function
    mock_download.return_value = MagicMock()

    feature_extractor._load_transformer_models()
    embeddings = feature_extractor.extract_image_features(sample_data['image_url'])

    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.shape[0] == len(sample_data)
    assert embeddings.shape[1] > 0
    assert mock_download.call_count == 2  # Called for the two valid URLs

# --- Tests for Fit and Transform ---

@patch('src.feature_extractor.FeatureExtractor._load_transformer_models')
@patch('src.feature_extractor.StandardScaler')
@patch('src.feature_extractor.OneHotEncoder')
@patch('src.feature_extractor.PCA')
def test_fit(mock_pca, mock_one_hot_encoder, mock_standard_scaler, mock_load_models, feature_extractor, sample_data):
    """
    Tests the fit method to ensure all components are fitted correctly.
    """
    # Mock the content column mapping
    with patch('src.feature_extractor.CONTENT_COLUMN_MAPPING', {
        'test_content': {
            'numerical_features': ['num_col'],
            'categorical_features': ['cat_col'],
            'text_features': ['text_col'],
            'image_url': 'image_url'
        }
    }):
        feature_extractor.fit(sample_data, 'test_content')

    mock_standard_scaler.return_value.fit_transform.return_value = np.array([[1.0], [2.0], [3.0]])
    mock_one_hot_encoder.return_value.fit_transform.return_value = np.array([[1, 0], [0, 1], [1, 0]])
    assert feature_extractor.column_transformer is not None

    # Check if PCA was called for text and image features
    assert mock_pca.call_count == 2
    assert feature_extractor.text_pca is not None
    assert feature_extractor.image_pca is not None

@patch('src.feature_extractor.FeatureExtractor.extract_text_features')
@patch('src.feature_extractor.FeatureExtractor.extract_image_features')
def test_transform(mock_extract_image, mock_extract_text, feature_extractor, sample_data):
    """
    Tests the transform method to ensure features are combined correctly.
    """
    # Mock fitted components
    mock_ct = MagicMock()
    mock_ct.transform.return_value = np.random.rand(3, 5)
    feature_extractor.column_transformer = mock_ct

    mock_text_pca = MagicMock()
    mock_text_pca.transform.return_value = np.random.rand(3, 10)
    feature_extractor.text_pca = mock_text_pca

    mock_image_pca = MagicMock()
    mock_image_pca.transform.return_value = np.random.rand(3, 15)
    feature_extractor.image_pca = mock_image_pca

    # Mock feature extraction returns
    mock_extract_text.return_value = torch.randn(3, 20)
    mock_extract_image.return_value = torch.randn(3, 25)

    # Set fitted columns
    feature_extractor.fitted_text_cols = ['text_col']
    feature_extractor.fitted_image_col = 'image_url'

    transformed_features = feature_extractor.transform(sample_data)

    assert isinstance(transformed_features, torch.Tensor)
    # 5 (tabular) + 10 (text) + 15 (image) = 30 features
    assert transformed_features.shape == (3, 30)

# --- Tests for Save and Load ---

@patch('joblib.dump')
@patch('src.feature_extractor.ensure_directory_exists')
@patch.object(FeatureExtractor, '_load_transformer_models')
def test_save(mock_load_models, mock_ensure_dir, mock_dump, feature_extractor, tmp_path):
    """
    Tests that the save method correctly saves all components.
    """
    # Mock fitted components
    feature_extractor.column_transformer = MagicMock()
    feature_extractor.text_pca = MagicMock()
    feature_extractor.image_pca = MagicMock()
    feature_extractor.text_model = MagicMock()
    feature_extractor.image_processor = MagicMock()
    feature_extractor.image_model = MagicMock()

    save_path = tmp_path / "feature_extractor"
    feature_extractor.save(str(save_path))

    mock_ensure_dir.assert_called_with(str(save_path))
    assert mock_dump.call_count == 4  # ct, text_pca, image_pca, metadata
    feature_extractor.text_model.save.assert_called_once()
    feature_extractor.image_processor.save_pretrained.assert_called_once()
    feature_extractor.image_model.save_pretrained.assert_called_once()

@patch('joblib.load')
@patch('src.feature_extractor.SentenceTransformer')
@patch('src.feature_extractor.AutoImageProcessor')
@patch('src.feature_extractor.AutoModel')
def test_load(mock_auto_model, mock_image_processor, mock_sentence_transformer, mock_load, tmp_path):
    """
    Tests that the load method correctly reconstructs the FeatureExtractor.
    """
    # Mock the metadata and components
    metadata = {
        'text_model_name': 'test_text_model',
        'text_model_revision': 'test_text_rev',
        'image_model_name': 'test_image_model',
        'image_model_revision': 'test_image_rev',
        'fitted_numerical_cols': ['num_col'],
        'fitted_categorical_cols': ['cat_col'],
        'fitted_text_cols': ['text_col'],
        'fitted_image_col': 'image_url'
    }
    mock_load.side_effect = [
        metadata,  # First call loads metadata
        MagicMock(),  # column_transformer
        MagicMock(),  # text_pca
        MagicMock()   # image_pca
    ]

    # Mock the paths to avoid actual file system checks
    with patch('os.path.exists', return_value=True):
        loaded_extractor = FeatureExtractor.load(str(tmp_path))

    assert loaded_extractor.text_model_name == 'test_text_model'
    assert loaded_extractor.image_model_name == 'test_image_model'
    assert loaded_extractor.column_transformer is not None
    assert loaded_extractor.text_pca is not None
    assert loaded_extractor.image_pca is not None
    mock_sentence_transformer.assert_called()
    mock_image_processor.from_pretrained.assert_called()
    mock_auto_model.from_pretrained.assert_called()
