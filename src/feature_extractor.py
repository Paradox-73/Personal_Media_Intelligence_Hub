# src/feature_extractor.py

from typing import Optional, List, Any, cast
from transformers.models.auto.processing_auto import AutoImageProcessor
from transformers.models.auto.modeling_auto import AutoModel
from transformers.modeling_utils import PreTrainedModel
# Removed: from transformers.image_processing_utils import ProcessorMixin # This caused the error
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # type: ignore
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
import joblib
from PIL import Image
import os
from tqdm import tqdm

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import clean_text, download_image_from_url, ensure_directory_exists
from src.data_loader import CONTENT_COLUMN_MAPPING

class FeatureExtractor:
    """
    A comprehensive feature extractor class for various content types (games, shows, movies, etc.),
    handling text, image, numerical, and categorical features.
    It uses pre-trained Transformer models for text and image embeddings.
    """
    def __init__(self,
                 text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # It takes the name of the text model we want to use.
                 text_model_revision: str = 'c9745ed',
                 image_model_name: str = 'google/vit-base-patch16-224-in21k',     # It also takes the name of the image model we want to use.
                 image_model_revision: str = 'b456956'):
        """
        Initializes the FeatureExtractor with specified Transformer models.

        Args:
            text_model_name (str): Hugging Face model name for text embeddings.
            image_model_name (str): Hugging Face model name for image embeddings.
        """
        self.text_model_name = text_model_name
        self.text_model_revision = text_model_revision
        self.image_model_name = image_model_name
        self.image_model_revision = image_model_revision

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.text_model: Optional[SentenceTransformer] = None
        # Revert type hint to AutoImageProcessor
        self.image_processor: Optional[Any] = None
        self.image_model: Optional[Any] = None
        self.column_transformer: Optional[ColumnTransformer] = None
        self.text_pca: Optional[PCA] = None
        self.image_pca: Optional[PCA] = None

        self.fitted_numerical_cols: List[str] = []
        self.fitted_categorical_cols: List[str] = []
        self.fitted_text_cols: List[str] = []
        self.fitted_image_col: Optional[str] = None
        self.ct_output_features: Optional[List[str]] = None

    def _load_transformer_models(self):
        """Loads or downloads the Transformer models and processors."""
        if self.text_model is None:
            print(f"Loading text model: {self.text_model_name}")
            self.text_model = SentenceTransformer(self.text_model_name, device=str(self.device), revision=self.text_model_revision)
            self.text_model.eval()

        if self.image_processor is None or self.image_model is None:
            print(f"Loading image model: {self.image_model_name}")
            if self.image_model_name is None:
                raise ValueError("Image model name must be provided.")
            self.image_processor = AutoImageProcessor.from_pretrained(self.image_model_name, revision=self.image_model_revision, use_fast=True) # type: ignore [operator] # nosec
            self.image_model = AutoModel.from_pretrained(self.image_model_name, revision=self.image_model_revision).to(str(self.device)) # nosec

    def extract_text_features(self, texts: pd.Series) -> torch.Tensor:
        """
        Extracts embeddings for a given Pandas Series of texts using the Sentence Transformer model.
        Handles missing or non-string values gracefully.
        """
        if self.text_model is None:
            self._load_transformer_models()
        if self.text_model is None:
            raise RuntimeError("Text model failed to load.")

        valid_texts = texts.fillna("").astype(str).tolist()

        embeddings = self.text_model.encode(valid_texts,
                                            show_progress_bar=True,
                                            convert_to_tensor=True,
                                            device=str(self.device),
                                            batch_size=32)
        return embeddings

    def extract_image_features(self, image_urls: pd.Series) -> torch.Tensor:
        """
        Downloads images from URLs and extracts visual features using a pre-trained image model.
        Handles missing URLs or download failures gracefully by returning zero vectors.
        """
        if self.image_processor is None or self.image_model is None:
            self._load_transformer_models()
            if self.image_model is None:
                raise ValueError("Image model should be loaded at this point.")
            if self.image_processor is None:
                raise ValueError("Image processor should be loaded at this point.")

        embeddings_list = []
        embedding_size = self.image_model.config.hidden_size
        for url in tqdm(image_urls, desc="Extracting image features"):
            image_embedding = torch.zeros(embedding_size, device=self.device)
            if pd.notna(url) and isinstance(url, str) and url.startswith(('http://', 'https://')):
                try:
                    image = download_image_from_url(url)
                    if image:
                        # self.image_processor is now typed as AutoImageProcessor (or whatever it is), MyPy will be ignored here
                        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.image_model(**inputs)
                        image_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                except Exception as e:
                    print(f"Error processing image from {url}: {e}")
            embeddings_list.append(image_embedding)
        return torch.stack(embeddings_list)


    def fit(self, X: pd.DataFrame, content_type: str):
        """
        Fits the feature extractor components (ColumnTransformer) to the input data.
        Text and image models are pre-trained and loaded, not fitted here.
        """
        print(f"--- Fitting FeatureExtractor for {content_type} ---")
        if content_type not in CONTENT_COLUMN_MAPPING:
            raise ValueError(f"Content type '{content_type}' not found in CONTENT_COLUMN_MAPPING.")

        config = CONTENT_COLUMN_MAPPING[content_type]

        numerical_features = [col for col in config["numerical_features"] if col in X.columns]
        categorical_features = [col for col in config["categorical_features"] if col in X.columns]
        text_features = [col for col in config["text_features"] if col in X.columns]
        image_col = "image_url" if "image_url" in X.columns and X["image_url"].notna().any() else None

        self.fitted_numerical_cols = numerical_features
        self.fitted_categorical_cols = categorical_features
        self.fitted_text_cols = text_features
        self.fitted_image_col = image_col

        transformers = []

        if self.fitted_numerical_cols:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, self.fitted_numerical_cols))
            print(f"Numerical features to fit: {self.fitted_numerical_cols}")

        if self.fitted_categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, self.fitted_categorical_cols))
            print(f"Categorical features to fit: {self.fitted_categorical_cols}")

        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        if transformers:
            print("Fitting ColumnTransformer...")
            self.column_transformer.fit(X)
            print("ColumnTransformer fitted.")
        else:
            print("No numerical or categorical features to fit ColumnTransformer.")

        self._load_transformer_models()
        if self.fitted_text_cols:
            print("Fitting PCA for text features...")
            combined_texts = X[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text)
            text_embeddings = self.extract_text_features(combined_texts).cpu().numpy()
            n_samples, n_features = text_embeddings.shape

            if n_samples > 1:
                max_components = min(n_samples - 1, n_features)
                n_components = min(50, max_components)

                if n_components > 0:
                    self.text_pca = PCA(n_components=n_components)
                    self.text_pca.fit(text_embeddings)
                    print(f"Text PCA fitted with {n_components} components.")
                else:
                    print("Not enough samples or features to fit PCA for text. Skipping.")
                    self.text_pca = None
            else:
                print("Not enough samples to fit PCA for text. Skipping.")
                self.text_pca = None

        if self.fitted_image_col:
            print("Fitting PCA for image features...")
            image_embeddings = self.extract_image_features(X[self.fitted_image_col]).cpu().numpy()
            n_samples, n_features = image_embeddings.shape

            if n_samples > 1:
                max_components = min(n_samples - 1, n_features)
                n_components = min(50, max_components)

                if n_components > 0:
                    self.image_pca = PCA(n_components=n_components)
                    self.image_pca.fit(image_embeddings)
                    print(f"Image PCA fitted with {n_components} components.")
                else:
                    print("Not enough samples or features to fit PCA for images. Skipping.")
                    self.image_pca = None
            else:
                print("Not enough samples to fit PCA for images. Skipping.")
                self.image_pca = None

        print("FeatureExtractor fitting complete.")

    def transform(self, X: pd.DataFrame) -> torch.Tensor:
        """
        Transforms the input DataFrame into a feature array using the fitted components.
        """
        print("--- Transforming data using FeatureExtractor ---")
        if self.column_transformer is None and self.text_pca is None and self.image_pca is None:
            raise RuntimeError("FeatureExtractor has not been fitted yet. Call .fit() first.")

        X_transformed = X.copy()

        ct_features = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device)
        if self.column_transformer and getattr(self.column_transformer, 'transformers_', []):
            ct_features = torch.tensor(self.column_transformer.transform(X_transformed),
                                       dtype=torch.float32, device=self.device)
            print(f"ColumnTransformer output shape (on {self.device}): {ct_features.shape}")

        text_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device)
        if self.fitted_text_cols:
            combined_texts = X_transformed[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text)
            text_features_raw = self.extract_text_features(combined_texts)
            if self.text_pca:
                text_features_raw_np = text_features_raw.cpu().numpy()
                text_embeddings = torch.tensor(self.text_pca.transform(text_features_raw_np), dtype=torch.float32, device=self.device)
            else:
                text_embeddings = text_features_raw
            print(f"Text embeddings shape (on {self.device}): {text_embeddings.shape}")

        image_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device)
        if self.fitted_image_col:
            image_features_raw = self.extract_image_features(X_transformed[self.fitted_image_col])
            if self.image_pca:
                image_features_raw_np = image_features_raw.cpu().numpy()
                image_embeddings = torch.tensor(self.image_pca.transform(image_features_raw_np), dtype=torch.float32, device=self.device)
            else:
                image_embeddings = image_features_raw
            print(f"Image embeddings shape (on {self.device}): {image_embeddings.shape}")

        feature_list = [f for f in [ct_features, text_embeddings, image_embeddings] if f.size(1) > 0]
        if not feature_list:
            raise ValueError("No features were extracted.")

        combined_features = torch.cat(feature_list, dim=1)
        print(f"Combined features shape (on {self.device}): {combined_features.shape}")
        return combined_features

    def save(self, path: str):
        """
        Saves the fitted FeatureExtractor components to the specified path.
        """
        ensure_directory_exists(path)
        print(f"Saving FeatureExtractor components to {path}...")

        if self.column_transformer:
            joblib.dump(self.column_transformer, os.path.join(path, 'column_transformer.pkl'))
        if self.text_pca:
            joblib.dump(self.text_pca, os.path.join(path, 'text_pca.pkl'))
        if self.image_pca:
            joblib.dump(self.image_pca, os.path.join(path, 'image_pca.pkl'))

        if self.text_model:
            self.text_model.save(os.path.join(path, 'sentence_transformer_model'))
        if self.image_processor and self.image_model:
            # Add type: ignore here
            self.image_processor.save_pretrained(os.path.join(path, 'image_processor')) # type: ignore [attr-defined]
            self.image_model.save_pretrained(os.path.join(path, 'image_model'))

        metadata = {
            'text_model_name': self.text_model_name,
            'text_model_revision': self.text_model_revision,
            'image_model_name': self.image_model_name,
            'image_model_revision': self.image_model_revision,
            'fitted_numerical_cols': self.fitted_numerical_cols,
            'fitted_categorical_cols': self.fitted_categorical_cols,
            'fitted_text_cols': self.fitted_text_cols,
            'fitted_image_col': self.fitted_image_col,
        }
        joblib.dump(metadata, os.path.join(path, 'feature_extractor_metadata.pkl'))
        print("FeatureExtractor components saved.")

    @classmethod
    def load(cls, path: str):
        """
        Loads a fitted FeatureExtractor from the specified path.
        """
        print(f"Loading FeatureExtractor components from {path}...")
        metadata = joblib.load(os.path.join(path, 'feature_extractor_metadata.pkl'))

        instance = cls(text_model_name=metadata['text_model_name'],
                       text_model_revision=metadata.get('text_model_revision', None),
                       image_model_name=metadata['image_model_name'],
                       image_model_revision=metadata.get('image_model_revision', None))

        if os.path.exists(os.path.join(path, 'column_transformer.pkl')):
            instance.column_transformer = joblib.load(os.path.join(path, 'column_transformer.pkl'))
        if os.path.exists(os.path.join(path, 'text_pca.pkl')):
            instance.text_pca = joblib.load(os.path.join(path, 'text_pca.pkl'))
        if os.path.exists(os.path.join(path, 'image_pca.pkl')):
            instance.image_pca = joblib.load(os.path.join(path, 'image_pca.pkl'))

        text_model_path = os.path.join(path, 'sentence_transformer_model')
        if os.path.exists(text_model_path):
            instance.text_model = SentenceTransformer(text_model_path, device=str(instance.device))
        else:
            print(f"Warning: Local text model not found. Downloading '{instance.text_model_name}'.")
            instance.text_model = SentenceTransformer(instance.text_model_name, device=str(instance.device))
        if instance.text_model is not None:
            instance.text_model.eval()

        image_processor_path = os.path.join(path, 'image_processor')
        image_model_path = os.path.join(path, 'image_model')
        if os.path.exists(image_processor_path) and os.path.exists(image_model_path):
            # No cast needed if we ignore the error
            instance.image_processor = AutoImageProcessor.from_pretrained(image_processor_path, revision=instance.image_model_revision) # type: ignore [operator] # nosec
            instance.image_model = AutoModel.from_pretrained(image_model_path, revision=instance.image_model_revision).to(instance.device) # type: ignore [operator] # nosec
        else:
            print(f"Warning: Local image model not found. Downloading '{instance.image_model_name}'.")
            # No cast needed if we ignore the error
            instance.image_processor = AutoImageProcessor.from_pretrained(instance.image_model_name, revision=instance.image_model_revision) # type: ignore [operator] # nosec
            instance.image_model = AutoModel.from_pretrained(instance.image_model_name, revision=instance.image_model_revision).to(instance.device) # nosec
        if instance.image_model is not None:
            instance.image_model.eval()

        instance.fitted_numerical_cols = metadata.get('fitted_numerical_cols', [])
        instance.fitted_categorical_cols = metadata.get('fitted_categorical_cols', [])
        instance.fitted_text_cols = metadata.get('fitted_text_cols', [])
        instance.fitted_image_col = metadata.get('fitted_image_col')

        print("FeatureExtractor components loaded successfully.")
        return instance