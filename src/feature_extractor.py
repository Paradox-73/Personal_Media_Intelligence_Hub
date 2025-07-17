import pandas as pd
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib
from PIL import Image
import os
from tqdm import tqdm # For progress bars

import sys

# Add the parent directory of the current script to sys.path
# This makes 'src' importable if you run from E:\\Show_ML\\GAME_PREDICT\\src\\\
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom utility functions and the content column mapping
from src.utils import clean_text, download_image_from_url, ensure_directory_exists
from src.data_loader import CONTENT_COLUMN_MAPPING # Import the mapping for feature selection

class FeatureExtractor:
    """
    A comprehensive feature extractor class for various content types (games, shows, movies, etc.),
    handling text, image, numerical, and categorical features.
    It uses pre-trained Transformer models for text and image embeddings.
    """
    def __init__(self,
                 text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 image_model_name: str = 'google/vit-base-patch16-224-in21k'):
        """
        Initializes the FeatureExtractor with specified Transformer models.

        Args:
            text_model_name (str): Hugging Face model name for text embeddings.
            image_model_name (str): Hugging Face model name for image embeddings.
        """
        self.text_model_name = text_model_name
        self.image_model_name = image_model_name
        
        # Determine device for PyTorch models
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Initialize models (will be loaded/downloaded during first use or fitting)
        self.text_model = None
        self.image_processor = None
        self.image_model = None
        self.column_transformer = None
        self.text_pca = None
        self.image_pca = None

        # Attributes to store which columns were actually fitted/processed
        self.fitted_numerical_cols = []
        self.fitted_categorical_cols = []
        self.fitted_text_cols = []
        self.fitted_image_col = None
        self.ct_output_features = None # To store the feature names from ColumnTransformer

    def _load_transformer_models(self):
        """Loads or downloads the Transformer models and processors."""
        if self.text_model is None:
            print(f"Loading text model: {self.text_model_name}")
            self.text_model = SentenceTransformer(self.text_model_name, device=self.device)
            self.text_model.eval() # Set to evaluation mode

        if self.image_processor is None or self.image_model is None:
            print(f"Loading image model: {self.image_model_name}")
            self.image_processor = AutoImageProcessor.from_pretrained(self.image_model_name, use_fast=True)
            self.image_model = AutoModel.from_pretrained(self.image_model_name).to(self.device)
            self.image_model.eval() # Set to evaluation mode

    def extract_text_features(self, texts: pd.Series) -> torch.Tensor: # Changed return type to torch.Tensor
        """
        Extracts embeddings for a given Pandas Series of texts using the Sentence Transformer model.
        Handles missing or non-string values gracefully.

        Args:
            texts (pd.Series): A Pandas Series containing text data.

        Returns:
            torch.Tensor: A PyTorch tensor of text embeddings on the configured device.
        """
        if self.text_model is None:
            self._load_transformer_models()

        # Filter out NaN/None and non-string types before encoding
        valid_texts = texts.fillna("").astype(str).tolist()
        
        # Encode in batches, ensuring output is a tensor on the correct device
        embeddings = self.text_model.encode(valid_texts, 
                                            show_progress_bar=True, 
                                            convert_to_tensor=True, # Ensure output is a tensor
                                            device=self.device, # Ensure it's on the correct device
                                            batch_size=32) 
        return embeddings

    def extract_image_features(self, image_urls: pd.Series) -> torch.Tensor: # Changed return type to torch.Tensor
        """
        Downloads images from URLs and extracts visual features using a pre-trained image model.
        Handles missing URLs or download failures gracefully by returning zero vectors.

        Args:
            image_urls (pd.Series): A Pandas Series containing image URLs.

        Returns:
            torch.Tensor: A PyTorch tensor of image embeddings on the configured device.
        """
        if self.image_processor is None or self.image_model is None:
            self._load_transformer_models()

        embeddings_list = []
        # Use tqdm for a progress bar during image processing
        # Get the expected embedding size from the model config
        embedding_size = self.image_model.config.hidden_size
        for url in tqdm(image_urls, desc="Extracting image features"):
            # Default zero vector as a PyTorch tensor on the correct device
            image_embedding = torch.zeros(embedding_size, device=self.device) 
            if pd.notna(url) and isinstance(url, str) and url.startswith(('http://', 'https://')):
                try:
                    image = download_image_from_url(url)
                    if image:
                        # Prepare image for the model, ensuring inputs are on the correct device
                        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
                        with torch.no_grad():
                            outputs = self.image_model(**inputs)
                        # Use the last hidden state's [CLS] token embedding as the feature
                        image_embedding = outputs.last_hidden_state[:, 0, :].squeeze() # Keep as tensor on device
                except Exception as e:
                    print(f"Error processing image from {url}: {e}")
            embeddings_list.append(image_embedding)
        return torch.stack(embeddings_list) # Stack tensors into a single tensor


    def fit(self, X: pd.DataFrame, content_type: str):
        """
        Fits the feature extractor components (ColumnTransformer) to the input data.
        Text and image models are pre-trained and loaded, not fitted here.

        Args:
            X (pd.DataFrame): The input DataFrame containing features.
            content_type (str): The type of content (e.g., "Game", "Show", "Movie").
        """
        print(f"--- Fitting FeatureExtractor for {content_type} ---")
        if content_type not in CONTENT_COLUMN_MAPPING:
            raise ValueError(f"Content type '{content_type}' not found in CONTENT_COLUMN_MAPPING.")

        config = CONTENT_COLUMN_MAPPING[content_type]

        # Filter features based on availability in the current DataFrame X
        numerical_features = [col for col in config["numerical_features"] if col in X.columns]
        categorical_features = [col for col in config["categorical_features"] if col in X.columns]
        text_features = [col for col in config["text_features"] if col in X.columns]
        image_col = "image_url" if "image_url" in X.columns and X["image_url"].notna().any() else None

        self.fitted_numerical_cols = numerical_features
        self.fitted_categorical_cols = categorical_features
        self.fitted_text_cols = text_features
        self.fitted_image_col = image_col

        transformers = []

        # Numerical pipeline (StandardScaler)
        if self.fitted_numerical_cols:
            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, self.fitted_numerical_cols))
            print(f"Numerical features to fit: {self.fitted_numerical_cols}")

        # Categorical pipeline (OneHotEncoder)
        if self.fitted_categorical_cols:
            # Handle unknown categories explicitly by setting them to a new category
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', categorical_transformer, self.fitted_categorical_cols))
            print(f"Categorical features to fit: {self.fitted_categorical_cols}")

        # Create ColumnTransformer
        # 'remainder='passthrough'' ensures columns not explicitly transformed are kept.
        # However, for ML models, we usually want only engineered features.
        # Given the "no standardized features" constraint, we'll implicitly drop
        # columns not specified, so we don't end up with hundreds of untransformed columns.
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop' # Drop columns not specified in transformers
        )

        # Fit the ColumnTransformer
        if transformers: # Only fit if there are numerical or categorical features
            print("Fitting ColumnTransformer...")
            self.column_transformer.fit(X)
            print("ColumnTransformer fitted.")
        else:
            print("No numerical or categorical features to fit ColumnTransformer.")

        # Load transformer models and fit PCA
        self._load_transformer_models()
        if self.fitted_text_cols:
            print("Fitting PCA for text features...")
            combined_texts = X[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text)
            text_embeddings = self.extract_text_features(combined_texts).cpu().numpy()
            n_samples = text_embeddings.shape[0]
            n_components = min(50, n_samples - 1)
            self.text_pca = PCA(n_components=n_components) # Reduce dynamically
            self.text_pca.fit(text_embeddings)
            print(f"Text PCA fitted with {n_components} components.")

        if self.fitted_image_col:
            print("Fitting PCA for image features...")
            image_embeddings = self.extract_image_features(X[self.fitted_image_col]).cpu().numpy()
            n_samples = image_embeddings.shape[0]
            n_components = min(50, n_samples - 1)
            self.image_pca = PCA(n_components=n_components) # Reduce dynamically
            self.image_pca.fit(image_embeddings)
            print(f"Image PCA fitted with {n_components} components.")

        print("FeatureExtractor fitting complete.")

    def transform(self, X: pd.DataFrame) -> torch.Tensor: # Changed return type to torch.Tensor
        """
        Transforms the input DataFrame into a feature array using the fitted components.

        Args:
            X (pd.DataFrame): The input DataFrame to transform.

        Returns:
            torch.Tensor: A PyTorch tensor of combined features on the configured device.
        """
        print("--- Transforming data using FeatureExtractor ---")
        if self.column_transformer is None:
            raise RuntimeError("FeatureExtractor has not been fitted yet. Call .fit() first.")
        
        # Ensure all fitted columns are present in X, fill missing with NaN for consistent transformation
        # The ColumnTransformer will handle these NaNs based on its fitted transformers
        # For categorical, OneHotEncoder handle_unknown='ignore' will output zeros for unknown categories
        # For numerical, StandardScaler will produce NaNs if input is NaN, which then needs handling by model or earlier.
        # data_loader is responsible for filling NaNs, so this should generally be clean.
        
        # Make a copy to avoid modifying the original DataFrame
        X_transformed = X.copy()

        # Numerical and Categorical features via ColumnTransformer
        ct_features = None
        if self.column_transformer.transformers_: # Check if CT has actually been fitted with transformers
            # ColumnTransformer outputs NumPy array, convert to PyTorch tensor and move to device
            ct_features = torch.tensor(self.column_transformer.transform(X_transformed), 
                                       dtype=torch.float32, device=self.device)
            print(f"ColumnTransformer output shape (on {self.device}): {ct_features.shape}")
        else:
            print("ColumnTransformer not fitted or no numerical/categorical features.")
            ct_features = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Empty tensor if no features for CT

        # Text features
        text_embeddings = None
        if self.fitted_text_cols:
            combined_texts = X_transformed[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text)
            text_features_raw = self.extract_text_features(combined_texts).cpu().numpy()
            text_embeddings = torch.tensor(self.text_pca.transform(text_features_raw), dtype=torch.float32, device=self.device)
            print(f"Text embeddings shape (on {self.device}): {text_embeddings.shape}")
        else:
            text_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Empty tensor

        # Image features
        image_embeddings = None
        if self.fitted_image_col:
            image_features_raw = self.extract_image_features(X_transformed[self.fitted_image_col]).cpu().numpy()
            image_embeddings = torch.tensor(self.image_pca.transform(image_features_raw), dtype=torch.float32, device=self.device)
            print(f"Image embeddings shape (on {self.device}): {image_embeddings.shape}")
        else:
            image_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Empty tensor


        # Combine all features - now all are PyTorch tensors on the same device
        feature_list = []
        if ct_features.size(1) > 0: # Check if there are actual features in ct_features
            feature_list.append(ct_features)
        if text_embeddings.size(1) > 0: # Check if there are actual features in text_embeddings
            feature_list.append(text_embeddings)
        if image_embeddings.size(1) > 0: # Check if there are actual features in image_embeddings
            feature_list.append(image_embeddings)

        if not feature_list:
            raise ValueError("No features were extracted. Check input data and feature configurations.")

        combined_features = torch.cat(feature_list, dim=1) # Use torch.cat for tensors
        print(f"Combined features shape (on {self.device}): {combined_features.shape}")
        return combined_features

    def save(self, path: str):
        """
        Saves the fitted FeatureExtractor components to the specified path.

        Args:
            path (str): Directory path to save the components.
        """
        ensure_directory_exists(path)
        print(f"Saving FeatureExtractor components to {path}...")

        # Save ColumnTransformer
        if self.column_transformer:
            joblib.dump(self.column_transformer, os.path.join(path, 'column_transformer.pkl'))
        if self.text_pca:
            joblib.dump(self.text_pca, os.path.join(path, 'text_pca.pkl'))
        if self.image_pca:
            joblib.dump(self.image_pca, os.path.join(path, 'image_pca.pkl'))
        
        # Save Transformer models (or their names to reload)
        # It's generally better to save and load directly from Hugging Face cache
        # or a local path after downloading. Saving the actual models can be large.
        # Here, we'll save the model weights and processor config locally.
        if self.text_model:
            self.text_model.save_pretrained(os.path.join(path, 'sentence_transformer_model'))
        if self.image_processor and self.image_model:
            self.image_processor.save_pretrained(os.path.join(path, 'image_processor'))
            self.image_model.save_pretrained(os.path.join(path, 'image_model'))

        # Save metadata
        metadata = {
            'text_model_name': self.text_model_name,
            'image_model_name': self.image_model_name,
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

        Args:
            path (str): Directory path where the components were saved.

        Returns:
            FeatureExtractor: The loaded and initialized FeatureExtractor instance.
        """
        print(f"Loading FeatureExtractor components from {path}...")
        metadata = joblib.load(os.path.join(path, 'feature_extractor_metadata.pkl'))
        
        # Re-initialize the FeatureExtractor class instance
        instance = cls(text_model_name=metadata['text_model_name'],
                       image_model_name=metadata['image_model_name'])

        # Load fitted components
        if os.path.exists(os.path.join(path, 'column_transformer.pkl')):
            instance.column_transformer = joblib.load(os.path.join(path, 'column_transformer.pkl'))
        if os.path.exists(os.path.join(path, 'text_pca.pkl')):
            instance.text_pca = joblib.load(os.path.join(path, 'text_pca.pkl'))
        if os.path.exists(os.path.join(path, 'image_pca.pkl')):
            instance.image_pca = joblib.load(os.path.join(path, 'image_pca.pkl'))
        
        # Load transformer models locally if saved, otherwise from Hugging Face Hub
        if os.path.exists(os.path.join(path, 'sentence_transformer_model')):
            instance.text_model = SentenceTransformer(instance.text_model_name, device='cpu')
            instance.text_model.eval()
        else:
            print(f"Warning: Local text model not found. Will try to download '{instance.text_model_name}'.")
            instance.text_model = SentenceTransformer(instance.text_model_name, device=instance.device)
            instance.text_model.eval()

        if os.path.exists(os.path.join(path, 'image_processor')) and \
           os.path.exists(os.path.join(path, 'image_model')):
            instance.image_processor = AutoImageProcessor.from_pretrained(os.path.join(path, 'image_processor'))
            instance.image_model = AutoModel.from_pretrained(os.path.join(path, 'image_model')).to('cpu')
            instance.image_model.eval()
        else:
            print(f"Warning: Local image processor/model not found. Will try to download '{instance.image_model_name}'.")
            instance.image_processor = AutoImageProcessor.from_pretrained(instance.image_model_name)
            instance.image_model = AutoModel.from_pretrained(instance.image_model_name).to('cpu')
            instance.image_model.eval()


        instance.fitted_numerical_cols = metadata.get('fitted_numerical_cols', [])
        instance.fitted_categorical_cols = metadata.get('fitted_categorical_cols', [])
        instance.fitted_text_cols = metadata['fitted_text_cols']
        instance.fitted_image_col = metadata['fitted_image_col']

        print("FeatureExtractor components loaded successfully.")
        return instance
