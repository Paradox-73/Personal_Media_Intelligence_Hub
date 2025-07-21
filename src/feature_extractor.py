import pandas as pd                                                                 # Imports the pandas library, used for creating and working with data tables (DataFrames).
import numpy as np                                                                  # Imports numpy, a library for numerical operations, especially with arrays.
import torch                                                                        # Imports PyTorch, a machine learning library, used here for deep learning models.
from transformers import AutoImageProcessor, AutoModel                              # From the 'transformers' library, imports tools to process images and a generic model.
from sentence_transformers import SentenceTransformer                               # Imports a special kind of model designed to understand the meaning of sentences.
from sklearn.preprocessing import StandardScaler, OneHotEncoder                     # Imports tools from scikit-learn for preparing data for ML.
from sklearn.compose import ColumnTransformer                                     # A tool to apply different preparations to different columns of data.
from sklearn.pipeline import Pipeline                                             # A tool to chain multiple data preparation steps together.
from sklearn.decomposition import PCA                                             # Imports PCA, a technique to reduce the number of features (simplify data).
import joblib                                                                       # Imports joblib, used for saving and loading Python objects, like our trained components.
from PIL import Image                                                               # Imports the Pillow library (PIL), used for opening and working with images.
import os                                                                           # Imports the os library to interact with the operating system (e.g., file paths).
from tqdm import tqdm                                                               # Imports tqdm, a library that creates progress bars for loops.

import sys

# This block adds the project's main folder to the list of places Python looks for files,
# allowing us to import our own custom Python files from the 'src' directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom utility functions and the content column mapping
from src.utils import clean_text, download_image_from_url, ensure_directory_exists  # Imports helper functions from our 'utils.py' file.
from src.data_loader import CONTENT_COLUMN_MAPPING                                  # Imports the main data configuration dictionary.

class FeatureExtractor:                                                             # Defines the FeatureExtractor class. Its job is to convert raw data into numerical features for the ML model.
    """
    A comprehensive feature extractor class for various content types (games, shows, movies, etc.),
    handling text, image, numerical, and categorical features.
    It uses pre-trained Transformer models for text and image embeddings.
    """
    def __init__(self,                                                              # This is the setup method, called when a new FeatureExtractor is created.
                 text_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',  # It takes the name of the text model we want to use.
                 image_model_name: str = 'google/vit-base-patch16-224-in21k'):     # It also takes the name of the image model we want to use.
        """
        Initializes the FeatureExtractor with specified Transformer models.

        Args:
            text_model_name (str): Hugging Face model name for text embeddings.
            image_model_name (str): Hugging Face model name for image embeddings.
        """
        self.text_model_name = text_model_name                                      # Stores the name of the text model.
        self.image_model_name = image_model_name                                    # Stores the name of the image model.
        
        # Determine device for PyTorch models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    # Checks if a powerful GPU (cuda) is available, otherwise uses the regular CPU.
        print(f"Using device: {self.device}")                                       # Prints which device is being used.

        # These are placeholders. The actual models and tools will be loaded when needed.
        self.text_model = None                                                      # Placeholder for the text model.
        self.image_processor = None                                                 # Placeholder for the image preparation tool.
        self.image_model = None                                                     # Placeholder for the image model.
        self.column_transformer = None                                              # Placeholder for the tool that handles numerical and categorical data.
        self.text_pca = None                                                        # Placeholder for the text data simplifier (PCA).
        self.image_pca = None                                                       # Placeholder for the image data simplifier (PCA).

        # These lists will keep track of which columns were used during training.
        self.fitted_numerical_cols = []                                             # Stores the names of the numerical columns used.
        self.fitted_categorical_cols = []                                           # Stores the names of the categorical columns used.
        self.fitted_text_cols = []                                                  # Stores the names of the text columns used.
        self.fitted_image_col = None                                                # Stores the name of the image URL column used.
        self.ct_output_features = None                                              # Stores the names of the features created by the column transformer.

    def _load_transformer_models(self):                                             # Defines a helper function to load the big AI models.
        """Loads or downloads the Transformer models and processors."""
        if self.text_model is None:                                                 # If the text model hasn't been loaded yet...
            print(f"Loading text model: {self.text_model_name}")                    # ...print a message.
            self.text_model = SentenceTransformer(self.text_model_name, device=self.device) # ...download/load the model and assign it to the correct device (CPU/GPU).
            self.text_model.eval()                                                  # ...set the model to "evaluation mode" (no training, just predicting).

        if self.image_processor is None or self.image_model is None:                # If the image model or its processor hasn't been loaded...
            print(f"Loading image model: {self.image_model_name}")                  # ...print a message.
            self.image_processor = AutoImageProcessor.from_pretrained(self.image_model_name, use_fast=True) # ...download/load the image processor.
            self.image_model = AutoModel.from_pretrained(self.image_model_name).to(self.device) # ...download/load the image model to the correct device.
            self.image_model.eval()                                                 # ...set it to evaluation mode.

    def extract_text_features(self, texts: pd.Series) -> torch.Tensor:              # Defines a function to turn text into numbers (vectors).
        """
        Extracts embeddings for a given Pandas Series of texts using the Sentence Transformer model.
        Handles missing or non-string values gracefully.
        """
        if self.text_model is None:                                                 # If the text model isn't loaded...
            self._load_transformer_models()                                         # ...load it now.

        valid_texts = texts.fillna("").astype(str).tolist()                        # Cleans up the text data: fills empty spots with "" and converts all to strings.
        
        # This is where the magic happens: the model "encodes" the text into numerical vectors.
        embeddings = self.text_model.encode(valid_texts,                            # Feeds the clean text to the model.
                                            show_progress_bar=True,                 # Shows a progress bar because this can be slow.
                                            convert_to_tensor=True,                 # Tells the model to output the results as PyTorch tensors.
                                            device=self.device,                     # Ensures the calculations happen on the right device.
                                            batch_size=32)                          # Processes the text in batches of 32 for efficiency.
        return embeddings                                                           # Returns the resulting numerical vectors (embeddings).

    def extract_image_features(self, image_urls: pd.Series) -> torch.Tensor:        # Defines a function to turn images (from URLs) into numbers.
        """
        Downloads images from URLs and extracts visual features using a pre-trained image model.
        Handles missing URLs or download failures gracefully by returning zero vectors.
        """
        if self.image_processor is None or self.image_model is None:                # If the image model isn't loaded...
            self._load_transformer_models()                                         # ...load it now.

        embeddings_list = []                                                        # Creates an empty list to store the vector for each image.
        embedding_size = self.image_model.config.hidden_size                        # Gets the size of the vector the model will create.
        for url in tqdm(image_urls, desc="Extracting image features"):               # Loops through each URL, with a progress bar.
            image_embedding = torch.zeros(embedding_size, device=self.device)       # Creates a default vector of all zeros in case the image fails.
            if pd.notna(url) and isinstance(url, str) and url.startswith(('http://', 'https://')): # Checks if the URL is a valid web address.
                try:                                                                # Starts a 'try' block to handle download/processing errors.
                    image = download_image_from_url(url)                            # Downloads the image from the URL.
                    if image:                                                       # If the download was successful...
                        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device) # ...use the processor to prepare the image for the model.
                        with torch.no_grad():                                       # Tells PyTorch not to track gradients (saves memory and speeds things up).
                            outputs = self.image_model(**inputs)                    # ...feed the prepared image to the model.
                        image_embedding = outputs.last_hidden_state[:, 0, :].squeeze() # ...extract the resulting numerical vector from the model's output.
                except Exception as e:                                              # If any error occurred...
                    print(f"Error processing image from {url}: {e}")               # ...print an error message.
            embeddings_list.append(image_embedding)                                 # Adds the resulting vector (or the zero vector on failure) to our list.
        return torch.stack(embeddings_list)                                         # Stacks all the individual vectors into a single large tensor.


    def fit(self, X: pd.DataFrame, content_type: str):                               # Defines the 'fit' method, which learns from the training data.
        """
        Fits the feature extractor components (ColumnTransformer) to the input data.
        Text and image models are pre-trained and loaded, not fitted here.
        """
        print(f"--- Fitting FeatureExtractor for {content_type} ---")               # Prints a status message.
        if content_type not in CONTENT_COLUMN_MAPPING:                              # Checks if the content type is valid.
            raise ValueError(f"Content type '{content_type}' not found in CONTENT_COLUMN_MAPPING.") # Raises an error if not.

        config = CONTENT_COLUMN_MAPPING[content_type]                               # Gets the configuration for this content type.

        # Determines which feature columns are actually present in the provided data table 'X'.
        numerical_features = [col for col in config["numerical_features"] if col in X.columns] # Finds available numerical columns.
        categorical_features = [col for col in config["categorical_features"] if col in X.columns] # Finds available categorical columns.
        text_features = [col for col in config["text_features"] if col in X.columns] # Finds available text columns.
        image_col = "image_url" if "image_url" in X.columns and X["image_url"].notna().any() else None # Checks if there is an image URL column with actual URLs.

        self.fitted_numerical_cols = numerical_features                             # Stores the list of numerical columns that will be used.
        self.fitted_categorical_cols = categorical_features                         # Stores the list of categorical columns.
        self.fitted_text_cols = text_features                                       # Stores the list of text columns.
        self.fitted_image_col = image_col                                           # Stores the name of the image column.

        transformers = []                                                           # Creates an empty list to hold our data preparation steps.

        # Sets up the preparation pipeline for numerical data.
        if self.fitted_numerical_cols:                                              # If there are numerical columns to process...
            numerical_transformer = Pipeline(steps=[                                 # ...create a pipeline.
                ('scaler', StandardScaler())                                        # The only step is 'StandardScaler', which standardizes numbers (e.g., mean 0, std dev 1).
            ])
            transformers.append(('num', numerical_transformer, self.fitted_numerical_cols)) # Adds this pipeline to our list of transformers.
            print(f"Numerical features to fit: {self.fitted_numerical_cols}")       # Prints which columns it will be applied to.

        # Sets up the preparation pipeline for categorical data.
        if self.fitted_categorical_cols:                                            # If there are categorical columns...
            categorical_transformer = Pipeline(steps=[                               # ...create a pipeline.
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # The step is 'OneHotEncoder', which converts categories (like "Action", "Comedy") into numerical 0s and 1s.
            ])
            transformers.append(('cat', categorical_transformer, self.fitted_categorical_cols)) # Adds this pipeline to the list.
            print(f"Categorical features to fit: {self.fitted_categorical_cols}")   # Prints which columns it will be applied to.

        self.column_transformer = ColumnTransformer(                                # Creates the main column transformer.
            transformers=transformers,                                              # It uses the pipelines we just defined.
            remainder='drop'                                                        # It will drop any columns that we haven't explicitly told it to process.
        )

        if transformers:                                                            # If we defined any pipelines (for num or cat data)...
            print("Fitting ColumnTransformer...")                                   # ...print a status message.
            self.column_transformer.fit(X)                                          # ...'fit' the transformer to the data, making it learn the necessary statistics (means, categories, etc.).
            print("ColumnTransformer fitted.")                                      # ...print a success message.
        else:
            print("No numerical or categorical features to fit ColumnTransformer.") # Otherwise, print a message that there was nothing to do.

        self._load_transformer_models()                                             # Make sure the AI models are loaded.
        if self.fitted_text_cols:                                                   # If there are text columns to process...
            print("Fitting PCA for text features...")                               # ...print a status message.
            combined_texts = X[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text) # ...combine all text columns into one, clean it up.
            text_embeddings = self.extract_text_features(combined_texts).cpu().numpy() # ...get the numerical vectors for the text.
            n_samples, n_features = text_embeddings.shape                           # ...get the dimensions of the resulting vectors.
            
            if n_samples > 1:                                                       # If we have more than one piece of data...
                max_components = min(n_samples - 1, n_features)                     # ...calculate the maximum number of PCA components we can have.
                n_components = min(50, max_components)                              # ...we will simplify the vectors down to 50 dimensions (or fewer if not possible).
                
                if n_components > 0:                                                # If we can create at least one component...
                    self.text_pca = PCA(n_components=n_components)                  # ...create the PCA tool.
                    self.text_pca.fit(text_embeddings)                              # ...'fit' the PCA tool to the text vectors, learning how to simplify them.
                    print(f"Text PCA fitted with {n_components} components.")       # ...print a success message.
                else:
                    print("Not enough samples or features to fit PCA for text. Skipping.") # Otherwise, skip PCA.
                    self.text_pca = None
            else:
                print("Not enough samples to fit PCA for text. Skipping.")         # Skip PCA if there's not enough data.
                self.text_pca = None

        if self.fitted_image_col:                                                   # If there is an image column to process...
            print("Fitting PCA for image features...")                              # ...print a status message.
            image_embeddings = self.extract_image_features(X[self.fitted_image_col]).cpu().numpy() # ...get the numerical vectors for the images.
            n_samples, n_features = image_embeddings.shape                          # ...get the dimensions of the vectors.

            if n_samples > 1:                                                       # If we have more than one image...
                max_components = min(n_samples - 1, n_features)                     # ...calculate the max possible PCA components.
                n_components = min(50, max_components)                              # ...we will simplify to 50 dimensions.

                if n_components > 0:                                                # If possible...
                    self.image_pca = PCA(n_components=n_components)                 # ...create the PCA tool for images.
                    self.image_pca.fit(image_embeddings)                            # ...'fit' it to the image vectors.
                    print(f"Image PCA fitted with {n_components} components.")      # ...print a success message.
                else:
                    print("Not enough samples or features to fit PCA for images. Skipping.") # Otherwise, skip.
                    self.image_pca = None
            else:
                print("Not enough samples to fit PCA for images. Skipping.")       # Skip if not enough data.
                self.image_pca = None

        print("FeatureExtractor fitting complete.")                                 # Print a final success message for the whole process.

    def transform(self, X: pd.DataFrame) -> torch.Tensor:                           # Defines the 'transform' method, which applies the learned rules to data.
        """
        Transforms the input DataFrame into a feature array using the fitted components.
        """
        print("--- Transforming data using FeatureExtractor ---")                   # Prints a status message.
        if self.column_transformer is None and self.text_pca is None and self.image_pca is None: # Checks if the extractor has been fitted.
            raise RuntimeError("FeatureExtractor has not been fitted yet. Call .fit() first.") # If not, raises an error.
        
        X_transformed = X.copy()                                                    # Makes a copy of the input data to avoid changing the original.

        ct_features = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Creates an empty tensor to hold the numerical/categorical features.
        if self.column_transformer and getattr(self.column_transformer, 'transformers_', []): # If the column transformer was fitted...
            ct_features = torch.tensor(self.column_transformer.transform(X_transformed), # ...use it to transform the data.
                                       dtype=torch.float32, device=self.device)   # ...and convert the result to a PyTorch tensor.
            print(f"ColumnTransformer output shape (on {self.device}): {ct_features.shape}") # Prints the shape of the resulting tensor.
        
        text_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Creates an empty tensor for text features.
        if self.fitted_text_cols:                                                   # If we have text columns...
            combined_texts = X_transformed[self.fitted_text_cols].fillna("").agg(' '.join, axis=1).apply(clean_text) # ...combine and clean the text.
            text_features_raw = self.extract_text_features(combined_texts)          # ...get the full numerical vectors for the text.
            if self.text_pca:                                                       # ...if PCA was fitted for text...
                text_features_raw_np = text_features_raw.cpu().numpy()              # ...convert tensor to a numpy array for PCA.
                text_embeddings = torch.tensor(self.text_pca.transform(text_features_raw_np), dtype=torch.float32, device=self.device) # ...apply PCA to simplify the vectors.
            else:                                                                   # ...if no PCA...
                text_embeddings = text_features_raw                                 # ...use the full, unsimplified vectors.
            print(f"Text embeddings shape (on {self.device}): {text_embeddings.shape}") # Prints the shape of the final text feature tensor.

        image_embeddings = torch.empty(len(X_transformed), 0, dtype=torch.float32, device=self.device) # Creates an empty tensor for image features.
        if self.fitted_image_col:                                                   # If we have an image column...
            image_features_raw = self.extract_image_features(X_transformed[self.fitted_image_col]) # ...get the full numerical vectors for the images.
            if self.image_pca:                                                      # ...if PCA was fitted for images...
                image_features_raw_np = image_features_raw.cpu().numpy()            # ...convert to numpy array.
                image_embeddings = torch.tensor(self.image_pca.transform(image_features_raw_np), dtype=torch.float32, device=self.device) # ...apply PCA to simplify.
            else:                                                                   # ...if no PCA...
                image_embeddings = image_features_raw                               # ...use the full vectors.
            print(f"Image embeddings shape (on {self.device}): {image_embeddings.shape}") # Prints the shape of the final image feature tensor.

        feature_list = [f for f in [ct_features, text_embeddings, image_embeddings] if f.size(1) > 0] # Gathers all the processed features into a list.
        if not feature_list:                                                        # If the list is empty (no features were created)...
            raise ValueError("No features were extracted.")                         # ...raise an error.

        combined_features = torch.cat(feature_list, dim=1)                          # Concatenates (joins) all the feature tensors together side-by-side into one big tensor.
        print(f"Combined features shape (on {self.device}): {combined_features.shape}") # Prints the shape of the final combined feature tensor.
        return combined_features                                                    # Returns the final result.

    def save(self, path: str):                                                      # Defines the method to save the fitted feature extractor.
        """
        Saves the fitted FeatureExtractor components to the specified path.
        """
        ensure_directory_exists(path)                                               # Makes sure the destination folder exists.
        print(f"Saving FeatureExtractor components to {path}...")                   # Prints a status message.

        if self.column_transformer:                                                 # If the column transformer was fitted...
            joblib.dump(self.column_transformer, os.path.join(path, 'column_transformer.pkl')) # ...save it to a file.
        if self.text_pca:                                                           # If text PCA was fitted...
            joblib.dump(self.text_pca, os.path.join(path, 'text_pca.pkl'))           # ...save it to a file.
        if self.image_pca:                                                          # If image PCA was fitted...
            joblib.dump(self.image_pca, os.path.join(path, 'image_pca.pkl'))         # ...save it to a file.
        
        if self.text_model:                                                         # If the text model was loaded...
            self.text_model.save(os.path.join(path, 'sentence_transformer_model')) # ...save the entire model to a folder.
        if self.image_processor and self.image_model:                               # If the image model and processor were loaded...
            self.image_processor.save_pretrained(os.path.join(path, 'image_processor')) # ...save the processor.
            self.image_model.save_pretrained(os.path.join(path, 'image_model'))     # ...save the model.

        metadata = {                                                                # Creates a dictionary to store metadata about the extractor.
            'text_model_name': self.text_model_name,                                # Saves the name of the text model used.
            'image_model_name': self.image_model_name,                              # Saves the name of the image model used.
            'fitted_numerical_cols': self.fitted_numerical_cols,                    # Saves the list of numerical columns.
            'fitted_categorical_cols': self.fitted_categorical_cols,                # Saves the list of categorical columns.
            'fitted_text_cols': self.fitted_text_cols,                              # Saves the list of text columns.
            'fitted_image_col': self.fitted_image_col,                              # Saves the name of the image column.
        }
        joblib.dump(metadata, os.path.join(path, 'feature_extractor_metadata.pkl')) # Saves the metadata to a file.
        print("FeatureExtractor components saved.")                                 # Prints a final success message.

    @classmethod                                                                    # Defines a class method, which can be called without creating an instance of the class first.
    def load(cls, path: str):                                                       # This method loads a previously saved FeatureExtractor.
        """
        Loads a fitted FeatureExtractor from the specified path.
        """
        print(f"Loading FeatureExtractor components from {path}...")               # Prints a status message.
        metadata = joblib.load(os.path.join(path, 'feature_extractor_metadata.pkl')) # Loads the metadata file first.
        
        instance = cls(text_model_name=metadata['text_model_name'],                 # Creates a new, empty FeatureExtractor instance using the saved model names.
                       image_model_name=metadata['image_model_name'])

        if os.path.exists(os.path.join(path, 'column_transformer.pkl')):            # If a saved column transformer exists...
            instance.column_transformer = joblib.load(os.path.join(path, 'column_transformer.pkl')) # ...load it.
        if os.path.exists(os.path.join(path, 'text_pca.pkl')):                      # If a saved text PCA exists...
            instance.text_pca = joblib.load(os.path.join(path, 'text_pca.pkl'))      # ...load it.
        if os.path.exists(os.path.join(path, 'image_pca.pkl')):                     # If a saved image PCA exists...
            instance.image_pca = joblib.load(os.path.join(path, 'image_pca.pkl'))    # ...load it.
        
        text_model_path = os.path.join(path, 'sentence_transformer_model')          # Defines the path to the saved text model folder.
        if os.path.exists(text_model_path):                                         # If the folder exists...
            instance.text_model = SentenceTransformer(text_model_path, device=instance.device) # ...load the model from the local files.
        else:                                                                       # Otherwise...
            print(f"Warning: Local text model not found. Downloading '{instance.text_model_name}'.") # ...print a warning.
            instance.text_model = SentenceTransformer(instance.text_model_name, device=instance.device) # ...download it from the internet again.
        instance.text_model.eval()                                                  # Sets the loaded model to evaluation mode.

        image_processor_path = os.path.join(path, 'image_processor')                # Defines the path to the saved image processor.
        image_model_path = os.path.join(path, 'image_model')                        # Defines the path to the saved image model.
        if os.path.exists(image_processor_path) and os.path.exists(image_model_path): # If both exist...
            instance.image_processor = AutoImageProcessor.from_pretrained(image_processor_path) # ...load the processor from local files.
            instance.image_model = AutoModel.from_pretrained(image_model_path).to(instance.device) # ...load the model from local files.
        else:                                                                       # Otherwise...
            print(f"Warning: Local image model not found. Downloading '{instance.image_model_name}'.") # ...print a warning.
            instance.image_processor = AutoImageProcessor.from_pretrained(instance.image_model_name) # ...download the processor again.
            instance.image_model = AutoModel.from_pretrained(instance.image_model_name).to(instance.device) # ...download the model again.
        instance.image_model.eval()                                                 # Sets the loaded model to evaluation mode.

        instance.fitted_numerical_cols = metadata.get('fitted_numerical_cols', [])  # Loads the list of numerical columns from metadata.
        instance.fitted_categorical_cols = metadata.get('fitted_categorical_cols', [])# Loads the list of categorical columns.
        instance.fitted_text_cols = metadata.get('fitted_text_cols', [])            # Loads the list of text columns.
        instance.fitted_image_col = metadata.get('fitted_image_col')              # Loads the name of the image column.

        print("FeatureExtractor components loaded successfully.")                   # Prints a final success message.
        return instance                                                             # Returns the fully loaded and ready-to-use FeatureExtractor.
