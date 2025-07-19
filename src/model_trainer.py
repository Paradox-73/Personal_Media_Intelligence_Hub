#   python src/model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import torch # To check CUDA availability
import optuna
import json
import sys

# Add the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from src.data_loader import load_content_data, CONTENT_COLUMN_MAPPING
from src.feature_extractor import FeatureExtractor
from src.utils import ensure_directory_exists

def objective(trial, X_train, y_train, X_test, y_test):
    """
    Objective function for Optuna to optimize.
    """
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'n_jobs': -1
    }
    
    model = XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    return r2

def train_model(content_type: str, data_base_dir: str, model_base_dir: str):
    """
    Loads data for a specific content type, extracts features, trains an XGBoost Regressor model,
    and saves the trained model and feature extractor components in a content-specific directory.

    Args:
        content_type (str): The type of content to train the model for (e.g., "Game", "Show", "Movie").
        data_base_dir (str): Base directory where content CSVs are located (e.g., '../data').
        model_base_dir (str): Base directory to save the trained model and feature extractor (e.g., '../models').
    """
    print(f"--- Starting Model Training for {content_type} ---\n")

    # 1. Load Data
    # The data_loader now handles the full path based on its internal logic,
    # so we just pass the content_type.
    df = load_content_data(content_type)
    print(f"\nShow rating distribution:\n{df['my_rating'].value_counts()}")
    print(f"Show like/dislike distribution:\n{df['like_dislike'].value_counts()}")
    if df.empty:
        print(f"Data loading failed for {content_type}. Exiting training.")
        return

    # Check if the target column 'my_rating' and 'like_dislike' were created
    my_rating_col = CONTENT_COLUMN_MAPPING[content_type].get("my_rating")
    if my_rating_col is None or 'like_dislike' not in df.columns:
        print(f"Skipping training for {content_type}: 'my_rating' or 'like_dislike' column not found or not applicable.")
        return
    
    # Ensure there's enough data for training after dropping NaNs
    if len(df) < 2: # Need at least two samples for train_test_split
        print(f"Not enough data for {content_type} after cleaning ({len(df)} samples). Skipping training.")
        return


    # Define features and target
    # X will be the DataFrame containing all relevant features for extraction
    X = df.drop(columns=[my_rating_col, 'like_dislike'], errors='ignore') # Drop target and derived target
    y = df[my_rating_col] # The actual rating for regression

    # 2. Split Data
    # Ensure consistent splitting with random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split: Train samples = {len(X_train)}, Test samples = {len(X_test)}")

    # 3. Initialize and Fit Feature Extractor
    feature_extractor = FeatureExtractor()
    # Fit the feature extractor specifically for this content type's data
    feature_extractor.fit(X_train, content_type=content_type)

    # 4. Transform Data
    print("Transforming training data...")
    # X_train_transformed will now be a torch.Tensor on the correct device
    X_train_transformed = feature_extractor.transform(X_train)
    print("Transforming test data...")
    # X_test_transformed will now be a torch.Tensor on the correct device
    X_test_transformed = feature_extractor.transform(X_test)
    
    # y_train_final and y_test_final should be numpy arrays for XGBoost
    if isinstance(y_train, torch.Tensor):
        y_train_final = y_train.cpu().numpy()
    else:
        y_train_final = y_train.values # Convert Series to numpy array
    if isinstance(y_test, torch.Tensor):
        y_test_final = y_test.cpu().numpy()
    else:
        y_test_final = y_test.values # Convert Series to numpy array


    # 5. Hyperparameter Tuning or Loading
    model_dir = os.path.join(model_base_dir, content_type.lower())
    params_path = os.path.join(model_dir, 'best_params.json')
    best_params = None

    if os.path.exists(params_path):
        print(f"Found existing best parameters for {content_type}. Loading them.")
        with open(params_path, 'r') as f:
            best_params = json.load(f)
        print("Loaded parameters:", best_params)
    else:
        print(f"No best parameters found for {content_type}. Running Optuna study.")
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, X_train_transformed, y_train_final, X_test_transformed, y_test_final), n_trials=100)

        print("Best trial:")
        trial = study.best_trial
        best_params = trial.params

        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # Save the best parameters for future runs
        ensure_directory_exists(model_dir)
        with open(params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        print(f"Saved best parameters to {params_path}")

    # 6. Train Final Model with Best Parameters
    print("Training XGBoost Regressor model with best parameters...")
    best_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    xgb_model = XGBRegressor(objective='reg:squarederror', **best_params)
    
    # Pass PyTorch tensors directly to xgb_model.fit. XGBoost with gpu_hist can handle them.
    xgb_model.fit(X_train_transformed, y_train_final, eval_set=[(X_test_transformed, y_test_final)],
    verbose=False)
    print("XGBoost model training complete.")

    # 7. Evaluate Model
    print("\n--- Evaluating Model ---")
    # Pass PyTorch tensors directly to xgb_model.predict
    y_pred = xgb_model.predict(X_test_transformed)

    # Ensure y_pred is a numpy array if it came from cupy (when using GPU in XGBoost)
    # This block handles potential CuPy arrays if XGBoost used GPU and returned them.
    try:
        import cupy
        if isinstance(y_pred, cupy.ndarray):
            y_pred = y_pred.get() # Convert cupy array to numpy array
    except ImportError:
        pass # cupy not installed, so it's not a cupy array


    mae = mean_absolute_error(y_test_final, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
    r2 = r2_score(y_test_final, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    # 8. Save Model and Feature Extractor
    # Create content-specific model directory
    diagnostics_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', content_type.lower())
    print(f"\n--- Saving trained model and feature extractor to {model_dir} ---\n")
    ensure_directory_exists(model_dir)
    ensure_directory_exists(diagnostics_dir)

    joblib.dump(xgb_model, os.path.join(model_dir, f'xgboost_model_{content_type.lower()}.pkl'))
    feature_extractor.save(os.path.join(model_dir, 'feature_extractor'))

    # 9. Save Diagnostics
    import matplotlib.pyplot as plt
    from xgboost import plot_importance

    fig, ax = plt.subplots(figsize=(10, 8))
    plot_importance(xgb_model, ax=ax, max_num_features=20, title=f'Feature Importance for {content_type}')
    plt.tight_layout()
    plot_path = os.path.join(diagnostics_dir, f'{content_type}_feature_importance.png')
    plt.savefig(plot_path)
    plt.close(fig) # Close the figure to free memory
    print(f"Saved feature importance plot to {plot_path}")

    print("Training process complete for", content_type)

if __name__ == "__main__":
    # Define base directories
    current_script_dir = os.path.dirname(__file__)
    data_base_path = os.path.join(current_script_dir, '..', 'data')
    model_base_path = os.path.join(current_script_dir, '..', 'models')

    # List content types to train models for
    # Only train for content types that have 'my_rating' as a target
    trainable_content_types = [
        "Game",
        "Show", # Combines shows_data.csv and neg_shows_data.csv
        "Movie",
        "Book",
        "Music", # Enabled for training
    ]

    for content_type in trainable_content_types:
        try:
            train_model(content_type, data_base_path, model_base_path)
            print(f"\nSuccessfully trained model for {content_type}.\n")
        except Exception as e:
            print(f"\nError training model for {content_type}: {e}\n")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
        print("\n" + "="*80 + "\n") # Separator for different content types
