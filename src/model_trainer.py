#   python src/model_trainer.py
from sklearn.model_selection import train_test_split  # type: ignore                                # Imports a function to split data into training and testing sets.
from xgboost import XGBRegressor                                                    # Imports XGBoost, a powerful machine learning algorithm for regression (predicting numbers).
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # type: ignore       # Imports metrics to evaluate how well our model performs.
import joblib  # type: ignore                                                                       # Imports joblib, used for saving and loading Python objects, like our trained models.
import os                                                                           # Imports the os library, which allows the script to interact with the operating system (e.g., file paths).
import torch                                                                        # Imports PyTorch, a machine learning library, used here to check for GPU availability.
import optuna                                                                       # Imports Optuna, a library for automating hyperparameter optimization (finding the best settings for our model).
import json                                                                         # Imports json, used for working with JSON data (like saving model settings).
import sys                                                                          # Imports the sys library, which provides access to system-specific parameters and functions.

# Add the parent directory of the current script to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Adds the project's main folder to Python's search path.

# Import custom modules
from src.data_loader import load_content_data, CONTENT_COLUMN_MAPPING               # Imports functions and configurations for loading and understanding our data.
from src.feature_extractor import FeatureExtractor                                  # Imports the FeatureExtractor class, which turns raw data into numbers for the model.
from src.utils import ensure_directory_exists                                       # Imports a utility function to make sure directories exist.

def objective(trial, X_train, y_train, X_test, y_test):                             # Defines the objective function for Optuna, which Optuna tries to optimize.
    """
    Objective function for Optuna to optimize.
    """
    param = {                                                                       # Defines a dictionary of parameters (settings) for the XGBoost model.
        'objective': 'reg:squarederror',                                            # Sets the model's goal: to minimize the squared error (good for regression).
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),               # Optuna will try different numbers of boosting rounds (trees) between 100 and 2000.
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),           # Optuna will try different learning rates (how much each tree contributes) between 0.01 and 0.3.
        'max_depth': trial.suggest_int('max_depth', 3, 10),                         # Optuna will try different maximum depths for each tree between 3 and 10.
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),                   # Optuna will try different subsample ratios (fraction of data used per tree) between 0.6 and 1.0.
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),     # Optuna will try different column subsample ratios (fraction of features used per tree) between 0.6 and 1.0.
        'tree_method': 'hist',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',                  # Tells XGBoost to use the GPU if available, otherwise the CPU.
        'n_jobs': -1                                                                # Uses all available CPU cores for parallel processing.
    }

    model = XGBRegressor(**param)                                                   # Creates an XGBoost Regressor model with the chosen parameters.
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)         # Trains the model using the training data and evaluates it on the test data.
    preds = model.predict(X_test)                                                   # Makes predictions on the test data.
    r2 = r2_score(y_test, preds)                                                    # Calculates the R-squared score, which measures how well the predictions match the actual values.
    return r2                                                                       # Optuna tries to maximize this R-squared score.

def train_model(content_type: str, data_base_dir: str, model_base_dir: str):        # Defines the main function to train a model for a specific content type.
    """
    Loads data for a specific content type, extracts features, trains an XGBoost Regressor model,
    and saves the trained model and feature extractor components in a content-specific directory.

    Args:
        content_type (str): The type of content to train the model for (e.g., "Game", "Show", "Movie").
        data_base_dir (str): Base directory where content CSVs are located (e.g., '../data').
        model_base_dir (str): Base directory to save the trained model and feature extractor (e.g., '../models').
    """
    print(f"--- Starting Model Training for {content_type} ---\n")                  # Prints a message indicating the start of training for a content type.

    # 1. Load Data
    df = load_content_data(content_type)                                            # Loads and preprocesses the data for the specified content type.
    print(f"\nShow rating distribution:\n{df['my_rating'].value_counts()}")       # Prints how many times each rating appears in the data.
    print(f"Show like/dislike distribution:\n{df['like_dislike'].value_counts()}") # Prints how many items are liked vs. disliked.
    if df.empty:                                                                    # Checks if the data loading failed.
        print(f"Data loading failed for {content_type}. Exiting training.")        # If so, prints an error.
        return                                                                      # And stops the training process.

    # Check if the target column 'my_rating' and 'like_dislike' were created
    my_rating_col = CONTENT_COLUMN_MAPPING[content_type].get("my_rating")           # Gets the name of the rating column from our configuration.
    if my_rating_col is None or 'like_dislike' not in df.columns:                   # Checks if the rating column or the like/dislike column is missing.
        print(f"Skipping training for {content_type}: 'my_rating' or 'like_dislike' column not found or not applicable.") # If so, prints a message.
        return                                                                      # And stops training.

    # Ensure there's enough data for training after dropping NaNs
    if len(df) < 2:                                                                 # Checks if there are at least 2 rows of data (needed for splitting).
        print(f"Not enough data for {content_type} after cleaning ({len(df)} samples). Skipping training.") # If not, prints a message.
        return                                                                      # And stops training.


    # Define features and target
    X = df.drop(columns=[my_rating_col, 'like_dislike'], errors='ignore')           # Creates 'X', the input features for the model, by removing the rating columns.
    y = df[my_rating_col]                                                           # Creates 'y', the target variable (the actual ratings) that the model will predict.

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splits the data into 80% for training and 20% for testing.
    print(f"Data split: Train samples = {len(X_train)}, Test samples = {len(X_test)}") # Prints the number of samples in each set.

    # 3. Initialize and Fit Feature Extractor
    feature_extractor = FeatureExtractor()                                          # Creates a new FeatureExtractor object.
    feature_extractor.fit(X_train, content_type=content_type)                       # Teaches the FeatureExtractor how to process the training data for this content type.

    # 4. Transform Data
    print("Transforming training data...")                                         # Prints a status message.
    X_train_transformed = feature_extractor.transform(X_train)                      # Uses the FeatureExtractor to convert the training data into numerical features.
    print("Transforming test data...")                                             # Prints a status message.
    X_test_transformed = feature_extractor.transform(X_test)                        # Uses the FeatureExtractor to convert the test data into numerical features.

    # Converts the target variables (y_train, y_test) into a format that XGBoost can use.
    if isinstance(y_train, torch.Tensor):                                           # If y_train is a PyTorch tensor...
        y_train_final = y_train.cpu().numpy()                                       # ...convert it to a NumPy array on the CPU.
    else:
        y_train_final = y_train.values                                              # Otherwise, convert it from a pandas Series to a NumPy array.
    if isinstance(y_test, torch.Tensor):                                            # Same conversion for y_test.
        y_test_final = y_test.cpu().numpy()
    else:
        y_test_final = y_test.values


    # 5. Hyperparameter Tuning or Loading
    model_dir = os.path.join(model_base_dir, content_type.lower())                  # Defines the directory where the model and its settings will be saved.
    params_path = os.path.join(model_dir, 'best_params.json')                       # Defines the file path for saving the best model settings.
    best_params = None                                                              # Initializes a variable to hold the best settings.

    if os.path.exists(params_path):                                                 # Checks if a file with best settings already exists.
        print(f"Found existing best parameters for {content_type}. Loading them.") # If so, prints a message.
        with open(params_path, 'r') as f:                                          # Opens the file for reading.
            best_params = json.load(f)                                              # Loads the settings from the JSON file.
        print("Loaded parameters:", best_params)                                    # Prints the loaded settings.
    else:                                                                           # If no best settings file is found...
        print(f"No best parameters found for {content_type}. Running Optuna study.") # ...prints a message that Optuna will run.
        study = optuna.create_study(direction='maximize')                           # Creates an Optuna study, aiming to maximize the R-squared score.
        study.optimize(lambda trial: objective(trial, X_train_transformed, y_train_final, X_test_transformed, y_test_final), n_trials=100) # Runs the optimization for 100 trials.

        print("Best trial:")                                                        # Prints a header for the best trial results.
        trial = study.best_trial                                                    # Gets the best trial found by Optuna.
        best_params = trial.params                                                  # Extracts the best parameters from that trial.

        print(f"  Value: {trial.value}")                                           # Prints the R-squared value achieved by the best parameters.
        print("  Params: ")                                                         # Prints a header for the parameters.
        for key, value in best_params.items():                                      # Loops through each best parameter.
            print(f"    {key}: {value}")                                           # Prints the parameter name and its value.

        ensure_directory_exists(model_dir)                                          # Makes sure the directory for saving the model exists.
        with open(params_path, 'w') as f:                                          # Opens a file for writing the best parameters.
            json.dump(best_params, f, indent=4)                                     # Saves the best parameters to the JSON file in a readable format.
        print(f"Saved best parameters to {params_path}")                           # Prints a confirmation message.

    # 6. Train Final Model with Best Parameters
    print("Training XGBoost Regressor model with best parameters...")              # Prints a status message.
    best_params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'          # Ensures the model uses the GPU if available.

    xgb_model = XGBRegressor(objective='reg:squarederror', **best_params)          # Creates the final XGBoost model using the best parameters found.

    xgb_model.fit(X_train_transformed, y_train_final, eval_set=[(X_test_transformed, y_test_final)], # Trains the final model.
    verbose=False)                                                                  # Does not print detailed training progress.
    print("XGBoost model training complete.")                                       # Prints a success message.

    # 7. Evaluate Model
    print("\n--- Evaluating Model ---")                                            # Prints a header for model evaluation.
    y_pred = xgb_model.predict(X_test_transformed)                                  # Uses the trained model to make predictions on the test data.

    # This block handles potential CuPy arrays if XGBoost used GPU and returned them.
    try:                                                                            # Starts a 'try' block.
        import cupy  # type: ignore                                                          # Tries to import CuPy (a library for GPU arrays).
        if isinstance(y_pred, cupy.ndarray):                                        # If the predictions are a CuPy array...
            y_pred = y_pred.get()                                                   # ...convert them to a NumPy array.
    except ImportError:                                                             # If CuPy is not installed...
        pass                                                                        # ...do nothing.


    mae = mean_absolute_error(y_test_final, y_pred)                                 # Calculates the Mean Absolute Error (average difference between predicted and actual).
    rmse = mean_squared_error(y_test_final, y_pred, squared=False)                        # Calculates the Root Mean Squared Error (another measure of prediction accuracy).
    r2 = r2_score(y_test_final, y_pred)                                             # Calculates the R-squared score again for the final model.

    print(f"Mean Absolute Error (MAE): {mae:.4f}")                                 # Prints the MAE, formatted to 4 decimal places.
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")                           # Prints the RMSE, formatted to 4 decimal places.
    print(f"R-squared (R2): {r2:.4f}")                                             # Prints the R-squared, formatted to 4 decimal places.

    # 8. Save Model and Feature Extractor
    diagnostics_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics', content_type.lower()) # Defines the directory for saving diagnostic plots.
    print(f"\n--- Saving trained model and feature extractor to {model_dir} ---\n") # Prints a status message.
    ensure_directory_exists(model_dir)                                              # Makes sure the model saving directory exists.
    ensure_directory_exists(diagnostics_dir)                                        # Makes sure the diagnostics directory exists.

    joblib.dump(xgb_model, os.path.join(model_dir, f'xgboost_model_{content_type.lower()}.pkl')) # Saves the trained XGBoost model to a file.
    feature_extractor.save(os.path.join(model_dir, 'feature_extractor'))            # Saves the fitted FeatureExtractor to a directory.

    # 9. Save Diagnostics
    import matplotlib.pyplot as plt                                                 # Imports matplotlib for plotting.
    from xgboost import plot_importance                                             # Imports a function from XGBoost to plot feature importance.

    fig, ax = plt.subplots(figsize=(10, 8))                                         # Creates a new plot figure and axes.
    plot_importance(xgb_model, ax=ax, max_num_features=20, title=f'Feature Importance for {content_type}') # Plots the importance of each feature in the model.
    plt.tight_layout()                                                              # Adjusts plot to prevent labels from overlapping.
    plot_path = os.path.join(diagnostics_dir, f'{content_type}_feature_importance.png') # Defines the file path for saving the plot.
    plt.savefig(plot_path)                                                          # Saves the plot to a PNG file.
    plt.close(fig)                                                                  # Closes the plot figure to free up memory.
    print(f"Saved feature importance plot to {plot_path}")                         # Prints a confirmation message.

    print("Training process complete for", content_type)                            # Prints a final message for the completed training.

if __name__ == "__main__":                                                          # Checks if this script is the main program being run.
    current_script_dir = os.path.join(os.path.dirname(__file__))                                  # Gets the directory where this script is located.
    data_base_path = os.path.join(current_script_dir, '..', 'data')                  # Defines the base directory for data files.
    model_base_path = os.path.join(current_script_dir, '..', 'models')               # Defines the base directory for saving models.

    # List content types to train models for
    trainable_content_types = [                                                     # Defines a list of content types for which models will be trained.
        "Game",
        "Show",
        "Movie",
        "Book",
        "Music",
    ]

    for content_type in trainable_content_types:                                    # Loops through each content type in the list.
        try:                                                                        # Starts a 'try' block to catch errors during training.
            train_model(content_type, data_base_path, model_base_path)              # Calls the train_model function for the current content type.
            print(f"\nSuccessfully trained model for {content_type}.\n")           # Prints a success message.
        except Exception as e:                                                      # If any error occurs during training...
            print(f"\nError training model for {content_type}: {e}\n")             # ...print an error message.
            import traceback                                                        # Imports traceback for detailed error info.
            traceback.print_exc()                                                   # Prints the full technical error details.
        print("\n" + "="*80 + "\n")                                                # Prints a separator line after each content type's training.
