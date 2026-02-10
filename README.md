# Personal Letterboxd Recommendation Engine (PLRE)

## 🎬 Introduction

The Personal Letterboxd Recommendation Engine (PLRE) is a machine learning-based application designed to predict a single user's movie ratings and provide content-based recommendations based on their Letterboxd viewing history. It enriches your data with information from TMDB and OMDB, visualizes your taste patterns through an interactive dashboard, and uses a hybrid prediction model to score unseen movies.

## ✨ Features

*   **Custom Data Ingestion**: Processes your Letterboxd `ratings.csv` (explicit ratings) and `liked.csv` (movies you've marked as liked, even if not rated).
*   **Dual-API Enrichment**: Fetches comprehensive movie metadata from TMDB (e.g., cast, crew, genres, keywords) and critic scores from OMDB (e.g., Rotten Tomatoes, Metacritic).
*   **Incremental & Resilient Data Fetching**: Only fetches data for new movies and saves progress after each movie, preventing data loss on interruption.
*   **Interactive EDA Dashboard**: Visualizes your rating behaviors against various metadata, helping you understand your unique cinematic preferences.
*   **Hybrid Recommendation Engine**:
    *   **Regression Score**: Predicts an exact rating (0.5 - 5.0) for new movies.
    *   **Classification Verdict**: Categorizes predictions into "Bad," "Ok," or "Great" with probability scores.
    *   **Content-Based Filtering**: Finds movies from your history that are most similar to a queried film.
*   **Streamlit Web Application**: An intuitive user interface to interact with the dashboard and the prediction "Oracle."

## 🚀 Setup

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/your-username/Personal_Media_Intelligence_Hub.git
cd Personal_Media_Intelligence_Hub
```

### 2. Install Dependencies

Install all required Python packages using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Configure API Keys

The application requires API keys for TMDB and OMDB to fetch movie data.

*   **TMDB API Key**: Obtain one from [The Movie Database (TMDB)](https://www.themoviedb.org/documentation/api).
*   **OMDB API Key**: Obtain one from [The Open Movie Database (OMDB)](http://www.omdbapi.com/apikey.aspx).

Create a file named `.env` in the root directory of the project and add your API keys as follows:

```
TMDB_API_KEY=your_tmdb_api_key_here
OMDB_API_KEY=your_omdb_api_key_here
```
**Important**: Replace `your_tmdb_api_key_here` and `your_omdb_api_key_here` with your actual keys.

### 4. Add Your Letterboxd Data

Place your Letterboxd movie export files into the `data/raw/` directory:

*   `ratings.csv`: This file should contain your explicit movie ratings.
    *   Expected columns: `Date`, `Name`, `Year`, `Letterboxd URI`, `Rating`
*   `liked.csv`: This file should contain movies you have liked but not necessarily rated.
    *   Expected columns: `Date`, `Name`, `Year`, `Letterboxd URI`

**Note**: Movies present in both `ratings.csv` and `liked.csv` will use the explicit rating from `ratings.csv`. Movies only in `liked.csv` will be assigned a default rating of `4.0` for training purposes.

## 🏃‍♀️ Usage

The pipeline is now modularized into separate steps for better organization and flexibility.

### Step 1: Run the Full Pipeline (Or individual steps)

You have two options to run the pipeline:

#### Option A: Run the entire pipeline sequentially (Recommended for first-time setup)

The `run_pipeline.py` script will execute all steps from data fetching to model training.

```bash
python run_pipeline.py
```

**Optional**: To quickly test the pipeline with a smaller dataset (e.g., the first 50 new movies), you can use the `--limit` argument:

```bash
python run_pipeline.py --limit 50
```

#### Option B: Run individual steps

You can run each step independently. This is useful for debugging or if you only need to re-run a specific part of the pipeline.

1.  **Fetch Data**: Gathers movie metadata from TMDB and OMDB, updating `data/processed/full_dataset.csv`.
    ```bash
    python fetch_data.py [--limit N]
    ```
    (Use `--limit N` as in `run_pipeline.py` if needed).

2.  **Preprocess Data**: Cleans and engineers features from `full_dataset.csv` into `data/processed/training_features.csv`.
    ```bash
    python preprocess_data.py
    ```

3.  **Train Model**: Trains the ML models using `training_features.csv` and saves the trained model to `models/plre_model_pipeline.pkl`.
    ```bash
    python train_model.py
    ```

### Step 2: Run the Streamlit Web Application

Once the model is trained, launch the interactive web application:

```bash
streamlit run app/main.py
```

Your web browser should automatically open to the Streamlit application. If not, open your browser and navigate to `http://localhost:8501`.

*   Navigate to the **Dashboard** page to explore visualizations of your movie taste.
*   Go to **The Oracle** page to get personalized rating predictions for any movie.

## 📁 Project Structure

```
.
├── README.md
├── requirements.txt
├── .env                  # API keys and environment variables
├── .gitignore
├── run_pipeline.py       # Orchestrates the full data pipeline and model training
├── fetch_data.py         # Script for data ingestion and API calls
├── preprocess_data.py    # Script for feature engineering and data cleaning
├── train_model.py        # Script for training the ML models
├── data/
│   ├── raw/
│   │   ├── ratings.csv   # Your Letterboxd ratings export
│   │   └── liked.csv     # Your Letterboxd liked movies export
│   ├── cache/            # Stores TMDB and OMDB JSON responses to prevent redundant API calls
│   │   ├── tmdb_movies.json
│   │   └── omdb_movies.json
│   └── processed/        # Stores processed dataframes
│       ├── full_dataset.csv      # Master enriched data (post-API calls)
│       └── training_features.csv # Data prepared for model training
├── models/               # Stores the trained machine learning model pipeline
│   └── plre_model_pipeline.pkl
├── src/                  # Core Python logic (functions/classes used by the scripts)
│   ├── config.py         # Global configuration settings
│   ├── data_loader.py    # Functions for data ingestion and API calls
│   ├── preprocessing.py  # Functions for feature engineering and data cleaning
│   └── models.py         # Functions for defining and training the ML prediction models
└── app/                  # Streamlit web application files
    ├── main.py           # Streamlit entry point (Home page)
    ├── utils.py          # UI helper functions (e.g., star rendering)
    └── pages/            # Individual Streamlit pages
        ├── 1_Dashboard.py
        └── 2_Oracle.py
```

## 🛠 Technologies Used

*   **Python**
*   **pandas**: Data manipulation and analysis
*   **scikit-learn**: Machine learning preprocessing and modeling
*   **XGBoost**: Gradient boosting for regression
*   **streamlit**: Web application framework
*   **plotly**: Interactive data visualizations
*   **tmdbv3api**: Python wrapper for TMDB API
*   **omdb**: Python wrapper for OMDB API
*   **python-dotenv**: Loading environment variables

## 💡 Future Improvements (Phase 2 from SRS)

*   Incorporate more complex features like `Writer`, `Cinematographer`, and `Production Company` if initial model accuracy is promising.
*   Further refine UI/UX based on user feedback.
*   Implement advanced model evaluation metrics and hyperparameter tuning.
