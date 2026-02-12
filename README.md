# Personal Movie Intelligence Hub

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Machine Learning & Recommender Systems](#machine-learning--recommender-systems)
  - [Detailed Feature Explanation (92 Features)](#detailed-feature-explanation-92-features)
- [Data Pipeline Steps](#data-pipeline-steps)
  - [API Data Enrichment Details](#api-data-enrichment-details)
- [Results and Achievements](#results-and-achievements)
  - [Interpretation of Results](#interpretation-of-results)
- [Code Explanation](#code-explanation)
- [Future Enhancements](#future-enhancements)

## Introduction
The Personal Movie Intelligence Hub is a Streamlit-based application designed to provide users with deep insights into their movie-watching habits and offer personalized movie recommendations. By leveraging user's Letterboxd data and enriching it with comprehensive metadata from TMDB and OMDb, the platform builds a personalized machine learning model to predict movie ratings and suggest similar films.

## Features
- **Dashboard:** Interactive visualizations and analytics of your movie viewing history, providing deep insights into your preferences and viewing patterns. See [Dashboard Visualizations](#dashboard-visualizations) for more details.
- **Oracle (Recommendation Engine):** Predicts your potential star rating for any movie and helps you discover new movies by finding films similar to those you've enjoyed.
- **Data Enrichment:** Automatically fetches detailed movie information (e.g., genre, cast, crew, plot, poster) from TMDB and OMDb APIs.
- **Personalized ML Models:** Trains an XGBoost Regressor to predict star ratings and a Classifier to categorize movies into 'Bad', 'Ok', or 'Great' based on your past preferences.
- **Explainable Recommendations:** Provides insights into why certain movies are recommended by comparing them to your previously watched films.

### Dashboard Visualizations
The Streamlit Dashboard (`app/pages/1_Dashboard.py`) uses the enriched data to generate a wide array of personal analytics and visualizations, offering unique insights into your movie consumption habits:

-   **Core Metrics:** High-level summaries such as total movies watched, your average rating across all films, and the total cumulative watch time.
-   **Time & Duration Analysis:**
    -   **Movies Watched Per Decade:** Bar charts illustrating your viewing activity across different cinematic eras, revealing preferred periods for film.
    -   **Runtime Distribution:** Histograms or density plots showing the distribution of movie runtimes you watch, indicating a preference for shorter, longer, or average-length films.
-   **Cast & Crew Analysis:**
    -   **Most Watched/Highest Rated Directors, Actors, and Writers:** Identification and ranking of individuals whose work you frequently watch or consistently rate highly, highlighting your affinities for specific creative talents.
-   **Critic & Financial Alignment:**
    -   **Your Ratings vs. Metacritic/Rotten Tomatoes:** Comparative plots to see how your personal ratings align with professional critics' scores, answering whether you tend to agree or disagree with critical consensus.
    -   **Your Ratings vs. Box Office:** Analysis of your ratings against a movie's financial performance, revealing if you prefer mainstream blockbusters or more niche, independent films.
-   **The 'Hipster Index':** A scatter plot comparing IMDb vote counts (a proxy for popularity) against your personal ratings. This helps you understand if you tend to favor highly popular films or discover and appreciate more obscure "hidden gems."
-   **'Hot Takes':** A direct comparison of your personal ratings versus IMDb ratings, specifically highlighting movies where your rating significantly deviates (either much higher or much lower) from the public's average, indicating your "hot takes" or unique perspectives.
-   **Sentiment Analysis:** Utilizes TextBlob to perform sentiment analysis on movie plot summaries (`overview` and `tagline`). Visualizations show whether your watched films tend to be more "dark/tragic," "happy/uplifting," or neutral, and whether the sentiment correlates with your ratings.

## Project Structure
The project is organized into logical directories to maintain clarity and scalability:

-   `app/`: Contains the Streamlit application code.
    -   `main.py`: The main entry point for the Streamlit application, defining the multi-page structure.
    -   `pages/`: Individual Streamlit pages.
        -   `1_Dashboard.py`: Implements the interactive data visualization dashboard.
        -   `2_Oracle.py`: Implements the movie prediction and recommendation engine.
-   `content_rec/`: (Likely virtual environment or dependency related files)
-   `data/`: Stores all data related to the project.
    -   `cache/`: Cached API responses or intermediate data.
    -   `letterboxd/`: Original Letterboxd export files.
    -   `predictions/`: Output of model predictions.
    -   `processed/`: Enriched and preprocessed data (`enriched_data.csv`).
    -   `raw/`: Raw input data (e.g., `ratings.csv`, `liked.csv`).
-   `models/`: Stores trained machine learning models and preprocessing objects.
-   `src/`: Source code for data processing, feature engineering, model training, and utility functions.
    -   `config.py`: Centralized configuration for file paths, API keys, and model parameters.
    -   `data_ingestion.py`: Script for ingesting raw data and enriching it with external APIs.
    -   `feature_engineering.py`: Script for creating features from enriched data for model training.
    -   `model_trainer.py`: Script for training and saving the machine learning models.
    -   `predict_ratings.py`: Script for batch prediction using trained models.

## Technology Stack
-   **Core Language:** Python
-   **Web Framework:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** XGBoost, Scikit-learn
-   **Data Visualization:** Plotly
-   **API Interaction:** `tmdbv3api`, `omdb` (used within `data_ingestion.py`)
-   **Utilities:** `tqdm` (progress bars), `python-dotenv` (environment variables), `joblib` (model persistence), `statsmodels` (statistical modeling - potentially for analysis in Dashboard)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Personal_Media_Intelligence_Hub.git
    cd Personal_Media_Intelligence_Hub
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv content_rec
    .\content_rec\Scripts\activate # On Windows
    # source content_rec/bin/activate # On macOS/Linux
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    *   Export your movie data from Letterboxd. Ensure you have `ratings.csv` and `liked.csv`.
    *   Place `ratings.csv` and `liked.csv` into the `data/raw/` directory.

5.  **Set up API Keys:**
    *   Obtain API keys from [TMDB](https://www.themoviedb.org/documentation/api) and [OMDb](http://www.omdbapi.com/apikey.aspx).
    *   Create a `.env` file in the project root directory with the following content:
        ```
        TMDB_API_KEY="YOUR_TMDB_API_KEY"
        OMDB_API_KEY="YOUR_OMDB_API_KEY"
        ```

6.  **Run Data Ingestion:**
    This step enriches your raw Letterboxd data with detailed movie information.
    ```bash
    python src/data_ingestion.py
    ```
    This will generate `enriched_data.csv` in the `data/processed/` directory.

7.  **Train Machine Learning Models:**
    This step trains the personalized rating prediction and classification models.
    ```bash
    python src/model_trainer.py
    ```
    This will save the trained models and a preprocessor object in the `models/` directory.

## Usage

1.  **Launch the Streamlit Application:**
    ```bash
    streamlit run app/main.py
    ```
2.  Open your web browser to the address provided by Streamlit (usually `http://localhost:8501`).
3.  Navigate between the "Dashboard" and "Oracle" pages to explore your data and get recommendations.

## Machine Learning & Recommender Systems
The core of the recommendation system involves:

1.  **Data Preprocessing and Feature Engineering (`src/feature_engineering.py`):**
    *   Enriched movie data is transformed into numerical features suitable for machine learning.
    *   This includes encoding categorical variables, scaling numerical features, and creating interaction terms.
    *   **Movie Similarity Calculation:** When finding similar movies in the Oracle page, `cosine_similarity` from `sklearn.metrics.pairwise` is used. This calculates the similarity between a given movie's feature vector and the feature vectors of all other movies in the dataset, which are generated on the fly.

2.  **Model Training (`src/model_trainer.py`):**
    *   **Rating Prediction (Regression):** An XGBoost Regressor is trained on your historical star ratings to predict how you would rate unseen movies.
    *   **Verdict Classification:** An XGBoost Classifier is trained to classify movies into broader categories (e.g., 'Bad', 'Ok', 'Great') based on your ratings, providing a simpler verdict.
    *   Trained models and the feature preprocessor are serialized using `joblib` and saved for later use by the Streamlit app.

3.  **Prediction and Recommendation (`app/pages/2_Oracle.py` & `src/predict_ratings.py`):**
    *   When you search for a movie in the Oracle, its data is fetched, preprocessed using the *same* preprocessor used during training, and fed into the trained models to get a predicted rating.
    *   Similar movie recommendations are generated by comparing the feature vector of the searched movie with your previously watched movies, using the engineered features.

### Detailed Feature Explanation (92 Features)
The 92 features used for model training are generated in the `process_features` function within `src/feature_engineering.py`. They are derived from data ingested and enriched by `src/data_ingestion.py`, which sources data from user-provided Letterboxd CSVs and enriches it with the TMDB and OMDb APIs. The features break down into the following categories:

-   **Numeric Features (7):** These are direct numerical attributes of the movies. Missing values are imputed with the median during preprocessing.
    -   `year`: The release year of the movie.
    -   `runtime`: The duration of the movie in minutes.
    -   `imdb_rating`: The rating from IMDb.
    -   `metascore`: The Metascore from Metacritic.
    -   `rotten_tomatoes_rating`: The rating from Rotten Tomatoes.
    -   `vote_average`: The average vote from TMDB.
    -   `popularity`: A popularity score from TMDB.

-   **Text Features (8):** These features capture the semantic content of the movie's description.
    -   `overview_tagline_pca_0` to `overview_tagline_pca_7`: The `overview` (plot summary) and `tagline` are combined into a single text field. A TF-IDF (Term Frequency-Inverse Document Frequency) vectorization is applied, generating a sparse matrix of up to 500 features. To reduce dimensionality and capture the most important textual patterns, Principal Component Analysis (PCA) is then applied to reduce these to 8 principal components. These 8 abstract features represent the underlying themes and topics in the movie's textual content.

-   **Director Features (Variable Number):** These are one-hot encoded features representing the movie's director(s).
    -   `dir_DirectorName`: A binary feature (0 or 1) for each director. To manage the high cardinality of directors, only directors who have directed at least 4 movies in the user's dataset get their own dedicated column.
    -   `dir_Other_Director`: A single binary feature that captures all other directors who do not meet the 4-movie threshold.

-   **Actor Features (Variable Number):** Similar to director features, these are one-hot encoded features for the primary actor(s).
    -   `act_ActorName`: A binary feature for each actor. To control dimensionality, an actor must have appeared in at least 8 movies in the user's dataset to get a dedicated column.
    -   `act_Other_Actor`: A single binary feature that captures all other actors who do not meet the 8-movie threshold.

-   **Genre Features (Variable Number):** These features represent the genres a movie belongs to using multi-hot encoding.
    -   `gen_GenreName`: A binary feature for each unique genre found in the dataset (e.g., `gen_Action`, `gen_Comedy`, `gen_Drama`). A movie can belong to multiple genres, so it will have a '1' for each relevant genre feature.

The total of 92 features is dynamically composed based on the specific dataset's unique directors, actors, and genres that meet the defined thresholds.

## Data Pipeline Steps
1.  **Raw Data Input:** User's Letterboxd `ratings.csv` and `liked.csv` in `data/raw`.
2.  **Data Ingestion (`src/data_ingestion.py`):**
    *   Reads raw data.
    *   **API Data Enrichment Details:** The `src/data_ingestion.py` script enriches the initial Letterboxd data (which typically contains only movie name, year, and user rating) by fetching comprehensive supplementary details from two external APIs:
        -   **TMDB (The Movie Database):** Provides rich production and content metadata. Key data points fetched include: `genres`, `cast`, `director`, `writer`, `overview` (plot summary), `tagline`, `production_companies`, `vote_average`, and `popularity`. This data is crucial for understanding a film's creative elements, thematic content, and general reception within the movie community.
        -   **OMDb (Open Movie Database):** Provides commercially-oriented and critical rating data. Key data points fetched include: `Metascore` (from Metacritic), `imdbRating`, `imdbVotes`, `BoxOffice` revenue, and `Rotten Tomatoes` scores. This data adds a layer of critical and public reception context, allowing for deeper analysis of how your preferences align with external evaluations and commercial success.
    *   Handles missing data and data cleaning.
    *   Outputs `enriched_data.csv` to `data/processed`.
3.  **Feature Engineering (`src/feature_engineering.py`):**
    *   Loads `enriched_data.csv`.
    *   Transforms raw features into machine-learning-ready features (e.g., one-hot encoding, numerical scaling, PCA for text).
    *   Outputs feature sets for training.
4.  **Model Training (`src/model_trainer.py`):**
    *   Loads engineered features.
    *   Trains XGBoost Regressor and Classifier models.
    *   Saves trained models and preprocessor to `models/`.
5.  **Batch Prediction (Optional - `src/predict_ratings.py`):**
    *   Loads trained models.
    *   Applies models to new data for bulk predictions.
    *   Generates a performance report.
6.  **Streamlit Application (`app/main.py`):**
    *   Loads `enriched_data.csv` for dashboard visualizations.
    *   Loads trained models and preprocessor for real-time predictions and similarity searches in the Oracle page.

## Results and Achievements
The machine learning models achieve strong performance in predicting user ratings and classifying movie verdicts, providing a robust foundation for personalized recommendations.

**ML Engineer Performance Report (from `src/predict_ratings.py`):**
```
🚀 Starting Batch Prediction...
📊 Features for prediction created. Shape: (779, 92)

========================================
📊  ML ENGINEER PERFORMANCE REPORT
========================================
Total Evaluated Samples: 779
----------------------------------------
📉 MAE (Mean Abs Error):  0.3475
📉 RMSE (Root Mean Sq):   0.4556
📈 R² Score:              0.7702
----------------------------------------
🎯 Error Distribution (Absolute Diff):
   Exact Match (0.0):      87  (11.2%)
   Tiny Diff   (0.1-0.4): 463  (59.4%)
   Small Diff  (0.5-0.9): 197  (25.3%)
   Large Diff  (>= 1.0):   32  (4.1%)
========================================
```

**Model Training Summary (from `src/model_trainer.py`):**
```
🤖 Starting Model Training...
   Training Regressor...
   📉 Regressor RMSE: 0.6204
   Training Classifier...
   🎯 Classifier Accuracy: 64.74%
✅ Models saved.
```

### Interpretation of Results
The reported metrics indicate a robust performance for the personalized movie rating prediction model:
-   **MAE (Mean Absolute Error): 0.3475:** This means, on average, the model's predictions are off by only about 0.35 of a star rating. This is a very good score for a 5-star rating system (assuming Letterboxd's half-star ratings), suggesting high precision in predicting user preferences.
-   **RMSE (Root Mean Squared Error): 0.4556:** RMSE penalizes larger errors more heavily than MAE. A value of 0.4556, close to the MAE, indicates that there aren't many significantly large, outlying errors, and the model's predictions are consistently close to the actual ratings.
-   **R² Score: 0.7702:** The R-squared value of 0.7702 (or 77%) signifies that approximately 77% of the variance in the user's movie ratings can be explained by the model. This is a strong indicator that the features and the XGBoost Regressor are effectively capturing the underlying patterns in the user's preferences.
-   **Error Distribution:**
    -   **Exact Match (0.0): 87  (11.2%):** The model perfectly predicted the rating for over 11% of the movies.
    -   **Tiny Diff (0.1-0.4): 463 (59.4%):** Nearly 60% of predictions were within 0.1 to 0.4 of a star, which is an extremely small and often imperceptible difference for a user.
    -   **Small Diff (0.5-0.9): 197 (25.3%):** A quarter of the predictions were off by 0.5 to 0.9 stars. While a noticeable difference, these are still within a reasonable range for personalized recommendations.
    -   **Large Diff (>= 1.0): 32 (4.1%):** Only a small percentage (4.1%) of predictions had an error of 1 star or more. This is positive, indicating that the model rarely makes significantly inaccurate predictions.

Overall, these results suggest that the model is highly effective and reliable in predicting personalized movie ratings, with a strong ability to understand and replicate individual user tastes. The low MAE and RMSE, coupled with a high R² score and a favorable error distribution, demonstrate the model's predictive power. The Classifier's 64.74% accuracy, while lower than the regressor's R-squared, is also a decent starting point for broad verdict classification.

## Code Explanation
-   `app/main.py`: Orchestrates the Streamlit application, serving as the main entry point and defining the overall layout.
-   `app/pages/1_Dashboard.py`: Contains the logic for generating interactive charts and statistics for the user's movie history. Utilizes `pandas` and `plotly` for data analysis and visualization.
-   `app/pages/2_Oracle.py`: Manages user interactions for movie search, prediction, and similarity-based recommendations. It loads pre-trained models and dynamically fetches movie data.
-   `src/config.py`: Defines constants and configurations, centralizing paths to data, models, and API keys. Essential for maintaining a consistent project structure.
-   `src/data_ingestion.py`: Handles the initial data processing, including reading raw Letterboxd CSVs and enriching movie entries with data from TMDB and OMDb. It's crucial for building a comprehensive dataset.
-   `src/feature_engineering.py`: Processes the enriched data, converting raw information into features suitable for machine learning models. This involves tasks like one-hot encoding, text processing, and feature scaling. It also contains the logic for calculating movie similarity.
-   `src/model_trainer.py`: Script responsible for building, training, and saving the XGBoost Regressor and Classifier models. It manages the training pipeline from feature input to model persistence.
-   `src/predict_ratings.py`: A utility script for performing batch predictions using the trained models and generating a performance report, useful for model evaluation and monitoring.

## Future Enhancements
To further enhance the Personal Movie Intelligence Hub, several avenues can be explored:

1.  **Advanced Model Techniques:**
    *   **Neural Networks:** Experiment with deep learning models (e.g., neural collaborative filtering) for potentially capturing more complex user-item interactions.
    *   **Ensemble Methods:** Combine predictions from multiple models (e.g., different regressors or classifiers) to improve robustness and accuracy.
    *   **Hyperparameter Optimization:** Implement more sophisticated hyperparameter tuning strategies (e.g., Bayesian optimization) for XGBoost and other models to squeeze out more performance.

2.  **Feature Engineering Improvements:**
    *   **Temporal Features:** Incorporate features related to when a movie was watched, release date relative to watch date, or trends in user preferences over time.
    *   **User Embeddings:** Generate user embeddings based on their watched movies or explicit feedback to capture more nuanced user preferences.
    *   **External Data Sources:** Integrate data from other movie platforms or review sites to enrich the dataset further.
    *   **More Sophisticated Text Features:** Explore transformer-based models (e.g., BERT embeddings) for movie overviews and taglines instead of TF-IDF + PCA.

3.  **Expanded Recommendation Paradigms:**
    *   **Hybrid Recommenders:** Combine content-based filtering (current approach) with collaborative filtering (e.g., finding users with similar tastes).
    *   **Diversity and Serendipity:** Introduce mechanisms to recommend movies that are not just similar, but also diverse or surprisingly relevant, to prevent filter bubbles.
    *   **Session-based Recommendations:** If more detailed interaction data becomes available, recommend based on the current viewing session.

4.  **Domain Expansion:**
    *   **TV Shows:** Extend the ingestion, feature engineering, and recommendation pipeline to include TV series, leveraging similar APIs and data structures.
    *   **Music:** Adapt the framework for music recommendations, using platforms like Last.fm or Spotify data.
    *   **Books/Games:** Explore integrating data from Goodreads or game platforms for broader media intelligence. This would involve significant changes to data ingestion and feature sets.

5.  **User Interface and Experience:**
    *   **Interactive Filtering:** Allow users to filter recommendations based on mood, genre, or specific criteria.
    *   **User Feedback Loop:** Implement a system where users can explicitly provide feedback on recommendations, which can then be used to retrain or fine-tune models.
    *   **Performance Optimization:** Optimize the Streamlit app for faster loading times and more responsive interactions, especially with larger datasets.
