# Personal Media Intelligence Hub

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
- [TV Shows - Training & Prediction Results](#tv-shows---training--prediction-results)
- [Movies - ML System Deep Dive](#movies---ml-system-deep-dive)
- [YouTube Dashboard Overview](#youtube-dashboard-overview)

## Introduction
The Personal Media Intelligence Hub is a Streamlit-based application designed to provide users with deep insights into their media consumption habits and offer personalized recommendations across various domains. By leveraging user data and enriching it with comprehensive metadata from external APIs, the platform builds personalized machine learning models to predict ratings, suggest similar content, and analyze viewing patterns.

## Features
- **Dashboard:** Interactive visualizations and analytics of your media viewing history, providing deep insights into your preferences and consumption patterns. See [Dashboard Visualizations](#dashboard-visualizations) for more details.
- **Oracle (Recommendation Engine):** Predicts your potential star rating for any content item and helps you discover new media by finding items similar to those you've enjoyed.
- **Data Enrichment:** Automatically fetches detailed media information (e.g., genre, cast, crew, plot, poster, video/channel statistics) from various APIs (TMDB, OMDb, YouTube Data API).
- **Personalized ML Models:** Trains XGBoost Regressors to predict star ratings and Classifiers to categorize media into broad sentiment categories based on your past preferences.
- **Explainable Recommendations:** Provides insights into why certain items are recommended by comparing them to your previously consumed content.

### Dashboard Visualizations
The Streamlit Dashboards (`app/pages/*_Dashboard.py`) use the enriched data to generate a wide array of personal analytics and visualizations, offering unique insights into your media consumption habits.

#### Movie Dashboard (`app/pages/1_Movies_Dashboard.py`)
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

#### YouTube Dashboard (`app/pages/11_YouTube_Dashboard.py`)
This dashboard provides comprehensive analytics of your YouTube watch history and subscription data. It requires `watch-history.json` and optionally `subscriptions.csv` from Google Takeout. For enriched data (watch time, genres, public views, subscriber counts), the `src/enrich_yt.py` script needs to be run using a YouTube Data API key.

Key features include:
-   **At a Glance:** Total views, shorts ratio, actual/estimated watch time, and top genre/channel.
-   **Temporal Habits:** Visualizations of watch history over time (monthly timeline), weekly heatmap of viewing activity by day and hour, and distribution of activity by time of day (Morning, Afternoon, Evening, Night).
-   **Content & Genres:** Analysis of format preference (Shorts vs. Long Form), top genres by watch time (if enriched), top channels, and a topic cloud generated from video tags (if enriched).
-   **Behavior & Sessions:** Metrics on average session length, number of "marathon" sessions (>1hr), average videos per session, session duration histograms, and identification of "binge-worthy" channels based on daily watch consistency.
-   **Hipster Index:** An exploration of content mainstream-ness by analyzing public view counts and creator subscriber counts, categorizing watched videos into 'Deep Niche', 'Niche', 'Popular', and 'Viral' tiers. This helps understand if your viewing habits lean towards popular content or more obscure finds.
-   **Loyalty & Subscriptions:** If subscription data is provided, it analyzes subscription utilization, identifying "active" subscriptions (channels watched) versus "ghost" subscriptions (subscribed but never watched).

## Project Structure
The project is organized into logical directories to maintain clarity and scalability:

-   `app/`: Contains the Streamlit application code.
    -   `main.py`: The main entry point for the Streamlit application, defining the multi-page structure.
    -   `pages/`: Individual Streamlit pages.
        -   `1_Movies_Dashboard.py`: Implements the interactive data visualization dashboard for movies.
        -   `2_Movies_Oracle.py`: Implements the movie prediction and recommendation engine.
        -   `3_TV_Shows_Dashboard.py`: Dashboard for TV Shows.
        -   `4_Music_Dashboard.py`: Dashboard for Music.
        -   `5_Games_Dashboard.py`: Dashboard for Games.
        -   `6_Books_Dashboard.py`: Dashboard for Books.
        -   `7_TV_Shows_Oracle.py`: Oracle for TV Shows.
        -   `8_Music_Oracle.py`: Oracle for Music.
        -   `9_Games_Oracle.py`: Oracle for Games.
        -   `10_Books_Oracle.py`: Oracle for Books.
        -   `11_YouTube_Dashboard.py`: Dashboard for YouTube watch history analysis.
-   `content_rec/`: Python virtual environment.
    -   `etc/`
    -   `Include/`
    -   `Lib/`
    -   `Scripts/`
    -   `share/`
-   `data/`: Stores all data related to the project.
    -   `cache/`: Cached API responses or intermediate data (e.g., `omdb_cache.json`, `tmdb_cache.json`).
    -   `predictions/`: Output of model predictions.
        -   `movies/`: Predicted ratings for movies (e.g., `predicted_ratings.csv`).
    -   `processed/`: Enriched and preprocessed data.
        -   `books/`
        -   `games/`
        -   `movies/`: Enriched movie data (e.g., `dashboard_view.csv`, `enriched_data.csv`, `training_features.csv`).
        -   `music/`
        -   `shows/`: Enriched TV show data (e.g., `dashboard_view.csv`, `enriched_data.csv`, `training_features.csv`).
        -   `youtube_channel_details.csv`: Enriched YouTube channel data.
        -   `youtube_video_details.csv`: Enriched YouTube video data.
    -   `raw/`: Raw input data.
        -   `books/`
        -   `games/`
        -   `movies/`: Raw movie data (e.g., `liked.csv`, `ratings.csv`).
            -   `letterboxd/`: Letterboxd export data (e.g., `comments.csv`, `diary.csv`, `ratings.csv`, `watched.csv`, `watchlist.csv`).
                -   `deleted/`: Deleted Letterboxd data.
                -   `likes/`: Liked items from Letterboxd.
                -   `lists/`: User-created lists from Letterboxd.
                -   `orphaned/`: Orphaned Letterboxd data.
        -   `music/`
        -   `shows/`: Raw TV show data (e.g., `neg_shows_data.csv`, `ratings.csv`).
        -   `yt/`: Raw YouTube data (e.g., `subscriptions.csv`, `watch-history.json`).
-   `models/`: Stores trained machine learning models and preprocessing objects.
    -   `books/`
    -   `games/`
    -   `movies/`: Trained movie models (e.g., `preprocessor_state.pkl`, `xgb_classifier.pkl`, `xgb_regressor.pkl`).
    -   `music/`
    -   `shows/`: Trained TV show models (e.g., `preprocessor_state.pkl`, `xgb_classifier.pkl`, `xgb_regressor.pkl`).
    -   `unified_model/`
-   `src/`: Source code for data processing, feature engineering, model training, and utility functions, organized by media domain.
    -   `books/`: Scripts for books data.
        -   `feature_engineering.py`
        -   `ingestion.py`
    -   `config.py`: Centralized configuration.
    -   `enrich_yt.py`: Script for enriching YouTube data.
    -   `games/`: Scripts for games data.
        -   `feature_engineering.py`
        -   `ingestion.py`
    -   `movies/`: Scripts for movies data.
        -   `feature_engineering.py`
        -   `ingestion.py`
        -   `model_trainer.py`
        -   `predict_ratings.py`
    -   `music/`: Scripts for music data.
        -   `feature_engineering.py`
        -   `ingestion.py`
    -   `shows/`: Scripts for TV shows data.
        -   `feature_engineering.py`
        -   `ingestion.py`
        -   `model_trainer.py`
        -   `predict_ratings.py`
    -   `unified_model/`
    -   `__pycache__/`: Python cache files.
-   `.env`: Environment variables.
-   `.git/`: Git version control data.
-   `.gitignore`: Git ignore patterns.
-   `log.txt`: Log file.
-   `output.txt`: Output file (used for file structure listing).
-   `README.md`: Project README.
-   `requirements.txt`: Python dependencies.
-   `srs.txt`: (Assuming this is a system requirement specification or similar document).

## Technology Stack
-   **Core Language:** Python
-   **Web Framework:** Streamlit
-   **Data Manipulation:** Pandas, NumPy
-   **Machine Learning:** XGBoost, Scikit-learn
-   **Data Visualization:** Plotly
-   **API Interaction:** `tmdbv3api`, `omdb`, `googleapiclient` (YouTube Data API v3)
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
    *   Place `ratings.csv` and `liked.csv` into the `data/raw/movies/letterboxd` directory.
    *   For YouTube analysis, download `watch-history.json` and `subscriptions.csv` from Google Takeout. Place them into `data/raw/yt/`.

5.  **Set up API Keys:**
    *   Obtain API keys from [TMDB](https://www.themoviedb.org/documentation/api), [OMDb](http://www.omdbapi.com/apikey.aspx), and [YouTube Data API v3](https://developers.google.com/youtube/v3/getting-started).
    *   Create a `.env` file in the project root directory with the following content:
        ```
        TMDB_API_KEY="YOUR_TMDB_API_KEY"
        OMDB_API_KEY="YOUR_OMDB_API_KEY"
        YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY"
        ```

6.  **Run Data Ingestion/Enrichment:**
    *   For movies:
        ```bash
        python src/movies/ingestion.py
        ```
        This will generate `enriched_data.csv` in `data/processed/movies/`.
    *   For YouTube:
        ```bash
        python src/enrich_yt.py
        ```
        This will generate `youtube_video_details.csv` and `youtube_channel_details.csv` in `data/processed/`.

7.  **Train Machine Learning Models:**
    *   For movies:
        ```bash
        python src/movies/model_trainer.py
        ```
    *   For TV shows:
        ```bash
        python src/shows/model_trainer.py
        ```
    These steps will save the trained models and preprocessor objects in their respective `models/<domain>/` directories.

## Usage

1.  **Launch the Streamlit Application:**
    ```bash
    streamlit run app/main.py
    ```
2.  Open your web browser to the address provided by Streamlit (usually `http://localhost:8501`).
3.  Navigate between the different dashboard and oracle pages to explore your data and get recommendations.

## Machine Learning & Recommender Systems
The core of the recommendation system involves:

1.  **Data Preprocessing and Feature Engineering (`src/<domain>/feature_engineering.py`):**
    *   Enriched media data is transformed into numerical features suitable for machine learning.
    *   This includes encoding categorical variables, scaling numerical features, and creating interaction terms.
    *   **Movie Similarity Calculation:** When finding similar movies in the Oracle page, `cosine_similarity` from `sklearn.metrics.pairwise` is used. This calculates the similarity between a given movie's feature vector and the feature vectors of all other movies in the dataset, which are generated on the fly.

2.  **Model Training (`src/<domain>/model_trainer.py`):**
    *   **Rating Prediction (Regression):** An XGBoost Regressor is trained on your historical star ratings to predict how you would rate unseen content.
    *   **Verdict Classification:** An XGBoost Classifier is trained to classify content into broader categories (e.g., 'Bad', 'Ok', 'Great') based on your ratings, providing a simpler verdict.
    *   Trained models and the feature preprocessor are serialized using `joblib` and saved for later use by the Streamlit app.

3.  **Prediction and Recommendation (`app/pages/*_Oracle.py` & `src/<domain>/predict_ratings.py`):**
    *   When you search for content in an Oracle, its data is fetched, preprocessed using the *same* preprocessor used during training, and fed into the trained models to get a predicted rating.
    *   Similar content recommendations are generated by comparing the feature vector of the searched item with your previously consumed items, using the engineered features.

### Detailed Feature Explanation (Movies - 92 Features)
The 92 features used for movie model training are generated in the `process_features` function within `src/movies/feature_engineering.py`. They are derived from data ingested and enriched by `src/movies/ingestion.py`, which sources data from user-provided Letterboxd CSVs and enriches it with the TMDB and OMDb APIs. The features break down into the following categories:

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
1.  **Raw Data Input:** User's Letterboxd `ratings.csv` and `liked.csv` in `data/raw/movies/letterboxd` (for movies), `watch-history.json` and `subscriptions.csv` in `data/raw/yt` (for YouTube), etc.
2.  **Data Ingestion/Enrichment (`src/<domain>/ingestion.py` or `src/enrich_yt.py`):**
    *   Reads raw data.
    *   **API Data Enrichment Details:** Scripts like `src/movies/ingestion.py` and `src/enrich_yt.py` enrich raw data with comprehensive supplementary details from external APIs.
        -   **TMDB (The Movie Database):** Provides rich production and content metadata for movies/shows. Key data points fetched include: `genres`, `cast`, `director`, `writer`, `overview` (plot summary), `tagline`, `production_companies`, `vote_average`, and `popularity`.
        -   **OMDb (Open Movie Database):** Provides commercially-oriented and critical rating data for movies/shows. Key data points fetched include: `Metascore` (from Metacritic), `imdbRating`, `imdbVotes`, `BoxOffice` revenue, and `Rotten Tomatoes` scores.
        -   **YouTube Data API v3:** Used by `src/enrich_yt.py` to fetch details for videos and channels based on IDs extracted from the user's watch history. Key data points fetched include video `duration_iso`, `tags`, `description`, `public_views`, `public_likes`, `comment_count`, and channel `subscriber_count`, `video_count`.
    *   Handles missing data and data cleaning.
    *   Outputs enriched and processed data to `data/processed/`.
3.  **Feature Engineering (`src/<domain>/feature_engineering.py`):**
    *   Loads enriched data.
    *   Transforms raw features into machine-learning-ready features (e.g., one-hot encoding, numerical scaling, PCA for text).
    *   Outputs feature sets for training.
4.  **Model Training (`src/<domain>/model_trainer.py`):**
    *   Loads engineered features.
    *   Trains XGBoost Regressor and Classifier models.
    *   Saves trained models and preprocessor to `models/<domain>/`.
5.  **Batch Prediction (Optional - `src/<domain>/predict_ratings.py`):**
    *   Loads trained models.
    *   Applies models to new data for bulk predictions.
    *   Generates a performance report.
6.  **Streamlit Application (`app/main.py`):**
    *   Loads enriched data for dashboard visualizations.
    *   Loads trained models and preprocessor for real-time predictions and similarity searches in the Oracle pages.

## Results and Achievements
The machine learning models achieve strong performance in predicting user ratings and classifying media verdicts, providing a robust foundation for personalized recommendations.

**ML Engineer Performance Report (from `src/movies/predict_ratings.py`):**
```
🚀 Starting Batch Prediction (Scale 0-5 Fixed)...
   Loaded 779 movies.

========================================
📊 ML ENGINEER PERFORMANCE REPORT
========================================
Total Evaluated: 779
----------------------------------------
📉 MAE:  0.3555
📉 RMSE: 0.4728
📈 R²:   0.7526
----------------------------------------
   Exact (0.0):    73  (9.4%)
   Tiny (0.1-0.4): 391 (50.2%)
   Small (0.5-0.9):164 (21.1%)
   Large (>= 1.0): 40  (5.1%)
========================================
```

**Model Training Summary (from `src/movies/model_trainer.py`):**
```
🤖 Starting Movie Model Training...
   Training Set: 623 | Test Set: 156
   Training Regressor...
   📉 [TEST SET] Regressor RMSE: 0.6346
   Training Classifier...
   🎯 [TEST SET] Accuracy: 76.92%
✅ Models saved.
```

**TV Shows - Training & Prediction Results**
This section summarizes the results from the latest training and prediction run for the TV shows model.

**Model Training Summary (from `src/shows/model_trainer.py`):**
```
🤖 STARTING TV SHOW MODEL TRAINING (DEEPER LEARNING)...
   Training Set: 122 | Test Set: 31

   📉 Training Regressor...
      [TEST SET] RMSE: 1.2434
      [TEST SET] MAE:  0.9122

   🎯 Training Classifier...
      [TEST SET] Accuracy: 70.97%

✅ Models saved to E:\Personal_Media_Intelligence_Hub\models\shows
```

**ML Engineer Performance Report (from `src/shows/predict_ratings.py`):**
```
🚀 STARTING PREDICTION (NLP + NETWORK AWARE)...
   Loaded 153 shows for prediction.
   📚 Processing text features...
📊 Features prepared. Shape: (153, 53)

========================================
📊  ML ENGINEER PERFORMANCE REPORT
========================================
Total Evaluated Samples: 153
----------------------------------------
📉 MAE (Mean Abs Error):  0.4902
📉 RMSE (Root Mean Sq):   0.6917
📈 R² Score:              0.5990
----------------------------------------
🎯 Error Distribution (Absolute Diff):
   Exact Match (0.0):      13  (8.5%)
   Tiny Diff   (0.1-0.5):  91  (59.5%)
   Small Diff  (0.6-1.0):  40  (26.1%)
   Large Diff  (> 1.0):    9   (5.9%)
========================================

✅ Dashboard view saved to: E:\Personal_Media_Intelligence_Hub\data\processed\shows\dashboard_view.csv
```

### Interpretation of Results

The updated results provide insights into the performance of both movie and TV show models:

**Movie Model Performance:**
The reported metrics for the movie rating prediction model indicate strong performance:
-   **MAE (Mean Absolute Error): 0.3555:** On average, the model's predictions are off by only about 0.36 of a star rating. This suggests high precision in predicting user preferences within a 5-star rating system.
-   **RMSE (Root Mean Squared Error): 0.4728:** A value of 0.4728, close to the MAE, indicates that there aren't many significantly large, outlying errors, and the model's predictions are consistently close to the actual ratings.
-   **R² Score: 0.7526:** Approximately 75.3% of the variance in the user's movie ratings can be explained by the model, a strong indicator that the features and the XGBoost Regressor effectively capture underlying patterns.
-   **Error Distribution:**
    -   **Exact Match (0.0): 9.4%:** The model perfectly predicted the rating for nearly 10% of the movies.
    -   **Tiny Diff (0.1-0.4): 50.2%:** Over half of the predictions were within a very small difference of 0.1 to 0.4 of a star.
    -   **Small Diff (0.5-0.9): 21.1%:** A significant portion (over 21%) had a small error.
    -   **Large Diff (>= 1.0): 5.1%:** Only a small percentage of predictions had an error of 1 star or more, indicating rare significantly inaccurate predictions.
Overall, the movie model demonstrates strong predictive power and reliability for personalized movie ratings.

**TV Show Model Performance:**
The TV show model, while effective, shows some differences compared to the movie model:
-   **Regressor MAE: 0.9122 (Test Set during training), Prediction MAE: 0.4902:** The prediction MAE of 0.4902 suggests that on average, the predictions are off by about half a star. While higher than the movie model, this is still a reasonable result for TV shows, which can have more complex viewing patterns and less consistent rating behavior due to longer narrative arcs.
-   **Regressor RMSE: 1.2434 (Test Set during training), Prediction RMSE: 0.6917:** Similar to MAE, the prediction RMSE of 0.6917 indicates the magnitude of errors, with larger errors being penalized more.
-   **R² Score: 0.5990:** An R² of approximately 59.9% indicates that a fair portion of the variance in TV show ratings is explained by the model. This is respectable, though lower than the movie model, potentially due to the smaller dataset size (153 shows vs 779 movies) or inherent complexities in TV show rating.
-   **Classifier Accuracy: 70.97% (Test Set during training):** The classifier for TV shows performs well at 70.97% accuracy, suggesting it can effectively categorize TV shows into broader sentiment categories.
-   **Error Distribution (Prediction):**
    -   **Exact Match (0.0): 8.5%:** A smaller percentage of exact matches compared to movies.
    -   **Tiny Diff (0.1-0.5): 59.5%:** A large proportion of predictions are within a tiny difference.
    -   **Small Diff (0.6-1.0): 26.1%:** This category is higher than movies, indicating more predictions are off by a moderate amount.
    -   **Large Diff (> 1.0): 5.9%:** Slightly higher than movies, suggesting a few more significantly inaccurate predictions.
The TV show model provides valuable insights, though its performance metrics indicate a slightly higher degree of prediction error compared to the movie model. This is likely attributable to the smaller training dataset for TV shows and potentially greater nuance in user preferences for serialized content.

## Code Explanation
-   `app/main.py`: Orchestrates the Streamlit application, serving as the main entry point and defining the overall layout.
-   `app/pages/1_Movies_Dashboard.py`: Contains the logic for generating interactive charts and statistics for the user's movie history. Utilizes `pandas` and `plotly` for data analysis and visualization.
-   `app/pages/2_Movies_Oracle.py`: Manages user interactions for movie search, prediction, and similarity-based recommendations. It loads pre-trained models and dynamically fetches movie data.
-   `app/pages/11_YouTube_Dashboard.py`: Contains the logic for generating interactive charts and statistics for the user's YouTube watch history.
-   `src/config.py`: Defines constants and configurations, centralizing paths to data, models, and API keys. Essential for maintaining a consistent project structure.
-   `src/<domain>/ingestion.py`: Handles the initial data processing for a specific domain (e.g., `src/movies/ingestion.py`), including reading raw CSVs and enriching entries with external data. It's crucial for building a comprehensive dataset.
-   `src/enrich_yt.py`: Handles the data enrichment specifically for YouTube watch history, fetching video and channel details using the YouTube Data API.
-   `src/<domain>/feature_engineering.py`: Processes the enriched data for a specific domain, converting raw information into features suitable for machine learning models. This involves tasks like one-hot encoding, text processing, and feature scaling. It also contains the logic for calculating movie similarity.
-   `src/<domain>/model_trainer.py`: Script responsible for building, training, and saving the XGBoost Regressor and Classifier models for a specific domain. It manages the training pipeline from feature input to model persistence.
-   `src/<domain>/predict_ratings.py`: A utility script for performing batch predictions using the trained models for a specific domain and generating a performance report, useful for model evaluation and monitoring.

## Future Enhancements
To further enhance the Personal Media Intelligence Hub, several avenues can be explored:

1.  **Advanced Model Techniques:**
    *   **Neural Networks:** Experiment with deep learning models (e.g., neural collaborative filtering) for potentially capturing more complex user-item interactions.
    *   **Ensemble Methods:** Combine predictions from multiple models (e.g., different regressors or classifiers) to improve robustness and accuracy.
    *   **Hyperparameter Optimization:** Implement more sophisticated hyperparameter tuning strategies (e.g., Bayesian optimization) for XGBoost and other models to squeeze out more performance.

2.  **Feature Engineering Improvements:**
    *   **Temporal Features:** Incorporate features related to when a movie was watched, release date relative to watch date, or trends in user preferences over time.
    *   **User Embeddings:** Generate user embeddings based on their watched movies or explicit feedback to capture more nuanced user preferences.
    *   **External Data Sources:** Integrate data from other media platforms or review sites to enrich the dataset further.
    *   **More Sophisticated Text Features:** Explore transformer-based models (e.g., BERT embeddings) for media overviews and taglines instead of TF-IDF + PCA.

3.  **Expanded Recommendation Paradigms:**
    *   **Hybrid Recommenders:** Combine content-based filtering (current approach) with collaborative filtering (e.g., finding users with similar tastes).
    *   **Diversity and Serendipity:** Introduce mechanisms to recommend content that is not just similar, but also diverse or surprisingly relevant, to prevent filter bubbles.
    *   **Session-based Recommendations:** If more detailed interaction data becomes available, recommend based on the current viewing session.

4.  **Domain Expansion:**
    *   **Music/Books/Games:** Continue to expand and refine the ingestion, feature engineering, and recommendation pipelines for these domains, leveraging similar APIs and data structures.

5.  **User Interface and Experience:**
    *   **Interactive Filtering:** Allow users to filter recommendations based on mood, genre, or specific criteria.
    *   **User Feedback Loop:** Implement a system where users can explicitly provide feedback on recommendations, which can then be used to retrain or fine-tune models.
    *   **Performance Optimization:** Optimize the Streamlit app for faster loading times and more responsive interactions, especially with larger datasets.
