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
- [Unified Media Intelligence - Performance Summary](#unified-media-intelligence---performance-summary)
- [Code Explanation](#code-explanation)
- [Future Enhancements](#future-enhancements)
- [TV Shows - Training & Prediction Results](#tv-shows---training--prediction-results)
- [Movies - ML System Deep Dive](#movies---ml-system-deep-dive)
- [YouTube Dashboard Overview](#youtube-dashboard-overview)
- [Games - Intelligence Pipeline](#games---intelligence-pipeline)

## Introduction
The Personal Media Intelligence Hub is a Streamlit-based application designed to provide users with deep insights into their media consumption habits and offer personalized recommendations across various domains. By leveraging user data and enriching it with comprehensive metadata from external APIs, the platform builds personalized machine learning models to predict ratings, suggest similar content, and analyze viewing patterns.

## Features
- **Dashboard:** Interactive visualizations and analytics of your media viewing history, providing deep insights into your preferences and consumption patterns.
- **Oracle (Recommendation Engine):** Predicts your potential star rating for any content item and helps you discover new media by finding items similar to those you've enjoyed.
- **Data Enrichment:** Automatically fetches detailed media information (e.g., genre, cast, crew, plot, poster, video/channel statistics) from various APIs (TMDB, OMDb, YouTube Data API, RAWG).
- **Personalized ML Models:** Trains XGBoost Regressors to predict star ratings and Classifiers to categorize media. Includes specialized domain models and a **Unified Ensemble** that learns across Movies, Shows, and Games.
- **Explainable Recommendations:** Provides insights into why certain items are recommended by comparing them to your previously consumed content.

### Dashboard Visualizations
The Streamlit Dashboards (`app/pages/*_Dashboard.py`) use the enriched data to generate a wide array of personal analytics and visualizations.

#### Movie Dashboard (`app/pages/1_Movies_Dashboard.py`)
- **Core Metrics:** Total movies, average rating, and cumulative watch time.
- **Time & Duration Analysis:** Movies per decade, runtime distribution, and release seasonality.
- **Cast & Crew Analysis:** Most watched and highest-rated directors, actors, and writers.
- **Critic & Financial Alignment:** Consensus tracking vs. Rotten Tomatoes, Metacritic, and Box Office performance.
- **Sentiment Analysis:** Emotional tone tracking of your library using VADER.

#### Games Dashboard (`app/pages/5_Games_Dashboard.py`)
- **Core Metrics:** Total games, average rating, average Metacritic score, and model error metrics.
- **Rating & Platform Analysis:** Distribution of your personal ratings and breakdown of games by platform.
- **Genre & Developer Insights:** Identification of your most played genres and developers.
- **Model Performance:** Visualization of actual vs. predicted ratings on your personal library.

#### YouTube Dashboard (`app/pages/11_YouTube_Dashboard.py`)
- **At a Glance:** Total views, shorts ratio, and estimated watch time.
- **Temporal Habits:** Weekly heatmap of viewing activity and timeline analysis.
- **Loyalty & Subscriptions:** active vs. "ghost" subscription analysis.

## Project Structure
- `app/`: Streamlit UI and dashboard pages.
- `data/`: Raw, processed, and cached data storage.
- `models/`: Serialized ML models and preprocessor states.
- `src/`: Core logic for ingestion, feature engineering, and training.

## Setup and Installation
1. **Clone the repository.**
2. **Create a virtual environment:** `python -m venv content_rec`
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Prepare data:** Place raw CSVs in `data/raw/`.
5. **Set up API Keys:** Add TMDB, OMDB, and RAWG keys to `.env`.
6. **Run Ingestion:** e.g., `python src/movies/ingestion.py` or `python src/games/ingestion.py`.
7. **Train Machine Learning Models:**
    * Movies: `python src/movies/model_trainer.py`
    * TV Shows: `python src/shows/model_trainer.py`
    * Games: `python src/games/model_trainer.py`
    * **Unified Model (Cross-Domain):**
        ```bash
        python src/unified_model/unified_feature_engineering.py
        python src/unified_model/advanced_unified_model_trainer.py
        ```

## Usage
1. **Launch the app:** `streamlit run app/main.py`
2. **Explore:** Navigate through dashboards to see personal analytics or use Oracles for predictions.

## Machine Learning & Recommender Systems
The system uses a **Stacking Ensemble** approach, combining XGBoost, CatBoost, and SVR models.

1. **Feature Engineering:** Converts metadata (runtimes, scores) and text (plots, tags) into numerical vectors. Text is processed using **SentenceTransformers** and reduced via **PCA**.
2. **Model Training:** Personalized models are trained on your historical ratings.
3. **Unified Intelligence:** The unified model (`src/unified_model/`) aligns schemas from different domains to learn universal taste patterns.

## Results and Achievements

### Unified Media Intelligence - Performance Summary
The Unified Model combines data from Movies, TV Shows, and Games into a single semantic feature space. This allows the model to leverage cross-domain insights (e.g., if you like "Cyberpunk" themes in games, it helps predict your rating for "Cyberpunk" movies).

**Triple-Domain Ensemble Results (987 records):**
```
📊 PERFORMANCE REPORT: 🚀 UNIFIED STACKING ENSEMBLE 🚀
==================================================
   📉 [TEST SET] Regressor MAE:  0.5076
   📉 [TEST SET] Regressor RMSE: 0.6657
   📈 [TEST SET] Regressor R²:   0.5313
   ----------------------------------------
   Exact (0.0):  29.3%
   ±0.5 Stars:   46.5%
   ±1.0 Stars:   18.2%
==================================================
```
Adding game data improved the overall $R^2$ from **0.50** to **0.53**, proving that tastes in interactive media (games) are highly correlated with preferences in passive media (film/TV) within this intelligence hub.

## Games - Intelligence Pipeline
The Games Intelligence Pipeline follows a similar structure to the movie pipeline but is optimized for smaller datasets (e.g., ~60 games).

1. **Data Ingestion (`src/games/ingestion.py`):** 
   - Uses the **RAWG API** to enrich your personal game list with metadata including genres, developers, publishers, Metacritic scores, and full descriptions.
2. **Feature Engineering (`src/games/feature_engineering.py`):**
   - Extracts numeric features (year, scores, counts).
   - One-hot encodes platforms.
   - Multi-hot encodes genres and developers (with frequency filtering).
   - Generates semantic embeddings for game descriptions and tags using **SentenceTransformers** (`all-MiniLM-L6-v2`) and reduces dimensionality with **PCA**.
3. **Model Training (`src/games/model_trainer.py`):**
   - Trains a conservative **XGBoost Regressor** (shallow trees, high regularization) to predict your ratings while preventing overfitting on small data.
   - Provides detailed performance metrics on an 80/20 split.
4. **Dashboard (`app/pages/5_Games_Dashboard.py`):**
   - Interactive Streamlit page to visualize your gaming library, your rating trends, and model predictions.

## Interpretation of Results
The models demonstrate high precision in predicting user preferences. An MAE of ~0.5 means the system is typically within half a star of your actual rating.

## Future Enhancements
- Integration of Music and Books into the Unified Model.
- Hybrid recommenders combining content-based filtering with collaborative filtering.
- Temporal preference tracking to see how tastes evolve over years.
