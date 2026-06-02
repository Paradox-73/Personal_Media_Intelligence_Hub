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
- [Books - Intelligence Pipeline](#books---intelligence-pipeline)

## Introduction
The Personal Media Intelligence Hub is a Streamlit-based application designed to provide users with deep insights into their media consumption habits and offer personalized recommendations across various domains. By leveraging user data and enriching it with comprehensive metadata from external APIs, the platform builds personalized machine learning models to predict ratings, suggest similar content, and analyze viewing patterns.

## Features
- **Dashboard:** Interactive visualizations and analytics of your media viewing history, providing deep insights into your preferences and consumption patterns.
- **Oracle (Recommendation Engine):** Predicts your potential star rating for any content item and helps you discover new media by finding items similar to those you've enjoyed.
- **Data Enrichment:** Automatically fetches detailed media information from various APIs (TMDB, OMDb, YouTube Data API, RAWG, Google Books).
- **Personalized ML Models:** Trains XGBoost Regressors to predict star ratings and Classifiers to categorize media. Includes specialized domain models and a **Unified Ensemble** that learns across Movies, Shows, Games, and Books.
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

#### Books Dashboard (`app/pages/6_Books_Dashboard.py`)
- **Core Metrics:** Total books read, average personal rating, and average Google Books rating.
- **Volume Analysis:** Page count distribution and decade-wise publishing trends.
- **Creative Breakdown:** Top authors and categories (genres).
- **Model Accuracy:** Scatter plots and dataframes comparing actual vs. predicted star ratings.

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
5. **Set up API Keys:** Add TMDB, OMDB, RAWG, and Google Books keys to `.env`.
6. **Run Ingestion:** e.g., `python src/movies/ingestion.py`, `python src/games/ingestion.py`, or `python src/books/ingestion.py`.
7. **Train Machine Learning Models:**
    * Movies: `python src/movies/model_trainer.py`
    * TV Shows: `python src/shows/model_trainer.py`
    * Games: `python src/games/model_trainer.py`
    * Books: `python src/books/model_trainer.py`
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

1. **Feature Engineering:** Converts metadata (runtimes, page counts, scores) and text (plots, descriptions, tags) into numerical vectors. Text is processed using **SentenceTransformers** and reduced via **PCA**.
2. **Model Training:** Personalized models are trained on your historical ratings.
3. **Unified Intelligence:** The unified model aligns schemas from different domains (e.g., mapping Book `authors` to `director` and `pageCount` to `runtime`) to learn universal taste patterns.

## Results and Achievements

### Unified Media Intelligence - Performance Summary
The Unified Model combines data from Movies, TV Shows, Games, and Books (1,048 records) into a single semantic feature space. This allows the model to leverage cross-domain insights.

**Quad-Domain Ensemble Results (Test Set):**
```
📊 PERFORMANCE REPORT: 🚀 UNIFIED STACKING ENSEMBLE 🚀
==================================================
   📉 [TEST SET] Regressor MAE:  0.5405
   📉 [TEST SET] Regressor RMSE: 0.7213
   📈 [TEST SET] Regressor R²:   0.4762
   ----------------------------------------
   Exact (0.0):  28.1%
   ±0.5 Stars:   45.7%
   ±1.0 Stars:   18.6%
==================================================
```
Adding book data brings the total intelligence hub to over **1,000 records**, creating a robust foundation for identifying deep semantic preferences across all entertainment mediums.

## Books - Intelligence Pipeline
The Books Intelligence Pipeline provides deep insights into your reading library.

1. **Data Ingestion (`src/books/ingestion.py`):** 
   - Uses the **Google Books API** to enrich your personal book list with metadata including categories, authors, page counts, and full descriptions.
2. **Feature Engineering (`src/books/feature_engineering.py`):**
   - Extracts numeric features (year, page counts, ratings).
   - Multi-hot encodes authors and categories.
   - Generates semantic embeddings for book descriptions using **SentenceTransformers**.
3. **Model Training (`src/books/model_trainer.py`):**
   - Trains an **XGBoost Regressor** to predict your star ratings.
   - **Individual Book Model Results (Test Set):**
     - **MAE:** 0.4615
     - **R²:** 0.4150
     - **Accuracy:** 77% of predictions within ±0.5 stars.
4. **Dashboard (`app/pages/6_Books_Dashboard.py`):**
   - Interactive Streamlit page to visualize your reading habits and model predictions.

## Interpretation of Results
The models demonstrate high precision in predicting user preferences. An MAE of ~0.5 means the system is typically within half a star of your actual rating.

## Future Enhancements
- Integration of Music into the Unified Model.
- Hybrid recommenders combining content-based filtering with collaborative filtering.
- Temporal preference tracking to see how tastes evolve over years.
