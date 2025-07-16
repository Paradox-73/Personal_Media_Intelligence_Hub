# Personal Media Intelligence Hub

[![Python Version](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A bespoke, machine learning-powered recommendation engine designed to unify and predict my personal ratings across a diverse range of entertainment domains: games, movies, TV shows, music, and books.**

This project is born from a personal desire to quantify and understand my own taste. It's a centralized hub for my media consumption, leveraging data I've collected over many years. The goal is to go beyond generic recommendations and build a system that truly understands my unique preferences.

---

## The Story Behind the Project

I've always been a meticulous tracker of the media I consume. This project is the culmination of that effort, bringing together disparate data sources into a single, intelligent system:

-   **Movies:** My complete viewing history, exported from Letterboxd.
-   **TV Shows:** A comprehensive list of every show I've watched over the years.
-   **Games:** A log of every game I've played.
-   **Music:** My entire listening history, pulled from Spotify using the Spotipy library.
-   **Books:** A catalog of my personal library, with ISBNs for each book, and my personal rating for each.

This rich, personal dataset is the foundation of the project, allowing me to train models that are uniquely tailored to my tastes.

---

## 🚀 Key Features

-   **Terminal-Based Interface:** A clean, menu-driven interface for interacting with the recommendation engine.
-   **ML-Powered Rating Prediction:** Utilizes a fine-tuned **XGBoost** model for each media category to predict my rating for new content.
-   **Advanced Feature Engineering:** Employs **Sentence-Transformers** to generate semantically rich vector embeddings from textual data (like descriptions, genres, and plots), enabling the model to understand nuanced contextual relationships.
-   **Real-Time Data Ingestion:** Fetches up-to-date information from external APIs (RAWG, OMDB, TVmaze, Spotify, Google Books) to enrich the data and provide recommendations for new content.
-   **Modular and Scalable Architecture:** The backend logic is designed to be robust and scalable, in preparation for future development.

---

## 🎯 Current Status & Development

This is an active, personal project. The current status is as follows:

-   **Fully Functional:** The recommendation engine is fully implemented and operational for **Games** and **Movies**.
-   **Under Development:** The modules for **Shows, Music, and Books** are currently being built.

---

## 🛠️ Tech Stack

-   **Core Language:** Python 3.12+
-   **Machine Learning:**
    -   Pandas & NumPy: For data manipulation.
    -   Scikit-learn: For machine learning pipelines.
    -   XGBoost: For the core gradient boosting models.
    -   Sentence-Transformers: For generating semantic embeddings.
    -   PyTorch: As the backend for Sentence-Transformers.
-   **APIs & Data:**
    -   Requests: For making API calls.
    -   python-dotenv: For managing API keys.

---

## ⚙️ Getting Started

### 1. Prerequisites

-   Python 3.12 or higher
-   Git

### 2. Clone the Repository

```bash
git clone https://github.com/Paradox-73/Personal_Media_Intelligence_Hub.git
cd Personal_Media_Intelligence_Hub
```

### 3. Set Up a Virtual Environment

```bash
# For Windows
python -m venv content_rec
content_rec\Scripts\activate

# For macOS/Linux
python3 -m venv content_rec
source content_rec/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

1.  Create a file named `.env` in the root directory.
2.  Populate it with your API keys.

```env
# .env file
RAWG_API_KEY="YOUR_RAWG_API_KEY"
OMDB_API_KEY="YOUR_OMDB_API_KEY"
SPOTIPY_CLIENT_ID="YOUR_SPOTIPY_CLIENT_ID"
SPOTIPY_CLIENT_SECRET="YOUR_SPOTIPY_CLIENT_SECRET"
GOOGLE_BOOKS_API_KEY="YOUR_GOOGLE_BOOKS_API_KEY"
```

### 6. A Note on Data and Model Training

The pre-trained models included in this repository were trained on my own extensive, personally-curated datasets. As such, their predictions reflect my specific tastes.

For other users, the models may not be accurate without being retrained on your own data. This project includes a sample data file, `data/sample_movies_data.csv`, to demonstrate how the application works.

### 7. Run the Application

```bash
python src/app.py
```

This will launch the terminal-based menu, where you can get recommendations.

---

## 📊 Example Prediction

Here is a sample of the output from the terminal application when predicting a rating for a game:

```text
--- Universal Content Rating Predictor ---
(Press Ctrl+C at any time to exit)


--- New Prediction ---
1️⃣ Select a Content Type:
   1. Game
   2. Show
   3. Movie
   4. Music
   5. Book
Enter the number of your choice: 1

2️⃣ Enter the name of the game to search for: fifa 24
Loading FeatureExtractor components from E:\Personal_Media_Intelligence_Hub\src\..\models\game\feature_extractor...
Using device: cpu
FeatureExtractor components loaded successfully.
✅ Successfully loaded model and extractor for Game.

Searching for 'fifa 24'...
Searching RAWG for: fifa 24

3️⃣ Select the correct item from the search results:
   1. EA SPORTS FC 24 (2023)
   2. FIFA 07 (2006)
   3. FIFA 2004 (2003)
   4. FIFA 22 (2021)
   5. FIFA 97 (1996)

Enter the number of the content you want to predict: 1

Fetching details for 'EA SPORTS FC 24 (2023)'...
Fetching details for game ID: 963218

Gathering features for prediction...
⚙️  Running prediction...
--- Transforming data using FeatureExtractor ---
ColumnTransformer output shape (on cpu): torch.Size([1, 79])
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  4.68it/s]
Text embeddings shape (on cpu): torch.Size([1, 384])
Combined features shape (on cpu): torch.Size([1, 463])

===================================
        ⭐ PREDICTION RESULT ⭐
===================================
  Content: EA SPORTS FC 24 (2023)
  Predicted Rating: 2.98 / 5.0
===================================
```

---

## 🔮 Future Roadmap

While the core engine is robust, this project is still evolving. Key future enhancements include:

-   **Web Interface:** Transitioning the user interface from the current terminal-based application to a dynamic and interactive web framework like **Streamlit** or **Flask**.
-   **CI/CD Pipeline:** Implementing a Continuous Integration/Continuous Delivery pipeline using **GitHub Actions** to automate testing and ensure code quality with every commit.
-   **Model Refinement:** Continuously improving the feature engineering and model tuning process as I gather more data.

---

## 📄 License

This project is licensed under the MIT License.
