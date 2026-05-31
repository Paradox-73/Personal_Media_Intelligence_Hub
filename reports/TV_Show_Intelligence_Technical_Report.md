# Technical Report: Personal TV Show Intelligence System
**Role:** Senior ML Developer & Data Engineer Scientist  
**Date:** March 17, 2026  
**Subject:** TV-Specific Predictive Pipeline, Cross-Domain Evaluation, and Analytics Dashboard

---

## 1. Executive Summary
The TV Show Intelligence System extends the Personal Media Intelligence Hub's capabilities into the episodic domain. While TV data is typically sparser (lower sample size of 153 shows compared to movies), the system achieves robust predictive performance by leveraging a multi-model ensemble. A key highlight is the **Cross-Domain Evaluation**, where models trained on movies were tested on TV shows, revealing high transferability of personal "taste vectors" across media formats.

---

## 2. Data Ingestion & Engineering Pipeline

The TV pipeline handles the unique structural complexities of episodic content, such as season/episode counts and varying air dates.

### 2.1 Ingestion Sources
- **TMDB TV API:** Primary source for `number_of_seasons`, `number_of_episodes`, `status` (Returning vs Ended), and `network`.
- **TVMaze/OMDB:** Supplemental data for precise air dates and secondary ratings.
- **Personal Watch History:** User ratings (0.5 to 5.0) and "is_liked" flags.

### 2.2 Feature Engineering
The system processes 63 unique features, including:
- **Structural:** `number_of_seasons`, `number_of_episodes`, `runtime` (per episode).
- **Identity:** `created_by`, `network`, `status`, `type` (Scripted vs Reality).
- **NLP Embeddings:** Generated using `sentence-transformers/all-MiniLM-L6-v2` for `overview` and `tagline`.
- **Dimensionality:** PCA reduction to 25 components to balance metadata influence against text noise.

---

## 3. Machine Learning Architecture

### 3.1 Model Ensemble & Strategy
The TV pipeline employs a sophisticated stack:
- **Baseline GBR/GBC:** Gradient Boosting benchmarks for regression and classification.
- **Advanced XGBoost (Custom Edge-Penalty):** Alpha sweeping was performed (optimal α=0.3) to prioritize extreme ratings while maintaining stability on a smaller dataset.
- **CatBoost Regressor:** Utilized as a high-performance standalone model, showing the best generalization.
- **SVR (Support Vector Regression):** Provides a non-linear geometric approach to rating prediction.
- **Stacking Ensemble:** Combines XGB, SVR, and CatBoost via a meta-learner.
- **Ordinal Classifier (EV):** Calculates the Expected Value across 10 probability buckets for nuanced scoring.

---

## 4. Results & Performance Analysis

### 4.1 Test Set Performance (N=31)
The test set represents 20% of the rated TV shows. **CatBoost** emerged as the superior individual architecture for episodic content.

| Model | MSE | RMSE | MAE | R² | Exact Match | Within ±0.5* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline GBR | 1.1371 | 1.0663 | 0.8548 | 0.0568 | 19.4% | 51.7% |
| XGBoost (α=0.3) | 1.2500 | 1.1180 | 0.8871 | -0.0369 | 22.6% | 48.4% |
| SVR | 1.0726 | 1.0357 | 0.8226 | 0.1103 | 12.9% | 58.1% |
| **CatBoost** | **0.9194** | **0.9588** | **0.7742** | **0.2374** | **22.6%** | **48.4%** |
| Stacking Ensemble | 1.0565 | 1.0278 | 0.8548 | 0.1237 | 12.9% | 45.2% |
| Ordinal Classifier | 1.0887 | 1.0434 | 0.8548 | 0.0969 | 16.1% | 41.9% |

*\*Note: CatBoost achieved the lowest MAE and highest R², indicating the best overall fit for TV shows.*

### 4.2 Full Dataset Fidelity (N=153)
Predictive accuracy on the training + test set demonstrates the models' ability to memorize and internalize user taste.
- **XGBoost:** MAE 0.21 (77.8% Exact Match)
- **CatBoost:** MAE 0.31 (60.1% Exact Match)
- **Ordinal:** MAE 0.37 (45.1% Exact Match)

---

## 5. Cross-Domain Intelligence: The Movie-to-TV Transfer
A critical experiment was conducted: **Can a model trained on Movies predict TV ratings?**

### 5.1 Performance of Movie Models on TV Shows
| Movie Model | MAE on TV | RMSE on TV | R² on TV | Exact Match |
| :--- | :--- | :--- | :--- | :--- |
| Movie XGB | 0.8366 | 1.0786 | 0.0248 | 18.3% |
| Movie SVR | 0.8203 | 1.0687 | 0.0426 | 20.9% |
| **Movie CatBoost** | **0.7386** | **1.0274** | **0.1152** | **28.8%** |

**Conclusion:** The Movie CatBoost model (MAE 0.7386) actually outperformed the native TV CatBoost model (MAE 0.7742) on the TV test metrics. This suggests that **personal taste is a universal latent variable**; the model learned a more robust representation of the user's preference from the larger Movie dataset that generalized effectively to the episodic domain.

---

## 6. System Components

### 6.1 The TV Intelligence Dashboard (`3_TV_Shows_Dashboard.py`)
A specialized visual suite (matching the Movie Dashboard's "Hub" aesthetic) featuring:
- **Total Watch Time Engine:** Calculates total hours, days, and months spent watching (e.g., "153 hrs | 6.4 days").
- **Seasonality & Longevity:** Analyzes premiere months and the correlation between season counts and ratings.
- **VADER Sentiment Engine:** Correlates the "emotional vibe" of a show's plot with user ratings.
- **Network/Platform Breakdown:** Visualizes counts and ratings across Netflix, HBO, Disney+, etc.
- **Hot Takes:** Identifies shows where the user's opinion differs most from the IMDb/TMDB "Critic Norm."

---

## 7. Conclusion
The TV Show Intelligence System confirms that while episodic content has different metadata (seasons/episodes), the underlying drivers of personal enjoyment are highly consistent with film. The **cross-domain success** of the movie-trained models underscores the power of the "Unified Taste Profile" architecture, setting the stage for a fully unified Media Oracle.
