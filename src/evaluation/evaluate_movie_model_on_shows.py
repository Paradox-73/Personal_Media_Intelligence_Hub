import pandas as pd
import numpy as np
import joblib
import sys
import ast
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

# --- HELPERS (Copied from src/movies/feature_engineering.py and src/movies/predict_ratings.py) ---
def clean_money(x):
    if pd.isna(x): return 0
    if isinstance(x, (int, float)): return x
    x = str(x).replace('$', '').replace(',', '').strip()
    try: return float(x)
    except: return 0

def parse_list(x):
    if pd.isna(x) or x == "": return []
    try:
        if isinstance(x, list): return x
        if "[" in str(x): return ast.literal_eval(str(x))
        return [str(x)]
    except: return []

def categorize_rating(r):
    r = str(r).upper()
    if 'R' in r or 'NC-17' in r or 'TV-MA' in r: return 'Adult'
    if 'PG' in r or 'TV-14' in r: return 'Teen'
    return 'General'

def sanitize_col(col_name):
    return re.sub(r"[\[\]<']", "", str(col_name))

# --- Performance Report Function (Copied from src/movies/advanced_movie_model_trainer.py) ---
def print_performance_report(model_name: str, y_true: pd.Series, y_pred: np.ndarray):
    """Calculates and prints a standardized performance report for a model."""
    print("="*50)
    print(f"📊 PERFORMANCE REPORT: {model_name}")
    print("="*50)
    
    # Clip and round predictions to valid rating values (0.5 increments)
    y_pred_rounded = np.round(np.clip(y_pred, 0, 5) * 2) / 2
    
    mse = mean_squared_error(y_true, y_pred_rounded)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred_rounded)
    r2 = r2_score(y_true, y_pred_rounded)

    print(f"   📉 [TEST SET] Regressor MSE:  {mse:.4f}")
    print(f"   📉 [TEST SET] Regressor RMSE: {rmse:.4f}")
    print(f"   📉 [TEST SET] Regressor MAE:  {mae:.4f}")
    print(f"   📈 [TEST SET] Regressor R²:   {r2:.4f}")

    # Detailed Difference Report
    diffs = np.abs(y_true - y_pred_rounded)
    total = len(diffs)
    print("" + "   " + "-"*40)
    print(f"   Exact (0.0):  {(diffs == 0.0).sum():<3} ({((diffs == 0.0).sum()/total)*100:.1f}%)")
    print(f"   Off by 0.5:   {(diffs == 0.5).sum():<3} ({((diffs == 0.5).sum()/total)*100:.1f}%)")
    print(f"   Off by 1.0:   {(diffs == 1.0).sum():<3} ({((diffs == 1.0).sum()/total)*100:.1f}%)")
    print(f"   Off by >1.0:  {(diffs > 1.0).sum():<3} ({((diffs > 1.0).sum()/total)*100:.1f}%)")
    print("   " + "-"*40)
    print("="*50)

def transform_show_data_for_movie_model(df_shows: pd.DataFrame, movie_preprocessor_state: dict) -> pd.DataFrame:
    """
    Transforms TV show DataFrame into a feature DataFrame suitable for the movie model.
    Handles mapping show features to movie features, filling missing movie-specific features
    with medians from the movie preprocessor state, and applying necessary encodings/embeddings.
    """
    print("   Transforming TV show data for movie model...")
    df = df_shows.copy()

    # Ensure movie-specific columns exist, filling with medians from movie preprocessor state
    # This dictionary was generated in src/movies/feature_engineering.py
    median_values_movie = movie_preprocessor_state['median_values']
    
    # 1. Numerical Features
    # Direct mappings or best approximations
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(median_values_movie.get('year', 0.0))
    
    # Calculate total show runtime: number_of_episodes * episode_runtime
    # Ensure both columns are numeric and handle NaNs before multiplication
    df['number_of_episodes'] = pd.to_numeric(df['number_of_episodes'], errors='coerce').fillna(1) # Default to 1 episode
    df['episode_runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(median_values_movie.get('runtime', 0.0)) # Use movie median as fallback for episode runtime if not available
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(median_values_movie.get('runtime', 0.0)) # Fill any remaining NaNs after multiplication
    
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(median_values_movie.get('imdb_rating', 0.0))
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(median_values_movie.get('vote_average', 0.0))
    
    # --- 3. Vote Count Distribution Scaling ---
    # Fetch movie medians
    movie_median_votes = median_values_movie.get('imdb_votes', 10000.0)
    movie_median_box = median_values_movie.get('box_office_clean', 10000000.0)
    df['imdb_votes'] = pd.to_numeric(df['imdb_votes'], errors='coerce').fillna(movie_median_votes)

    # Use the ACTUAL imdb_votes from your dataset
    show_votes = pd.to_numeric(df['imdb_votes'], errors='coerce').fillna(movie_median_votes)
    
    # Calculate blockbuster status: How many times more votes does this show have vs the average movie?
    vote_ratio = (show_votes / movie_median_votes).clip(0, 15) 
    
    # Generate Box Office based on engagement
    synthetic_box_office = movie_median_box * vote_ratio
    df['box_office_log'] = np.log1p(synthetic_box_office)
    

    # Movie-specific numerical features, fill with movie medians (or 0.0 if median not in state)
    # Note: These are movie-specific, so their values for shows are effectively imputed/neutralized.
    # --- 2. Synthetic Critic Scores Proxy ---
    # Base the missing critic scores on the existing 1-10 IMDB rating, scaled to 100
    base_critic_score = df['imdb_rating'] * 10 
    
    # If a show has a 8.0 IMDB (80/100), we synthesize a Metascore of ~68 and Rotten Tomatoes of ~72
    synthetic_metascore = base_critic_score * 0.85 
    synthetic_rt = base_critic_score * 0.90
    
    df['metascore'] = pd.to_numeric(df.get('metascore', pd.Series(dtype=float)), errors='coerce').fillna(synthetic_metascore)
    df['rotten_tomatoes_rating'] = pd.to_numeric(df.get('rotten_tomatoes_rating', pd.Series(dtype=float)), errors='coerce').fillna(synthetic_rt)
    df['popularity'] = pd.to_numeric(df.get('popularity', pd.Series(dtype=float)), errors='coerce').fillna(median_values_movie.get('popularity', 0.0))
    
    # --- 1. Synthetic Box Office Proxy ---
    # Fetch the movie medians (adding fallbacks to prevent division by zero)
    movie_median_pop = median_values_movie.get('popularity', 15.0) 
    movie_median_box = median_values_movie.get('box_office_clean', 10000000.0)

    show_popularity = pd.to_numeric(df.get('popularity', pd.Series(dtype=float)), errors='coerce').fillna(movie_median_pop)
    
    # Calculate how many times more/less popular this show is compared to the average movie
    # We clip at 10 to prevent mega-viral shows from creating massive mathematical outliers
    pop_ratio = (show_popularity / movie_median_pop).clip(0, 10) 
    
    # Generate a synthetic box office and log it
    synthetic_box_office = movie_median_box * pop_ratio
    df['box_office_log'] = np.log1p(synthetic_box_office)

    # Awards features: extract from show 'awards' column
    df['total_wins'] = df.get('awards', pd.Series(dtype=str)).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(median_values_movie.get('total_wins', 0.0))
    df['total_nominations'] = df.get('awards', pd.Series(dtype=str)).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(median_values_movie.get('total_nominations', 0.0))

    # Critic average feature calculation, using current (potentially imputed) values
    df['imdb_rating_100'] = df['imdb_rating'] * 10
    df['vote_average_100'] = df['vote_average'] * 10
    
    # Only include non-zero critic scores in the mean to avoid averaging with imputed 0s for missing features
    critic_scores_100 = pd.DataFrame({
        'imdb_rating_100': df['imdb_rating_100'],
        'metascore': df['metascore'],
        'rotten_tomatoes_rating': df['rotten_tomatoes_rating'],
        'vote_average_100': df['vote_average_100']
    }).replace(0, np.nan) # Replace 0s with NaN for correct mean calculation
    
    df['critic_avg_100'] = critic_scores_100.mean(axis=1).fillna(0) # Fill NaN from mean with 0
    df['critic_avg_5'] = (df['critic_avg_100'] / 100) * 5.0 
    df['critic_avg_5'] = df['critic_avg_5'].fillna(median_values_movie.get('critic_avg_5', 0.0)) # Final fill for this feature

    # Create the X_num part of the feature set
    num_cols_movie_model = [
        'year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating',
        'vote_average', 'imdb_votes', 'box_office_log', 'popularity',
        'total_wins', 'total_nominations', 'critic_avg_5'
    ]
    # Ensure all columns exist and are numeric, fill any remaining NaNs with 0 if not handled by median_values
    for col in num_cols_movie_model:
        if col not in df.columns:
            df[col] = 0.0 # Add missing columns
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(median_values_movie.get(col, 0.0))
    
    X_num = df[num_cols_movie_model].reset_index(drop=True)

    # 2. Language One-Hot Encoding (using movie's top languages)
    def map_language_for_movie_model(lang_str, top_langs):
        if pd.isna(lang_str): return 'Other'
        # Convert list-like string to list
        if isinstance(lang_str, str) and ('[' in lang_str or ',' in lang_str):
            langs = [l.strip() for l in ast.literal_eval(lang_str) if isinstance(l, str)] if '[' in lang_str else [l.strip() for l in lang_str.split(',')]
        elif isinstance(lang_str, str):
            langs = [lang_str.strip()]
        else:
            return 'Other'
        
        for lang in langs:
            if lang in top_langs:
                return lang
        return 'Other'
    
    df['language_cleaned'] = df['language'].apply(lambda x: map_language_for_movie_model(x, movie_preprocessor_state['top_languages']))
    X_lang = pd.get_dummies(df['language_cleaned'], prefix='lang')
    # Ensure all language columns from training are present, fill missing with 0
    for lang_col in [f'lang_{lang}' for lang in movie_preprocessor_state['top_languages']] + ['lang_Other']:
        if lang_col not in X_lang.columns:
            X_lang[lang_col] = 0

    # 3. Text Representation (Sentence Transformers + PCA) - using movie's models
    print("   Generating text embeddings for TV shows...")
    df['text_content'] = "Title: " + df['name'].fillna('Unknown') + \
                         ". Directed by: " + df['created_by'].fillna('Unknown') + \
                         ". Starring: " + df['actors'].fillna('Unknown') + \
                         ". Written by: " + df['writer'].fillna('Unknown') + \
                         ". Produced by: " + df['production_companies'].fillna('Unknown') + \
                         ". " + df['tagline'].fillna('') + \
                         " " + df['overview'].fillna('')
    
    transformer_model = SentenceTransformer(movie_preprocessor_state['sentence_transformer'])
    text_embeddings = transformer_model.encode(df['text_content'].tolist(), show_progress_bar=False) # No progress bar for batch
    
    pca_movie = movie_preprocessor_state['pca']
    X_text_pca = pca_movie.transform(text_embeddings)
    X_text = pd.DataFrame(X_text_pca, columns=[f'pca_{i}' for i in range(pca_movie.n_components_)], index=df.index)

    # 4. Genres (using movie's MLB)
    genre_mapping = {
        'Action & Adventure': ['Action', 'Adventure'],
        'Sci-Fi & Fantasy': ['Science Fiction', 'Fantasy'],
        'War & Politics': ['War', 'History'], # Or whichever closest movie equivalents exist
        'Kids': ['Family', 'Animation']
    }

    def map_tv_genres(genre_list):
        new_genres = []
        for g in genre_list:
            if g in genre_mapping:
                new_genres.extend(genre_mapping[g])
            else:
                new_genres.append(g)
        return list(set(new_genres)) # Remove duplicates

    # 4. Genres (using movie's MLB)
    df['genre_list'] = df['genres'].apply(parse_list)
    df['genre_list'] = df['genre_list'].apply(map_tv_genres) # Apply mapping here
    mlb_movie = movie_preprocessor_state['mlb_genre']
    genre_encoded = mlb_movie.transform(df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in mlb_movie.classes_], index=df.index)


    # 5. MPAA Rating (using movie's categorization)
    df['mpaa_cat'] = df['age_rating'].apply(categorize_rating) # Shows have 'age_rating'
    X_mpaa = pd.get_dummies(df['mpaa_cat'], prefix='rated')
    # Ensure all rated columns from training are present, fill missing with 0
    for rated_col in [f'rated_{cat}' for cat in ['Adult', 'Teen', 'General']]:
        if rated_col not in X_mpaa.columns:
            X_mpaa[rated_col] = 0


    # 6. Combine all features, align columns with movie training columns
    # Need to make sure the order and presence of columns match exactly what the movie model was trained on
    X_combined = pd.concat([X_num, X_lang, X_genre, X_mpaa, X_text], axis=1)
    X_combined.columns = [sanitize_col(col) for col in X_combined.columns]
    X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()] # Drop duplicate columns if any

    # Final alignment: create a DataFrame with all training columns, fill missing with 0, and match order
    movie_training_columns = movie_preprocessor_state['training_columns']
    X_final = pd.DataFrame(0, index=df.index, columns=movie_training_columns)
    
    # Copy over features that exist in the combined and training columns
    common_cols = list(set(X_combined.columns) & set(movie_training_columns))
    X_final[common_cols] = X_combined[common_cols]

    print(f"   Feature transformation complete. Final shape: {X_final.shape}")
    return X_final

def evaluate_movie_model_on_shows():
    print("🚀 Evaluating Movie Model on TV Show Dataset...")

    # --- 1. Load Movie Models and Preprocessor State ---
    ensemble_dir = config.MODEL_DIR / "movies" / "ensemble"
    movie_model_paths = {
        "XGB": ensemble_dir / "xgb_base_regressor.joblib",
        "SVR": ensemble_dir / "svr_base_regressor.joblib",
        "CatBoost": ensemble_dir / "catboost_base_regressor.joblib",
        "Stacking": ensemble_dir / "stacking_ensemble_regressor.joblib",
        "Ordinal": ensemble_dir / "ordinal_classifier.joblib"
    }
    
    movie_models = {}
    for name, path in movie_model_paths.items():
        if not path.exists():
            print(f"❌ ERROR: Movie model file not found for '{name}' at {path}.")
            print("   Please ensure movie models are trained (run src/movies/advanced_movie_model_trainer.py).")
            return
        movie_models[name] = joblib.load(path)
    
    movie_preprocessor_state_path = config.PREPROCESSOR_STATE # This is for movies
    if not movie_preprocessor_state_path.exists():
        print(f"❌ ERROR: Movie Preprocessor State not found at {movie_preprocessor_state_path}.")
        print("   Please ensure movie features are engineered (run src/movies/feature_engineering.py).")
        return
        
    movie_preprocessor_state = joblib.load(movie_preprocessor_state_path)
    print(f"✅ Loaded movie models and movie preprocessor state.")

    # --- 2. Load TV Show Enriched Data ---
    tv_shows_enriched_path = config.TV_SHOWS_ENRICHED_DATA_PATH
    if not tv_shows_enriched_path.exists():
        print(f"❌ ERROR: TV Show Enriched Data not found at {tv_shows_enriched_path}.")
        print("   Please ensure TV show data is ingested (run src/shows/ingestion.py).")
        return

    df_shows_raw = pd.read_csv(tv_shows_enriched_path)
    
    # Filter for shows that have a user rating to evaluate performance
    df_shows_rated = df_shows_raw.dropna(subset=['user_rating']).copy()
    if df_shows_rated.empty:
        print("❌ No TV shows with user ratings found in the enriched data. Cannot evaluate.")
        return

    print(f"   Loaded {len(df_shows_rated)} rated TV shows for evaluation.")

    # --- 3. Transform TV Show Data for Movie Model ---
    X_shows_transformed = transform_show_data_for_movie_model(df_shows_rated, movie_preprocessor_state)

    # --- 4. Predict Ratings using Movie Models ---
    print("   Predicting ratings for TV shows using movie models...")
    df_shows_rated['name'] = df_shows_rated['name'].fillna('Unknown Show') # Ensure a name for reports
    
    # Load the ordinal class mapping if it exists
    ordinal_classes_path = ensemble_dir / "ordinal_classes.joblib"
    if ordinal_classes_path.exists():
        unique_classes = joblib.load(ordinal_classes_path)
        bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
        present_bucket_vals = np.array([bucket_map[c] for c in unique_classes])
    else:
        # Fallback to standard 10 buckets
        present_bucket_vals = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    for name, model in movie_models.items():
        if name == "Ordinal":
            ord_probs = model.predict_proba(X_shows_transformed)
            preds = np.sum(ord_probs * present_bucket_vals, axis=1)
        else:
            preds = model.predict(X_shows_transformed)
        
        df_shows_rated[f'pred_movie_{name}'] = np.round(np.clip(preds, 0, 5) * 2) / 2 # Movie ratings are 0.5-5.0
        
    # --- 5. Performance Report ---
    y_true = df_shows_rated['user_rating']
    
    y_true_scaled = y_true  

    print("\n" + "="*55)
    print("📊 MOVIE MODEL PERFORMANCE ON TV SHOWS (ENSEMBLE REPORT)")
    print("="*55)
    print(f"Total TV Shows Evaluated: {len(df_shows_rated)}")
    
    for name in movie_models.keys():
        pred_col = f'pred_movie_{name}'
        y_pred = df_shows_rated[pred_col]
        
        # Call the common print_performance_report function
        print_performance_report(f"Movie {name} on TV Shows", y_true_scaled, y_pred)

    # --- 6. Visualization (KDE Plot) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(16, 8))
    
    sns.kdeplot(x=y_true_scaled, ax=ax, label='Actual TV Show Ratings (Scaled)', color='black', linewidth=3, fill=True, alpha=0.1)
    
    for name in movie_models.keys():
        sns.kdeplot(data=df_shows_rated, x=f'pred_movie_{name}', ax=ax, label=f'Movie {name} Preds', linestyle='--', linewidth=2)
            
    ax.set_title('Movie Model Predictions vs. Actual TV Show Ratings (KDE)', fontsize=18, pad=20)
    ax.set_xlabel('Rating (0.5 - 5.0)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_xticks(np.arange(0.5, 5.5, 0.5))
    ax.legend(title='Model', fontsize='medium')
    plt.tight_layout()
    
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path_kde = config.PREDICTIONS_DIR / "movie_model_on_shows_kde.png"
    plt.savefig(plot_path_kde, dpi=150)
    print(f"📈 KDE distribution plot saved to {plot_path_kde}")
    plt.close(fig) 

    # --- 7. Visualization (Histogram Grid) ---
    data_cols = [y_true_scaled.name] + [f'pred_movie_{name}' for name in movie_models.keys()]
    titles = ['Actual TV Show Ratings (Scaled)'] + [f'Movie {name} Predictions' for name in movie_models.keys()]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10), sharey=True)
    axes = axes.flatten()
    
    # Assign the scaled y_true to a temporary column for plotting with data=df
    df_shows_rated['user_rating_scaled_for_plot'] = y_true_scaled

    for i, col in enumerate(data_cols):
        # Use the scaled column for actual ratings
        if i == 0:
            sns.histplot(df_shows_rated['user_rating_scaled_for_plot'], bins=np.arange(0.25, 5.75, 0.5), 
                         ax=axes[i], kde=False)
        else:
            sns.histplot(df_shows_rated[col], bins=np.arange(0.25, 5.75, 0.5), 
                         ax=axes[i], kde=False)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Rating (0.5 - 5.0)')
        axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide any unused subplots
    for i in range(len(data_cols), len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('Histogram of Rating Distributions (Movie Model on TV Shows)', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path_hist = config.PREDICTIONS_DIR / "movie_model_on_shows_histograms.png"
    plt.savefig(plot_path_hist, dpi=150)
    print(f"📈 Histogram grid plot saved to {plot_path_hist}")
    plt.close(fig)
    
    # --- 8. Save Results ---
    out_cols = ['name', 'year', 'user_rating'] + [f'pred_movie_{name}' for name in movie_models.keys()]
    final_df_results = df_shows_rated[[c for c in out_cols if c in df_shows_rated.columns]].sort_values(
        by='user_rating', ascending=False # Sort by actual user rating
    )
    
    output_dir = config.PREDICTIONS_DIR / "movie_model_on_shows"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path = output_dir / "movie_model_on_shows_predictions.csv"
    final_df_results.to_csv(output_csv_path, index=False)
    print(f"✅ Detailed predictions saved to {output_csv_path}")


if __name__ == "__main__":
    evaluate_movie_model_on_shows()
