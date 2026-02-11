import pandas as pd
import numpy as np
import joblib
import ast
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config

# --- UPDATED CONFIG (Stricter Thresholds) ---
MIN_DIRECTOR_COUNT = 4  # Increased to reduce columns
MIN_ACTOR_COUNT = 8     # Increased to reduce columns
PCA_COMPONENTS = 8      # Reduced dimensionality

def clean_percentage(x):
    if pd.isna(x): return np.nan
    x = str(x).replace('%', '').strip()
    try: return float(x)
    except: return np.nan

def parse_list(x):
    # Handle explicit None or numpy NaN
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    # If it's already a list, return it directly
    if isinstance(x, list):
        return x
    # If it's a pandas Series/array-like with one element, extract the element
    if isinstance(x, (pd.Series, np.ndarray)):
        if len(x) == 1:
            x = x.iloc[0] if isinstance(x, pd.Series) else x[0]
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return []
        else: # If it's a multi-element array, which is not expected for a single movie attribute
            return []
    
    # Now x should be a scalar (str, int, etc.)
    s_x = str(x).strip()
    if not s_x: # Handle empty string after stripping
        return []

    try:
        # Attempt to evaluate as a literal if it looks like a string representation of a list
        if s_x.startswith('[') and s_x.endswith(']'):
            return ast.literal_eval(s_x)
        # Otherwise, treat as a single string element in a list
        return [s_x]
    except (ValueError, SyntaxError):
        # If ast.literal_eval fails, treat the string as a single element
        return [s_x]
    except Exception:
        # Catch any other unexpected errors and return an empty list
        return []

def process_features():
    print("🛠️ Starting Feature Engineering...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(config.ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv not found.")
        return

    # 2. Clean Target
    df = df.dropna(subset=['user_rating'])
    y_reg = df['user_rating']
    
    def classify(r):
        if r <= 2.5: return 0
        elif r < 4.0: return 1
        else: return 2
    y_class = df['user_rating'].apply(classify)

    # 3. Numeric Features
    if df['rotten_tomatoes_rating'].dtype == object:
        df['rotten_tomatoes_rating'] = df['rotten_tomatoes_rating'].apply(clean_percentage)
    
    num_cols = ['year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'popularity']
    for col in num_cols:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    X_num = df[num_cols].reset_index(drop=True)

    # 4. Categorical: Director
    df['director_list'] = df['director'].apply(parse_list)
    df['primary_director'] = df['director_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    dir_counts = df['primary_director'].value_counts()
    valid_dirs = dir_counts[dir_counts >= MIN_DIRECTOR_COUNT].index
    df['primary_director'] = df['primary_director'].apply(lambda x: x if x in valid_dirs else 'Other_Director')
    
    X_dir = pd.get_dummies(df['primary_director'], prefix='dir')

    # 5. Categorical: Actors
    df['actors_list'] = df['actors'].apply(parse_list)
    df['primary_actor'] = df['actors_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    act_counts = df['primary_actor'].value_counts()
    valid_actors = act_counts[act_counts >= MIN_ACTOR_COUNT].index
    df['primary_actor'] = df['primary_actor'].apply(lambda x: x if x in valid_actors else 'Other_Actor')
    
    X_act = pd.get_dummies(df['primary_actor'], prefix='act')

    # 6. Categorical: Genres
    df['genre_list'] = df['genre'].apply(parse_list)
    mlb = MultiLabelBinarizer()
    X_genre = pd.DataFrame(mlb.fit_transform(df['genre_list']), columns=mlb.classes_, index=df.index)
    X_genre.columns = [f"gen_{c}" for c in X_genre.columns]

    # 7. Text Features
    df['text_content'] = df['overview'].fillna('') + " " + df['tagline'].fillna('')
    
    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['text_content'])
    
    pca = PCA(n_components=PCA_COMPONENTS)
    pca_features = pca.fit_transform(tfidf_matrix.toarray())
    X_text = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(PCA_COMPONENTS)])

    # 8. Combine
    X_final = pd.concat([X_num, X_dir.reset_index(drop=True), X_act.reset_index(drop=True), X_genre.reset_index(drop=True), X_text.reset_index(drop=True)], axis=1)
    
    print(f"📊 Training Data Shape: {X_final.shape}")
    
    # 9. Save
    state = {
        'valid_directors': valid_dirs,
        'valid_actors': valid_actors,
        'mlb_genre': mlb,
        'tfidf': tfidf,
        'pca': pca,
        'training_columns': X_final.columns.tolist(),
        'median_values': df[num_cols].median().to_dict()
    }
    joblib.dump(state, config.PREPROCESSOR_STATE)
    
    X_final['target_reg'] = y_reg.reset_index(drop=True)
    X_final['target_class'] = y_class.reset_index(drop=True)
    
    X_final['target_class'] = y_class.reset_index(drop=True)
    
    X_final.to_csv(config.TRAINING_DATA_PATH, index=False)
    print(f"✅ Features saved to {config.TRAINING_DATA_PATH}")

def transform_single_movie(raw_movie_data: dict, preprocessor_state: dict) -> pd.DataFrame:
    """
    Transforms raw movie data into a feature vector using the loaded preprocessor state.
    
    Args:
        raw_movie_data (dict): A dictionary containing raw movie data (e.g., from TMDB/OMDB).
        preprocessor_state (dict): The loaded preprocessor state from joblib.
        
    Returns:
        pd.DataFrame: A DataFrame with a single row representing the feature vector.
    """
    training_cols = preprocessor_state['training_columns']
    feature_cols = [c for c in training_cols if c not in ['target_reg', 'target_class']]
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # A. Numeric Features
    num_cols = ['year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'popularity']
    
    # Handle Rotten Tomatoes (string "87%" -> float 87.0)
    rt = raw_movie_data.get('rotten_tomatoes_rating')
    if isinstance(rt, str) and '%' in rt:
        rt = float(rt.replace('%', ''))
    
    num_map = {
        'year': raw_movie_data.get('year'),
        'runtime': raw_movie_data.get('runtime'),
        'imdb_rating': raw_movie_data.get('imdb_rating'),
        'metascore': raw_movie_data.get('metascore'),
        'vote_average': raw_movie_data.get('vote_average'),
        'popularity': raw_movie_data.get('popularity'),
        'rotten_tomatoes_rating': rt
    }
    
    for col in num_cols:
        val = num_map.get(col)
        if col in input_df.columns:
            if val is None or val == '' or str(val).lower() == 'nan':
                input_df.loc[0, col] = preprocessor_state['median_values'].get(col, 0)
            else:
                input_df.loc[0, col] = float(val)

    # B. Categorical: Director
    dirs = parse_list(raw_movie_data.get('director'))
    primary_dir = dirs[0] if dirs else 'Unknown'
    if primary_dir not in preprocessor_state['valid_directors']:
        primary_dir = 'Other_Director'
    
    dir_col = f"dir_{primary_dir}"
    if dir_col in input_df.columns:
        input_df.loc[0, dir_col] = 1

    # C. Categorical: Actor
    acts = parse_list(raw_movie_data.get('actors'))
    primary_act = acts[0] if acts else 'Unknown'
    if primary_act not in preprocessor_state['valid_actors']:
        primary_act = 'Other_Actor'
        
    act_col = f"act_{primary_act}"
    if act_col in input_df.columns:
        input_df.loc[0, act_col] = 1
        
    # D. Categorical: Genres (Multi-hot)
    genres = parse_list(raw_movie_data.get('genre'))
    # Use the fitted MultiLabelBinarizer to ensure consistent encoding
    if 'mlb_genre' in preprocessor_state:
        mlb = preprocessor_state['mlb_genre']
        genre_encoded = mlb.transform([genres])
        for i, genre_name in enumerate(mlb.classes_):
            col_name = f"gen_{genre_name}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = genre_encoded[0, i]
    else: # Fallback if mlb_genre is not in state (shouldn't happen if state is properly saved)
        for g in genres:
            gen_col = f"gen_{g}"
            if gen_col in input_df.columns:
                input_df.loc[0, gen_col] = 1

    # E. Text Features (PCA)
    text = (str(raw_movie_data.get('overview', '')) + " " + str(raw_movie_data.get('tagline', ''))).strip()
    if text and 'tfidf' in preprocessor_state and 'pca' in preprocessor_state:
        tfidf_vec = preprocessor_state['tfidf'].transform([text])
        pca_vec = preprocessor_state['pca'].transform(tfidf_vec.toarray())
        
        for i in range(pca_vec.shape[1]):
            col_name = f"pca_{i}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = pca_vec[0][i]

    return input_df

def find_similar_movies(raw_input_movie_data: dict, input_movie_df: pd.DataFrame, preprocessor_state: dict, n: int = 5):
    """
    Finds movies from the enriched data that are most similar to the input movie.
    
    Args:
        raw_input_movie_data (dict): The raw data of the movie currently being queried.
        input_movie_df (pd.DataFrame): DataFrame of the input movie's feature vector.
        preprocessor_state (dict): The loaded preprocessor state from joblib.
        n (int): Number of top similar movies to return.
        
    Returns:
        list: A list of dictionaries, each containing movie title, year, similarity score, and raw_movie_data.
    """
    try:
        enriched_df = pd.read_csv(config.ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv not found for similarity calculation.")
        return []

    input_title = raw_input_movie_data.get('title')
    input_year = raw_input_movie_data.get('year')
    
    similarities_with_raw_data = []
    all_movie_features = []
    
    for index, row in enriched_df.iterrows():
        movie_raw_data = {k: row.get(k) for k in row.index if k in ['title', 'year', 'director', 'actors', 'genre', 'overview', 'tagline', 
                                                  'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 
                                                  'vote_average', 'popularity']}
        
        # Ensure 'title' and 'year' are present for comparison
        if not movie_raw_data.get('title') or not movie_raw_data.get('year'):
            continue

        if movie_raw_data['title'] == input_title and movie_raw_data['year'] == input_year:
            continue
            
        try:
            features = transform_single_movie(movie_raw_data, preprocessor_state)
            if not features.empty:
                all_movie_features.append({
                    'title': movie_raw_data['title'],
                    'year': movie_raw_data['year'],
                    'features': features,
                    'raw_data': movie_raw_data # Store raw data for explanation
                })
        except Exception as e:
            print(f"Error processing movie {movie_raw_data.get('title')}: {e}")
            continue

    if not all_movie_features:
        return []
        
    all_features_df = pd.concat([mf['features'] for mf in all_movie_features], ignore_index=True)

    sim_scores = cosine_similarity(input_movie_df, all_features_df)[0]
    
    for i, score in enumerate(sim_scores):
        similarities_with_raw_data.append({
            'title': all_movie_features[i]['title'],
            'year': all_movie_features[i]['year'],
            'similarity': score,
            'raw_data': all_movie_features[i]['raw_data']
        })

    similarities_with_raw_data = sorted(similarities_with_raw_data, key=lambda x: x['similarity'], reverse=True)
    return similarities_with_raw_data[:n]

def explain_similarity(raw_movie_1: dict, raw_movie_2: dict, preprocessor_state: dict) -> str:
    """
    Generates a textual explanation of the similarity between two movies.
    
    Args:
        raw_movie_1 (dict): Raw data for the first movie.
        raw_movie_2 (dict): Raw data for the second movie.
        preprocessor_state (dict): The loaded preprocessor state.
        
    Returns:
        str: A human-readable explanation of their similarities.
    """
    explanations = []

    # Directors
    dirs1 = parse_list(raw_movie_1.get('director'))
    dirs2 = parse_list(raw_movie_2.get('director'))
    common_dirs = list(set(dirs1) & set(dirs2))
    if common_dirs:
        explanations.append(f"Both directed by: {', '.join(common_dirs)}.")

    # Actors
    acts1 = parse_list(raw_movie_1.get('actors'))
    acts2 = parse_list(raw_movie_2.get('actors'))
    common_acts = list(set(acts1) & set(acts2))
    if common_acts:
        explanations.append(f"Both star: {', '.join(common_acts[:3])}{'...' if len(common_acts) > 3 else ''}.")

    # Genres
    genres1 = parse_list(raw_movie_1.get('genre'))
    genres2 = parse_list(raw_movie_2.get('genre'))
    common_genres = list(set(genres1) & set(genres2))
    if common_genres:
        explanations.append(f"Share genres like: {', '.join(common_genres)}.")

    # Numeric Features (simplified comparison)
    year1, year2 = raw_movie_1.get('year'), raw_movie_2.get('year')
    if year1 and year2 and abs(year1 - year2) <= 5: # Within 5 years
        explanations.append(f"Released around the same time (within {abs(year1 - year2)} years).")
    
    runtime1, runtime2 = raw_movie_1.get('runtime'), raw_movie_2.get('runtime')
    if runtime1 and runtime2 and abs(runtime1 - runtime2) <= 20: # Within 20 minutes
        explanations.append(f"Have similar runtimes (within {abs(runtime1 - runtime2)} minutes).")

    # Textual content similarity (qualitative, based on presence)
    text1_present = bool(raw_movie_1.get('overview') or raw_movie_1.get('tagline'))
    text2_present = bool(raw_movie_2.get('overview') or raw_movie_2.get('tagline'))
    if text1_present and text2_present:
        # Since PCA components are abstract, we can't easily explain "why" from them directly.
        # Just acknowledge that their textual content was used in similarity.
        explanations.append("Their plot descriptions also contributed to their similarity.")

    if not explanations:
        return "Their similarity is based on subtle combinations of features."
    
    return " ".join(explanations)

def transform_dataframe_for_prediction(df: pd.DataFrame, preprocessor_state: dict) -> pd.DataFrame:
    """
    Transforms a DataFrame of raw movie data into a feature-engineered DataFrame
    suitable for model prediction, using the loaded preprocessor state.

    Args:
        df (pd.DataFrame): DataFrame containing raw movie data.
        preprocessor_state (dict): The loaded preprocessor state from joblib.

    Returns:
        pd.DataFrame: A feature-engineered DataFrame ready for prediction.
    """
    processed_df = df.copy()

    # 1. Numeric Features
    num_cols = ['year', 'runtime', 'imdb_rating', 'metascore', 'rotten_tomatoes_rating', 'vote_average', 'popularity']
    
    for col in num_cols:
        if col not in processed_df.columns:
            processed_df[col] = np.nan
        # Handle Rotten Tomatoes (string "87%" -> float 87.0)
        if col == 'rotten_tomatoes_rating' and processed_df[col].dtype == object:
            processed_df[col] = processed_df[col].apply(clean_percentage)
        
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        # Fill with median from preprocessor state
        processed_df[col] = processed_df[col].fillna(preprocessor_state['median_values'].get(col, 0))

    X_num = processed_df[num_cols].reset_index(drop=True)

    # 2. Categorical: Director
    processed_df['director_list'] = processed_df['director'].apply(parse_list)
    processed_df['primary_director'] = processed_df['director_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    processed_df['primary_director'] = processed_df['primary_director'].apply(
        lambda x: x if x in preprocessor_state['valid_directors'] else 'Other_Director'
    )
    X_dir = pd.get_dummies(processed_df['primary_director'], prefix='dir')

    # 3. Categorical: Actors
    processed_df['actors_list'] = processed_df['actors'].apply(parse_list)
    processed_df['primary_actor'] = processed_df['actors_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    processed_df['primary_actor'] = processed_df['primary_actor'].apply(
        lambda x: x if x in preprocessor_state['valid_actors'] else 'Other_Actor'
    )
    X_act = pd.get_dummies(processed_df['primary_actor'], prefix='act')

    # 4. Categorical: Genres
    processed_df['genre_list'] = processed_df['genre'].apply(parse_list)
    mlb = preprocessor_state['mlb_genre']
    genre_encoded = mlb.transform(processed_df['genre_list'])
    X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in mlb.classes_], index=processed_df.index)

    # 5. Text Features
    processed_df['text_content'] = processed_df['overview'].fillna('') + " " + processed_df['tagline'].fillna('')
    tfidf = preprocessor_state['tfidf']
    pca = preprocessor_state['pca']
    
    tfidf_matrix = tfidf.transform(processed_df['text_content'])
    pca_features = pca.transform(tfidf_matrix.toarray())
    X_text = pd.DataFrame(pca_features, columns=[f'pca_{i}' for i in range(pca_features.shape[1])], index=processed_df.index)

    # 6. Combine all features
    X_final = pd.concat([X_num, X_dir, X_act, X_genre, X_text], axis=1)
    
    # 7. Align columns with training data to ensure consistency
    training_columns = [c for c in preprocessor_state['training_columns'] if c not in ['target_reg', 'target_class']]
    
    # Add missing columns with 0
    missing_cols = set(training_columns) - set(X_final.columns)
    for col in missing_cols:
        X_final[col] = 0
    
    # Reorder columns to match training data
    X_final = X_final[training_columns]

    return X_final

if __name__ == "__main__":
    process_features()