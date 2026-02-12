import pandas as pd
import numpy as np
import joblib
import ast
import sys
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MultiLabelBinarizer

# Add Project Root to Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent)) # Adjusted path for src/games
from src import config

# --- CONFIG ---
MIN_DEVELOPER_COUNT = 2
PCA_COMPONENTS = 0 # No text features for now

def clean_year(val):
    try:
        s = str(val).replace('.0', '').strip()
        if not s or s.lower() in ['nan', 'none', '']: return None
        return int(s[:4])
    except:
        return None

def parse_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, (pd.Series, np.ndarray)):
        if len(x) == 1:
            x = x.iloc[0] if isinstance(x, pd.Series) else x[0]
            if x is None or (isinstance(x, float) and np.isnan(x)):
                return []
        else:
            return []
    
    s_x = str(x).strip()
    if not s_x:
        return []

    try:
        if s_x.startswith('[') and s_x.endswith(']'):
            return ast.literal_eval(s_x)
        return [s_x]
    except (ValueError, SyntaxError):
        return [s_x]
    except Exception:
        return []

def process_features():
    print("🛠️ Starting Game Feature Engineering...")
    
    # 1. Load Data
    try:
        df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv for games not found.")
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
    num_cols = ['year']
    for col in num_cols:
        if col not in df.columns: df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    X_num = df[num_cols].reset_index(drop=True)

    # 4. Categorical: Platform
    df['platform'] = df['platform'].fillna('Unknown_Platform')
    X_platform = pd.get_dummies(df['platform'], prefix='plat')

    # 5. Categorical: Developer
    df['developer_list'] = df['developer'].apply(parse_list)
    df['primary_developer'] = df['developer_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    
    dev_counts = df['primary_developer'].value_counts()
    valid_devs = dev_counts[dev_counts >= MIN_DEVELOPER_COUNT].index
    df['primary_developer'] = df['primary_developer'].apply(lambda x: x if x in valid_devs else 'Other_Developer')
    
    X_dev = pd.get_dummies(df['primary_developer'], prefix='dev')

    # 6. Categorical: Genres (if available)
    if 'genre' in df.columns and not df['genre'].isnull().all():
        df['genre_list'] = df['genre'].apply(parse_list)
        mlb_genre = MultiLabelBinarizer()
        X_genre = pd.DataFrame(mlb_genre.fit_transform(df['genre_list']), columns=mlb_genre.classes_, index=df.index)
        X_genre.columns = [f"gen_{c}" for c in X_genre.columns]
    else:
        X_genre = pd.DataFrame(index=df.index) # Empty DataFrame if no genre data
        mlb_genre = None

    # 7. Text Features (Placeholder for now)
    X_text = pd.DataFrame(index=df.index)
    tfidf = None
    pca = None

    # 8. Combine
    feature_dfs = [X_num, X_platform.reset_index(drop=True), X_dev.reset_index(drop=True)]
    if not X_genre.empty:
        feature_dfs.append(X_genre.reset_index(drop=True))
    if not X_text.empty:
        feature_dfs.append(X_text.reset_index(drop=True))

    X_final = pd.concat(feature_dfs, axis=1)
    
    print(f"📊 Training Data Shape (Games): {X_final.shape}")
    
    # 9. Save
    state = {
        'valid_platforms': df['platform'].unique().tolist(), # Save all unique platforms
        'valid_developers': valid_devs,
        'mlb_genre': mlb_genre,
        'tfidf': tfidf,
        'pca': pca,
        'training_columns': X_final.columns.tolist(),
        'median_values': df[num_cols].median().to_dict()
    }
    joblib.dump(state, config.GAMES_MODEL_PREPROCESSOR_STATE)
    
    X_final['target_reg'] = y_reg.reset_index(drop=True)
    X_final['target_class'] = y_class.reset_index(drop=True)
    
    X_final.to_csv(config.GAMES_TRAINING_DATA_PATH, index=False)
    print(f"✅ Features saved to {config.GAMES_TRAINING_DATA_PATH}")

def transform_single_game(raw_game_data: dict, preprocessor_state: dict) -> pd.DataFrame:
    """
    Transforms raw game data into a feature vector using the loaded preprocessor state.
    """
    training_cols = preprocessor_state['training_columns']
    feature_cols = [c for c in training_cols if c not in ['target_reg', 'target_class']]
    
    input_df = pd.DataFrame(0, index=[0], columns=feature_cols)

    # A. Numeric Features
    num_cols = ['year']
    num_map = {
        'year': raw_game_data.get('year'),
    }
    
    for col in num_cols:
        val = num_map.get(col)
        if col in input_df.columns:
            if val is None or val == '' or str(val).lower() == 'nan':
                input_df.loc[0, col] = preprocessor_state['median_values'].get(col, 0)
            else:
                input_df.loc[0, col] = float(val)

    # B. Categorical: Platform
    platform = raw_game_data.get('platform', 'Unknown_Platform')
    plat_col = f"plat_{platform}"
    if plat_col in input_df.columns:
        input_df.loc[0, plat_col] = 1

    # C. Categorical: Developer
    developers = parse_list(raw_game_data.get('developer'))
    primary_developer = developers[0] if developers else 'Unknown'
    if primary_developer not in preprocessor_state['valid_developers']:
        primary_developer = 'Other_Developer'
    
    dev_col = f"dev_{primary_developer}"
    if dev_col in input_df.columns:
        input_df.loc[0, dev_col] = 1
        
    # D. Categorical: Genres (Multi-hot)
    genres = parse_list(raw_game_data.get('genre'))
    if 'mlb_genre' in preprocessor_state and preprocessor_state['mlb_genre'] is not None:
        mlb = preprocessor_state['mlb_genre']
        genre_encoded = mlb.transform([genres])
        for i, genre_name in enumerate(mlb.classes_):
            col_name = f"gen_{genre_name}"
            if col_name in input_df.columns:
                input_df.loc[0, col_name] = genre_encoded[0, i]
    else:
        for g in genres:
            gen_col = f"gen_{g}"
            if gen_col in input_df.columns:
                input_df.loc[0, gen_col] = 1

    # E. Text Features (Not used for now)
    # No text features for now, so input_df remains 0 for pca_ components.

    return input_df

def find_similar_games(raw_input_game_data: dict, input_game_df: pd.DataFrame, preprocessor_state: dict, n: int = 5):
    """
    Finds games from the enriched data that are most similar to the input game.
    """
    try:
        enriched_df = pd.read_csv(config.GAMES_ENRICHED_DATA_PATH)
    except FileNotFoundError:
        print("❌ Error: enriched_data.csv for games not found for similarity calculation.")
        return []

    input_title = raw_input_game_data.get('title')
    input_platform = raw_input_game_data.get('platform')
    input_year = raw_input_game_data.get('year')
    
    similarities_with_raw_data = []
    all_game_features = []
    
    for index, row in enriched_df.iterrows():
        game_raw_data = {k: row.get(k) for k in row.index if k in ['title', 'platform', 'year', 'developer', 'genre']}
        
        if not game_raw_data.get('title') or not game_raw_data.get('year'):
            continue

        if game_raw_data['title'] == input_title and 
           game_raw_data['platform'] == input_platform and 
           game_raw_data['year'] == input_year:
            continue
            
        try:
            features = transform_single_game(game_raw_data, preprocessor_state)
            if not features.empty:
                all_game_features.append({
                    'title': game_raw_data['title'],
                    'platform': game_raw_data['platform'],
                    'year': game_raw_data['year'],
                    'features': features,
                    'raw_data': game_raw_data
                })
        except Exception as e:
            print(f"Error processing game {game_raw_data.get('title')}: {e}")
            continue

    if not all_game_features:
        return []
        
    all_features_df = pd.concat([mf['features'] for mf in all_game_features], ignore_index=True)

    all_cols = list(set(input_game_df.columns) | set(all_features_df.columns))
    input_game_df_aligned = input_game_df.reindex(columns=all_cols, fill_value=0)
    all_features_df_aligned = all_features_df.reindex(columns=all_cols, fill_value=0)
    
    sim_scores = cosine_similarity(input_game_df_aligned, all_features_df_aligned)[0]
    
    for i, score in enumerate(sim_scores):
        similarities_with_raw_data.append({
            'title': all_game_features[i]['title'],
            'platform': all_game_features[i]['platform'],
            'year': all_game_features[i]['year'],
            'similarity': score,
            'raw_data': all_game_features[i]['raw_data']
        })

    similarities_with_raw_data = sorted(similarities_with_raw_data, key=lambda x: x['similarity'], reverse=True)
    return similarities_with_raw_data[:n]

def explain_similarity_games(raw_game_1: dict, raw_game_2: dict, preprocessor_state: dict) -> str:
    """
    Generates a textual explanation of the similarity between two games.
    """
    explanations = []

    # Developers
    devs1 = parse_list(raw_game_1.get('developer'))
    devs2 = parse_list(raw_game_2.get('developer'))
    common_devs = list(set(devs1) & set(devs2))
    if common_devs:
        explanations.append(f"Both developed by: {', '.join(common_devs)}.")

    # Platforms
    plat1 = raw_game_1.get('platform')
    plat2 = raw_game_2.get('platform')
    if plat1 == plat2:
        explanations.append(f"Both available on: {plat1}.")

    # Genres
    genres1 = parse_list(raw_game_1.get('genre'))
    genres2 = parse_list(raw_game_2.get('genre'))
    common_genres = list(set(genres1) & set(genres2))
    if common_genres:
        explanations.append(f"Share genres like: {', '.join(common_genres)}.")

    # Numeric Features (simplified comparison)
    year1, year2 = raw_game_1.get('year'), raw_game_2.get('year')
    if year1 and year2 and abs(year1 - year2) <= 5: # Within 5 years
        explanations.append(f"Released around the same time (within {abs(year1 - year2)} years).")

    if not explanations:
        return "Their similarity is based on subtle combinations of features."
    
    return " ".join(explanations)


def transform_dataframe_for_prediction(df: pd.DataFrame, preprocessor_state: dict) -> pd.DataFrame:
    """
    Transforms a DataFrame of raw game data into a feature-engineered DataFrame
    suitable for model prediction, using the loaded preprocessor state.
    """
    processed_df = df.copy()

    # 1. Numeric Features
    num_cols = ['year']
    
    for col in num_cols:
        if col not in processed_df.columns:
            processed_df[col] = np.nan
        processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        processed_df[col] = processed_df[col].fillna(preprocessor_state['median_values'].get(col, 0))

    X_num = processed_df[num_cols].reset_index(drop=True)

    # 2. Categorical: Platform
    processed_df['platform'] = processed_df['platform'].fillna('Unknown_Platform')
    X_platform = pd.get_dummies(processed_df['platform'], prefix='plat')

    # 3. Categorical: Developer
    processed_df['developer_list'] = processed_df['developer'].apply(parse_list)
    processed_df['primary_developer'] = processed_df['developer_list'].apply(lambda x: x[0] if len(x) > 0 else 'Unknown')
    processed_df['primary_developer'] = processed_df['primary_developer'].apply(
        lambda x: x if x in preprocessor_state['valid_developers'] else 'Other_Developer'
    )
    X_dev = pd.get_dummies(processed_df['primary_developer'], prefix='dev')

    # 4. Categorical: Genres
    if 'genre' in processed_df.columns and preprocessor_state['mlb_genre'] is not None:
        processed_df['genre_list'] = processed_df['genre'].apply(parse_list)
        mlb = preprocessor_state['mlb_genre']
        genre_encoded = mlb.transform(processed_df['genre_list'])
        X_genre = pd.DataFrame(genre_encoded, columns=[f"gen_{c}" for c in mlb.classes_], index=processed_df.index)
    else:
        X_genre = pd.DataFrame(index=processed_df.index)

    # 5. Text Features (Not used for now)
    X_text = pd.DataFrame(index=processed_df.index)

    # 6. Combine all features
    feature_dfs = [X_num, X_platform, X_dev]
    if not X_genre.empty:
        feature_dfs.append(X_genre)
    if not X_text.empty:
        feature_dfs.append(X_text)

    X_final = pd.concat(feature_dfs, axis=1)
    
    # 7. Align columns with training data to ensure consistency
    training_columns = [c for c in preprocessor_state['training_columns'] if c not in ['target_reg', 'target_class']]
    
    missing_cols = set(training_columns) - set(X_final.columns)
    for col in missing_cols:
        X_final[col] = 0
    
    X_final = X_final[training_columns]

    return X_final

if __name__ == "__main__":
    process_features()