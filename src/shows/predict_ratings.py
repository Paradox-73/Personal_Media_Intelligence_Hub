import pandas as pd
import numpy as np
import joblib
import sys
import ast
import re
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

def safe_eval(val):
    if pd.isna(val) or val == "": return []
    try:
        if isinstance(val, list): return val
        if "[" in str(val): return ast.literal_eval(str(val))
        return [str(val)]
    except: return []

def get_primary_network(val):
    if pd.isna(val) or val == "": return "Other"
    try:
        if "[" in str(val):
            val_list = ast.literal_eval(str(val))
            if val_list: return val_list[0].strip()
        return str(val).split(',')[0].strip()
    except:
        return "Other"

def run_consolidated_predictions():
    print("🚀 STARTING CONSOLIDATED TV SHOW PREDICTIONS & EVALUATION...")

    # 1. Load Artifacts
    ensemble_dir = config.TV_SHOWS_MODEL_DIR / "ensemble"
    try:
        models = {
            "Baseline": joblib.load(config.TV_SHOWS_MODEL_REGRESSOR),
            "XGB": joblib.load(ensemble_dir / "xgb_base_regressor.joblib"),
            "SVR": joblib.load(ensemble_dir / "svr_base_regressor.joblib"),
            "CatBoost": joblib.load(ensemble_dir / "catboost_base_regressor.joblib"),
            "Stacking": joblib.load(ensemble_dir / "stacking_ensemble_regressor.joblib"),
            "Ordinal": joblib.load(ensemble_dir / "ordinal_classifier.joblib")
        }
        unique_classes = joblib.load(ensemble_dir / "ordinal_classes.joblib")
        state = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
        print("   ✅ Loaded all models and preprocessor state.")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        print("   Have you run src/shows/model_trainer.py?")
        return

    # 2. Load and Prepare Data
    df = pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    print(f"   Loaded {len(df)} shows for prediction.")
    
    # Feature Engineering
    df['vote_count_log'] = np.log1p(pd.to_numeric(df['vote_count'], errors='coerce').fillna(0))
    _medians = state.get('median_values', {})
    df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(_medians.get('vote_average'))
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(_medians.get('year'))
    df['season_count'] = pd.to_numeric(df['number_of_seasons'], errors='coerce').fillna(1)
    df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce').fillna(_medians.get('imdb_rating'))
    df['imdb_votes_log'] = np.log1p(pd.to_numeric(df['imdb_votes'], errors='coerce').fillna(0))
    df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce').fillna(_medians.get('runtime'))
    df['episode_count'] = pd.to_numeric(df['number_of_episodes'], errors='coerce').fillna(_medians.get('episode_count'))
    df['total_wins'] = df['awards'].astype(str).str.extract(r'(\d+)\s+win', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['total_nominations'] = df['awards'].astype(str).str.extract(r'(\d+)\s+nomination', flags=re.IGNORECASE)[0].astype(float).fillna(0)
    df['is_adult'] = df['age_rating'].apply(lambda x: 1 if str(x) in ['TV-MA', 'R', '18+'] else 0)

    df['primary_network'] = df['network'].apply(get_primary_network)
    df['network_clean'] = df['primary_network'].apply(lambda x: x if x in state['top_networks'] else "Other")
    # Country one-hots (mirror feature_engineering)
    _top_ctry = state.get('top_countries', [])
    df['primary_country'] = (df['country'].fillna('').astype(str)
                             .str.replace(r"[\[\]'\"<>]", "", regex=True)
                             .str.split(',').str[0].str.strip())
    df['country_clean'] = df['primary_country'].apply(lambda x: x if x in _top_ctry else 'Other')
    for _ctry in _top_ctry:
        df[f"ctry_{_ctry}"] = (df['country_clean'] == _ctry).astype(int)
    df["ctry_Other"] = (df['country_clean'] == "Other").astype(int)
    
    df['genres_list'] = df['genres'].apply(safe_eval)
    for col_name in state['kept_genres']:
        raw_genre = col_name.replace('g_', '')
        df[col_name] = df['genres_list'].apply(lambda x: 1 if raw_genre in x else 0)

    for net in state['top_networks']:
        df[f"net_{net}"] = (df['network_clean'] == net).astype(int)
    if "net_Other" not in df.columns:
        df["net_Other"] = (df['network_clean'] == "Other").astype(int)

    # Text features: mirror feature_engineering exactly (MiniLM embeddings -> fitted PCA).
    if 'pca' in state and 'pca_cols' in state:
        from sentence_transformers import SentenceTransformer
        text_content = ("Title: " + df['name'].fillna('Unknown') +
                        ". Network: " + df['primary_network'].fillna('Unknown') +
                        ". Creator: " + df['created_by'].fillna('').astype(str).str.slice(0, 80) +
                        ". Cast: " + df['actors'].fillna('').astype(str).str.slice(0, 120) +
                        ". " + df['overview'].fillna(''))
        _stm = SentenceTransformer(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
        _emb = _stm.encode(text_content.tolist(), show_progress_bar=False)
        _pca = state['pca'].transform(_emb)
        pca_df = pd.DataFrame(_pca, columns=state['pca_cols'], index=df.index)
        df = pd.concat([df, pca_df], axis=1)
    elif 'tfidf_model' in state:  # legacy fallback
        df['overview'] = df['overview'].fillna('')
        tfidf_matrix = state['tfidf_model'].transform(df['overview'])
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=state['tfidf_cols'], index=df.index)
        df = pd.concat([df, tfidf_df], axis=1)

    X = df[state['features_columns']].fillna(0)

    # 3. Predict using all models
    print(f"📊 Predicting for {len(df)} shows...")
    
    bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
    present_bucket_vals = np.array([bucket_map[c] for c in unique_classes])

    for name, model in models.items():
        if name == "Ordinal":
            probs = model.predict_proba(X)
            preds = np.sum(probs * present_bucket_vals, axis=1)
        else:
            preds = model.predict(X)
        df[f'pred_{name.lower()}'] = np.round(np.clip(preds, 0, 5) * 2) / 2

    # Main prediction column (Using Stacking for best accuracy)
    df['predicted_rating'] = df['pred_stacking']

    # 4. Evaluation and Visuals
    results_dir = config.BASE_DIR / "results" / "shows"
    results_dir.mkdir(parents=True, exist_ok=True)

    if 'user_rating' in df.columns and not df['user_rating'].isna().all():
        eval_df = df.dropna(subset=['user_rating']).copy()
        y_true = eval_df['user_rating']
        
        print("\n" + "="*60)
        print("📊 SHOW ENSEMBLE PERFORMANCE REPORT (FULL DATASET)")
        print("="*60)
        
        for name in models.keys():
            y_pred = eval_df[f'pred_{name.lower()}']
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            diffs = np.abs(y_true - y_pred)
            
            print(f"Model: {name:<10} | MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
            print(f"   Accuracy -> Exact: {((diffs == 0.0).sum()/len(y_true))*100:.1f}% | ±0.5: {((diffs <= 0.5).sum()/len(y_true))*100:.1f}%")
        print("="*60 + "\n")

        # Visualizations
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # KDE Plot
        fig, ax = plt.subplots(figsize=(14, 7))
        sns.kdeplot(data=eval_df, x='user_rating', label='Actual', color='black', linewidth=3, fill=True, alpha=0.1)
        for name in ["Baseline", "Stacking", "Ordinal"]:
            sns.kdeplot(data=eval_df, x=f'pred_{name.lower()}', label=f'{name} Preds', linestyle='--')
        ax.set_title('TV Show Rating Distributions (KDE)', fontsize=16)
        ax.set_xticks(np.arange(0, 5.5, 0.5))
        plt.legend()
        plt.savefig(results_dir / "shows_distribution_kde.png", dpi=150)
        plt.close()

        # Histogram Grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
        axes = axes.flatten()
        plot_models = ["Baseline", "XGB", "SVR", "CatBoost", "Stacking", "Ordinal"]
        for i, name in enumerate(plot_models):
            sns.histplot(eval_df[f'pred_{name.lower()}'], bins=np.arange(0.25, 5.75, 0.5), ax=axes[i], kde=False, color='skyblue', edgecolor='black')
            axes[i].set_title(f'{name} Predictions')
            axes[i].set_xticks(np.arange(0.5, 5.5, 0.5))
        fig.suptitle('Histogram of TV Show Rating Predictions', fontsize=20)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(results_dir / "shows_distribution_histograms.png", dpi=150)
        plt.close()

    # 5. Save Results
    # Dashboard view (Cleaned up)
    output_cols = ['name', 'year', 'user_rating', 'predicted_rating', 'primary_network']
    if 'user_rating' in df.columns:
        df['abs_diff'] = np.abs(df['user_rating'] - df['predicted_rating'])
        output_cols.append('abs_diff')
    
    final_df = df[output_cols].sort_values(by='predicted_rating', ascending=False)
    final_df.to_csv(config.TV_SHOWS_FULL_VIEW_PATH, index=False)
    
    # Full results
    full_results_path = results_dir / "show_predictions_full.csv"
    df.to_csv(full_results_path, index=False)
    
    print(f"✅ Dashboard view saved to: {config.TV_SHOWS_FULL_VIEW_PATH}")
    print(f"✅ Full results and graphs saved to: {results_dir}")

if __name__ == "__main__":
    run_consolidated_predictions()
