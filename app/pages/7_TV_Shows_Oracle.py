import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
import ast
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import tmdb_tv, get_tv_show_metadata_ultimate
from src.unified_model.unified_oracle import predict_unified
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="TV Shows Oracle", page_icon="🔮", layout="wide")


# --- HELPERS ---
def safe_eval(val):
    if isinstance(val, list): return val
    if pd.isna(val) or val == "": return []
    try:
        if "[" in str(val): return ast.literal_eval(str(val))
        return [x.strip() for x in str(val).split(',') if x.strip()]
    except Exception:
        return []

def get_primary_network(val):
    if isinstance(val, list):
        return val[0].strip() if val else "Other"
    if pd.isna(val) or val == "": return "Other"
    try:
        if "[" in str(val):
            vl = ast.literal_eval(str(val))
            if vl: return vl[0].strip()
        return str(val).split(',')[0].strip()
    except Exception:
        return "Other"


@st.cache_resource
def get_transformer():
    return SentenceTransformer('all-MiniLM-L6-v2')


# --- PARITY TRANSFORMATION (mirrors src/shows/feature_engineering.py exactly) ---
def transform_for_parity(meta, state):
    med = state.get('median_values', {})
    X = pd.DataFrame(0.0, index=[0], columns=state['features_columns'])

    # Numericals
    vals = {
        'year': pd.to_numeric(meta.get('year'), errors='coerce'),
        'vote_average': pd.to_numeric(meta.get('vote_average'), errors='coerce'),
        'vote_count_log': np.log1p(pd.to_numeric(meta.get('vote_count'), errors='coerce') or 0),
        'season_count': pd.to_numeric(meta.get('number_of_seasons'), errors='coerce'),
        'is_adult': 1 if str(meta.get('age_rating')) in ['TV-MA', 'R', '18+'] else 0,
    }
    for k, v in vals.items():
        if k in X.columns:
            if pd.isna(v):
                v = med.get(k, 1 if k == 'season_count' else 0)
            X.at[0, k] = float(v)

    # Genres (kept set only)
    glist = safe_eval(meta.get('genres'))
    for col in state.get('kept_genres', []):
        if col in X.columns:
            X.at[0, col] = 1 if col.replace('g_', '') in glist else 0

    # Network one-hot
    net = get_primary_network(meta.get('network'))
    net_clean = net if net in state.get('top_networks', []) else "Other"
    ncol = f"net_{net_clean}"
    if ncol in X.columns:
        X.at[0, ncol] = 1

    # MiniLM embedding -> PCA
    text = (f"Title: {meta.get('name', 'Unknown')}. "
            f"Network: {net}. {meta.get('overview', '') or ''}")
    try:
        emb = get_transformer().encode([text])
        pcav = state['pca'].transform(emb)[0]
        for i, c in enumerate(state.get('pca_cols', [])):
            if c in X.columns:
                X.at[0, c] = pcav[i]
    except Exception:
        pass

    return X[state['features_columns']]


# --- LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    art = {}
    ensemble_dir = config.TV_SHOWS_MODEL_DIR / "ensemble"
    try:
        art['state'] = joblib.load(config.TV_SHOWS_PREPROCESSOR_STATE)
        art['stacking'] = joblib.load(ensemble_dir / "stacking_ensemble_regressor.joblib")
        art['xgb'] = joblib.load(ensemble_dir / "xgb_base_regressor.joblib")
        art['svr'] = joblib.load(ensemble_dir / "svr_base_regressor.joblib")
        art['cat'] = joblib.load(ensemble_dir / "catboost_base_regressor.joblib")
        art['ordinal'] = joblib.load(ensemble_dir / "ordinal_classifier.joblib")
        art['ordinal_classes'] = joblib.load(ensemble_dir / "ordinal_classes.joblib")
    except Exception as e:
        art['error'] = str(e)
    if config.TV_SHOWS_MODEL_CLASSIFIER.exists():
        try: art['clf'] = joblib.load(config.TV_SHOWS_MODEL_CLASSIFIER)
        except Exception: art['clf'] = None
    return art


@st.cache_data
def load_library():
    try:
        return pd.read_csv(config.TV_SHOWS_ENRICHED_DATA_PATH)
    except Exception:
        return pd.DataFrame()


def find_similar_shows(meta, n=3):
    """Cosine similarity over MiniLM embeddings of your rated shows."""
    lib = load_library()
    if lib.empty:
        return []
    lib = lib[lib['user_rating'].notna()].copy() if 'user_rating' in lib.columns else lib.copy()
    if lib.empty:
        return []
    model = get_transformer()
    net = get_primary_network(meta.get('network'))
    target_text = f"Title: {meta.get('name', '')}. Network: {net}. {meta.get('overview', '') or ''}"
    lib['_text'] = ("Title: " + lib['name'].fillna('Unknown') +
                    ". Network: " + lib['network'].fillna('').apply(get_primary_network) +
                    ". " + lib['overview'].fillna(''))
    embs = model.encode([target_text] + lib['_text'].tolist())
    tv, lv = embs[0], embs[1:]
    sims = lv @ tv / ((np.linalg.norm(lv, axis=1) * np.linalg.norm(tv)) + 1e-9)
    target_name = str(meta.get('name', '')).lower().strip()
    out = []
    for idx in np.argsort(sims)[::-1]:
        row = lib.iloc[idx]
        if str(row.get('name', '')).lower().strip() == target_name:
            continue
        shared = set(safe_eval(meta.get('genres'))) & set(safe_eval(row.get('genres')))
        out.append({
            'name': row.get('name', 'Unknown'),
            'year': row.get('year', ''),
            'score': float(sims[idx]),
            'shared': ", ".join(sorted(shared)) if shared else "overall vibe",
        })
        if len(out) >= n:
            break
    return out


# --- UI ---
st.header("🔮 The TV Show Oracle")
st.markdown("Predict your rating for any series using a Stacking Ensemble, an Ordinal Probability Engine, and Semantic Text Embeddings.")

artifacts = load_artifacts()
if 'error' in artifacts:
    st.error(f"❌ Show ensemble not found ({artifacts['error']}). Run `python -m src.shows.model_trainer` first.")
    st.stop()
state = artifacts['state']

# Session state
for k, v in {'tv_query': "", 'tv_results': [], 'tv_sel_id': None, 'tv_raw': None, 'tv_last': ""}.items():
    st.session_state.setdefault(k, v)

search_col, _ = st.columns([3, 1])
with search_col:
    term = st.text_input("Search for a TV Show", value=st.session_state['tv_query'], key="tv_search_input")
    st.session_state['tv_query'] = term
    if term and term != st.session_state['tv_last']:
        st.session_state['tv_last'] = term
        try:
            res = tmdb_tv.search(term)
            st.session_state['tv_results'] = [
                {'id': r.id, 'name': r.name,
                 'year': (getattr(r, 'first_air_date', '') or '????')[:4],
                 'poster_path': getattr(r, 'poster_path', None),
                 'overview': getattr(r, 'overview', '')}
                for r in list(res)[:5]
            ]
        except Exception as e:
            st.session_state['tv_results'] = []
            st.warning(f"Search failed: {e}")
        st.session_state['tv_sel_id'] = None
        st.session_state['tv_raw'] = None

    if st.session_state['tv_results']:
        st.write("Search Results:")
        for r in st.session_state['tv_results']:
            c_img, c_info, c_btn = st.columns([1, 4, 1])
            with c_img:
                if r['poster_path']:
                    st.image(f"https://image.tmdb.org/t/p/w92{r['poster_path']}")
            with c_info:
                st.markdown(f"**{r['name']}** ({r['year']})")
            with c_btn:
                if st.button("Select", key=f"tv_sel_{r['id']}"):
                    st.session_state['tv_sel_id'] = r['id']
                    st.session_state['tv_results'] = []
                    st.rerun()
    elif term:
        st.info("No results — try another title.")

# Fetch full metadata for the selected show
if st.session_state['tv_sel_id'] and not st.session_state['tv_raw']:
    with st.spinner("Fetching Ultimate Metadata (TMDB + TVMaze + OMDb)..."):
        meta = get_tv_show_metadata_ultimate(None, None, tmdb_id=st.session_state['tv_sel_id'])
        if meta and meta.get('name'):
            st.session_state['tv_raw'] = meta
        else:
            st.error("Failed to fetch details.")
            st.session_state['tv_sel_id'] = None

# --- PREDICTION UI ---
if st.session_state['tv_raw']:
    raw = st.session_state['tv_raw']
    st.divider()
    c_img, c_meta, c_pred = st.columns([1, 2, 2])

    with c_img:
        poster = raw.get('poster_path')
        if poster and str(poster).startswith('/'):
            st.image(f"https://image.tmdb.org/t/p/w500{poster}")
        elif poster and str(poster).startswith('http'):
            st.image(poster)

    with c_meta:
        st.subheader(f"{raw.get('name')} ({raw.get('year')})")
        st.write(f"**Network:** {get_primary_network(raw.get('network'))}")
        st.write(f"**Seasons:** {raw.get('number_of_seasons', 'N/A')}")
        genres = safe_eval(raw.get('genres'))
        if genres:
            st.write(f"**Genres:** {', '.join(genres)}")
        st.caption((raw.get('overview') or '')[:300])
        consult_btn = st.button("Consult the Oracle 🔮", type="primary", use_container_width=True)

    with c_pred:
        if consult_btn:
            with st.spinner("Analyzing Embeddings & Ensembles..."):
                X = transform_for_parity(raw, state)

                def round_half(x): return np.round(np.clip(x, 0, 5) * 2) / 2
                v_stack = round_half(artifacts['stacking'].predict(X)[0])

                # Base models are a cosmetic breakdown; skip any that aren't usable
                # (e.g. a base estimator persisted unfitted) rather than crashing.
                def safe_pred(key):
                    try:
                        return round_half(artifacts[key].predict(X)[0])
                    except Exception:
                        return None
                v_svr, v_cat, v_xgb = safe_pred('svr'), safe_pred('cat'), safe_pred('xgb')

                # Ordinal EV over present classes
                bucket_map = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0, 6: 3.5, 7: 4.0, 8: 4.5, 9: 5.0}
                present = np.array([bucket_map[c] for c in artifacts['ordinal_classes']])
                ord_probs = artifacts['ordinal'].predict_proba(X)[0]
                v_ord = round_half(np.sum(ord_probs * present))

                # Binary "above your median" verdict
                verdict = "N/A"
                if artifacts.get('clf') is not None:
                    try:
                        verdict = "Above your median 👍" if int(artifacts['clf'].predict(X)[0]) == 1 else "Below your median 👎"
                    except Exception:
                        pass

                st.markdown("### The Verdict")
                c1, c2, c3 = st.columns(3)
                base_bits = [f"{lbl}: {v:.1f}" for lbl, v in
                             [("XGBoost", v_xgb), ("SVR", v_svr), ("CatBoost", v_cat)] if v is not None]
                hover = "Base models →  " + "  ·  ".join(base_bits) if base_bits else "Stacking ensemble prediction."
                c1.metric("Stacking Ensemble", f"⭐ {v_stack:.1f}", help=hover)
                c2.metric("Ordinal (EV)", f"⭐ {v_ord:.1f}", help="Expected value of the rating-bucket distribution.")
                c3.metric("Verdict", verdict, help="XGBoost binary classifier vs. your median rating.")

                # Unified cross-domain model (separate, 397-feature shared space)
                u = predict_unified(raw, 'tv')
                if u is not None:
                    st.metric("🌐 Unified Model (cross-domain)", f"⭐ {np.round(u * 2) / 2:.1f}",
                              help="Independent prediction from the 397-feature Unified Model trained across "
                                   "all domains — a cross-domain sanity check against the TV stacking ensemble.")

                # SHAP
                import shap
                import matplotlib.pyplot as plt
                st.markdown("#### Why this prediction?")
                try:
                    explainer = shap.TreeExplainer(artifacts['xgb'])
                    shap_values = explainer(X)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except Exception as e:
                    st.info(f"Feature explanation unavailable. ({e})")

                # Confidence engine
                st.markdown("#### Confidence Engine (Ordinal Spread)")
                full_probs = {f"{bucket_map[c]:.1f}": p for c, p in zip(artifacts['ordinal_classes'], ord_probs)}
                prob_df = pd.DataFrame({
                    "Rating": [f"{v:.1f}" for v in np.arange(0.5, 5.5, 0.5)],
                    "Probability (%)": [full_probs.get(f"{v:.1f}", 0.0) * 100 for v in np.arange(0.5, 5.5, 0.5)],
                })
                st.bar_chart(prob_df.set_index("Rating"), height=160)

    if consult_btn:
        st.divider()
        st.subheader("💡 Semantic Matches")
        st.caption("Shows you've already rated with similar vibes (title, network, plot embeddings).")
        sims = find_similar_shows(raw, n=3)
        if sims:
            cols = st.columns(len(sims))
            for i, (col, s) in enumerate(zip(cols, sims)):
                with col:
                    st.markdown(f"**{i+1}. {s['name']} ({s['year']})**")
                    st.markdown(f"*Match: {s['score']*100:.1f}%*")
                    st.caption(f"Shared: {s['shared']}")
        else:
            st.info("No similar shows found in your watch history.")
