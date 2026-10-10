import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config
from src.shows.ingestion import tmdb_tv, get_tv_show_metadata_ultimate
from src.unified_model.unified_oracle import predict_unified
from src.shows.feature_engineering import (transform_single_show, safe_eval, get_primary_network,
                                           _sentence_model, _show_header, _show_body)
from src.common.parsing import round_half

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="TV Shows Oracle", page_icon="🔮", layout="wide")
apply_mobile_css()


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
    model = _sentence_model(state.get('sentence_transformer', 'all-MiniLM-L6-v2'))
    net = get_primary_network(meta.get('network'))
    from src.common.text_embedding import encode_pooled
    # Same header/body helpers as training, so "similar shows" is measured in the
    # space the model actually learned in rather than a third ad-hoc string.
    _target = dict(meta)
    _target['primary_network'] = net
    from src.shows.season_data import attach_season_narrative
    lib = attach_season_narrative(lib)
    _lib_recs = lib.to_dict('records')
    for _r in _lib_recs:
        _r['primary_network'] = get_primary_network(_r.get('network'))
    headers = [_show_header(_target)] + [_show_header(r) for r in _lib_recs]
    bodies = [_show_body(_target)] + [_show_body(r) for r in _lib_recs]
    target_text = f"{headers[0]} {bodies[0]}"
    lib['_text'] = [f"{h} {b}" for h, b in zip(headers[1:], bodies[1:])]

    embs = encode_pooled(headers, bodies, model)
    tv, lv = embs[0], embs[1:]
    # encode_pooled returns unit vectors, so the dot product IS the cosine.
    sims = lv @ tv
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
    do_search = st.button("🔍 Search", key="tv_search_btn", use_container_width=True)
    if term and (do_search or term != st.session_state['tv_last']):
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
                X = transform_single_show(raw, state)
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
