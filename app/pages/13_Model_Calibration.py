import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

st.set_page_config(page_title="Model Calibration", page_icon="📈", layout="wide")

st.header("📈 Model Calibration & Reliability")
st.markdown("""
This dashboard evaluates how "honest" the model's confidence is. 
In a perfectly calibrated model, when it predicts a rating of 4.5, your actual average rating for those items should be exactly 4.5.
""")

@st.cache_data
def load_prediction_data():
    # Load unified predictions if available
    path = config.UNIFIED_PREDICTIONS_DIR / "unified_predictions_ensemble.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

df = load_prediction_data()

# The unified prediction file exposes several model columns; prefer the headline
# Mean Ensemble, then fall back to whatever prediction column is present.
PREFERRED_PRED_COLS = ['pred_MeanEnsemble', 'pred_Stacking', 'pred_XGB_Base',
                       'pred_CatBoost_Base', 'pred_Ordinal_EV']

def pick_pred_col(frame):
    for c in PREFERRED_PRED_COLS:
        if c in frame.columns:
            return c
    fallbacks = [c for c in frame.columns if c.startswith('pred_')]
    return fallbacks[0] if fallbacks else None

if df is not None:
    pred_col = pick_pred_col(df)
    if pred_col and 'user_rating' in df.columns:
        st.caption(f"Calibrating on **`{pred_col}`** (the deployed unified model).")
        eval_df = df.dropna(subset=['user_rating', pred_col]).copy()
        eval_df['user_rating'] = pd.to_numeric(eval_df['user_rating'], errors='coerce')
        eval_df = eval_df.dropna(subset=['user_rating'])

        # Calibration Curve
        # Bin predictions
        bins = np.arange(0.25, 5.75, 0.5)
        eval_df['pred_bin'] = pd.cut(eval_df[pred_col], bins=bins, labels=np.arange(0.5, 5.5, 0.5))
        
        calibration = eval_df.groupby('pred_bin')['user_rating'].agg(['mean', 'std', 'count']).reset_index()
        calibration['pred_bin'] = calibration['pred_bin'].astype(float)
        
        fig = go.Figure()
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0.5, 5.0], y=[0.5, 5.0],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='black', dash='dash')
        ))
        
        # Actual calibration line
        fig.add_trace(go.Scatter(
            x=calibration['pred_bin'], y=calibration['mean'],
            mode='lines+markers',
            name='Ensemble Performance',
            error_y=dict(type='data', array=calibration['std'], visible=True),
            line=dict(color='red', width=3)
        ))
        
        fig.update_layout(
            title="Reliability Diagram (Predicted vs. Actual)",
            xaxis_title="Predicted Rating",
            yaxis_title="Actual Average Rating",
            xaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5),
            yaxis=dict(tickmode='linear', tick0=0.5, dtick=0.5),
            height=600,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution of predictions
        st.subheader("Prediction Frequency")
        fig_hist = go.Figure(data=[go.Histogram(x=eval_df[pred_col], xbins=dict(start=0.25, end=5.25, size=0.5))])
        fig_hist.update_layout(title="Histogram of Predicted Ratings", xaxis_title="Rating Bucket", yaxis_title="Count")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning(f"No usable prediction/`user_rating` columns in unified_predictions_ensemble.csv "
                   f"(found: {', '.join(df.columns[:8])}…)")
else:
    st.error("Unified predictions data not found. Please run the Unified Prediction script first.")
