import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src import config

sys.path.append(str(Path(__file__).resolve().parent.parent))  # app/ dir for shared helpers
from mobile import apply_mobile_css
st.set_page_config(page_title="Model Calibration", page_icon="📈", layout="wide")
apply_mobile_css()

st.header("📈 Model Calibration & Reliability")
st.markdown("""
This dashboard evaluates how "honest" the model's confidence is. 
In a perfectly calibrated model, when it predicts a rating of 4.5, your actual average rating for those items should be exactly 4.5.
""")

@st.cache_data
def load_prediction_data():
    # Out-of-fold unified predictions: each comes from a model that never trained on
    # that item. Averaged over the CV repeats, one row per item.
    path = config.BASE_DIR / "reports" / "oof_predictions.csv"
    if not path.exists():
        return None
    oof = pd.read_csv(path)
    return (oof.groupby(['media_type', 'source_id'], as_index=False)
               .agg(user_rating=('target_reg', 'first'), pred=('pred', 'mean')))

df = load_prediction_data()

if df is not None:
    pred_col = 'pred'
    st.caption("Calibrating on the unified model's **out-of-fold** predictions "
               "(`reports/oof_predictions.csv`, averaged over CV repeats).")
    eval_df = df.dropna(subset=['user_rating', pred_col]).copy()

    # Skill = 1 - MSE / variance: 0 is "always predict my mean rating".
    skill = eval_df.groupby('media_type').apply(
        lambda g: pd.Series({'items': len(g),
                             'skill': 1 - ((g['user_rating'] - g[pred_col]) ** 2).mean()
                                      / g['user_rating'].var(ddof=0)}),
        include_groups=False)
    st.dataframe(skill.style.format({'items': '{:.0f}', 'skill': '{:.3f}'}))

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
    st.error("reports/oof_predictions.csv not found. Run `python -m src.unified_model.comprehensive_evaluator` first.")
