import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import shap

from src.data_loader import load_data
from src.feature_engineering import prepare_features
from src.utils import load_artifacts, get_risk_score

st.set_page_config(layout="wide", page_title="Fraud Investigation Dashboard")

@st.cache_resource
def load_resources():
    df = load_data('data/creditcard.csv')
    model, scaler, explainer, threshold = load_artifacts('models/')
    return df, model, scaler, explainer, threshold

df, model, scaler, explainer, threshold = load_resources()

@st.cache_data
def score_all_transactions():
    sample_df = df.sample(n=min(1000, len(df)), random_state=42).reset_index(drop=True)
    X, _, _ = prepare_features(sample_df, scaler=scaler, fit_scaler=False)
    
    probs = model.predict_proba(X)[:, 1]
    scores = [get_risk_score(p) for p in probs]
    preds = (probs >= threshold).astype(int)
    
    sample_df['risk_score'] = scores
    sample_df['prediction'] = preds
    sample_df['probability'] = probs
    
    for col in X.columns:
        sample_df[col] = X[col].values
    return sample_df, X.columns.tolist()

scored_df, feature_cols = score_all_transactions()

st.sidebar.header("Filters")
risk_min = st.sidebar.slider("Minimum Risk Score", 0, 100, 0)
risk_max = st.sidebar.slider("Maximum Risk Score", 0, 100, 100)
pred_filter = st.sidebar.selectbox("Prediction", ["All", "Fraud", "Legitimate"])
time_filter = st.sidebar.slider("Hour of Day", 0, 23, (0, 23))

filtered = scored_df[
    (scored_df['risk_score'] >= risk_min) &
    (scored_df['risk_score'] <= risk_max) &
    (scored_df['Hour'] >= time_filter[0]) &
    (scored_df['Hour'] <= time_filter[1])
]

if pred_filter != "All":
    filtered = filtered[filtered['prediction'] == (1 if pred_filter == "Fraud" else 0)]

st.title("🕵️ Smart Credit Card Fraud Investigation")
st.markdown("**Explainable AI for fraud detection**")

col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(filtered))
col2.metric("Fraudulent (predicted)", filtered['prediction'].sum())
col3.metric("Average Risk Score", f"{filtered['risk_score'].mean():.1f}")

st.subheader("🏆 Top Risky Transactions")
top_n = st.slider("Number of top transactions to show", 5, 50, 10)
top_risky = filtered.nlargest(top_n, 'risk_score')[['Time', 'Amount', 'risk_score', 'prediction', 'Hour'] + [f'V{i}' for i in range(1, 29)]]
st.dataframe(top_risky, use_container_width=True)

st.subheader("🔍 Transaction Explanation")
if len(filtered) > 0:
    trans_id = st.selectbox("Select a transaction (index)", filtered.index.tolist())
    trans = filtered.loc[trans_id]
    
    # --- NEW: HACKATHON MOCK CUSTOMER DATA ---
    # We use the transaction ID as a seed so the fake data stays consistent if you click away and come back
    np.random.seed(trans_id) 
    mock_names = ["Emma Watson", "John Doe", "Aisha Khan", "Liam Chen", "Sophia Patel", "Jackson Smith"]
    customer_name = np.random.choice(mock_names)
    past_avg = max(15.0, trans['Amount'] * np.random.uniform(0.3, 1.2)) # Fake past average
    past_std = past_avg * np.random.uniform(0.1, 0.5) # Fake standard deviation
    
    # --- UI: CUSTOMER PROFILE DASHBOARD ---
    st.markdown("### 👤 Customer & Transaction Profile")
    
    # Create 4 clean columns for the top metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cardholder Name", customer_name)
    m2.metric("Transaction Amount", f"${trans['Amount']:.2f}")
    m3.metric("Fraud Probability", f"{trans['probability'] * 100:.2f}%")
    m4.metric("Risk Decision", "🛑 BLOCKED" if trans['prediction'] == 1 else "✅ APPROVED")
    
    # Create 2 columns for the historical context
    h1, h2 = st.columns(2)
    h1.info(f"**Historical Average:** ${past_avg:.2f}")
    h2.warning(f"**Historical Std Dev:** ±${past_std:.2f}")
    
    st.divider() # Adds a nice visual line

    # --- UI: SHAP EXPLANATION ---
    st.markdown("### 🧠 AI Explanation (Why is this suspicious?)")
    
    X_single = pd.DataFrame([trans[feature_cols].values], columns=feature_cols)
    shap_values = explainer.shap_values(X_single)
    
    shap_vals_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    base_val_to_plot = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    # Waterfall chart
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals_to_plot,
            base_values=base_val_to_plot,
            data=X_single.iloc[0].values,
            feature_names=feature_cols
        ),
        show=False,
        max_display=10
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    st.warning("No transactions match the filters.")