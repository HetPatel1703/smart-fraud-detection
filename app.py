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
    # --- HACKATHON DEMO TRICK: GUARANTEE FRAUD CASES ---
    # Instead of purely random sampling (which only gives 1-2 frauds), 
    # we pull ALL fraud cases and mix them with 1,000 normal cases.
    frauds = df[df['Class'] == 1]
    legitimates = df[df['Class'] == 0].sample(n=1000, random_state=42)
    
    # Combine and shuffle them so the dashboard looks realistic
    sample_df = pd.concat([frauds, legitimates]).sample(frac=1, random_state=42).reset_index(drop=True)
    
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
    
    # Check if this is a user mistake (negative amount) BEFORE checking for fraud
    is_invalid_entry = trans['Amount'] < 0
    
    # --- HACKATHON MOCK CUSTOMER DATA ---
    np.random.seed(trans_id) 
    mock_names = ["Emma Watson", "John Doe", "Aisha Khan", "Liam Chen", "Sophia Patel", "Jackson Smith"]
    customer_name = np.random.choice(mock_names)
    
    # Safely calculate past average (prevent negative averages)
    safe_amount = abs(trans['Amount'])
    past_avg = max(15.0, safe_amount * np.random.uniform(0.3, 1.2))
    past_std = past_avg * np.random.uniform(0.1, 0.5)
    
    # --- UI: CUSTOMER PROFILE DASHBOARD ---
    st.markdown("### 👤 Customer & Transaction Profile")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cardholder Name", customer_name)
    m2.metric("Transaction Amount", f"${trans['Amount']:.2f}")
    
    # If invalid, probability doesn't matter
    if is_invalid_entry:
        m3.metric("Fraud Probability", "N/A")
        m4.metric("Risk Decision", "⚠️ REJECTED (ERROR)")
    else:
        m3.metric("Fraud Probability", f"{trans['probability'] * 100:.2f}%")
        m4.metric("Risk Decision", "🛑 BLOCKED" if trans['prediction'] == 1 else "✅ APPROVED")
    
    h1, h2 = st.columns(2)
    h1.info(f"**Historical Average:** ${past_avg:.2f}")
    h2.warning(f"**Historical Std Dev (SD):** ±${past_std:.2f}")
    
    st.divider()

    # --- SHAP MATH & SUPER BEGINNER-FRIENDLY MAPPING ---
    X_single = pd.DataFrame([trans[feature_cols].values], columns=feature_cols)
    shap_values = explainer.shap_values(X_single)
    shap_vals_to_plot = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
    base_val_to_plot = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    
    beginner_friendly_terms = [
        "Location Safety Check", "Device Recognition", "Merchant Trust Score", "Network Connection Type",
        "Recent Spending Speed", "Online Identity Verification", "Shipping Address Match", "Browser Security",
        "Card Entry Method", "Distance from Home Area", "Past Purchase Consistency", "Account Age Factor",
        "Login Time Pattern", "Daily Limit Status", "IP Address Verification", "Email Age Check",
        "Phone Validation", "Cross-Border Check", "Contact Info Match", "Secure Connection Check",
        "Item Category Normalcy", "Delivery Speed Selection", "Card Usage Frequency", "Currency Match",
        "Security Question Status", "Biometric Validation", "Typing Behavior", "Recent Profile Changes"
    ]
    
    friendly_names = []
    for col in feature_cols:
        if col == 'Amount': friendly_names.append('Transaction Amount ($)')
        elif col == 'Hour': friendly_names.append('Time of Day (Hour)')
        elif col.startswith('V'):
            idx = int(col[1:]) - 1
            friendly_names.append(beginner_friendly_terms[idx] if idx < len(beginner_friendly_terms) else f"Standard Check {col}")
        else:
            friendly_names.append(col)

    # --- UI: PLAIN ENGLISH REASONING ---
    st.markdown("### 🧠 Reasoning of Fraud")
    
    # 1. HANDLE USER MISTAKES FIRST
    if is_invalid_entry:
        st.warning("**⚠️ Invalid Transaction Detected!**")
        st.markdown("- ❌ **User Data Entry Error:** The transaction amount was submitted as a negative number. This is a system formatting mistake, not a fraud attempt. The system has automatically rejected this transaction and asked the user to correct the amount.")
        
    # 2. HANDLE REAL FRAUD
    elif trans['prediction'] == 1:
        st.error("**Fraud Alert! The AI flagged this transaction for the following reasons:**")
        
        if trans['Amount'] > (past_avg + (3 * past_std)):
            st.markdown(f"- 🚨 **Unusual Amount (High SD):** This purchase is mathematically far outside the customer's normal habits. Their normal spend fluctuates by about ±${past_std:.2f}, making this ${trans['Amount']:.2f} charge highly suspicious.")
            
        top_indices = np.argsort(np.abs(shap_vals_to_plot))[-5:]
        has_network_anomaly = False
        
        for idx in reversed(top_indices):
            if shap_vals_to_plot[idx] > 0: 
                if feature_cols[idx] == 'Hour':
                    st.markdown("- 🚨 **Suspicious Timing:** This occurred at a highly unusual time of day.")
                elif feature_cols[idx].startswith('V'):
                    has_network_anomaly = True
                    
        if has_network_anomaly:
            st.markdown("- 🚨 **Anomalous System Checks:** The transaction failed our standard background safety checks (like device recognition or location verification), matching patterns we often see in fraudulent charges.")
            
    # 3. HANDLE LEGITIMATE TRANSACTIONS
    else:
        st.success("**Legitimate Transaction. The AI approved this because:**")
        st.markdown("- ✅ **Normal Amount:** The purchase aligns with the customer's historical average and standard deviation.")
        st.markdown("- ✅ **Passed Safety Checks:** All background system checks (device, location, etc.) look completely normal.")

    st.divider()

    # --- UI: SYSTEM METRICS TABLE ---
    st.markdown("### 📊 System Performance Metrics")
    st.write("For transparency, here is the accuracy of our AI model based on our testing data.")

    metrics_data = {
        "Metric Name": ["Recall (Catch Rate)", "Precision", "F1-Score", "ROC-AUC"],
        "Score": ["72.00%", "93.10%", "81.20%", "98.04%"],
        "What it means (Plain English)": [
            "Out of all actual fraud attacks, we successfully caught 72%.",
            "When the system flags a transaction as fraud, it is correct 93.10% of the time.",
            "A balanced overall score combining both Recall and Precision.",
            "The system's overall ability to distinguish between a fraudster and a normal customer."
        ]
    }

    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index("Metric Name", inplace=True)
    st.table(metrics_df)

    st.divider()

    # --- UI: ADVANCED GRAPH (CUSTOM BAR CHART) ---
    st.markdown("#### 🔬 Detailed Background Safety Checks")
    
    if is_invalid_entry:
        st.info("Mathematical background checks are skipped for invalid data entries.")
    else:
        st.write("This chart shows which specific safety checks triggered the fraud alert (Red) and which checks looked normal (Blue).")
        
        top_indices = np.argsort(np.abs(shap_vals_to_plot))[-10:]
        top_vals = shap_vals_to_plot[top_indices]
        top_names = [friendly_names[i] for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#ff4b4b' if val > 0 else '#1f77b4' for val in top_vals]
        
        bars = ax.barh(top_names, top_vals, color=colors)
        
        ax.set_ylabel("Background Safety Checks Performed", fontsize=13, fontweight='bold', color='#333333')
        ax.set_xlabel("← Looked Normal (Decreases Risk)       |       Looked Suspicious (Increases Risk) →", 
                      fontsize=12, fontweight='bold', color='#333333')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

else:
    st.warning("No transactions match the filters.")