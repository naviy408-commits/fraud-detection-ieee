
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_all():
    with open("/content/best_model.pkl", "rb") as f:
        best = pickle.load(f)
    with open("/content/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
    return best["model"], best["threshold"], list(data["X_test"].columns)

model, THRESHOLD, FEATURE_COLS = load_all()

st.title("🔍 Fraud Detection System")
st.markdown("### IEEE-CIS Dataset — XGBoost Model")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model",          "XGBoost")
col2.metric("Dataset",        "IEEE-CIS")
col3.metric("Threshold",      str(THRESHOLD))
col4.metric("Total Features", str(len(FEATURE_COLS)))

st.markdown("---")

st.sidebar.title("About")
st.sidebar.markdown("This dashboard predicts whether a transaction is fraudulent.")
st.sidebar.markdown("**Model:** XGBoost")
st.sidebar.markdown("**Dataset:** IEEE-CIS Kaggle")
st.sidebar.markdown("**Rows:** 590,540")
st.sidebar.markdown("**Technique:** SMOTE + Feature Engineering")
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use:**")
st.sidebar.markdown("1. Enter transaction details")
st.sidebar.markdown("2. Click Check for Fraud")
st.sidebar.markdown("3. See prediction result")

st.subheader("Enter Transaction Details")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Transaction Info**")
    amount   = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=100000.0, value=150.0, step=10.0)
    product  = st.selectbox("Product Code", options=[0,1,2,3,4], index=0)
    is_round = st.checkbox("Round Amount")

with c2:
    st.markdown("**Card Info**")
    card1      = st.number_input("Card1", min_value=0, max_value=20000, value=5000)
    card2      = st.number_input("Card2", min_value=0.0, max_value=600.0, value=200.0)
    card1_freq = st.slider("Card1 Usage Frequency", min_value=1, max_value=500, value=10)

with c3:
    st.markdown("**Time Info**")
    hour       = st.slider("Hour of Day", min_value=0, max_value=23, value=14)
    is_night   = st.checkbox("Night Transaction (12am-6am)")
    is_weekend = st.checkbox("Weekend Transaction")

st.markdown("---")

if st.button("🔎 Check for Fraud", type="primary", use_container_width=True):

    row = pd.DataFrame(
        [np.zeros(len(FEATURE_COLS))],
        columns=FEATURE_COLS
    )

    mapping = [
        ("TransactionAmt", float(amount)),
        ("ProductCD",      int(product)),
        ("card1",          int(card1)),
        ("card2",          float(card2)),
        ("hour",           int(hour)),
        ("day",            5 if is_weekend else 1),
        ("amt_log",        float(np.log1p(amount))),
        ("is_round",       int(is_round)),
        ("is_night",       int(is_night)),
        ("is_weekend",     int(is_weekend)),
        ("card1_freq",     int(card1_freq)),
    ]

    for col, val in mapping:
        if col in FEATURE_COLS:
            row[col] = val

    proba    = model.predict_proba(row)[0][1]
    is_fraud = proba >= THRESHOLD

    st.markdown("---")
    st.subheader("Prediction Result")

    if is_fraud:
        st.error("FRAUD DETECTED")
        st.markdown(f"### Fraud Confidence: {proba:.2%}")
    else:
        st.success("LEGITIMATE TRANSACTION")
        st.markdown(f"### Legitimate Confidence: {1-proba:.2%}")

    if proba > 0.7:
        risk = "HIGH RISK"
    elif proba > 0.4:
        risk = "MEDIUM RISK"
    else:
        risk = "LOW RISK"

    st.markdown(f"**Risk Level: {risk}**")
    st.markdown("---")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Fraud Probability", f"{proba:.4f}")
    m2.metric("Threshold",         f"{THRESHOLD}")
    m3.metric("Decision",          "FRAUD" if is_fraud else "LEGIT")
    m4.metric("Risk Level",        risk)

    st.markdown("**Fraud Probability Meter:**")
    st.progress(float(proba))

    st.markdown("---")
    st.subheader("Transaction Summary")

    summary = {
        "Field": ["Amount", "Product Code", "Card1", "Hour", "Night", "Weekend", "Round Amount"],
        "Value": [f"${amount:.2f}", str(product), str(card1),
                  f"{hour}:00", str(is_night), str(is_weekend), str(is_round)]
    }
    st.table(summary)
