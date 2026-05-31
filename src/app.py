
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fraud Detection API")

# Load model and get feature names
with open("best_model.pkl", "rb") as f:
    best = pickle.load(f)

with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

model        = best["model"]
THRESHOLD    = best["threshold"]
FEATURE_COLS = data["X_test"].columns.tolist()  # all 225 features

class Transaction(BaseModel):
    TransactionAmt: float = 100.0
    ProductCD:      int   = 0
    card1:          int   = 5000
    hour:           int   = 12

@app.get("/")
def root():
    return {"message": "Fraud Detection API", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(txn: Transaction):
    # Create a row of zeros with all 225 feature columns
    row = pd.DataFrame([np.zeros(len(FEATURE_COLS))], columns=FEATURE_COLS)

    # Fill in the values we have
    row["TransactionAmt"] = txn.TransactionAmt
    row["ProductCD"]      = txn.ProductCD
    row["card1"]          = txn.card1
    row["hour"]           = txn.hour
    row["amt_log"]        = np.log1p(txn.TransactionAmt)
    row["is_round"]       = int(txn.TransactionAmt % 1 == 0)
    row["is_night"]       = int(txn.hour <= 6)

    proba    = model.predict_proba(row)[0][1]
    is_fraud = int(proba >= THRESHOLD)

    return {
        "is_fraud":   is_fraud,
        "confidence": round(float(proba), 4),
        "threshold":  THRESHOLD,
        "risk_level": "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.4 else "LOW"
    }
