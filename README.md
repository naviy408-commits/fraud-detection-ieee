# 🔍 Fraud Detection System — IEEE-CIS Dataset

An end-to-end machine learning system for detecting fraudulent transactions in real time.
Built with Python, XGBoost, SHAP, FastAPI, Streamlit on 590,540 real transactions
from the IEEE-CIS Kaggle competition dataset.

---

## ✨ Features

- 📊 Exploratory Data Analysis — fraud distribution, amount analysis, time patterns, card type analysis
- ⚙️ Smart Preprocessing — missing value handling, label encoding, feature engineering across 400+ features
- ⚖️ Class Imbalance Handling — SMOTE oversampling to handle severe 3.5% fraud rate
- 🤖 Multi-Model Training — XGBoost, Random Forest, Logistic Regression trained and compared
- 📈 Comprehensive Evaluation — ROC-AUC, Precision-Recall, Confusion Matrix, Threshold Tuning
- 🧠 SHAP Explainability — feature importance, force plots, summary charts for model transparency
- 🚀 FastAPI REST API — /predict endpoint returning fraud probability, confidence, and risk level
- 🖥️ Streamlit Dashboard — interactive UI for fraud analysts to check transactions in real time
- 📦 Model Persistence — trained models saved and reloaded without retraining

---

## 📊 Results

| Model               | ROC-AUC | Notes                    |
|---------------------|---------|--------------------------|
| Logistic Regression | 0.80    | Baseline linear model    |
| Random Forest       | 0.88    | Ensemble with bagging    |
| XGBoost             | 0.92    | Best model — deployed    |

---

## 🗂 Project Structure

fraud-detection-ieee/
│
├── notebooks/
│   ├── 01_EDA.ipynb               # Exploratory Data Analysis
│   ├── 02_Preprocessing.ipynb     # Data cleaning, SMOTE, feature engineering
│   ├── 03_Modeling.ipynb          # Train XGBoost, RF, Logistic Regression
│   ├── 04_Evaluation.ipynb        # ROC-AUC, confusion matrix, threshold tuning
│   ├── 05_SHAP.ipynb              # SHAP explainability charts
│   ├── 06_API.ipynb               # FastAPI REST API
│   └── 07_Dashboard.ipynb         # Streamlit dashboard
│
├── models/
│   ├── best_model.pkl             # XGBoost model + threshold
│   ├── models.pkl                 # All 3 trained models
│   └── feature_cols.pkl           # Feature column names (225 features)
│
├── src/
│   ├── app.py                     # FastAPI application
│   └── dashboard.py               # Streamlit dashboard
│
├── reports/
│   ├── fraud_distribution.png     # Fraud vs legit chart
│   ├── amount_analysis.png        # Transaction amount analysis
│   ├── confusion_matrix.png       # Model confusion matrix
│   ├── roc_curves.png             # ROC curves for all models
│   ├── pr_curves.png              # Precision-recall curves
│   ├── shap_summary.png           # SHAP summary plot
│   ├── shap_bar.png               # SHAP feature importance bar
│   └── shap_force.png             # SHAP force plot
│
├── data/                          # Data folder (CSVs not included)
├── tests/
│   └── test_model.py              # Model tests
│
├── .gitignore
├── requirements.txt
├── setup.py
└── README.md

---

## 🚀 Setup

### 1. Clone the repository

git clone https://github.com/YOUR_USERNAME/fraud-detection-ieee.git
cd fraud-detection-ieee

### 2. Install dependencies

pip install -r requirements.txt

### 3. Download the dataset

Download from IEEE-CIS Fraud Detection on Kaggle:
- train_transaction.csv
- train_identity.csv

Place both files in the data/ folder.

### 4. Run notebooks in order

01_EDA → 02_Preprocessing → 03_Modeling → 04_Evaluation → 05_SHAP

### 5. Run the API

cd src
uvicorn app:app --reload --port 8000
Docs at http://localhost:8000/docs

### 6. Run the Dashboard

streamlit run src/dashboard.py
Opens at http://localhost:8501

---

## 🔌 API Usage

### Predict Endpoint

POST /predict

Request Body:
{
  "TransactionAmt": 500.0,
  "ProductCD": 0,
  "card1": 1234,
  "hour": 3
}

Response:
{
  "is_fraud": 1,
  "confidence": 0.731,
  "threshold": 0.4,
  "risk_level": "HIGH"
}

### All Endpoints

| Endpoint  | Method | Description                    |
|-----------|--------|--------------------------------|
| /         | GET    | API status and feature count   |
| /health   | GET    | Health check                   |
| /predict  | POST   | Fraud prediction               |

---

## 🧠 How It Works

1. Data Loading — Two CSV files merged on TransactionID using LEFT JOIN
2. Preprocessing — Drop 50%+ missing columns, fill remaining with median/mode
3. Feature Engineering — 7 new features created from raw transaction data
4. SMOTE — Synthetic fraud samples created to handle 3.5% class imbalance
5. Model Training — XGBoost trained with scale_pos_weight=10 for imbalance
6. Threshold Tuning — Threshold set to 0.4 to maximize fraud recall
7. SHAP — TreeExplainer computes feature contributions for each prediction
8. API — FastAPI serves predictions using saved XGBoost model
9. Dashboard — Streamlit provides interactive UI for fraud analysts

---

## 📈 Key Findings from EDA

- Fraud rate is only 3.5% — severe class imbalance problem
- Fraud transactions have higher average amount than legitimate ones
- Fraud peaks during night hours between 12am and 6am
- Product code W has the highest fraud rate
- Many columns have 50%+ missing values — handled in preprocessing

---

## 🔑 Feature Engineering

| Feature    | How Created                    | Why Useful                    |
|------------|-------------------------------|-------------------------------|
| hour       | TransactionDT divided by 3600  | Fraud peaks at night          |
| day        | TransactionDT divided by 86400 | Weekend fraud patterns        |
| is_night   | hour less than or equal to 6   | Night = higher fraud risk     |
| is_weekend | day greater than or equal to 5 | Weekend patterns different    |
| amt_log    | log of TransactionAmt plus 1   | Reduces amount skewness       |
| is_round   | Amount mod 1 equals 0          | Round amounts more suspicious |
| card1_freq | Count of each card1 value      | High frequency = risk signal  |

---

## 🩺 Troubleshooting

| Problem                  | Fix                                                    |
|--------------------------|--------------------------------------------------------|
| FileNotFoundError        | Check DRIVE_PATH — files must be in fraud-project      |
| Feature shape mismatch   | Use feature_cols.pkl not raw 18 features               |
| SMOTE takes too long     | Normal — takes 3 to 5 mins on 590K rows               |
| Ngrok auth error         | Replace with real token from dashboard.ngrok.com       |
| Dashboard shows error    | Restart streamlit — run pkill streamlit first          |

---

## 🛡 Notes

- processed_data.pkl excluded from GitHub (1.9GB) — regenerate by running Notebook 2
- train_transaction.csv and train_identity.csv excluded — download from Kaggle
- Model threshold set to 0.4 not 0.5 — optimized for fraud recall over precision
- SHAP computed on 500 row sample for speed — increase for full analysis
- All notebooks designed to run on Google Colab with Google Drive

---

## 💻 Tech Stack

| Category        | Tools                                    |
|-----------------|------------------------------------------|
| Language        | Python 3.12                              |
| ML Models       | XGBoost, Scikit-learn, imbalanced-learn  |
| Explainability  | SHAP                                     |
| API             | FastAPI, Uvicorn, Pydantic               |
| Dashboard       | Streamlit                                |
| Data Processing | Pandas, NumPy                            |
| Visualization   | Matplotlib, Seaborn                      |
| Platform        | Google Colab, Google Drive               |
| Version Control | GitHub                                   |
| Dataset         | IEEE-CIS Kaggle Competition              |
