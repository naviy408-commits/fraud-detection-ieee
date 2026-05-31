import pickle
import numpy as np
import pandas as pd

def test_model_loads():
    with open("models/best_model.pkl", "rb") as f:
        best = pickle.load(f)
    assert "model" in best
    assert "threshold" in best
    print("✅ Model loads correctly")

def test_feature_cols():
    with open("models/feature_cols.pkl", "rb") as f:
        cols = pickle.load(f)
    assert len(cols) == 225
    print(f"✅ Feature cols correct: {len(cols)}")

def test_prediction():
    with open("models/best_model.pkl", "rb") as f:
        best = pickle.load(f)
    with open("models/feature_cols.pkl", "rb") as f:
        cols = pickle.load(f)

    model     = best["model"]
    threshold = best["threshold"]

    row   = pd.DataFrame([np.zeros(len(cols))], columns=cols)
    proba = model.predict_proba(row)[0][1]

    assert 0 <= proba <= 1
    print(f"✅ Prediction works: {proba:.4f}")

if __name__ == "__main__":
    test_model_loads()
    test_feature_cols()
    test_prediction()
    print("All tests passed!")
