"""
Predict whether a transaction is fraudulent using the saved model.
Usage:
    python predict.py                        # runs with a built-in sample transaction
    python predict.py --file new_data.csv    # predict on a CSV file
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "fraud_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "fraud_model.pkl not found.\n"
        "Run fraud_detection.py first to train and save the model."
    )

model = joblib.load(MODEL_PATH)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    df = df.copy()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])
    df["Time"]   = scaler.fit_transform(df[["Time"]])
    return df[FEATURE_COLS]

def predict(df: pd.DataFrame) -> pd.DataFrame:
    X = preprocess(df)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["PredictedClass"]   = preds
    df["FraudProbability"] = probs.round(4)
    df["Verdict"]          = df["PredictedClass"].map({0: "LEGIT", 1: "FRAUD"})
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Transaction Predictor")
    parser.add_argument("--file", type=str, help="Path to CSV file with transactions")
    args = parser.parse_args()

    if args.file:
        if not os.path.exists(args.file):
            raise FileNotFoundError(f"File not found: {args.file}")
        df = pd.read_csv(args.file)
        print(f"Loaded {len(df)} transactions from {args.file}")
    else:
        # Built-in sample transaction (typical legit transaction values)
        sample = {f"V{i}": [round(np.random.randn(), 6)] for i in range(1, 29)}
        sample["Amount"] = [149.62]
        sample["Time"]   = [406.0]
        df = pd.DataFrame(sample)
        print("No file provided — running on built-in sample transaction.\n")

    results = predict(df)

    print("\n── Prediction Results ──")
    print(results[["Amount", "PredictedClass", "FraudProbability", "Verdict"]].to_string(index=False))

    out_path = "predict_output.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
