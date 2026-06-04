# Fraud Detection

Detects fraudulent credit card transactions using Logistic Regression and Random Forest with SMOTE to handle class imbalance.

## Dataset

Download `creditcard.csv` from Kaggle and place it in this folder:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python fraud_detection.py
```

## Output

- Classification report + ROC-AUC score for both models
- Best model saved as `fraud_model.pkl`
- Plots saved in `plots/`:
  - `confusion_matrices.png`
  - `roc_curves.png`
  - `precision_recall_curves.png`
  - `feature_importance.png`

## Project Structure

```
Fraud-Detection/
├── fraud_detection.py   ← main script
├── requirements.txt
├── README.md
├── creditcard.csv       ← download from Kaggle (not in repo)
├── fraud_model.pkl      ← generated after running
└── plots/               ← generated after running
```
