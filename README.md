# Fraud Detection

Detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest** with **SMOTE**, **cost-sensitive learning**, and **SHAP explanations**.

## Dataset

Download `creditcard.csv` from Kaggle and place it in this folder:
👉 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Train the model:**
```bash
python fraud_detection.py
```

**Predict on a new transaction:**
```bash
python predict.py
```

**Predict on your own CSV:**
```bash
python predict.py --file new_transactions.csv
```

## How It Works

1. Loads and scales the credit card dataset (~284K transactions, 0.17% fraud)
2. Applies **SMOTE** to balance the heavily imbalanced classes
3. Trains with **cost-sensitive learning** (`class_weight=balanced`) to penalize misclassifying fraud more
4. Evaluates using **ROC-AUC** and **AUC-PR** (better metric for imbalanced data)
5. Generates **SHAP explanations** showing which features drive fraud predictions
6. Saves best model, predictions CSV, and full model report

## Output Files

| File | Description |
|------|-------------|
| `fraud_model.pkl` | Saved best model |
| `output.csv` | Predictions on test set |
| `model_report.txt` | Full classification report + AUC scores |
| `plots/confusion_matrices.png` | Confusion matrix comparison |
| `plots/roc_curves.png` | ROC curve comparison |
| `plots/precision_recall_curves.png` | AUC-PR curve comparison |
| `plots/feature_importance.png` | Top 15 features (Random Forest) |
| `plots/shap_summary.png` | SHAP beeswarm — feature impact per prediction |
| `plots/shap_bar.png` | SHAP bar — mean feature importance |

## Project Structure

```
Fraud-Detection/
├── fraud_detection.py   ← train, evaluate, SHAP, save
├── predict.py           ← predict on new transactions
├── requirements.txt
├── README.md
├── sample_output.csv    ← preview results
├── creditcard.csv       ← download from Kaggle
├── fraud_model.pkl      ← generated after training
├── model_report.txt     ← generated after training
├── output.csv           ← generated after training
└── plots/               ← generated after training
```

## Expected Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | ~0.97 | ~0.71 |
| Random Forest | ~0.99 | ~0.87 |
