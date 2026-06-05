# Fraud Detection

Detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest** with **SMOTE** to handle severe class imbalance (~0.17% fraud rate).

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

**Predict on a new transaction (sample built-in):**
```bash
python predict.py
```

**Predict on your own CSV file:**
```bash
python predict.py --file new_transactions.csv
```

## How It Works

1. Loads and scales the credit card dataset
2. Applies SMOTE to balance the heavily imbalanced classes
3. Trains Logistic Regression and Random Forest
4. Compares both models using ROC-AUC score
5. Saves the best model as `fraud_model.pkl`
6. Exports predictions and a full model report

## Output Files

| File | Description |
|------|-------------|
| `fraud_model.pkl` | Saved best model |
| `output.csv` | Predictions on test set |
| `model_report.txt` | Full classification report + AUC scores |
| `predict_output.csv` | Results from predict.py |
| `plots/confusion_matrices.png` | Confusion matrix comparison |
| `plots/roc_curves.png` | ROC curve comparison |
| `plots/precision_recall_curves.png` | Precision-Recall curves |
| `plots/feature_importance.png` | Top 15 features (Random Forest) |

## Project Structure

```
Fraud-Detection/
├── fraud_detection.py   ← train, evaluate, save model
├── predict.py           ← predict on new transactions
├── requirements.txt
├── README.md
├── sample_output.csv    ← preview of expected results
├── creditcard.csv       ← download from Kaggle (not in repo)
├── fraud_model.pkl      ← generated after training
├── model_report.txt     ← generated after training
├── output.csv           ← generated after training
└── plots/               ← generated after training
```

## Results (Expected)

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | ~0.97 |
| Random Forest | ~0.99 |
