import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
import shap

# ── 1. Load Data ──────────────────────────────────────────────────────────────
DATA_PATH = "creditcard.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "creditcard.csv not found.\n"
        "Download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "Then place it in this folder."
    )

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
print(f"Fraud %: {df['Class'].mean() * 100:.4f}%")

# ── 2. Preprocess ─────────────────────────────────────────────────────────────
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df[["Amount"]])
df["Time"]   = scaler.fit_transform(df[["Time"]])

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Handle Imbalance with SMOTE ────────────────────────────────────────────
print("\nApplying SMOTE to balance training data...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"Resampled training set size: {X_train_res.shape[0]}")

# ── 4. Compute class weight for cost-sensitive learning ───────────────────────
fraud_count = int(y_train.sum())
legit_count = int((y_train == 0).sum())
scale_pos_weight = legit_count / fraud_count
print(f"\nCost-sensitive weight (legit/fraud): {scale_pos_weight:.2f}")

# ── 5. Train Models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42,
        class_weight="balanced"          # cost-sensitive
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1,
        class_weight="balanced"          # cost-sensitive
    ),
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "model":   model,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc":  average_precision_score(y_test, y_prob),
    }

    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"ROC-AUC : {results[name]['roc_auc']:.4f}")
    print(f"PR-AUC  : {results[name]['pr_auc']:.4f}")

# ── 6. Save Best Model ────────────────────────────────────────────────────────
best_name  = max(results, key=lambda n: results[n]["roc_auc"])
best_model = results[best_name]["model"]
joblib.dump(best_model, "fraud_model.pkl")
print(f"\nBest model: {best_name} (AUC={results[best_name]['roc_auc']:.4f}) saved as fraud_model.pkl")

# ── 7. Plots ──────────────────────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrices.png", dpi=150)
plt.close()

# ROC curves
plt.figure(figsize=(7, 5))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.4f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150)
plt.close()

# Precision-Recall curves
plt.figure(figsize=(7, 5))
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res["y_prob"])
    plt.plot(recall, precision, label=f"{name} (PR-AUC={res['pr_auc']:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (AUC-PR)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/precision_recall_curves.png", dpi=150)
plt.close()

# Feature importance (Random Forest)
rf       = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).nlargest(15)
plt.figure(figsize=(8, 5))
feat_imp.sort_values().plot(kind="barh", color="steelblue")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("plots/feature_importance.png", dpi=150)
plt.close()

print("\nPlots saved to plots/")

# ── 8. SHAP Explanations ──────────────────────────────────────────────────────
print("\nGenerating SHAP explanations (Random Forest)...")
rf_model   = results["Random Forest"]["model"]
explainer  = shap.TreeExplainer(rf_model)
X_sample   = X_test.iloc[:200]           # use 200 samples for speed
shap_values = explainer.shap_values(X_sample)

# SHAP summary plot
shap_fraud = shap_values[1] if isinstance(shap_values, list) else shap_values
plt.figure()
shap.summary_plot(shap_fraud, X_sample, show=False)
plt.tight_layout()
plt.savefig("plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

# SHAP bar plot
plt.figure()
shap.summary_plot(shap_fraud, X_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("plots/shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()

print("SHAP plots saved to plots/shap_summary.png and plots/shap_bar.png")

# ── 9. Export Predictions to CSV ──────────────────────────────────────────────
best_res   = results[best_name]
output_df  = X_test.copy()
output_df["ActualClass"]      = y_test.values
output_df["PredictedClass"]   = best_res["y_pred"]
output_df["FraudProbability"] = best_res["y_prob"].round(4)
output_df["Correct"]          = output_df["ActualClass"] == output_df["PredictedClass"]
output_df.insert(0, "TransactionID", range(1, len(output_df) + 1))
output_df[["TransactionID", "ActualClass", "PredictedClass", "FraudProbability", "Correct"]].to_csv("output.csv", index=False)
print("Predictions saved to output.csv")

# ── 10. Save Model Report ─────────────────────────────────────────────────────
with open("model_report.txt", "w") as f:
    f.write("FRAUD DETECTION MODEL REPORT\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset Shape      : {df.shape}\n")
    f.write(f"Fraud Transactions : {int(y.sum())} ({y.mean()*100:.4f}%)\n")
    f.write(f"Legit Transactions : {int((y==0).sum())}\n")
    f.write(f"Train Size         : {len(X_train_res)} (after SMOTE)\n")
    f.write(f"Test Size          : {len(X_test)}\n")
    f.write(f"Cost-Sensitive     : class_weight=balanced\n\n")
    f.write("=" * 50 + "\n")
    for name, res in results.items():
        f.write(f"\nModel: {name}\n")
        f.write("-" * 40 + "\n")
        f.write(classification_report(y_test, res["y_pred"], target_names=["Legit", "Fraud"]))
        f.write(f"ROC-AUC : {res['roc_auc']:.4f}\n")
        f.write(f"PR-AUC  : {res['pr_auc']:.4f}\n")
    f.write("\n" + "=" * 50 + "\n")
    f.write(f"Best Model : {best_name}\n")
    f.write(f"Best AUC   : {results[best_name]['roc_auc']:.4f}\n")
    f.write(f"Saved As   : fraud_model.pkl\n")
print("Model report saved to model_report.txt")
print("\nDone!")
