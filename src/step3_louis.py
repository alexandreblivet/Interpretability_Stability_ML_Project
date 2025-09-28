# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

# %%
# ----------------------------
# 1. Load model and data
# ----------------------------
root = os.path.dirname(os.path.dirname(__file__))  # adapte si besoin
models_dir = os.path.join(root, "models")
data_path = os.path.join(root, "data", "dataproject2025.csv")

# load pipeline (déjà complet : vectorizer + modèle)
model_path = os.path.join(models_dir, "xgb_default_model.pkl")  # ou .joblib
model = load(model_path)

# load data
df = pd.read_csv(data_path)

# Assurer que la date est bien au format datetime
df["issue_d"] = pd.to_datetime(df["issue_d"])

# Trier par date
df = df.sort_values("issue_d")

# Définir le seuil de split (par ex. 80% train / 20% test)
split_date = df["issue_d"].quantile(0.8)

# Split temporel
df_train = df[df["issue_d"] <= split_date]
df_test  = df[df["issue_d"] > split_date]

X_train = df_train.drop(columns=["target"])
y_train = df_train["target"]

X_test = df_test.drop(columns=["target"])
y_test = df_test["target"]

# Drop 'issue_d' because model cannot use datetime
X_train = X_train.drop(columns=["issue_d"])
X_test  = X_test.drop(columns=["issue_d"])

# %%
# ----------------------------
# 2. Overall forecasting performance
# ----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC-ROC": roc_auc_score(y_test, y_proba),
    "F1-score": f1_score(y_test, y_pred, pos_label=1),
    "Precision": precision_score(y_test, y_pred, pos_label=1),
    "Recall": recall_score(y_test, y_pred, pos_label=1),
    "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist()
}

print("Overall Performance Metrics:")
for k, v in metrics.items():
    print(f"{k}: {v}")


# %%
# ----------------------------
# 3. Structural stability
# ----------------------------

# (a) Temporal stability (if 'issue_d' or similar date column exists)
if "issue_d" in X.columns:
    df_stab = X.copy()
    df_stab["target"] = y
    df_stab["pred"] = model.predict(X)

    yearly_auc = (
        df_stab.groupby(df_stab["issue_d"])  # replace with pd.to_datetime if needed
        .apply(lambda d: roc_auc_score(d["target"], model.predict_proba(d.drop(columns=["target", "pred"]))[:,1]))
    )

    plt.figure(figsize=(8,4))
    yearly_auc.plot(marker="o")
    plt.title("Temporal Stability of AUC by issue_d")
    plt.ylabel("AUC-ROC")
    plt.xlabel("Year")
    plt.grid(True)
    plt.show()

# (b) Subgroup stability: example by income quantiles
df["pred"] = model.predict(X)
df["proba"] = model.predict_proba(X)[:,1]

df["income_group"] = pd.qcut(df["annual_inc"], q=4, labels=["Low", "Mid-Low", "Mid-High", "High"])

subgroup_auc = df.groupby("income_group").apply(
    lambda d: roc_auc_score(d["target"], d["proba"])
)

print("\nAUC by income quartile:")
print(subgroup_auc)

# (c) Feature stability: retrain on bootstrap samples and compare top features
# (optional, more advanced – requires SHAP or feature_importances_)

# ----------------------------
# 4. Takeaways
# ----------------------------
print("\nInterpretation:")
print("- Overall metrics show how well the model forecasts default.")
print("- Temporal stability checks if performance drifts across years.")
print("- Subgroup stability checks consistency across borrower profiles (e.g. income).")

# %%
