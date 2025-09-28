# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer  # For automated preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

# %%
# ----------------------------
# 1. Load model and data
# ----------------------------
root = os.path.dirname(os.path.dirname(__file__))  # adapte si besoin
models_dir = os.path.join(root, "models")

# load pipeline (déjà complet : vectorizer + modèle)
model_path = os.path.join(models_dir, "xgb_default_model.pkl")  # ou .joblib
model = load(model_path)

data_path = os.path.join(root, "data", "dataproject2025.csv")
df = pd.read_csv(data_path)

# Define columns to exclude from features
exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']

# Prepare feature matrix (all columns except excluded ones)
feature_cols = [col for col in df.columns if col not in exclude_cols]
X_raw = df[feature_cols].copy()
y = df['Predicted probabilities']  # This is our DP (Default Probability)

# Check for missing values
missing_counts = X_raw.isnull().sum()

# Clean data for TableVectorizer preprocessing
# Remove rows with missing values completely to ensure clean data
df_clean = df.dropna()  # Remove all rows with any missing values
X_raw_clean = df_clean[feature_cols].copy()
y_clean = df_clean['Predicted probabilities']

# Initialize TableVectorizer with default parameters
# TableVectorizer automatically detects and handles different column types
vectorizer = TableVectorizer()

# Split data for training and testing surrogate models based on year
# Use data from 2018, 2019, and 2020 as the test set
test_years = [2018, 2019, 2020]
test_indices = X_raw_clean[X_raw_clean['issue_d'].isin(test_years)].index
train_indices = X_raw_clean[~X_raw_clean['issue_d'].isin(test_years)].index

X_train = X_raw_clean.loc[train_indices]
y_train = y_clean.loc[train_indices]
X_test = X_raw_clean.loc[test_indices]
y_test = y_clean.loc[test_indices]

# Fit the TableVectorizer on training data
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)



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
