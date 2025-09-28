# %%
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from skrub import TableVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
import re
import warnings
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score, classification_report, confusion_matrix
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('../data/dataproject2025.csv')

# %%
# Prepare data using TableVectorizer for automated preprocessing
exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
feature_cols = [col for col in df.columns if col not in exclude_cols]
X_raw = df[feature_cols].copy()
y = df['target']

# Clean data for TableVectorizer preprocessing
df_clean = df.dropna()
X_raw_clean = df_clean[feature_cols].copy()
y_clean = df_clean['target']

# Initialize TableVectorizer
vectorizer = TableVectorizer()

# %%
# Split data by years for train/test
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
# Clean feature names for XGBoost
raw_feature_names = vectorizer.get_feature_names_out()

def clean_feature_names(names):
    cleaned = []
    for name in names:
        clean_name = re.sub(r'[<>\[\](){}]', '_', str(name))
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')
        if not clean_name or clean_name.isdigit():
            clean_name = f'feature_{len(cleaned)}'
        cleaned.append(clean_name)
    return cleaned

feature_names_clean = clean_feature_names(raw_feature_names)

# %%
# Configure XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=0.5,
    gamma=0.1,
    min_child_weight=3,
    random_state=42,
    verbosity=0,
    tree_method='hist',
    objective='binary:logistic',
    eval_metric='logloss'
)

# %%
classification_threshold = 0.5  # this can be adjusted

# Convert target to binary classification
y_train_binary = (y_train >= classification_threshold).astype(int)
y_test_binary = (y_test >= classification_threshold).astype(int)

# Convert to numpy arrays and fit the model
X_train_np = X_train_vectorized.values if hasattr(X_train_vectorized, 'values') else X_train_vectorized
X_test_np = X_test_vectorized.values if hasattr(X_test_vectorized, 'values') else X_test_vectorized
y_train_np = y_train_binary.values if hasattr(y_train_binary, 'values') else y_train_binary
y_test_np = y_test_binary.values if hasattr(y_test_binary, 'values') else y_test_binary

# Fit the model
xgb_model.fit(X_train_np, y_train_np)

# %%
# Make predictions
y_pred_binary = xgb_model.predict(X_test_np)
y_pred_proba = xgb_model.predict_proba(X_test_np)[:, 1]

# %%
# Overview of the results : Evaluate the model
f1 = f1_score(y_test_np, y_pred_binary)
accuracy = accuracy_score(y_test_np, y_pred_binary)
precision = precision_score(y_test_np, y_pred_binary)
recall = recall_score(y_test_np, y_pred_binary)
auc_roc = roc_auc_score(y_test_np, y_pred_binary)

# Print results
print("Model Performance on Test Set")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")

# Classification report
print(classification_report(y_test_np, y_pred_binary, target_names=['No Default', 'Default']))

# %%
# Optimize F1-score by testing different thresholds
from sklearn.metrics import precision_recall_curve

# Get precision, recall, and thresholds from precision-recall curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test_np, y_pred_proba)

# Calculate F1-score for each threshold
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)

# Find the optimal threshold that maximizes F1-score
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]
optimal_precision = precision_curve[optimal_idx]
optimal_recall = recall_curve[optimal_idx]

print(f"\nThreshold Optimization Results:")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Optimal F1-score: {optimal_f1:.4f}")
print(f"Optimal Precision: {optimal_precision:.4f}")
print(f"Optimal Recall: {optimal_recall:.4f}")

# Test a range of thresholds for comparison
test_thresholds = np.arange(0.1, 0.9, 0.05)
threshold_results = []

for thresh in test_thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    f1_thresh = f1_score(y_test_np, y_pred_thresh)
    acc_thresh = accuracy_score(y_test_np, y_pred_thresh)
    prec_thresh = precision_score(y_test_np, y_pred_thresh)
    rec_thresh = recall_score(y_test_np, y_pred_thresh)

    threshold_results.append({
        'threshold': thresh,
        'f1_score': f1_thresh,
        'accuracy': acc_thresh,
        'precision': prec_thresh,
        'recall': rec_thresh
    })

# Convert to DataFrame for easier analysis
threshold_df = pd.DataFrame(threshold_results)

# Find the best threshold from our test range
best_idx = threshold_df['f1_score'].idxmax()
best_threshold = threshold_df.loc[best_idx, 'threshold']
best_f1 = threshold_df.loc[best_idx, 'f1_score']

print(f"\nBest threshold from test range: {best_threshold:.3f}")
print(f"Best F1-score from test range: {best_f1:.4f}")

# Show top 5 thresholds
print(f"\nTop 5 thresholds by F1-score:")
top_5 = threshold_df.nlargest(5, 'f1_score')
for i, row in top_5.iterrows():
    print(f"Threshold: {row['threshold']:.3f} | F1: {row['f1_score']:.4f} | "
          f"Precision: {row['precision']:.4f} | Recall: {row['recall']:.4f}")

# Visualize threshold optimization
plt.figure(figsize=(12, 4))

# Plot 1: F1-score vs Threshold
plt.subplot(1, 2, 1)
plt.plot(threshold_df['threshold'], threshold_df['f1_score'], 'b-', linewidth=2, label='F1-score')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.3f}')
plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best from test: {best_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('F1-score')
plt.title('F1-score vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Precision and Recall vs Threshold
plt.subplot(1, 2, 2)
plt.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', linewidth=2, label='Precision')
plt.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', linewidth=2, label='Recall')
plt.axvline(x=optimal_threshold, color='k', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.3f}')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Precision and Recall vs Classification Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Apply the optimal threshold to get final predictions
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)

# Calculate final metrics with optimal threshold
final_f1 = f1_score(y_test_np, y_pred_optimal)
final_accuracy = accuracy_score(y_test_np, y_pred_optimal)
final_precision = precision_score(y_test_np, y_pred_optimal)
final_recall = recall_score(y_test_np, y_pred_optimal)

print(f"\nFinal Model Performance with Optimal Threshold ({optimal_threshold:.3f}):")
print(f"Accuracy : {final_accuracy:.4f}")
print(f"F1-score : {final_f1:.4f}")
print(f"Precision: {final_precision:.4f}")
print(f"Recall   : {final_recall:.4f}")

# %%
# Classification report
print(classification_report(y_test_np, y_pred_binary, target_names=['No Default', 'Default']))

# Confusion matrix
cm = confusion_matrix(y_test_np, y_pred_binary)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test_np, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_roc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% Saving model
from sklearn.pipeline import Pipeline
from joblib import dump

# Build a pipeline: preprocessing + model
final_pipeline = Pipeline([
    ("vectorizer", vectorizer),
    ("model", xgb_model)
])

# Save the full pipeline as a .pkl file
dump(final_pipeline, '../models/xgboost_black_box_pipeline_vf.pkl')

print("Full pipeline (vectorizer + XGBoost) saved as '../models/xgboost_black_box_pipeline_vf.pkl'")



# %%
# Model Stability Analysis

# ----------------------------
# 10.1 Temporal Stability Analysis
# ----------------------------

if 'issue_d' in df_clean.columns:
    print("\n=== TEMPORAL STABILITY ANALYSIS ===")

    # Get all years in the dataset
    if df_clean['issue_d'].dtype in ['int64', 'int32', 'float64']:
        all_years = sorted(df_clean['issue_d'].unique())
    else:
        all_years = sorted(df_clean['issue_d'].dt.year.unique())

    print(f"Analyzing stability across years: {all_years}")

    # Calculate metrics per year
    yearly_metrics = []

    for year in all_years:
        # Get data for this year
        if df_clean['issue_d'].dtype in ['int64', 'int32', 'float64']:
            year_mask = df_clean['issue_d'] == year
        else:
            year_mask = df_clean['issue_d'].dt.year == year

        if year_mask.sum() < 10:  # Skip years with too few samples
            continue

        X_year = X_raw_clean[year_mask]
        y_year = y_clean[year_mask]

        # Vectorize the data for this year (vectorizer is already fitted)
        X_year_vectorized = vectorizer.transform(X_year)
        X_year_np = X_year_vectorized.values if hasattr(X_year_vectorized, 'values') else X_year_vectorized

        # Convert to binary classification using the optimal threshold
        y_year_binary = (y_year >= optimal_threshold).astype(int)

        # Make predictions for this year
        y_pred_year = xgb_model.predict(X_year_np)
        y_pred_proba_year = xgb_model.predict_proba(X_year_np)[:, 1]

        # Calculate metrics
        accuracy_year = accuracy_score(y_year_binary, y_pred_year)
        auc_year = roc_auc_score(y_year_binary, y_pred_proba_year)

        # Precision, Recall, F1
        precision_year = precision_score(y_year_binary, y_pred_year, zero_division=0)
        recall_year = recall_score(y_year_binary, y_pred_year, zero_division=0)
        f1_year = f1_score(y_year_binary, y_pred_year, zero_division=0)

        yearly_metrics.append({
            'year': year,
            'n_samples': year_mask.sum(),
            'accuracy': accuracy_year,
            'auc': auc_year,
            'precision': precision_year,
            'recall': recall_year,
            'f1': f1_year,
            'default_rate': y_year_binary.mean()
        })

    # Convert to DataFrame
    yearly_df = pd.DataFrame(yearly_metrics)

    if len(yearly_df) > 0:
        print(f"\nYearly Performance Summary:")
        print(yearly_df.round(4))

        # Plot temporal stability
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # AUC over time
        axes[0, 0].plot(yearly_df['year'], yearly_df['auc'], marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('AUC-ROC Over Time')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('AUC-ROC')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=yearly_df['auc'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {yearly_df["auc"].mean():.3f}')
        axes[0, 0].legend()

        # Precision over time
        axes[0, 1].plot(yearly_df['year'], yearly_df['precision'], marker='o', linewidth=2, markersize=6, color='green')
        axes[0, 1].set_title('Precision Over Time')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=yearly_df['precision'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {yearly_df["precision"].mean():.3f}')
        axes[0, 1].legend()

        # F1 over time
        axes[1, 0].plot(yearly_df['year'], yearly_df['f1'], marker='o', linewidth=2, markersize=6, color='orange')
        axes[1, 0].set_title('F1-Score Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=yearly_df['f1'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {yearly_df["f1"].mean():.3f}')
        axes[1, 0].legend()

        # Default rate over time
        axes[1, 1].plot(yearly_df['year'], yearly_df['default_rate'], marker='o', linewidth=2, markersize=6, color='purple')
        axes[1, 1].set_title('Default Rate Over Time')
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Default Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=yearly_df['default_rate'].mean(), color='red', linestyle='--', alpha=0.7, label=f'Mean: {yearly_df["default_rate"].mean():.3f}')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

        # Stability statistics
        print(f"\n=== TEMPORAL STABILITY STATISTICS ===")
        print(f"AUC - Mean: {yearly_df['auc'].mean():.4f}, Std: {yearly_df['auc'].std():.4f}, Range: {yearly_df['auc'].max() - yearly_df['auc'].min():.4f}")
        print(f"Precision - Mean: {yearly_df['precision'].mean():.4f}, Std: {yearly_df['precision'].std():.4f}, Range: {yearly_df['precision'].max() - yearly_df['precision'].min():.4f}")
        print(f"F1 - Mean: {yearly_df['f1'].mean():.4f}, Std: {yearly_df['f1'].std():.4f}, Range: {yearly_df['f1'].max() - yearly_df['f1'].min():.4f}")

        # Identify years with significant performance drops
        auc_mean = yearly_df['auc'].mean()
        auc_std = yearly_df['auc'].std()
        threshold = 2 * auc_std  # 2 standard deviations

        unstable_years = yearly_df[abs(yearly_df['auc'] - auc_mean) > threshold]
        if len(unstable_years) > 0:
            print(f"\nYears with significant AUC deviation (>2σ):")
            print(unstable_years[['year', 'auc']].round(4))
        else:
            print(f"\nNo years with significant AUC deviation detected.")

# %%
# 10.2 Subgroup Stability Analysis
# ----------------------------

print(f"\n=== SUBGROUP STABILITY ANALYSIS ===")

# Define relevant subgroups based on your dataset features
subgroups = {}

# Grade-based subgroups
if 'grade' in X_raw_clean.columns:
    subgroups['grade'] = X_raw_clean['grade'].unique()

# Home ownership subgroups
if 'home_ownership' in X_raw_clean.columns:
    subgroups['home_ownership'] = X_raw_clean['home_ownership'].unique()

# Purpose subgroups (top purposes only)
if 'purpose' in X_raw_clean.columns:
    top_purposes = X_raw_clean['purpose'].value_counts().head(5).index
    subgroups['purpose'] = top_purposes

# Income-based subgroups (if annual_inc exists)
if 'annual_inc' in X_raw_clean.columns:
    income_quartiles = pd.qcut(X_raw_clean['annual_inc'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    subgroups['income_quartile'] = income_quartiles.unique()

# Loan amount subgroups (if funded_amnt exists)
if 'funded_amnt' in X_raw_clean.columns:
    loan_quartiles = pd.qcut(X_raw_clean['funded_amnt'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')
    subgroups['loan_amount_quartile'] = loan_quartiles.unique()

# Analyze each subgroup
subgroup_results = []

for subgroup_name, subgroup_values in subgroups.items():
    print(f"\n--- Analyzing {subgroup_name} subgroups ---")

    if subgroup_name in ['income_quartile', 'loan_amount_quartile']:
        # For quartile-based subgroups
        if subgroup_name == 'income_quartile':
            X_temp = X_raw_clean.copy()
            X_temp['quartile'] = pd.qcut(X_raw_clean['annual_inc'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        else:
            X_temp = X_raw_clean.copy()
            X_temp['quartile'] = pd.qcut(X_raw_clean['funded_amnt'], q=4, labels=['Low', 'Medium', 'High', 'Very High'], duplicates='drop')

        for value in subgroup_values:
            if pd.isna(value):
                continue

            mask = X_temp['quartile'] == value
            if mask.sum() < 50:  # Skip subgroups with too few samples
                continue

            X_subgroup = X_raw_clean[mask]
            y_subgroup = y_clean[mask]

            # Vectorize the subgroup data (vectorizer is already fitted)
            X_subgroup_vectorized = vectorizer.transform(X_subgroup)
            X_subgroup_np = X_subgroup_vectorized.values if hasattr(X_subgroup_vectorized, 'values') else X_subgroup_vectorized

            # Convert to binary classification using the optimal threshold
            y_subgroup_binary = (y_subgroup >= optimal_threshold).astype(int)

            # Make predictions
            y_pred_sub = xgb_model.predict(X_subgroup_np)
            y_pred_proba_sub = xgb_model.predict_proba(X_subgroup_np)[:, 1]

            # Calculate metrics
            accuracy_sub = accuracy_score(y_subgroup_binary, y_pred_sub)
            auc_sub = roc_auc_score(y_subgroup_binary, y_pred_proba_sub)
            precision_sub = precision_score(y_subgroup_binary, y_pred_sub, zero_division=0)
            recall_sub = recall_score(y_subgroup_binary, y_pred_sub, zero_division=0)
            f1_sub = f1_score(y_subgroup_binary, y_pred_sub, zero_division=0)

            subgroup_results.append({
                'subgroup': subgroup_name,
                'value': str(value),
                'n_samples': mask.sum(),
                'accuracy': accuracy_sub,
                'auc': auc_sub,
                'precision': precision_sub,
                'recall': recall_sub,
                'f1': f1_sub,
                'default_rate': y_subgroup_binary.mean()
            })

    else:
        # For categorical subgroups
        for value in subgroup_values:
            if pd.isna(value):
                continue

            mask = X_raw_clean[subgroup_name] == value
            if mask.sum() < 50:  # Skip subgroups with too few samples
                continue

            X_subgroup = X_raw_clean[mask]
            y_subgroup = y_clean[mask]

            # Vectorize the subgroup data (vectorizer is already fitted)
            X_subgroup_vectorized = vectorizer.transform(X_subgroup)
            X_subgroup_np = X_subgroup_vectorized.values if hasattr(X_subgroup_vectorized, 'values') else X_subgroup_vectorized

            # Convert to binary classification using the optimal threshold
            y_subgroup_binary = (y_subgroup >= optimal_threshold).astype(int)

            # Make predictions
            y_pred_sub = xgb_model.predict(X_subgroup_np)
            y_pred_proba_sub = xgb_model.predict_proba(X_subgroup_np)[:, 1]

            # Calculate metrics
            accuracy_sub = accuracy_score(y_subgroup_binary, y_pred_sub)
            auc_sub = roc_auc_score(y_subgroup_binary, y_pred_proba_sub)
            precision_sub = precision_score(y_subgroup_binary, y_pred_sub, zero_division=0)
            recall_sub = recall_score(y_subgroup_binary, y_pred_sub, zero_division=0)
            f1_sub = f1_score(y_subgroup_binary, y_pred_sub, zero_division=0)

            subgroup_results.append({
                'subgroup': subgroup_name,
                'value': str(value),
                'n_samples': mask.sum(),
                'accuracy': accuracy_sub,
                'auc': auc_sub,
                'precision': precision_sub,
                'recall': recall_sub,
                'f1': f1_sub,
                'default_rate': y_subgroup_binary.mean()
            })

# Convert to DataFrame and analyze
if subgroup_results:
    subgroup_df = pd.DataFrame(subgroup_results)

    print(f"\nSubgroup Performance Summary:")
    print(subgroup_df.round(4))

    # Plot subgroup stability
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # AUC by subgroup
    for subgroup in subgroup_df['subgroup'].unique():
        sub_data = subgroup_df[subgroup_df['subgroup'] == subgroup]
        axes[0, 0].scatter(sub_data['value'], sub_data['auc'], label=subgroup, alpha=0.7, s=60)
    axes[0, 0].set_title('AUC-ROC by Subgroup')
    axes[0, 0].set_xlabel('Subgroup Value')
    axes[0, 0].set_ylabel('AUC-ROC')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Precision by subgroup
    for subgroup in subgroup_df['subgroup'].unique():
        sub_data = subgroup_df[subgroup_df['subgroup'] == subgroup]
        axes[0, 1].scatter(sub_data['value'], sub_data['precision'], label=subgroup, alpha=0.7, s=60)
    axes[0, 1].set_title('Precision by Subgroup')
    axes[0, 1].set_xlabel('Subgroup Value')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # F1 by subgroup
    for subgroup in subgroup_df['subgroup'].unique():
        sub_data = subgroup_df[subgroup_df['subgroup'] == subgroup]
        axes[1, 0].scatter(sub_data['value'], sub_data['f1'], label=subgroup, alpha=0.7, s=60)
    axes[1, 0].set_title('F1-Score by Subgroup')
    axes[1, 0].set_xlabel('Subgroup Value')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Default rate by subgroup
    for subgroup in subgroup_df['subgroup'].unique():
        sub_data = subgroup_df[subgroup_df['subgroup'] == subgroup]
        axes[1, 1].scatter(sub_data['value'], sub_data['default_rate'], label=subgroup, alpha=0.7, s=60)
    axes[1, 1].set_title('Default Rate by Subgroup')
    axes[1, 1].set_xlabel('Subgroup Value')
    axes[1, 1].set_ylabel('Default Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # Subgroup stability statistics
    print(f"\n=== SUBGROUP STABILITY STATISTICS ===")
    for subgroup in subgroup_df['subgroup'].unique():
        sub_data = subgroup_df[subgroup_df['subgroup'] == subgroup]
        print(f"\n{subgroup.upper()}:")
        print(f"  AUC - Mean: {sub_data['auc'].mean():.4f}, Std: {sub_data['auc'].std():.4f}, Range: {sub_data['auc'].max() - sub_data['auc'].min():.4f}")
        print(f"  Precision - Mean: {sub_data['precision'].mean():.4f}, Std: {sub_data['precision'].std():.4f}, Range: {sub_data['precision'].max() - sub_data['precision'].min():.4f}")
        print(f"  F1 - Mean: {sub_data['f1'].mean():.4f}, Std: {sub_data['f1'].std():.4f}, Range: {sub_data['f1'].max() - sub_data['f1'].min():.4f}")

        # Identify subgroups with significant performance differences
        auc_mean = sub_data['auc'].mean()
        auc_std = sub_data['auc'].std()
        threshold = 2 * auc_std

        unstable_subgroups = sub_data[abs(sub_data['auc'] - auc_mean) > threshold]
        if len(unstable_subgroups) > 0:
            print(f"  Subgroups with significant AUC deviation (>2σ):")
            for _, row in unstable_subgroups.iterrows():
                print(f"    {row['value']}: AUC = {row['auc']:.4f}")

# %%
# ----------------------------
# 10.3 Overall Stability Summary
# ----------------------------

print(f"\n{'='*60}")
print(f"OVERALL STABILITY SUMMARY")
print(f"{'='*60}")

# Calculate overall stability metrics
if 'issue_d' in df_clean.columns and len(yearly_df) > 0:
    temporal_auc_std = yearly_df['auc'].std()
    temporal_precision_std = yearly_df['precision'].std()

    print(f"TEMPORAL STABILITY:")
    print(f"  AUC Standard Deviation: {temporal_auc_std:.4f}")
    print(f"  Precision Standard Deviation: {temporal_precision_std:.4f}")

    if temporal_auc_std < 0.05:
        print(f"  Model shows GOOD temporal stability (AUC std < 0.05)")
    elif temporal_auc_std < 0.1:
        print(f"  Model shows MODERATE temporal stability (AUC std < 0.1)")
    else:
        print(f"  Model shows POOR temporal stability (AUC std >= 0.1)")

if subgroup_results:
    overall_auc_std = subgroup_df['auc'].std()
    overall_precision_std = subgroup_df['precision'].std()

    print(f"\nSUBGROUP STABILITY:")
    print(f"  AUC Standard Deviation: {overall_auc_std:.4f}")
    print(f"  Precision Standard Deviation: {overall_precision_std:.4f}")

    if overall_auc_std < 0.05:
        print(f"  Model shows GOOD subgroup stability (AUC std < 0.05)")
    elif overall_auc_std < 0.1:
        print(f"  Model shows MODERATE subgroup stability (AUC std < 0.1)")
    else:
        print(f"  Model shows POOR subgroup stability (AUC std >= 0.1)")

print(f"\n{'='*60}")
print(f"STABILITY ANALYSIS COMPLETED")
print(f"{'='*60}")

# %%
