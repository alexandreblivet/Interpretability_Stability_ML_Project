# %%
# STEP 2 : Estimate your own black-box machine learning model to forecast default.
# This script builds an XGBoost classifier with categorical encoding, hyperparameter search,
# evaluation metrics, and model persistence. It is organized as interactive cells.

# %% Imports
from __future__ import annotations
import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from joblib import dump

from xgboost import XGBClassifier

# %% Utility functions

def _infer_feature_types(df: pd.DataFrame, target_col: str) -> Tuple[list[str], list[str]]:
    """Infer categorical and numerical feature columns from a dataframe.

    - Categorical: dtype 'object' or 'category'
    - Numerical: all remaining columns except target
    """
    candidate_cols = [c for c in df.columns if c != target_col]
    categorical_cols = [c for c in candidate_cols if str(df[c].dtype) in ("object", "category")]
    numerical_cols = [c for c in candidate_cols if c not in categorical_cols]
    return categorical_cols, numerical_cols


def _drop_non_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop common non-informative columns if present and columns with a single unique value.

    This is a conservative cleanup; adjust as needed for your dataset.
    """
    to_drop = [
        col for col in [
            "id", "ID", "Id", "index", "Index", "Unnamed: 0",
        ] if col in df.columns
    ]
    df = df.drop(columns=to_drop, errors="ignore")

    # Drop constant columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df = df.drop(columns=constant_cols, errors="ignore")
    return df


def _sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infs with NaN and standardize string spacing for object/category columns."""
    df = df.replace([np.inf, -np.inf], np.nan)
    # Trim whitespace in string columns to avoid duplicate categories due to spacing
    obj_cols = [c for c in df.columns if str(df[c].dtype) in ("object", "category")]
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()
    return df


def _prepare_binary_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Coerce the target to a clean binary 0/1 series and drop rows with missing target.

    Rules:
      - Drop rows where target is NA
      - If target is object/category: lowercase/strip and map common values; otherwise factorize
      - If numeric with two unique values: map {min->0, max->1}
      - If already 0/1 ints: keep
    """
    # Drop missing targets first
    df = df.copy()
    mask_notna = df[target_col].notna()
    if mask_notna.sum() < len(df):
        df = df.loc[mask_notna].copy()

    t = df[target_col]

    # Object/category handling
    if str(t.dtype) in ("object", "category", "string"):
        t_norm = t.astype("string").str.strip().str.lower()
        mapping = {
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "default": 1,
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0, "non-default": 0,
        }
        mapped = t_norm.map(mapping)
        if mapped.isna().any():
            # Fallback: factorize to 0/1 if exactly two classes
            codes, uniques = pd.factorize(t_norm, sort=True)
            if len(uniques) != 2:
                raise ValueError(f"Target must be binary; found classes: {list(uniques)}")
            y = pd.Series(codes, index=df.index)
        else:
            y = mapped
        y = y.astype(int)
        return df, y

    # Numeric handling
    # Ensure finite
    t = pd.to_numeric(t, errors="coerce")
    if t.isna().any():
        # Missing after coercion -> cannot be used; drop those rows
        df = df.loc[t.notna()].copy()
        t = t.loc[df.index]

    unique_vals = sorted(pd.unique(t))
    if set(unique_vals).issubset({0, 1}):
        y = t.astype(int)
        return df, y
    if len(unique_vals) == 2:
        # Map min->0, max->1
        y = t.map({min(unique_vals): 0, max(unique_vals): 1}).astype(int)
        return df, y

    raise ValueError(f"Numeric target is not binary; unique values: {unique_vals}")


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder with correct argument name across sklearn versions.

    sklearn >=1.2 uses 'sparse_output'; older uses 'sparse'.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Fallback for older versions
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _print_feature_summary(df: pd.DataFrame, target_col: str) -> None:
    """Print column-wise summary: dtype, #unique, NA%, and sample unique values."""
    print("=== Column summary (before preprocessing) ===")
    for col in df.columns:
        dtype = df[col].dtype
        na_ratio = float(df[col].isna().mean())
        nunique = int(df[col].nunique(dropna=True))
        sample_vals = df[col].dropna().unique()[:5]
        print(f"- {col}: dtype={dtype}, unique={nunique}, na%={na_ratio:.3f}, sample={sample_vals}")
    if target_col in df.columns:
        print("Target value counts (non-null):")
        print(df[target_col].dropna().value_counts().head(10))


def _split_and_cast_feature_types(
    df: pd.DataFrame,
    target_col: str,
    numeric_parse_threshold: float = 0.98,
) -> Tuple[list[str], list[str], pd.DataFrame]:
    """Split features into categorical/numerical and cast numeric-like object columns.

    For object/category columns, attempt numeric coercion; if >= threshold values parse to
    numbers and there are at least 2 distinct numeric values, cast to float and mark as numerical.
    Otherwise, keep as categorical.
    """
    df = df.copy()
    candidate_cols = [c for c in df.columns if c != target_col]
    categorical_cols: list[str] = []
    numerical_cols: list[str] = []

    for col in candidate_cols:
        dtype = df[col].dtype
        if str(dtype) in ("object", "category", "string"):
            parsed = pd.to_numeric(df[col], errors="coerce")
            ratio = float(parsed.notna().mean())
            if ratio >= numeric_parse_threshold and parsed.nunique(dropna=True) >= 2:
                df[col] = parsed.astype(float)
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    print("Numerical columns:", numerical_cols)
    print("Categorical columns:", categorical_cols)
    return categorical_cols, numerical_cols, df


# %% Core function

def train_xgboost_model(data_path: str) -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Train an XGBoost classifier for default prediction and return model and metrics.

    Steps:
      - Load dataset (expects a 'target' column)
      - Clean and prepare features (OHE for categoricals)
      - Train/test split (70/30)
      - RandomizedSearchCV over key hyperparameters with 5-fold CV
      - Evaluate metrics and plot ROC + Precision-Recall
      - Save best model to models/xgb_default_model.pkl

    Returns
    -------
    model: XGBClassifier (best estimator)
    metrics: dict with accuracy, auc_roc, f1, confusion_matrix, best_params
    """
    # 1) Load data
    df = pd.read_csv(data_path)
    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    df = _drop_non_useful_columns(df)
    df = _sanitize_dataframe(df)

    # Diagnostics: print types and basic stats before any coercion of target
    _print_feature_summary(df, target_col="target")

    # 2) Prepare target first (drop rows with missing/invalid target and coerce to 0/1)
    df, y = _prepare_binary_target(df, target_col="target")

    # 2a) Robustly identify feature types and cast numeric-like objects
    categorical_cols, numerical_cols, df_cast = _split_and_cast_feature_types(df, target_col="target")

    X = df_cast.drop(columns=["target"])  # features

    # 2b) Preprocessing pipelines with imputers
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", _make_one_hot_encoder()),
    ])
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", numerical_pipeline, numerical_cols),
        ],
        remainder="drop",
    )

    # 3) Model and pipeline
    xgb_clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=os.cpu_count() or 4,
        verbosity=0,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", xgb_clf),
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # 3b) Hyperparameter search space (RandomizedSearchCV)
    param_distributions = {
        "model__n_estimators": randint(200, 800),
        "model__max_depth": randint(3, 12),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.6, 0.4),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=5,
        scoring="roc_auc",
        n_jobs=os.cpu_count() or 4,
        cv=cv,
        verbose=1,
        random_state=42,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_model: Pipeline = search.best_estimator_

    # 4) Evaluation
    y_pred = best_model.predict(X_test)
    # For ROC/PR we need probabilities
    y_proba = best_model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": float(acc),
        "auc_roc": float(auc),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "best_params": search.best_params_,
    }

    # Plot ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0])
    axes[0].set_title("ROC Curve")
    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("Precision-Recall Curve")
    fig.tight_layout()
    plt.show()

    # 5) Save model
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "xgb_default_model.pkl")
    dump(best_model, model_path)

    print(f"Model saved to: {model_path}")

    return best_model, metrics


# %% Quick run cell (adjust the path if needed)
# Example: data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dataproject2025.csv')
data_path = "../data/dataproject2025.csv"

# %% Train and evaluate
if os.path.exists(data_path):
    model, eval_metrics = train_xgboost_model(data_path)
else:
    print(f"CSV not found at: {data_path}. Please set 'data_path' to the correct location.")

# %% Display metrics
try:
    if 'eval_metrics' in globals():
        print("Best params:")
        for k, v in eval_metrics["best_params"].items():
            print(f"  {k}: {v}")
        print("\nMetrics:")
        print({k: v for k, v in eval_metrics.items() if k != 'best_params'})
except Exception as e:
    print(f"Unable to display metrics: {e}")

# %%
