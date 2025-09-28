# %%
import os
import json
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from joblib import load

# %%
# Optional: if JSON booster is preferred
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None  # type: ignore

from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# %%
# LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    _has_lime = True
except Exception:
    _has_lime = False


def _find_model_path(models_dir: str) -> str:
    """Pick an available saved black-box model file with preference order."""
    candidates = [
        os.path.join(models_dir, "xgb_default_model.pkl"),  # prefer pipeline with preprocessing
        os.path.join(models_dir, "xgboost_black_box_model.joblib"),
        os.path.join(models_dir, "xgboost_black_box_model.pkl"),
        os.path.join(models_dir, "xgboost_black_box_model.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No saved model found in {models_dir}. Expected one of: {candidates}")


def _load_model(model_path: str):
    """Load a saved model. Supports joblib/pkl pipelines, or XGBoost JSON booster."""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".joblib", ".pkl"):
        return load(model_path)
    if ext == ".json":
        if XGBClassifier is None:
            raise RuntimeError("xgboost not available to load JSON model")
        model = XGBClassifier()
        model.load_model(model_path)
        return model
    raise ValueError(f"Unsupported model format: {ext}")


def _infer_feature_names(df: pd.DataFrame, target_col: str = "target") -> List[str]:
    return [c for c in df.columns if c != target_col]


def _prepare_dataset(data_path: str, target_col: str = "target") -> tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Dataset must contain '{target_col}' column.")
    feature_names = _infer_feature_names(df, target_col)
    X = df[feature_names].copy()
    y = df[target_col].copy()
    return X, y, feature_names


def _predict_proba_wrapper(model):
    """Return a function X->probas compatible with LIME/ICE, handling pipelines."""
    def predict_proba(X: np.ndarray) -> np.ndarray:
        # Some models may not support predict_proba (then use decision_function/sigmoid)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            # Convert to 2-class proba-like
            scores = np.asarray(scores).reshape(-1, 1)
            scores = 1 / (1 + np.exp(-scores))
            return np.hstack([1 - scores, scores])
        # Fallback: predict as label and make degenerate proba
        labels = np.asarray(model.predict(X)).reshape(-1)
        return np.vstack([1 - labels, labels]).T
    return predict_proba

def generate_lime_explanations(model, X_df: pd.DataFrame, feature_names: List[str],
                               class_names: Optional[List[str]] = None,
                               num_samples: int = 5,
                               output_dir: str = "interpretability_outputs") -> None:
    if not _has_lime:
        print("LIME not installed; skipping LIME explanations.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Train/test split pour exemples
    X_train, X_test = train_test_split(X_df, test_size=0.2, random_state=42)

    # Utiliser l'espace transformé du Pipeline pour éviter les strings bruts
    from sklearn.pipeline import Pipeline as SkPipeline
    if not isinstance(model, SkPipeline) or "preprocess" not in model.named_steps or "model" not in model.named_steps:
        print("Warning: LIME expects a sklearn Pipeline with steps 'preprocess' and 'model'. Skipping LIME.")
        return

    preprocess = model.named_steps["preprocess"]
    final_estimator = model.named_steps["model"]

    # Transform training set to numeric space used by the final estimator
    Z_train = preprocess.transform(X_train)
    if hasattr(Z_train, "toarray"):
        Z_train = Z_train.toarray()

    # Feature names after preprocessing
    try:
        feature_names_out = list(preprocess.get_feature_names_out())
    except Exception:
        feature_names_out = [f"f_{i}" for i in range(Z_train.shape[1])]

    explainer = LimeTabularExplainer(
        training_data=Z_train,
        feature_names=feature_names_out,
        class_names=class_names or ["no", "yes"],
        discretize_continuous=True,
        mode="classification",
    )

    # Predict function that works in transformed space directly on the final estimator
    def predict_fn_transformed(Z: np.ndarray) -> np.ndarray:
        return final_estimator.predict_proba(Z)

    # Choose random instances from test set
    np.random.seed(42)
    idxs = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)

    for i, idx in enumerate(idxs, start=1):
        x_raw = X_test.iloc[idx:idx+1]
        z = preprocess.transform(x_raw)
        if hasattr(z, "toarray"):
            z = z.toarray()
        z1 = z[0]

        exp = explainer.explain_instance(z1, predict_fn_transformed, num_features=min(10, len(feature_names_out)))

        # Save HTML explanation
        html_path = os.path.join(output_dir, f"lime_explanation_{i}.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(exp.as_html())
        print(f"Saved LIME explanation: {html_path}")

        # Save PNG
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f"lime_explanation_{i}.png")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved LIME plot: {fig_path}")


def generate_ice_plots(model, X_df: pd.DataFrame, feature_names: List[str], features_to_plot: Optional[List[str]] = None, output_dir: str = "interpretability_outputs_step6") -> None:
    os.makedirs(output_dir, exist_ok=True)

    # If the model is not a sklearn Pipeline with preprocessing, PDP may not align with raw X
    from sklearn.pipeline import Pipeline as SkPipeline  # local import to avoid dependency at top
    if not isinstance(model, SkPipeline):
        print("Warning: ICE expects a sklearn Pipeline with preprocessing. Skipping ICE for non-pipeline model.")
        return

    # Pick up to 6 numeric features if not provided
    if features_to_plot is None:
        numeric_candidates = [c for c in feature_names if np.issubdtype(X_df[c].dtype, np.number)]
        features_to_plot = numeric_candidates[:6]

    if not features_to_plot:
        print("No numeric features available for ICE plots.")
        return

    # sklearn's PartialDependenceDisplay can do ICE (individual=True)
    fig, axes = plt.subplots(nrows=int(np.ceil(len(features_to_plot) / 3)), ncols=3, figsize=(15, 4 * int(np.ceil(len(features_to_plot) / 3))))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features_to_plot):
        try:
            PartialDependenceDisplay.from_estimator(
                estimator=model,
                X=X_df,
                features=[feat],
                kind="individual",  # ICE curves
                response_method="auto",
                target=1,
                ax=ax,
            )
            ax.set_title(f"ICE for {feat}")
        except Exception as e:
            ax.set_visible(False)
            print(f"Skipping ICE for {feat}: {e}")

    # Hide unused axes
    for ax in axes[len(features_to_plot):]:
        ax.set_visible(False)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "ice_plots.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ICE plots: {fig_path}")


def main():
    # Paths
    root = os.path.dirname(os.path.dirname(__file__))
    models_dir = os.path.join(root, "models")

    # Preferred data path (align with run_analysis/ml_models)
    default_data_paths = [
        os.path.join(root, "data", "dataproject2025 (1).csv"),
        os.path.join(root, "data", "dataproject2025.csv"),
    ]
    data_path = next((p for p in default_data_paths if os.path.exists(p)), None)
    if data_path is None:
        raise FileNotFoundError(f"Could not find data CSV in: {default_data_paths}")

    # Load model
    model_path = _find_model_path(models_dir)
    print(f"Loading model from: {model_path}")
    model = _load_model(model_path)

    # Prepare dataset
    X, y, feature_names = _prepare_dataset(data_path, target_col="target")

    # Output dir
    out_dir = os.path.join(root, "interpretability_outputs")
    os.makedirs(out_dir, exist_ok=True)

    # LIME explanations
    generate_lime_explanations(model, X, feature_names, class_names=["no_default", "default"], num_samples=5, output_dir=out_dir)

    # ICE plots
    # If the model is a pipeline with preprocessing, sklearn PDP will call transform internally
    generate_ice_plots(model, X, feature_names, features_to_plot=None, output_dir=out_dir)

    print("Done. Check 'interpretability_outputs' for artifacts.")

# %%
if __name__ == "__main__":
    main()

# %%
