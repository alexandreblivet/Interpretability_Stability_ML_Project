"""
SHAP Analysis Script for XGBoost Credit Default Model
"""

import os
import warnings
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def ensure_plots_dir(path="plots"):
    """Create plots directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


def load_model():
    """Load XGBoost model from JSON format"""
    model_path = "models/xgboost_black_box_model.json"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model, model_path


def load_data():
    """Load dataset"""
    data_path = "data/dataproject2025.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found: {data_path}")

    return pd.read_csv(data_path), data_path


def prepare_data(df):
    """Prepare data for SHAP analysis with TableVectorizer"""
    from skrub import TableVectorizer
    import re

    # Exclude columns
    exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if not feature_cols:
        raise ValueError("No feature columns found after exclusion")

    print(f"Features: {len(feature_cols)} columns")

    # Select features
    X_raw = df[feature_cols].copy()

    print(f"Original dataset: {len(X_raw)} rows")

    # Clean data
    X_raw = X_raw.dropna()
    print(f"Cleaned dataset: {len(X_raw)} rows")

    # Sample data for performance
    if len(X_raw) > 100000:
        X_raw = X_raw.sample(n=100000, random_state=RANDOM_STATE)
        print(f"Sampled: {len(X_raw)} rows")

    # Apply TableVectorizer preprocessing
    print("Applying TableVectorizer...")
    vectorizer = TableVectorizer()
    X_vectorized = vectorizer.fit_transform(X_raw)

    # Convert to array and create DataFrame with clean names
    if hasattr(X_vectorized, 'toarray'):
        X_array = X_vectorized.toarray()
    else:
        X_array = np.array(X_vectorized)

    # Get and clean feature names
    feature_names = vectorizer.get_feature_names_out()
    cleaned_names = []
    seen = set()

    for i, name in enumerate(feature_names):
        clean_name = re.sub(r'[<>\[\](){}]', '_', str(name))
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name).strip('_')

        if not clean_name or clean_name.isdigit():
            clean_name = f'feature_{i}'

        # Ensure uniqueness
        original_name = clean_name
        counter = 1
        while clean_name in seen:
            clean_name = f"{original_name}_{counter}"
            counter += 1

        seen.add(clean_name)
        cleaned_names.append(clean_name)

    X_processed = pd.DataFrame(X_array, columns=cleaned_names)

    print(f"After vectorization: {X_raw.shape[1]} -> {X_processed.shape[1]} features")

    return X_processed


def build_explainer(model, X_sample):
    """Build SHAP explainer with background data"""
    # Sample background data
    background_size = min(100, len(X_sample))
    np.random.seed(RANDOM_STATE)
    background_indices = np.random.choice(len(X_sample), background_size, replace=False)
    background_data = X_sample.iloc[background_indices]

    # Create explainer
    explainer = shap.Explainer(model, background_data)

    return explainer


def compute_shap_values(explainer, X_sample):
    """Compute SHAP values for sample data"""
    # Sample for SHAP computation
    sample_size = min(2000, len(X_sample))

    if len(X_sample) > sample_size:
        print(f"SHAP sampling: {sample_size} rows from {len(X_sample)}")
        np.random.seed(RANDOM_STATE + 1)
        indices = np.random.choice(len(X_sample), sample_size, replace=False)
        X_shap = X_sample.iloc[indices]
    else:
        X_shap = X_sample

    print(f"Computing SHAP values on {len(X_shap)} samples")

    # Calculate SHAP values
    shap_values = explainer(X_shap)

    return shap_values, X_shap


def create_visualizations(shap_values, X_sample, plots_dir):
    """Generate and save SHAP visualizations"""

    # 1. Summary plot (beeswarm)
    plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Bar plot (feature importance)
    plt.figure(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Waterfall plot (individual explanation)
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(shap_values[0], max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_waterfall.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Dependence plots for top features
    feature_importance = np.abs(shap_values.values).mean(0)
    top_features_idx = np.argsort(feature_importance)[-10:]  # Top 10 features

    for idx in top_features_idx:
        feature_name = X_sample.columns[idx]
        plt.figure(figsize=(8, 6))
        shap.plots.scatter(shap_values[:, idx], show=False)
        plt.title(f"SHAP Dependence: {feature_name}")
        plt.tight_layout()
        # Clean feature name for filename
        safe_name = feature_name.replace('/', '_').replace(' ', '_').replace(':', '_')
        plt.savefig(os.path.join(plots_dir, f"shap_dependence_{safe_name}.png"),
                   dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Feature importance ranking
    plt.figure(figsize=(12, 8))
    feature_names = X_sample.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True).tail(20)

    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Top 20 Features by SHAP Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_feature_ranking.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. SHAP values distribution
    plt.figure(figsize=(10, 6))
    plt.hist(shap_values.values.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('SHAP Values')
    plt.ylabel('Frequency')
    plt.title('Distribution of SHAP Values')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "shap_values_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


def analyze_shap_insights(shap_values, X_sample):
    """Analyze and print key insights from SHAP values"""
    print("\n" + "="*80)
    print("DETAILED SHAP ANALYSIS RESULTS")
    print("="*80)

    # Calculate feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    feature_names = X_sample.columns

    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance,
        'mean_shap': shap_values.values.mean(0),
        'std_shap': shap_values.values.std(0)
    }).sort_values('importance', ascending=False)

    print("\nTOP 15 MOST IMPORTANT FEATURES:")
    print("-" * 60)
    for i, (_, row) in enumerate(importance_df.head(15).iterrows(), 1):
        direction = "INCREASES" if row['mean_shap'] > 0 else "DECREASES"
        print(f"{i:2d}. {row['feature'][:40]:40} | Impact: {row['importance']:.4f} | {direction} risk")

    # Analyze positive vs negative impacts
    positive_impact = importance_df[importance_df['mean_shap'] > 0]
    negative_impact = importance_df[importance_df['mean_shap'] < 0]

    print(f"\nIMPACT DISTRIBUTION:")
    print(f"Features increasing risk: {len(positive_impact)} ({len(positive_impact)/len(importance_df)*100:.1f}%)")
    print(f"Features decreasing risk: {len(negative_impact)} ({len(negative_impact)/len(importance_df)*100:.1f}%)")

    print(f"\nTOP 5 RISK-INCREASING FACTORS:")
    for i, (_, row) in enumerate(positive_impact.head(5).iterrows(), 1):
        print(f"{i}. {row['feature'][:50]:50} (Impact: +{row['mean_shap']:.4f})")

    print(f"\nTOP 5 RISK-DECREASING FACTORS:")
    for i, (_, row) in enumerate(negative_impact.head(5).iterrows(), 1):
        print(f"{i}. {row['feature'][:50]:50} (Impact: {row['mean_shap']:.4f})")

    # Statistical insights
    total_variance = shap_values.values.var(axis=0).sum()
    top_10_variance = importance_df.head(10)['std_shap'].pow(2).sum()

    print(f"\nSTATISTICAL INSIGHTS:")
    print(f"Variance explained by top 10 features: {top_10_variance/total_variance*100:.1f}%")
    print(f"Mean SHAP value: {shap_values.values.mean():.4f}")
    print(f"SHAP values standard deviation: {shap_values.values.std():.4f}")

    # Base value insight
    print(f"\nBASE VALUE (without features): {shap_values.base_values[0]:.4f}")
    print(f"This represents the model's average prediction")

    return importance_df


def main():
    """Main function"""
    print("Starting SHAP Analysis")

    # Setup
    plots_dir = ensure_plots_dir()

    print("Loading model...")
    model, model_path = load_model()
    print(f"Model loaded from: {model_path}")

    print("Loading data...")
    df, data_path = load_data()
    print(f"Data loaded from: {data_path}")

    print("Preparing data...")
    X_sample = prepare_data(df)

    print("Building SHAP explainer...")
    explainer = build_explainer(model, X_sample)

    print("Computing SHAP values...")
    shap_values, X_shap = compute_shap_values(explainer, X_sample)

    print("Creating visualizations...")
    create_visualizations(shap_values, X_shap, plots_dir)

    print("Analyzing insights...")
    importance_df = analyze_shap_insights(shap_values, X_shap)

    print(f"\nAnalysis complete! Results saved in '{plots_dir}/'")
    print(f"SHAP analysis completed on {len(X_shap)} samples")

    # Save detailed results
    importance_df.to_csv(os.path.join(plots_dir, "shap_feature_importance.csv"), index=False)
    print(f"Detailed results saved as 'shap_feature_importance.csv'")


if __name__ == "__main__":
    main()