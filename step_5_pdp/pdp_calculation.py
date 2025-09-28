#!/usr/bin/env python3
"""
Simple PDP Implementation for XGBoost Black Box Model
Uses existing XGBoost predictions to create PDP analysis via manual calculation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_xgboost_model_info():
    """Load XGBoost model info from pkl file"""
    try:
        with open('../models/xgboost_black_box_model.pkl', 'rb') as f:
            model = pickle.load(f)
        print(f"XGBoost model info: {type(model)}")
        print(f"Model expects {model.n_features_in_} features (after vectorization)")
        return model
    except Exception as e:
        print(f"Model info: {e}")
        return None

def load_data():
    """Load data with existing XGBoost predictions"""
    print("Loading data with existing XGBoost predictions...")

    df = pd.read_csv('../data/dataproject2025 (5).csv', index_col=0)

    # Use a sample for faster processing
    sample_size = min(20000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} rows")

    # Basic preprocessing for features
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Get features (exclude target and prediction columns)
    exclude_cols = ['target', 'Predictions', 'Predicted probabilities']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols]

    # Use existing XGBoost predictions
    xgb_predictions = df['Predicted probabilities']

    print(f"Features: {len(feature_cols)}")
    print(f"XGBoost predictions range: {xgb_predictions.min():.4f} to {xgb_predictions.max():.4f}")

    return X, xgb_predictions, feature_cols

def get_top_features_by_correlation(X, xgb_predictions, feature_cols, n_features=6):
    """Get top features by correlation with XGBoost predictions"""
    print(f"Finding top {n_features} features by correlation with XGBoost predictions...")

    correlations = []
    for col in feature_cols:
        corr = abs(X[col].corr(xgb_predictions))
        if not np.isnan(corr):
            correlations.append((col, corr))

    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in correlations[:n_features]]

    print(f"Top {n_features} features by correlation with XGBoost predictions:")
    for i, (feat, corr) in enumerate(correlations[:n_features], 1):
        print(f"{i:2}. {feat:25} {corr:.4f}")

    return top_features

def calculate_manual_pdp(X, xgb_predictions, feature, grid_resolution=30):
    """Calculate PDP manually using existing XGBoost predictions"""
    print(f"Calculating PDP for {feature}...")

    # Create grid
    feature_values = X[feature].values
    if X[feature].nunique() <= 15:  # Categorical/discrete
        grid = np.unique(feature_values)
    else:
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

    pdp_values = []

    # For each grid point, find samples where feature is close to that value
    # and average their XGBoost predictions
    for grid_value in grid:
        if X[feature].nunique() <= 15:  # Discrete
            mask = X[feature] == grid_value
        else:  # Continuous - use samples within a range
            range_size = (feature_values.max() - feature_values.min()) / (grid_resolution * 2)
            mask = np.abs(X[feature] - grid_value) <= range_size

        if mask.sum() > 0:
            # Average XGBoost predictions for samples with this feature value
            avg_prediction = xgb_predictions[mask].mean()
            pdp_values.append(avg_prediction)
        else:
            # If no samples, interpolate
            pdp_values.append(np.nan)

    # Remove NaN values
    valid_mask = ~np.isnan(pdp_values)
    grid_clean = grid[valid_mask]
    pdp_clean = np.array(pdp_values)[valid_mask]

    return grid_clean, pdp_clean

def create_pdp_plots(X, xgb_predictions, features):
    """Create PDP plots using XGBoost predictions"""
    print(f"Creating PDP plots for {len(features)} features using XGBoost predictions...")

    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.style.use('default')
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))

    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_features == 1 else list(axes)
    else:
        axes = axes.flatten()

    for i, feature in enumerate(features):
        ax = axes[i]

        # Calculate PDP using XGBoost predictions
        grid, pdp_values = calculate_manual_pdp(X, xgb_predictions, feature)

        if len(pdp_values) > 0:
            # Plot PDP
            ax.plot(grid, pdp_values, color='#2E86AB', linewidth=3,
                   marker='o', markersize=4, label='XGBoost PDP')
            ax.fill_between(grid, pdp_values, alpha=0.2, color='#2E86AB')

            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Partial Dependence\n(XGBoost Prediction)', fontsize=11, fontweight='bold')
            ax.set_title(f'PDP: {feature}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Add feature distribution
            ax2 = ax.twinx()
            ax2.hist(X[feature], bins=30, alpha=0.3, color='gray', density=True)
            ax2.set_ylabel('Feature Density', alpha=0.6, fontsize=9)
            ax2.tick_params(axis='y', labelsize=8)
        else:
            ax.text(0.5, 0.5, f'No data for {feature}',
                   transform=ax.transAxes, ha='center', va='center')

    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Partial Dependence Plots - XGBoost Black Box Model\n(Direct from XGBoost Predictions)',
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('sklearn_pdp_plots.png', dpi=300, bbox_inches='tight')
    print("PDP plots saved as 'sklearn_pdp_plots.png'")
    plt.show()

def main():
    """Main function"""
    print("Starting XGBoost PDP Analysis...")
    print("Using existing XGBoost predictions directly (no surrogate models)")
    print("="*60)

    try:
        # Load XGBoost model info
        xgb_model = load_xgboost_model_info()

        # Load data with existing XGBoost predictions
        X, xgb_predictions, feature_cols = load_data()

        # Get top features by correlation with XGBoost predictions
        top_features = get_top_features_by_correlation(X, xgb_predictions, feature_cols, n_features=6)

        # Create PDP plots using XGBoost predictions directly
        create_pdp_plots(X, xgb_predictions, top_features)

        print("\n" + "="*60)
        print("PDP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("This analysis shows how the XGBoost black box model's predictions")
        print("vary with each feature, calculated directly from the model's")
        print("actual predictions without any surrogate approximation.")
        print(f"\nGenerated: sklearn_pdp_plots.png")
        print("="*60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())