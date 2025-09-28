#!/usr/bin/env python3
"""
Simplified PDP Implementation for XGBoost Black Box Pipeline
Directly uses the xgboost_black_box_pipeline_vf.pkl model for predictions
Uses the same preprocessing pipeline from steps_2_&_3_vf.py
No surrogate models - analyzes the actual XGBoost pipeline results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
import warnings
warnings.filterwarnings('ignore')

def load_xgboost_pipeline():
    """Load XGBoost pipeline from vf pkl file"""
    try:
        pipeline = load('../models/xgboost_black_box_pipeline_vf.pkl')
        print(f"XGBoost pipeline loaded: {type(pipeline)}")
        print(f"Pipeline has steps: {[step[0] for step in pipeline.steps]}")
        return pipeline
    except Exception as e:
        print(f"Pipeline loading error: {e}")
        return None

def load_data():
    """Load raw data for pipeline processing using same format as steps_2_&_3_vf.py"""
    print("Loading raw data for XGBoost pipeline...")

    df = pd.read_csv('../data/dataproject2025 (5).csv', index_col=0)

    # Use a sample for faster processing
    sample_size = min(20000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} rows")

    # Get features using same format as original training (steps_2_&_3_vf.py)
    exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    print(f"Raw features: {len(feature_cols)}")
    print(f"Sample data shape: {X.shape}")

    return X, feature_cols

def get_top_features_by_importance(pipeline, X, feature_cols, n_features=6):
    """Get top features using the pipeline predictions"""
    print(f"Getting predictions from XGBoost pipeline to find top {n_features} features...")

    # Get predictions from the pipeline
    try:
        pipeline_predictions = pipeline.predict_proba(X)[:, 1]  # Get positive class probabilities
        print(f"Pipeline predictions range: {pipeline_predictions.min():.4f} to {pipeline_predictions.max():.4f}")
    except Exception as e:
        print(f"Error getting pipeline predictions: {e}")
        # Fallback to first few features
        return feature_cols[:n_features]

    # Find features with highest correlation to pipeline predictions
    correlations = []
    for col in feature_cols:
        if col in X.columns:
            # For numerical columns, use directly
            if X[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                x_values = X[col].fillna(0)  # Fill NaN with 0
            else:
                # For categorical, encode as numerical for correlation
                x_values = pd.Categorical(X[col]).codes

            try:
                corr = abs(np.corrcoef(x_values, pipeline_predictions)[0, 1])
                if not np.isnan(corr):
                    correlations.append((col, corr))
            except:
                continue

    correlations.sort(key=lambda x: x[1], reverse=True)
    top_features = [feat for feat, _ in correlations[:n_features]]

    print(f"Top {n_features} features by correlation with pipeline predictions:")
    for i, (feat, corr) in enumerate(correlations[:n_features], 1):
        print(f"{i:2}. {feat:25} {corr:.4f}")

    return top_features

def calculate_pdp_with_pipeline(pipeline, X, feature, grid_resolution=20):
    """Calculate PDP using XGBoost pipeline predictions"""
    print(f"Calculating PDP for {feature} using pipeline...")

    # Create grid based on feature type
    feature_values = X[feature].dropna()
    if feature_values.nunique() <= 10 or X[feature].dtype == 'object':  # Categorical/discrete
        grid = sorted(feature_values.unique())
        if len(grid) > 15:  # Limit categorical values
            grid = grid[:15]
    else:
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

    pdp_values = []

    # For each grid point, create modified dataset and get pipeline predictions
    for grid_value in grid:
        try:
            # Create copy of data with feature fixed to grid_value
            X_modified = X.copy()
            X_modified[feature] = grid_value

            # Get predictions from pipeline
            predictions = pipeline.predict_proba(X_modified)[:, 1]
            avg_prediction = predictions.mean()
            pdp_values.append(avg_prediction)
        except Exception as e:
            print(f"Error at grid value {grid_value}: {e}")
            pdp_values.append(np.nan)

    # Remove NaN values
    valid_mask = ~np.isnan(pdp_values)
    grid_clean = np.array(grid)[valid_mask]
    pdp_clean = np.array(pdp_values)[valid_mask]

    return grid_clean, pdp_clean

def create_pdp_plots(pipeline, X, features):
    """Create PDP plots using XGBoost pipeline"""
    print(f"Creating PDP plots for {len(features)} features using XGBoost pipeline...")

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

        # Calculate PDP using XGBoost pipeline
        grid, pdp_values = calculate_pdp_with_pipeline(pipeline, X, feature)

        if len(pdp_values) > 0:
            # Plot PDP
            ax.plot(grid, pdp_values, color='#2E86AB', linewidth=3,
                   marker='o', markersize=4, label='XGBoost PDP')
            ax.fill_between(grid, pdp_values, alpha=0.2, color='#2E86AB')

            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel('Partial Dependence\n(Pipeline Prediction)', fontsize=11, fontweight='bold')
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

    plt.suptitle('Partial Dependence Plots - XGBoost Pipeline (steps_2_&_3_vf.py)\n(Direct from xgboost_black_box_pipeline_vf.pkl)',
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('sklearn_pdp_plots.png', dpi=300, bbox_inches='tight')
    print("PDP plots saved as 'sklearn_pdp_plots.png'")
    plt.show()

def main():
    """Main function"""
    print("Starting Simplified XGBoost PDP Analysis...")
    print("Using xgboost_black_box_pipeline_vf.pkl directly (no surrogate models)")
    print("="*60)

    try:
        # Load XGBoost pipeline
        pipeline = load_xgboost_pipeline()
        if pipeline is None:
            print("Failed to load pipeline. Exiting.")
            return 1

        # Load raw data
        X, feature_cols = load_data()

        # Get top features using pipeline predictions
        top_features = get_top_features_by_importance(pipeline, X, feature_cols, n_features=6)

        # Create PDP plots using pipeline directly
        create_pdp_plots(pipeline, X, top_features)

        print("\n" + "="*60)
        print("SIMPLIFIED PDP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("This analysis shows how the XGBoost pipeline's predictions")
        print("vary with each feature, calculated directly from the")
        print("xgboost_black_box_pipeline_vf.pkl model without surrogate models.")
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