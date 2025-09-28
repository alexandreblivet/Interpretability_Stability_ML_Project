#!/usr/bin/env python3
"""
Simplified Fairness Partial Dependence Plot (FPDP) Implementation
Directly uses the xgboost_black_box_pipeline_vf.pkl model for predictions
Uses the same preprocessing pipeline from steps_2_&_3_vf.py
Analyzes predictions across different demographic groups
No surrogate models - analyzes the actual XGBoost pipeline results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    # Use a smaller sample for faster processing in fairness analysis
    sample_size = min(5000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} rows")

    # Store original data for group definition (before any preprocessing)
    df_original = df.copy()

    # Get features using same format as original training (steps_2_&_3_vf.py)
    exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()

    print(f"Raw features: {len(feature_cols)}")
    print(f"Sample data shape: {X.shape}")

    return X, feature_cols, df_original

def define_demographic_groups(df_original):
    """Define demographic groups using same approach as step_9_fairness"""
    protected_col = 'Pct_afro_american'

    if protected_col not in df_original.columns:
        print(f"Warning: {protected_col} not found. Available columns: {list(df_original.columns)}")
        # Fallback to any suitable column
        for col in df_original.columns:
            if df_original[col].nunique() <= 8 and col not in ['target', 'Predictions', 'Predicted probabilities']:
                protected_col = col
                break

    print(f"Using protected attribute: {protected_col}")
    raw_groups = df_original[protected_col]

    # Bin continuous protected attributes into quartiles (same as step_9)
    if pd.api.types.is_numeric_dtype(raw_groups) and raw_groups.nunique() > 10:
        print(f"Attribute '{protected_col}' is continuous. Binning into quartiles.")
        groups = pd.qcut(raw_groups, q=4, labels=["Q1_low", "Q2", "Q3", "Q4_high"], duplicates='drop')
        groups.name = f"{protected_col}_quartile"
    else:
        # For categorical, keep as is
        groups = raw_groups.astype(str)
        groups.name = protected_col

    # Show distribution
    dist = groups.value_counts()
    print(f"\nGroup distribution for {groups.name}:")
    for group, count in dist.items():
        pct = count / len(groups) * 100
        print(f"  {group}: {count} ({pct:.1f}%)")

    return groups

def get_top_features_by_importance(pipeline, X, feature_cols, n_features=4):
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

def calculate_group_pdp_with_pipeline(pipeline, X, feature, groups, grid_resolution=10):
    """Calculate PDP for different demographic groups using XGBoost pipeline"""
    group_values = sorted(groups.unique())
    group_pdps = {}

    # Create value grid based on feature type
    feature_values = X[feature].dropna()
    if feature_values.nunique() <= 8 or X[feature].dtype == 'object':  # Categorical/discrete
        grid = sorted(feature_values.unique())
        if len(grid) > 12:  # Limit categorical values
            grid = grid[:12]
    else:
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

    for group in group_values:
        # Filter data for this group
        group_mask = groups == group
        X_group = X[group_mask]

        if len(X_group) < 30:  # Skip small groups
            print(f"  Skipping group {group} (only {len(X_group)} samples)")
            continue

        print(f"  Processing group {group} ({len(X_group)} samples)")

        # Calculate PDP for this group using pipeline
        pdp_values = []
        for grid_value in grid:
            try:
                # Create modified dataset with feature fixed to grid_value
                X_modified = X_group.copy()
                X_modified[feature] = grid_value

                # Get predictions from pipeline
                predictions = pipeline.predict_proba(X_modified)[:, 1]
                avg_prediction = predictions.mean()
                pdp_values.append(avg_prediction)
            except Exception as e:
                print(f"    Error at grid value {grid_value}: {e}")
                pdp_values.append(np.nan)

        # Remove NaN values
        valid_mask = ~np.isnan(pdp_values)
        if valid_mask.sum() > 0:
            group_pdps[group] = np.array(pdp_values)[valid_mask]

    return grid, group_pdps

def create_fairness_pdp_plots(pipeline, X, features, groups):
    """Create fairness PDP plots comparing different demographic groups"""
    print(f"Creating Fairness PDP plots for {len(features)} features...")
    print(f"Comparing across {groups.name} groups using XGBoost pipeline")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Use a nice color palette
    unique_groups = sorted(groups.unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_groups)))

    for i, feature in enumerate(features):
        ax = axes[i]

        print(f"Calculating FPDP for {feature}...")
        # Calculate fairness PDP using XGBoost pipeline
        grid, group_pdps = calculate_group_pdp_with_pipeline(pipeline, X, feature, groups)

        # Plot PDP for each group
        for j, (group, pdp_values) in enumerate(group_pdps.items()):
            ax.plot(grid, pdp_values, color=colors[j % len(colors)], linewidth=2.5,
                   label=f'{group}', marker='o', markersize=4, alpha=0.8)

        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Partial Dependence\n(Pipeline Prediction)', fontsize=11, fontweight='bold')
        ax.set_title(f'Fairness PDP: {feature}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, title=groups.name)

    plt.suptitle(f'Fairness Partial Dependence Plots\nSensitive Attribute: {groups.name}\n(xgboost_black_box_pipeline_vf.pkl)',
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('fairness_pdp_plots.png', dpi=300, bbox_inches='tight')
    print("Fairness PDP plots saved as 'fairness_pdp_plots.png'")
    plt.show()

def calculate_fairness_metrics(pipeline, X, features, groups):
    """Calculate fairness metrics based on PDP differences"""
    print(f"\nCalculating fairness metrics for {groups.name}...")

    fairness_results = {}

    for feature in features:
        grid, group_pdps = calculate_group_pdp_with_pipeline(pipeline, X, feature, groups)

        if len(group_pdps) < 2:
            continue

        # Calculate group means
        group_means = {}
        for group, pdp_values in group_pdps.items():
            group_means[group] = np.mean(pdp_values)

        if len(group_means) >= 2:
            means_list = list(group_means.values())
            max_diff = max(means_list) - min(means_list)
            std_diff = np.std(means_list)

            fairness_results[feature] = {
                'max_group_difference': max_diff,
                'std_group_difference': std_diff,
                'group_means': group_means
            }

    # Create summary plot
    if fairness_results:
        features_list = list(fairness_results.keys())
        max_diffs = [fairness_results[f]['max_group_difference'] for f in features_list]

        plt.figure(figsize=(12, 8))
        bars = plt.barh(features_list, max_diffs, color='#e74c3c', alpha=0.7, edgecolor='black')
        plt.xlabel('Maximum Group Difference', fontsize=12, fontweight='bold')
        plt.ylabel('Features', fontsize=12, fontweight='bold')
        plt.title(f'Fairness Analysis: Maximum PDP Differences\nSensitive Attribute: {groups.name}\n(xgboost_black_box_pipeline_vf.pkl)',
                  fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, max_diffs):
            plt.text(val + max(max_diffs)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', ha='left', fontsize=10)

        plt.tight_layout()
        plt.savefig('fairness_summary.png', dpi=300, bbox_inches='tight')
        print("Fairness summary saved as 'fairness_summary.png'")
        plt.show()

        # Print results
        print(f"\nFairness Analysis Results:")
        print(f"{'='*50}")
        for feature, results in fairness_results.items():
            print(f"\n{feature}:")
            print(f"  Max group difference: {results['max_group_difference']:.6f}")
            print(f"  Std group difference: {results['std_group_difference']:.6f}")
            print(f"  Group means:")
            for group, mean in results['group_means'].items():
                print(f"    {group}: {mean:.6f}")

    return fairness_results

def main():
    """Main function"""
    print("Starting Simplified XGBoost Fairness PDP Analysis...")
    print("Using xgboost_black_box_pipeline_vf.pkl directly (no surrogate models)")
    print("Using same demographic groups as step_9_fairness")
    print("="*60)

    try:
        # Load XGBoost pipeline
        pipeline = load_xgboost_pipeline()
        if pipeline is None:
            print("Failed to load pipeline. Exiting.")
            return 1

        # Load raw data
        X, feature_cols, df_original = load_data()

        # Define demographic groups (same as step_9_fairness)
        groups = define_demographic_groups(df_original)

        # Get top features using pipeline predictions
        top_features = get_top_features_by_importance(pipeline, X, feature_cols, n_features=4)

        # Create fairness PDP plots using pipeline directly
        create_fairness_pdp_plots(pipeline, X, top_features, groups)

        # Calculate fairness metrics
        fairness_results = calculate_fairness_metrics(pipeline, X, top_features, groups)

        print("\n" + "="*60)
        print("SIMPLIFIED FAIRNESS PDP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("This analysis shows how the XGBoost pipeline's predictions vary")
        print("across different demographic groups, calculated directly from")
        print("the xgboost_black_box_pipeline_vf.pkl model without surrogate models.")
        print(f"Analyzed groups: {groups.name}")
        print("Generated files:")
        print("- fairness_pdp_plots.png")
        print("- fairness_summary.png")
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