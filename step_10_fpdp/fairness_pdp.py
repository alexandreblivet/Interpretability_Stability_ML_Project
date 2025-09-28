#!/usr/bin/env python3
"""
Fairness Partial Dependence Plot (FPDP) Implementation
Analyzes XGBoost model predictions across different demographic groups
Uses same approach as step_9_fairness with Pct_afro_american quartiles
Works directly with existing XGBoost predictions (no surrogate models)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    sample_size = min(15000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    print(f"Using sample of {sample_size} rows")

    # Store original data before preprocessing for group definition
    df_original = df.copy()

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

    return X, xgb_predictions, feature_cols, df_original

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

def get_top_features_by_correlation(X, xgb_predictions, feature_cols, n_features=4):
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

def calculate_group_pdp_manual(X, xgb_predictions, feature, groups, grid_resolution=20):
    """Calculate PDP for different demographic groups using XGBoost predictions"""
    group_values = sorted(groups.unique())
    group_pdps = {}

    # Create value grid
    feature_values = X[feature].values
    if X[feature].nunique() <= 10:  # Discrete
        grid = np.unique(feature_values)
    else:
        grid = np.linspace(feature_values.min(), feature_values.max(), grid_resolution)

    for group in group_values:
        # Filter data for this group
        group_mask = groups == group
        X_group = X[group_mask]
        pred_group = xgb_predictions[group_mask]

        if len(X_group) < 30:  # Skip small groups
            print(f"  Skipping group {group} (only {len(X_group)} samples)")
            continue

        print(f"  Processing group {group} ({len(X_group)} samples)")

        # Calculate PDP for this group using XGBoost predictions
        pdp_values = []
        for grid_value in grid:
            if X[feature].nunique() <= 10:  # Discrete
                mask = X_group[feature] == grid_value
            else:  # Continuous
                range_size = (feature_values.max() - feature_values.min()) / (grid_resolution * 2)
                mask = np.abs(X_group[feature] - grid_value) <= range_size

            if mask.sum() > 0:
                # Average XGBoost predictions for this feature value in this group
                avg_prediction = pred_group[mask].mean()
                pdp_values.append(avg_prediction)
            else:
                pdp_values.append(np.nan)

        # Remove NaN values
        valid_mask = ~np.isnan(pdp_values)
        if valid_mask.sum() > 0:
            group_pdps[group] = np.array(pdp_values)[valid_mask]
            grid = grid[valid_mask]

    return grid, group_pdps

def create_fairness_pdp_plots(X, xgb_predictions, features, groups):
    """Create fairness PDP plots comparing different demographic groups"""
    print(f"Creating Fairness PDP plots for {len(features)} features...")
    print(f"Comparing across {groups.name} groups using XGBoost predictions")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # Use a nice color palette
    unique_groups = sorted(groups.unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_groups)))

    for i, feature in enumerate(features):
        ax = axes[i]

        print(f"Calculating FPDP for {feature}...")
        # Calculate fairness PDP using XGBoost predictions
        grid, group_pdps = calculate_group_pdp_manual(X, xgb_predictions, feature, groups)

        # Plot PDP for each group
        for j, (group, pdp_values) in enumerate(group_pdps.items()):
            ax.plot(grid, pdp_values, color=colors[j % len(colors)], linewidth=2.5,
                   label=f'{group}', marker='o', markersize=4, alpha=0.8)

        ax.set_xlabel(feature, fontsize=11, fontweight='bold')
        ax.set_ylabel('Partial Dependence\n(XGBoost Prediction)', fontsize=11, fontweight='bold')
        ax.set_title(f'Fairness PDP: {feature}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, title=groups.name)

    plt.suptitle(f'Fairness Partial Dependence Plots\nSensitive Attribute: {groups.name}\n(Direct from XGBoost Predictions)',
                fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('fairness_pdp_plots.png', dpi=300, bbox_inches='tight')
    print("Fairness PDP plots saved as 'fairness_pdp_plots.png'")
    plt.show()

def calculate_fairness_metrics(X, xgb_predictions, features, groups):
    """Calculate fairness metrics based on PDP differences"""
    print(f"\nCalculating fairness metrics for {groups.name}...")

    fairness_results = {}

    for feature in features:
        grid, group_pdps = calculate_group_pdp_manual(X, xgb_predictions, feature, groups)

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
        plt.title(f'Fairness Analysis: Maximum PDP Differences\nSensitive Attribute: {groups.name}\n(XGBoost Black Box Model)',
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
    print("Starting XGBoost Fairness PDP Analysis...")
    print("Using same demographic groups as step_9_fairness")
    print("Working directly with XGBoost predictions (no surrogate models)")
    print("="*60)

    try:
        # Load XGBoost model info
        xgb_model = load_xgboost_model_info()

        # Load data with existing XGBoost predictions
        X, xgb_predictions, feature_cols, df_original = load_data()

        # Define demographic groups (same as step_9_fairness)
        groups = define_demographic_groups(df_original)

        # Get top features by correlation with XGBoost predictions
        top_features = get_top_features_by_correlation(X, xgb_predictions, feature_cols, n_features=4)

        # Create fairness PDP plots using XGBoost predictions directly
        create_fairness_pdp_plots(X, xgb_predictions, top_features, groups)

        # Calculate fairness metrics
        fairness_results = calculate_fairness_metrics(X, xgb_predictions, top_features, groups)

        print("\n" + "="*60)
        print("FAIRNESS PDP ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("This analysis shows how the XGBoost model's predictions vary")
        print("across different demographic groups, calculated directly from")
        print("the model's actual predictions (no surrogate approximation).")
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