#!/usr/bin/env python3
"""
Terminal runner for ML Interpretability and Stability Analysis
Converts the notebook analysis into a command-line executable script
"""

import sys
import os

# Add src directory to path
sys.path.append('src')

from ml_models import DefaultProbabilityAnalysis
import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("ML INTERPRETABILITY AND STABILITY PROJECT")
    print("=" * 60)

    # Check if data file exists
    data_path = 'data/dataproject2025 (1).csv'

    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        if os.path.exists('data'):
            print("Available files in data directory:")
            for file in os.listdir('data'):
                print(f"  - {file}")
        else:
            print("Data directory does not exist")
        return 1

    print(f"SUCCESS: Data file found: {data_path}")

    try:
        # Initialize analysis
        print("\nInitializing analysis...")
        analysis = DefaultProbabilityAnalysis(data_path)

        # Step 1: Load and preprocess data
        print("\nLoading and preprocessing data...")
        df = analysis.load_and_preprocess_data()
        print(f"   Dataset shape: {df.shape}")
        print(f"   Number of features: {len(analysis.feature_cols)}")
        print(f"   Default rate: {df['target'].mean():.2%}")

        # Step 2: Exploratory analysis
        print("\nPerforming exploratory analysis...")
        analysis.exploratory_analysis()

        # Step 3: Implement surrogate models (Project Step 1)
        print("\nImplementing surrogate models...")
        lr_surrogate, dt_surrogate, linear_importance, tree_importance = analysis.implement_surrogate_models()

        print("\n   SURROGATE MODELS SUMMARY:")
        print("   - Linear Regression: Global interpretability via coefficients")
        print("   - Decision Tree: Rule-based interpretability")

        print(f"\n   Top 5 Linear Model Features:")
        for i, (_, row) in enumerate(linear_importance.head().iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

        print(f"\n   Top 5 Decision Tree Features:")
        for i, (_, row) in enumerate(tree_importance.head().iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

        # Step 4: Build black-box models (Project Step 2)
        print("\nBuilding black-box models...")
        rf_model, lr_model, rf_importance = analysis.build_blackbox_model()

        print("\n   BLACK-BOX MODELS SUMMARY:")
        print("   - Random Forest: High predictive power ensemble")
        print("   - Logistic Regression: Baseline linear classifier")

        print(f"\n   Top 5 Random Forest Features:")
        for i, (_, row) in enumerate(rf_importance.head().iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f}")

        # Step 5: Feature importance comparison
        print("\nAnalyzing feature importance across models...")

        top_linear = set(linear_importance.head(10)['feature'])
        top_tree = set(tree_importance.head(10)['feature'])
        top_rf = set(rf_importance.head(10)['feature'])

        common_features = top_linear.intersection(top_tree).intersection(top_rf)
        two_models = (top_linear.intersection(top_tree).union(
                      top_linear.intersection(top_rf)).union(
                      top_tree.intersection(top_rf)))

        print(f"\n   Features important across ALL models ({len(common_features)}):")
        for feature in sorted(common_features):
            print(f"   * {feature}")

        print(f"\n   Features important in at least 2 models ({len(two_models)}):")
        for feature in sorted(two_models - common_features):
            print(f"   - {feature}")

        # Step 6: Generate summary report
        print("\nGenerating summary report...")
        analysis.generate_summary_report()

        # Step 7: Model validation
        print("\nValidating models...")
        test_features = analysis.df[analysis.feature_cols].iloc[:3]
        test_features_scaled = analysis.scaler.transform(test_features)

        lr_pred = lr_surrogate.predict(test_features_scaled)
        dt_pred = dt_surrogate.predict(test_features)
        rf_pred = rf_model.predict(test_features)
        lr_pred_bb = lr_model.predict(test_features_scaled)

        print("   Sample predictions (first 3 observations):")
        print(f"   Linear Surrogate (DP): {lr_pred}")
        print(f"   Tree Surrogate (DP): {dt_pred}")
        print(f"   Random Forest (Default): {rf_pred}")
        print(f"   Logistic Regression (Default): {lr_pred_bb}")

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

        print("\nSUMMARY:")
        print("Step 1: Surrogate models implemented for DP interpretation")
        print("Step 2: Black-box models built for default forecasting")
        print("Check the 'plots' directory for generated visualizations")
        print("Check 'analysis_summary.txt' for detailed results")

        return 0

    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)