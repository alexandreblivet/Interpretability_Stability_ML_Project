#!/usr/bin/env python3
"""
Partial Dependence Plot (PDP) Implementation for Global Interpretability
Step 5: PDP Analysis for XGBoost Model with dataproject2025 dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class PDPAnalysisXGBoost:
    def __init__(self, model_path=None, data_path=None):
        """Initialize PDP Analysis for XGBoost model"""
        self.model_path = model_path or '../models/xgboost_black_box_model.pkl'
        self.data_path = data_path or '../data/dataproject2025 (5).csv'
        self.model = None
        self.df = None
        self.feature_cols = None
        self.le_dict = {}

    def load_xgboost_model(self):
        """Load the XGBoost model or create alternative model"""
        model_paths = [
            '../models/xgboost_black_box_model.pkl',
            '../models/xgb_default_model.pkl'
        ]

        for path in model_paths:
            try:
                print(f"Attempting to load model from: {path}")

                with open(path, 'rb') as f:
                    self.model = pickle.load(f)

                print(f"Model loaded successfully: {type(self.model)}")

                # Verify it's a proper model with predict methods
                if hasattr(self.model, 'predict') and hasattr(self.model, 'predict_proba'):
                    print("Model has both predict and predict_proba methods - Good for PDP!")
                    return True
                elif hasattr(self.model, 'predict'):
                    print("Model has predict method - Can work with PDP")
                    return True
                else:
                    print("Model doesn't have required prediction methods")
                    continue

            except Exception as e:
                print(f"Error loading from {path}: {e}")
                continue

        # If XGBoost models fail to load, create a surrogate model for PDP analysis
        print("XGBoost models failed to load. Creating surrogate model for PDP analysis...")
        return self.create_surrogate_model()

    def create_surrogate_model(self):
        """Create a surrogate model using existing data for PDP analysis"""
        try:
            print("Creating surrogate Random Forest model...")

            # Load data first
            if not self.load_and_preprocess_data():
                return False

            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Sample data if it's too large (>100k rows)
            if len(self.df) > 100000:
                print(f"Dataset is large ({len(self.df)} rows). Sampling 50,000 rows for efficiency...")
                self.df = self.df.sample(n=50000, random_state=42)

            # Use existing predictions if available, otherwise use target
            if 'Predicted probabilities' in self.df.columns:
                # Use predicted probabilities as target (surrogate approach)
                y = (self.df['Predicted probabilities'] > 0.5).astype(int)
                print("Using existing predicted probabilities as target for surrogate model")
            elif 'target' in self.df.columns:
                y = self.df['target']
                print("Using actual target for surrogate model")
            else:
                print("No target variable found!")
                return False

            X = self.df[self.feature_cols]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Create and train smaller Random Forest model for efficiency
            self.model = RandomForestClassifier(
                n_estimators=50,  # Reduced for speed
                max_depth=10,     # Reduced for speed
                random_state=42,
                n_jobs=-1
            )

            print("Training surrogate Random Forest model...")
            self.model.fit(X_train, y_train)

            # Evaluate
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)

            print(f"Surrogate model created successfully!")
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Test accuracy: {test_score:.4f}")

            return True

        except Exception as e:
            print(f"Error creating surrogate model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_and_preprocess_data(self):
        """Load and preprocess the dataproject2025 dataset"""
        print(f"Loading dataset: {self.data_path}")

        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded successfully: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")

            # Identify feature columns (exclude target and prediction columns if they exist)
            exclude_cols = ['target', 'Predictions', 'Predicted probabilities']
            self.feature_cols = [col for col in self.df.columns if col not in exclude_cols and not col.startswith('Unnamed')]

            print(f"Feature columns identified: {len(self.feature_cols)}")
            print(f"Features: {self.feature_cols[:10]}...")  # Show first 10

            # Handle categorical variables with label encoding
            categorical_cols = []
            for col in self.feature_cols:
                if self.df[col].dtype == 'object':
                    categorical_cols.append(col)

            print(f"Categorical columns to encode: {categorical_cols}")

            for col in categorical_cols:
                try:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.le_dict[col] = le
                    print(f"Encoded {col}: {len(le.classes_)} unique values")
                except Exception as e:
                    print(f"Error encoding {col}: {e}")
                    # Remove problematic columns
                    self.feature_cols.remove(col)

            # Handle missing values
            print("Handling missing values...")
            for col in self.feature_cols:
                if self.df[col].isnull().sum() > 0:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(0, inplace=True)

            print(f"Final feature count: {len(self.feature_cols)}")
            return True

        except Exception as e:
            print(f"Error loading/preprocessing data: {e}")
            return False

    def identify_important_features(self, top_k=10):
        """Identify most important features for PDP analysis"""
        print(f"Identifying top {top_k} important features...")

        # Method 1: Use model's feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            try:
                feature_importance = pd.DataFrame({
                    'feature': self.feature_cols,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)

                print("Top features by XGBoost importance:")
                print(feature_importance.head(top_k))
                return feature_importance.head(top_k)['feature'].tolist()

            except Exception as e:
                print(f"Error using model importance: {e}")

        # Method 2: Use correlation with target if available
        if 'target' in self.df.columns:
            try:
                correlations = []
                target = self.df['target']

                for col in self.feature_cols:
                    if col in self.df.columns:
                        corr = abs(self.df[col].corr(target))
                        if not np.isnan(corr):
                            correlations.append((col, corr))

                correlations.sort(key=lambda x: x[1], reverse=True)
                top_features = [feat for feat, _ in correlations[:top_k]]

                print("Top features by correlation with target:")
                for i, (feat, corr) in enumerate(correlations[:top_k], 1):
                    print(f"{i}. {feat}: {corr:.4f}")

                return top_features

            except Exception as e:
                print(f"Error using correlation method: {e}")

        # Method 3: Use variance as fallback
        print("Using variance-based selection...")
        numerical_features = []
        for col in self.feature_cols:
            if self.df[col].dtype in ['int64', 'float64']:
                var = self.df[col].var()
                if not np.isnan(var) and var > 0:
                    numerical_features.append((col, var))

        numerical_features.sort(key=lambda x: x[1], reverse=True)
        return [feat for feat, _ in numerical_features[:top_k]]

    def calculate_pdp_manual(self, feature, grid_resolution=30, sample_size=5000):
        """Calculate PDP manually for a feature (optimized for large datasets)"""
        print(f"Calculating PDP for feature: {feature}")

        # Sample data for PDP calculation if dataset is large
        df_sample = self.df
        if len(self.df) > sample_size:
            df_sample = self.df.sample(n=sample_size, random_state=42)
            print(f"Using {sample_size} samples for PDP calculation")

        # Get feature values and create grid
        feature_values = df_sample[feature].values
        feature_min, feature_max = feature_values.min(), feature_values.max()

        # Create grid
        if len(np.unique(feature_values)) <= 15:  # Categorical/discrete
            grid = np.unique(feature_values)
        else:
            grid = np.linspace(feature_min, feature_max, grid_resolution)

        # Prepare data for predictions
        X = df_sample[self.feature_cols].copy()
        pdp_values = []

        # Calculate PDP values with progress tracking
        total_grid = len(grid)
        for i, grid_value in enumerate(grid):
            if i % 5 == 0:  # Progress tracking
                print(f"  Progress: {i+1}/{total_grid}")

            X_modified = X.copy()
            X_modified[feature] = grid_value

            # Get predictions
            try:
                if hasattr(self.model, 'predict_proba'):
                    predictions = self.model.predict_proba(X_modified)
                    if predictions.shape[1] > 1:
                        predictions = predictions[:, 1]  # Get positive class probabilities
                    else:
                        predictions = predictions[:, 0]
                else:
                    predictions = self.model.predict(X_modified)

                pdp_values.append(np.mean(predictions))

            except Exception as e:
                print(f"Error predicting for {feature}={grid_value}: {e}")
                pdp_values.append(np.nan)

        return grid, np.array(pdp_values)

    def create_individual_pdp_plots(self, features, save_plots=True):
        """Create individual PDP plots for each feature with enhanced styling"""
        print(f"Creating individual PDP plots for {len(features)} features")

        n_features = len(features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols

        # Enhanced figure size and styling
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        fig.suptitle('Partial Dependence Plots - Individual Features', fontsize=20, fontweight='bold', y=0.98)

        # Handle single subplot case
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_features == 1 else list(axes)
        else:
            axes = axes.flatten()

        for i, feature in enumerate(features):
            ax = axes[i]

            try:
                # Calculate PDP
                grid, pdp_values = self.calculate_pdp_manual(feature)

                # Remove NaN values
                valid_mask = ~np.isnan(pdp_values)
                grid_clean = grid[valid_mask]
                pdp_clean = pdp_values[valid_mask]

                if len(pdp_clean) > 0:
                    # Plot PDP with enhanced styling
                    ax.plot(grid_clean, pdp_clean, color='#2E86AB', linewidth=3, alpha=0.9, label='PDP', marker='o', markersize=4, markevery=max(1, len(grid_clean)//20))
                    ax.fill_between(grid_clean, pdp_clean, alpha=0.2, color='#2E86AB')
                    ax.set_xlabel(feature, fontsize=12, fontweight='bold')
                    ax.set_ylabel('Partial Dependence', fontsize=12, fontweight='bold')
                    ax.set_title(f'PDP: {feature}', fontsize=14, fontweight='bold', pad=15)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    # Add feature distribution with better styling
                    ax2 = ax.twinx()
                    ax2.hist(self.df[feature], bins=40, alpha=0.25, color='#A23B72', density=True, edgecolor='white', linewidth=0.5)
                    ax2.set_ylabel('Feature Distribution (Density)', alpha=0.8, color='#A23B72', fontsize=10, fontweight='bold')
                    ax2.tick_params(axis='y', labelcolor='#A23B72', labelsize=9)
                    ax2.spines['top'].set_visible(False)

                else:
                    ax.text(0.5, 0.5, f'No valid data for {feature}',
                           transform=ax.transAxes, ha='center', va='center')

            except Exception as e:
                print(f"Error plotting PDP for {feature}: {e}")
                ax.text(0.5, 0.5, f'Error: {str(e)[:50]}...',
                       transform=ax.transAxes, ha='center', va='center')

        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        if save_plots:
            plt.savefig('individual_pdp_plots.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print("Enhanced individual PDP plots saved as 'individual_pdp_plots.png'")

        plt.show()
        return fig

    def create_sklearn_pdp_plots(self, features, save_plots=True):
        """Create enhanced PDP plots using sklearn's PartialDependenceDisplay"""
        print(f"Creating enhanced sklearn PDP plots for {len(features)} features")

        try:
            X = self.df[self.feature_cols]

            # Limit features for readability
            features_to_plot = features[:6]

            # Enhanced styling
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(2, 3, figsize=(20, 14))
            ax = ax.flatten()

            display = PartialDependenceDisplay.from_estimator(
                self.model,
                X,
                features=features_to_plot,
                grid_resolution=60,
                ax=ax,
                line_kw={'color': '#2E86AB', 'linewidth': 3},
                ice_lines_kw={'alpha': 0.1, 'linewidth': 0.5}
            )

            display.figure_.suptitle('Partial Dependence Plots (sklearn) - XGBoost Model', fontsize=20, fontweight='bold', y=0.98)

            # Enhanced individual axis styling
            for i, axis in enumerate(ax[:len(features_to_plot)]):
                axis.grid(True, alpha=0.3, linestyle='--')
                axis.spines['top'].set_visible(False)
                axis.spines['right'].set_visible(False)
                axis.set_xlabel(axis.get_xlabel(), fontsize=12, fontweight='bold')
                axis.set_ylabel(axis.get_ylabel(), fontsize=12, fontweight='bold')
                axis.set_title(axis.get_title(), fontsize=14, fontweight='bold')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.subplots_adjust(hspace=0.25, wspace=0.25)

            if save_plots:
                plt.savefig('sklearn_pdp_plots.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print("Enhanced sklearn PDP plots saved as 'sklearn_pdp_plots.png'")

            plt.show()
            return display

        except Exception as e:
            print(f"Error creating sklearn PDP plots: {e}")
            return None

    def create_2d_pdp_plot(self, feature1, feature2, save_plots=True):
        """Create 2D PDP interaction plot"""
        print(f"Creating 2D PDP interaction plot: {feature1} vs {feature2}")

        try:
            X = self.df[self.feature_cols]

            # Get feature indices
            feat1_idx = self.feature_cols.index(feature1)
            feat2_idx = self.feature_cols.index(feature2)

            pd_results = partial_dependence(
                self.model,
                X,
                features=[(feat1_idx, feat2_idx)],
                grid_resolution=40,
                kind='average'
            )

            # Create enhanced 2D plot
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, ax = plt.subplots(figsize=(12, 10))

            XX, YY = np.meshgrid(pd_results['grid'][0], pd_results['grid'][1])
            Z = pd_results['average'][0].T

            contour = ax.contourf(XX, YY, Z, levels=25, cmap='RdYlBu_r', alpha=0.9)
            contour_lines = ax.contour(XX, YY, Z, levels=15, colors='black', alpha=0.4, linewidths=0.8)
            ax.clabel(contour_lines, inline=True, fontsize=9, fmt='%.3f')

            cbar = plt.colorbar(contour, ax=ax, label='Partial Dependence', shrink=0.8)
            cbar.ax.tick_params(labelsize=10)
            cbar.set_label('Partial Dependence', fontsize=12, fontweight='bold')

            ax.set_xlabel(feature1, fontsize=14, fontweight='bold')
            ax.set_ylabel(feature2, fontsize=14, fontweight='bold')
            ax.set_title(f'2D PDP Interaction: {feature1} vs {feature2}', fontsize=16, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='--')

            if save_plots:
                filename = f'2d_pdp_{feature1}_vs_{feature2}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"Enhanced 2D PDP plot saved as '{filename}'")

            plt.show()
            return fig

        except Exception as e:
            print(f"Error creating 2D PDP plot: {e}")
            return None

    def analyze_pdp_insights(self, features):
        """Analyze PDP results for insights"""
        print("\n" + "="*60)
        print("PDP ANALYSIS INSIGHTS")
        print("="*60)

        insights = {}

        for feature in features:
            try:
                grid, pdp_values = self.calculate_pdp_manual(feature)

                # Remove NaN values
                valid_mask = ~np.isnan(pdp_values)
                if valid_mask.sum() == 0:
                    continue

                grid_clean = grid[valid_mask]
                pdp_clean = pdp_values[valid_mask]

                # Calculate insights
                pdp_range = pdp_clean.max() - pdp_clean.min()
                pdp_mean = pdp_clean.mean()

                max_impact_idx = np.argmax(pdp_clean)
                min_impact_idx = np.argmin(pdp_clean)

                max_impact_value = grid_clean[max_impact_idx]
                min_impact_value = grid_clean[min_impact_idx]

                # Check monotonicity
                diff = np.diff(pdp_clean)
                monotonic_increasing = np.all(diff >= -1e-6)
                monotonic_decreasing = np.all(diff <= 1e-6)

                insights[feature] = {
                    'range': pdp_range,
                    'mean_effect': pdp_mean,
                    'max_impact_value': max_impact_value,
                    'max_impact_effect': pdp_clean[max_impact_idx],
                    'min_impact_value': min_impact_value,
                    'min_impact_effect': pdp_clean[min_impact_idx],
                    'monotonic_increasing': monotonic_increasing,
                    'monotonic_decreasing': monotonic_decreasing
                }

                print(f"\n{feature}:")
                print(f"  Effect Range: {pdp_range:.6f}")
                print(f"  Max Effect at {feature}={max_impact_value:.2f}: {pdp_clean[max_impact_idx]:.6f}")
                print(f"  Min Effect at {feature}={min_impact_value:.2f}: {pdp_clean[min_impact_idx]:.6f}")

                if monotonic_increasing:
                    print(f"  Relationship: Monotonic Increasing")
                elif monotonic_decreasing:
                    print(f"  Relationship: Monotonic Decreasing")
                else:
                    print(f"  Relationship: Non-monotonic")

            except Exception as e:
                print(f"Error analyzing {feature}: {e}")
                continue

        return insights

    def create_summary_plot(self, features, insights, save_plots=True):
        """Create a summary plot showing feature importance vs PDP range"""
        print("Creating PDP summary visualization...")

        try:
            # Extract data for summary plot
            feature_names = []
            pdp_ranges = []
            mean_effects = []

            for feature in features:
                if feature in insights:
                    feature_names.append(feature)
                    pdp_ranges.append(insights[feature]['range'])
                    mean_effects.append(insights[feature]['mean_effect'])

            if not feature_names:
                print("No valid insights for summary plot")
                return None

            # Create summary plot
            plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Plot 1: PDP Range (Feature Importance)
            bars1 = ax1.barh(feature_names, pdp_ranges, color='#2E86AB', alpha=0.8, edgecolor='white', linewidth=1)
            ax1.set_xlabel('PDP Range (Effect Size)', fontsize=12, fontweight='bold')
            ax1.set_title('Feature Impact on Model Predictions', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars1, pdp_ranges)):
                ax1.text(val + max(pdp_ranges)*0.01, i, f'{val:.4f}',
                        va='center', ha='left', fontsize=10)

            # Plot 2: Mean Effects
            colors = ['#A23B72' if x >= 0 else '#F18F01' for x in mean_effects]
            bars2 = ax2.barh(feature_names, mean_effects, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax2.set_xlabel('Mean PDP Effect', fontsize=12, fontweight='bold')
            ax2.set_title('Average Feature Effects', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars2, mean_effects)):
                offset = max(abs(min(mean_effects)), abs(max(mean_effects))) * 0.02
                ha = 'left' if val >= 0 else 'right'
                x_pos = val + offset if val >= 0 else val - offset
                ax2.text(x_pos, i, f'{val:.4f}', va='center', ha=ha, fontsize=10)

            plt.suptitle('PDP Analysis Summary - Feature Effects Overview', fontsize=16, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_plots:
                plt.savefig('pdp_summary_plot.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print("PDP summary plot saved as 'pdp_summary_plot.png'")

            plt.show()
            return fig

        except Exception as e:
            print(f"Error creating summary plot: {e}")
            return None

    def run_complete_pdp_analysis(self, top_k_features=8):
        """Run complete PDP analysis workflow"""
        print("="*60)
        print("XGBOOST MODEL - PARTIAL DEPENDENCE PLOT ANALYSIS")
        print("="*60)

        # Step 1: Load XGBoost model
        if not self.load_xgboost_model():
            print("Failed to load XGBoost model. Exiting.")
            return False

        # Step 2: Load and preprocess data
        if not self.load_and_preprocess_data():
            print("Failed to load/preprocess data. Exiting.")
            return False

        # Step 3: Identify important features
        important_features = self.identify_important_features(top_k_features)
        print(f"\nSelected features for PDP analysis: {important_features}")

        if not important_features:
            print("No important features identified. Exiting.")
            return False

        # Step 4: Create enhanced individual PDP plots
        print("\n" + "="*50)
        print("CREATING ENHANCED INDIVIDUAL PDP PLOTS")
        print("="*50)
        self.create_individual_pdp_plots(important_features)

        # Step 5: Create enhanced sklearn PDP plots
        print("\n" + "="*50)
        print("CREATING ENHANCED SKLEARN PDP PLOTS")
        print("="*50)
        self.create_sklearn_pdp_plots(important_features)

        # Step 6: Create enhanced 2D interaction plot
        if len(important_features) >= 2:
            print("\n" + "="*50)
            print("CREATING ENHANCED 2D INTERACTION PLOT")
            print("="*50)
            self.create_2d_pdp_plot(important_features[0], important_features[1])

        # Step 7: Analyze insights
        print("\n" + "="*50)
        print("ANALYZING PDP INSIGHTS")
        print("="*50)
        insights = self.analyze_pdp_insights(important_features)

        # Step 8: Create summary visualization
        print("\n" + "="*50)
        print("CREATING SUMMARY VISUALIZATION")
        print("="*50)
        self.create_summary_plot(important_features, insights)

        # Step 9: Generate report
        self.generate_report(important_features, insights)

        print("\n" + "="*60)
        print("PDP ANALYSIS COMPLETED!")
        print("="*60)
        print("Generated files:")
        print("- individual_pdp_plots.png (Enhanced individual PDP plots)")
        print("- sklearn_pdp_plots.png (Enhanced sklearn PDP plots)")
        print("- 2d_pdp_*.png (Enhanced 2D interaction plots)")
        print("- pdp_summary_plot.png (Feature effects summary visualization)")
        print("- pdp_analysis_report.txt (Comprehensive analysis report)")

        return True

    def generate_report(self, features, insights):
        """Generate comprehensive PDP analysis report"""
        report = f"""
XGBoost Model - PDP Analysis Report
{'='*50}

Model Information:
- Model Type: {type(self.model).__name__}
- Model Path: {self.model_path}
- Dataset: {self.data_path}
- Dataset Shape: {self.df.shape}
- Features Analyzed: {len(features)}
- Features: {', '.join(features)}

Feature Analysis Summary:
{'='*30}
"""

        for feature in features:
            if feature in insights:
                info = insights[feature]
                relationship = ('Monotonic Increasing' if info['monotonic_increasing']
                             else 'Monotonic Decreasing' if info['monotonic_decreasing']
                             else 'Non-monotonic')

                report += f"""
{feature}:
  Effect Range: {info['range']:.6f}
  Most Positive Impact: {info['max_impact_value']:.2f} -> {info['max_impact_effect']:.6f}
  Most Negative Impact: {info['min_impact_value']:.2f} -> {info['min_impact_effect']:.6f}
  Relationship: {relationship}
"""

        report += f"""
Key Insights:
{'='*15}
1. PDP shows average marginal effects of each feature on model predictions
2. Range indicates how much each feature influences predictions
3. Monotonic relationships suggest linear-like effects
4. Non-monotonic relationships indicate complex feature interactions
5. 2D plots reveal feature interactions

Files Generated:
- individual_pdp_plots.png: Enhanced individual PDP for each feature
- sklearn_pdp_plots.png: Enhanced PDP plots using sklearn
- 2d_pdp_*.png: Enhanced feature interaction plots
- pdp_analysis_report.txt: This comprehensive report

Plot Enhancements:
- Enhanced color schemes and styling for better readability
- Feature distribution overlays for context
- Improved grid lines and axis formatting
- Higher resolution (300 DPI) for publication quality
- Better spacing and layout optimization
- Contour lines with labels for 2D plots

Analysis completed successfully.
"""

        # Save report
        with open('pdp_analysis_report.txt', 'w') as f:
            f.write(report)

        print(report)
        return report

def main():
    """Main function to run PDP analysis"""
    print("Starting XGBoost PDP Analysis...")

    # Initialize analyzer
    analyzer = PDPAnalysisXGBoost()

    # Run analysis
    success = analyzer.run_complete_pdp_analysis(top_k_features=8)

    if success:
        print("\nPDP Analysis completed successfully!")
        return 0
    else:
        print("\nPDP Analysis failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())