<<<<<<< HEAD
"""
SHAP Analysis Script for XGBoost Credit Default Model
"""

import os
import warnings
=======
import os
import warnings
import joblib
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
<<<<<<< HEAD
import joblib

warnings.filterwarnings('ignore')
RANDOM_STATE = 42


def ensure_plots_dir(path="plots"):
    """Create plots directory if it doesn't exist"""
=======
import re
from skrub import TableVectorizer


def _ensure_plots_dir(path: str = "plots") -> str:
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


<<<<<<< HEAD
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
=======
def _load_model():
    import xgboost as xgb

    # Essayer d'abord le format JSON natif XGBoost
    json_path = "models/xgboost_black_box_model.json"
    if os.path.exists(json_path):
        model = xgb.XGBClassifier()
        model.load_model(json_path)
        return model, json_path

    # Fallback sur les formats pickle/joblib
    candidates = [
        "models/xgboost_black_box_model.joblib",
        "models/xgb_default_model.pkl",
        "models/xgboost_black_box_model.pkl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p), p

    raise FileNotFoundError(
        "Aucun modèle trouvé. Attendu l'un de: " + json_path + ", " + ", ".join(candidates)
    )


def _load_data():
    candidates = [
        "data/dataproject2025.csv",
        "data/dataproject2025 (1).csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return pd.read_csv(p), p
    raise FileNotFoundError(
        "Aucun fichier de données trouvé. Attendu l'un de: " + ", ".join(candidates)
    )


def _prepare_data_with_tablevectorizer(df: pd.DataFrame):
    """Prépare les données exactement comme dans le notebook avec TableVectorizer"""
    # Colonnes à exclure (exactement comme dans le notebook)
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c
    exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if not feature_cols:
<<<<<<< HEAD
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
=======
        raise ValueError("Aucune colonne de features détectée après exclusion.")

    print(f"Features pour TableVectorizer: {len(feature_cols)} colonnes")
    print(f"Colonnes: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Colonnes: {feature_cols}")

    # Préparer les données brutes
    X_raw = df[feature_cols].copy()

    # Nettoyer les données (supprimer les lignes avec valeurs manquantes)
    df_clean = df.dropna()
    X_raw_clean = df_clean[feature_cols].copy()

    print(f"Dataset original: {len(df)} lignes")
    print(f"Dataset nettoyé: {len(df_clean)} lignes")
    print(f"Lignes supprimées: {len(df) - len(df_clean)}")

    # Initialiser TableVectorizer avec paramètres par défaut
    vectorizer = TableVectorizer()

    # Échantillonner pour performance (comme dans le notebook)
    sample_size = min(50000, len(X_raw_clean))
    if len(X_raw_clean) > sample_size:
        print(f"Échantillonnage: {sample_size} lignes sur {len(X_raw_clean)}")
        sample_indices = X_raw_clean.sample(n=sample_size, random_state=42).index
        X_sample = X_raw_clean.loc[sample_indices]
    else:
        X_sample = X_raw_clean

    # Appliquer TableVectorizer
    print("Application de TableVectorizer...")
    X_vectorized = vectorizer.fit_transform(X_sample)

    print(f"Après vectorisation:")
    print(f"- Features originales: {X_sample.shape[1]}")
    print(f"- Features après preprocessing: {X_vectorized.shape[1]}")
    print(f"- Expansion: {X_vectorized.shape[1] / X_sample.shape[1]:.1f}x")

    # Nettoyer les noms de features (exactement comme dans le notebook)
    raw_feature_names = vectorizer.get_feature_names_out()
    feature_names_clean = _clean_feature_names(raw_feature_names)

    return X_vectorized, feature_names_clean


def _clean_feature_names(names):
    """Nettoie les noms de features et garantit l'unicité"""
    cleaned = []
    seen = set()

    for i, name in enumerate(names):
        # Remplacer les caractères problématiques
        clean_name = re.sub(r'[<>\[\](){}]', '_', str(name))
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', clean_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c

        if not clean_name or clean_name.isdigit():
            clean_name = f'feature_{i}'

<<<<<<< HEAD
        # Ensure uniqueness
=======
        # Garantir l'unicité
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c
        original_name = clean_name
        counter = 1
        while clean_name in seen:
            clean_name = f"{original_name}_{counter}"
            counter += 1

        seen.add(clean_name)
<<<<<<< HEAD
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
=======
        cleaned.append(clean_name)

    return cleaned


def _build_explainer(model, X_for_background: pd.DataFrame):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            background = shap.sample(X_for_background, 100) if len(X_for_background) > 100 else X_for_background
            explainer = shap.Explainer(model, background)
        return explainer
    except Exception:
        # Fallback pour modèles arborescents purs
        return shap.TreeExplainer(model)


def _compute_shap_values(explainer, X, feature_names):
    """Calcule les valeurs SHAP avec les données préprocessées"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Convertir en array numpy si nécessaire
        if hasattr(X, 'toarray'):  # Si c'est une matrice sparse
            X_array = X.toarray()
        else:
            X_array = np.array(X)

        # Échantillonnage pour rapidité si dataset très grand
        sample_size = min(5000, len(X_array))
        if len(X_array) > sample_size:
            print(f"Échantillonnage SHAP: {sample_size} lignes sur {len(X_array)}")
            indices = np.random.choice(len(X_array), sample_size, replace=False)
            X_sample = X_array[indices]  # Indexation numpy correcte
        else:
            X_sample = X_array

        # Convertir en DataFrame avec noms de colonnes nettoyés
        X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

        print(f"Calcul SHAP sur {len(X_sample_df)} échantillons avec {len(feature_names)} features")
        shap_values = explainer(X_sample_df)

    return shap_values, X_sample_df


def _plot_and_save(shap_values, X, out_dir: str):
    # Beeswarm (summary plot)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Bar plot (importance globale moyenne)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Dependence plots pour top features
    try:
        # Identifie les features les plus influentes
        if hasattr(shap_values, "values") and isinstance(shap_values.values, np.ndarray):
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            feature_order = np.argsort(-mean_abs)
            top_idx = feature_order[:5]
            feature_names = X.columns
        else:
            imp = shap.utils.get_feature_importance(shap_values)
            top = imp.sort_values("importance", ascending=False)["feature"].head(5).tolist()
            feature_names = X.columns
            top_idx = [list(feature_names).index(f) for f in top if f in feature_names]

        for i in top_idx:
            fname = X.columns[i]
            plt.figure(figsize=(8, 5))
            shap.plots.scatter(shap_values[:, i], color=shap_values, show=False)
            plt.title(f"Dependence plot: {fname}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"shap_dependence_{fname}.png"), dpi=150, bbox_inches="tight")
            plt.close()
    except Exception:
        pass

    # Force plot / Waterfall pour une observation
    try:
        idx = 0
        sv0 = shap_values[idx]
        # Waterfall (si supporté)
        plt.figure(figsize=(8, 6))
        shap.plots.waterfall(sv0, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_waterfall_sample0.png"), dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        pass


def main():
    plots_dir = _ensure_plots_dir()
    print("Chargement du modèle…")
    model, model_path = _load_model()
    print(f"Modèle chargé depuis: {model_path}")

    print("Chargement des données…")
    df, data_path = _load_data()
    print(f"Données chargées depuis: {data_path}")

    print("Préparation des données avec TableVectorizer (comme dans le notebook)…")
    X_vectorized, feature_names = _prepare_data_with_tablevectorizer(df)

    print("Construction de l'explainer SHAP…")
    # Utiliser un échantillon pour le background du TreeExplainer
    background_sample = X_vectorized[:100] if len(X_vectorized) > 100 else X_vectorized
    if hasattr(background_sample, 'toarray'):
        background_sample = background_sample.toarray()
    background_df = pd.DataFrame(background_sample, columns=feature_names)

    # Utiliser TreeExplainer directement car on a un XGBoost
    explainer = shap.TreeExplainer(model)

    print("Calcul des valeurs SHAP (échantillon)…")
    shap_values, X_sample = _compute_shap_values(explainer, X_vectorized, feature_names)

    print("Génération des visualisations…")
    _plot_and_save(shap_values, X_sample, plots_dir)

    print("Visuels enregistrés dans le dossier 'plots/'.")
    print(f"Analyse SHAP terminée sur {len(X_sample)} échantillons avec {len(feature_names)} features.")


if __name__ == "__main__":
    main()
>>>>>>> 190f71933fde3208f4d401534203d48cac572d5c
