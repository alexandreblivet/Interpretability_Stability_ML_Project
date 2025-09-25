import os
import warnings
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import re
from skrub import TableVectorizer


def _ensure_plots_dir(path: str = "plots") -> str:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


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
    exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    if not feature_cols:
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

        if not clean_name or clean_name.isdigit():
            clean_name = f'feature_{i}'

        # Garantir l'unicité
        original_name = clean_name
        counter = 1
        while clean_name in seen:
            clean_name = f"{original_name}_{counter}"
            counter += 1

        seen.add(clean_name)
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
