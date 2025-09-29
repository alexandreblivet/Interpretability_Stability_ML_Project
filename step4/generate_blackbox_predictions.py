#!/usr/bin/env python3
"""
Script pour générer les prédictions du black-box XGBoost
et les sauvegarder dans un format compatible avec le notebook surrogate
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_xgboost_pipeline():
    """Charge le pipeline XGBoost en gérant les incompatibilités de version"""
    try:
        # Essayer de charger directement
        pipeline = joblib.load('../models/xgboost_black_box_pipeline_vf.pkl')
        return pipeline
    except Exception as e:
        print(f"Erreur lors du chargement du pipeline: {e}")
        print("Tentative de chargement avec gestion d'erreur...")
        
        # Essayer de charger avec pickle directement
        import pickle
        try:
            with open('../models/xgboost_black_box_pipeline_vf.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            return pipeline
        except Exception as e2:
            print(f"Erreur pickle: {e2}")
            return None

def generate_predictions():
    """Génère les prédictions du black-box"""
    print("Chargement des données...")
    df = pd.read_csv('../data/dataproject2025.csv')
    print(f"Données chargées: {df.shape}")
    
    print("Chargement du pipeline XGBoost...")
    pipeline = load_xgboost_pipeline()
    
    if pipeline is None:
        print("❌ Impossible de charger le pipeline XGBoost")
        return None, None
    
    print(f"✅ Pipeline chargé: {type(pipeline)}")
    print(f"Étapes du pipeline: {[step[0] for step in pipeline.steps]}")
    
    print("Préparation des données (même preprocessing que l'entraînement)...")
    try:
        # Utiliser exactement le même preprocessing que dans steps_2_&_3_vf.py
        exclude_cols = ['Unnamed: 0', 'Predictions', 'Predicted probabilities', 'target']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        X_raw = df[feature_cols].copy()
        
        # Clean data for TableVectorizer preprocessing (comme dans l'entraînement)
        df_clean = df.dropna()
        X_raw_clean = df_clean[feature_cols].copy()
        
        print(f"Données nettoyées: {X_raw_clean.shape}")
        print("Génération des prédictions...")
        
        # Générer les prédictions avec le pipeline complet
        y_bb = pipeline.predict_proba(X_raw_clean)[:, 1]  # Probabilité de la classe positive
        print(f"✅ Prédictions générées: {y_bb.shape}")
        print(f"Stats des prédictions: min={y_bb.min():.4f}, max={y_bb.max():.4f}, mean={y_bb.mean():.4f}")
        
        return X_raw_clean, y_bb
        
    except Exception as e:
        print(f"❌ Erreur lors de la génération des prédictions: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_predictions(df, y_bb):
    """Sauvegarde les prédictions dans un fichier CSV"""
    # Créer un DataFrame avec les features et les prédictions
    df_with_preds = df.copy()
    df_with_preds['y_pred_bb'] = y_bb
    
    # Sauvegarder
    output_path = 'black_box_with_preds.csv'
    df_with_preds.to_csv(output_path, index=False)
    print(f"✅ Prédictions sauvegardées dans: {output_path}")
    print(f"Shape du fichier: {df_with_preds.shape}")
    
    return output_path

if __name__ == "__main__":
    print("=" * 60)
    print("GÉNÉRATION DES PRÉDICTIONS BLACK-BOX")
    print("=" * 60)
    
    # Générer les prédictions
    df, y_bb = generate_predictions()
    
    if df is not None and y_bb is not None:
        # Sauvegarder
        output_path = save_predictions(df, y_bb)
        print(f"\n✅ Succès! Utilisez maintenant: {output_path}")
    else:
        print("\n❌ Échec de la génération des prédictions")
