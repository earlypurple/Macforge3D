import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import joblib
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import json
import pandas as pd
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    """Classe pour l'analyse intelligente des résultats de simulation."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialise l'analyseur de résultats.
        
        Args:
            model_path: Chemin vers le modèle entraîné (optionnel)
        """
        self.scaler = StandardScaler()
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        # Charger un modèle pré-entraîné si disponible
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            
        self.feature_names = [
            'max_stress',
            'max_displacement',
            'min_safety_factor',
            'max_temperature',
            'cooling_rate',
            'time_above_glass'
        ]
        
    def preprocess_results(self, results: Dict[str, Any]) -> np.ndarray:
        """
        Prétraite les résultats pour l'analyse.
        
        Args:
            results: Dictionnaire des résultats
            
        Returns:
            np.ndarray: Features normalisés
        """
        features = []
        
        # Extraire les caractéristiques pertinentes
        for feature in self.feature_names:
            value = results.get(feature, 0.0)
            if isinstance(value, (list, dict)):
                # Pour les courbes de température, prendre la moyenne
                if isinstance(value, dict) and 'max_temps' in value:
                    value = np.mean(value['max_temps'])
                else:
                    value = np.mean(value)
            features.append(float(value))
            
        features = np.array(features).reshape(1, -1)
        return self.scaler.transform(features)
        
    def detect_anomalies(self, features: np.ndarray) -> List[str]:
        """
        Détecte les anomalies dans les résultats.
        
        Args:
            features: Features normalisés
            
        Returns:
            List[str]: Liste des anomalies détectées
        """
        predictions = self.anomaly_detector.predict(features)
        anomalies = []
        
        if predictions[0] == -1:  # Anomalie détectée
            # Identifier les features problématiques
            feature_scores = np.abs(features[0])
            for fname, score in zip(self.feature_names, feature_scores):
                if score > 2.0:  # Plus de 2 écarts-types
                    anomalies.append(f"Valeur anormale pour {fname}: {score:.2f}")
                    
        return anomalies
        
    def analyze_results(
        self,
        results: Dict[str, Any],
        material_props: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyse complète des résultats de simulation.
        
        Args:
            results: Résultats de la simulation
            material_props: Propriétés du matériau
            
        Returns:
            Dict[str, Any]: Analyse et recommandations
        """
        try:
            # Prétraiter les résultats
            features = self.preprocess_results(results)
            
            # Détecter les anomalies
            anomalies = self.detect_anomalies(features)
            
            # Initialiser les recommandations
            recommendations = []
            risk_level = "faible"
            
            # Analyse structurelle
            if "min_safety_factor" in results:
                sf = results["min_safety_factor"]
                if sf < 1.2:
                    risk_level = "élevé"
                    recommendations.append(
                        "Facteur de sécurité critique. Renforcer la structure immédiatement."
                    )
                elif sf < 1.5:
                    risk_level = "moyen"
                    recommendations.append(
                        "Facteur de sécurité faible. Considérer un renforcement."
                    )
                    
            # Analyse thermique
            if "max_temperature" in results:
                max_temp = results["max_temperature"]
                melt_temp = material_props.get("melting_point", float('inf'))
                
                if max_temp > melt_temp:
                    risk_level = "élevé"
                    recommendations.append(
                        f"Température maximale ({max_temp:.1f}°C) au-dessus du point "
                        f"de fusion ({melt_temp}°C)"
                    )
                    
            # Analyse du refroidissement
            if "cooling_rate" in results:
                cooling_rate = abs(results["cooling_rate"])
                if cooling_rate > 10.0:
                    risk_level = "élevé"
                    recommendations.append(
                        f"Refroidissement trop rapide ({cooling_rate:.1f}°C/s). "
                        "Risque de déformation."
                    )
                    
            # Ajouter les anomalies détectées aux recommandations
            if anomalies:
                recommendations.extend(anomalies)
                if risk_level == "faible":
                    risk_level = "moyen"
                    
            # Générer des suggestions d'optimisation
            suggestions = self.generate_optimization_suggestions(
                results,
                material_props,
                risk_level
            )
            
            return {
                "risk_level": risk_level,
                "recommendations": recommendations,
                "optimization_suggestions": suggestions,
                "timestamp": datetime.now().isoformat(),
                "anomalies_detected": len(anomalies) > 0
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse des résultats: {str(e)}")
            return {
                "error": str(e),
                "risk_level": "inconnu",
                "recommendations": [
                    "Erreur lors de l'analyse. Vérifier les données d'entrée."
                ]
            }
            
    def generate_optimization_suggestions(
        self,
        results: Dict[str, Any],
        material_props: Dict[str, Any],
        risk_level: str
    ) -> List[str]:
        """
        Génère des suggestions d'optimisation basées sur les résultats.
        
        Args:
            results: Résultats de la simulation
            material_props: Propriétés du matériau
            risk_level: Niveau de risque identifié
            
        Returns:
            List[str]: Suggestions d'optimisation
        """
        suggestions = []
        
        # Suggestions structurelles
        if "max_stress" in results and "yield_strength" in material_props:
            stress_ratio = results["max_stress"] / material_props["yield_strength"]
            
            if stress_ratio > 0.8:
                suggestions.append(
                    "Considérer un matériau plus résistant ou augmenter l'épaisseur"
                )
            elif stress_ratio < 0.2:
                suggestions.append(
                    "Potentiel de réduction de matière pour optimiser le poids"
                )
                
        # Suggestions thermiques
        if "cooling_rate" in results:
            cooling_rate = abs(results["cooling_rate"])
            
            if cooling_rate > 5.0:
                suggestions.append(
                    "Ajouter des supports pour améliorer la dissipation thermique"
                )
                if cooling_rate > 10.0:
                    suggestions.append("Envisager un système de refroidissement actif")
                    
        # Suggestions basées sur le niveau de risque
        if risk_level == "élevé":
            suggestions.append("Effectuer des tests physiques avant la production")
            suggestions.append("Réviser la conception pour améliorer la fiabilité")
            
        return suggestions
        
    def save_model(self, path: str) -> bool:
        """
        Sauvegarde le modèle entraîné.
        
        Args:
            path: Chemin de sauvegarde
            
        Returns:
            bool: True si la sauvegarde réussit
        """
        try:
            model_data = {
                'scaler': self.scaler,
                'anomaly_detector': self.anomaly_detector,
                'classifier': self.classifier,
                'feature_names': self.feature_names
            }
            
            joblib.dump(model_data, path)
            logger.info(f"Modèle sauvegardé: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            return False
            
    def load_model(self, path: str) -> bool:
        """
        Charge un modèle entraîné.
        
        Args:
            path: Chemin vers le modèle
            
        Returns:
            bool: True si le chargement réussit
        """
        try:
            model_data = joblib.load(path)
            
            self.scaler = model_data['scaler']
            self.anomaly_detector = model_data['anomaly_detector']
            self.classifier = model_data['classifier']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Modèle chargé: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {str(e)}")
            return False
            
    def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """
        Entraîne le modèle sur un ensemble de données.
        
        Args:
            training_data: Liste de résultats de simulation
            
        Returns:
            bool: True si l'entraînement réussit
        """
        try:
            # Préparer les données
            X = []
            for data in training_data:
                features = []
                for feature in self.feature_names:
                    value = data.get(feature, 0.0)
                    if isinstance(value, (list, dict)):
                        value = np.mean(value)
                    features.append(float(value))
                X.append(features)
                
            X = np.array(X)
            
            # Entraîner le scaler
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Entraîner le détecteur d'anomalies
            self.anomaly_detector.fit(X_scaled)
            
            logger.info("Entraînement terminé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {str(e)}")
            return False
