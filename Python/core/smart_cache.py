"""
Système de cache intelligent et prédiction ML pour MacForge3D.
"""

import os
import json
import hashlib
import logging
import numpy as np
import pickle
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées."""
    key: str
    result: Any
    parameters: Dict
    creation_time: datetime
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    computation_time: float = 0.0
    hash_value: str = ""

class SmartCache:
    """Cache intelligent avec prédiction et optimisation."""
    
    def __init__(
        self,
        base_path: str,
        max_size_gb: float = 10.0,
        max_entries: int = 1000,
        min_hit_rate: float = 0.1
    ):
        self.base_path = base_path
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.max_entries = max_entries
        self.min_hit_rate = min_hit_rate
        
        # Créer les dossiers
        self.cache_dir = os.path.join(base_path, "cache")
        self.meta_dir = os.path.join(base_path, "metadata")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        
        # Charger les métadonnées
        self.metadata: Dict[str, CacheEntry] = {}
        self._load_metadata()
        
        # Statistiques
        self.hits = 0
        self.misses = 0
        
        # Prédicteur de temps
        self.time_predictor = None
        self.time_scaler = StandardScaler()
        self._load_predictor()
        
    def get(self, parameters: Dict[str, Any]) -> Optional[Any]:
        """Récupère un résultat du cache."""
        key = self._compute_key(parameters)
        
        if key in self.metadata:
            entry = self.metadata[key]
            
            # Vérifier le hash
            current_hash = self._compute_hash(
                os.path.join(self.cache_dir, f"{key}.pkl")
            )
            if current_hash != entry.hash_value:
                logger.warning(f"Hash mismatch for {key}, invalidating cache")
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Mettre à jour les statistiques
            entry.access_count += 1
            entry.last_access = datetime.now()
            self._save_metadata()
            
            # Charger le résultat
            try:
                with open(os.path.join(self.cache_dir, f"{key}.pkl"), 'rb') as f:
                    self.hits += 1
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement du cache: {str(e)}")
                self._remove_entry(key)
                self.misses += 1
                return None
        
        self.misses += 1
        return None
        
    def put(
        self,
        parameters: Dict[str, Any],
        result: Any,
        computation_time: float
    ):
        """Stocke un résultat dans le cache."""
        key = self._compute_key(parameters)
        
        try:
            # Sauvegarder le résultat
            result_path = os.path.join(self.cache_dir, f"{key}.pkl")
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Créer l'entrée
            size = os.path.getsize(result_path)
            hash_value = self._compute_hash(result_path)
            
            entry = CacheEntry(
                key=key,
                result=result,
                parameters=parameters,
                creation_time=datetime.now(),
                size_bytes=size,
                computation_time=computation_time,
                hash_value=hash_value
            )
            
            # Nettoyer si nécessaire
            self._cleanup_if_needed(size)
            
            # Sauvegarder les métadonnées
            self.metadata[key] = entry
            self._save_metadata()
            
            # Mettre à jour le prédicteur
            self._update_predictor(parameters, computation_time)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise en cache: {str(e)}")
            if os.path.exists(result_path):
                os.remove(result_path)
                
    def predict_computation_time(
        self,
        parameters: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Prédit le temps de calcul et l'intervalle de confiance."""
        if not self.time_predictor or len(self.metadata) < 10:
            # Pas assez de données, utiliser une estimation basique
            return self._basic_time_estimate(parameters), 0.5
            
        try:
            # Préparer les features
            features = self._extract_features(parameters)
            features_scaled = self.time_scaler.transform([features])
            
            # Prédire
            pred = self.time_predictor.predict(features_scaled)[0]
            
            # Estimer l'incertitude (std des prédictions individuelles)
            predictions = np.array([
                estimator.predict(features_scaled)[0]
                for estimator in self.time_predictor.estimators_
            ])
            uncertainty = np.std(predictions)
            
            return max(0, pred), min(1.0, uncertainty)
            
        except Exception as e:
            logger.error(f"Erreur de prédiction: {str(e)}")
            return self._basic_time_estimate(parameters), 1.0
            
    def get_statistics(self) -> Dict:
        """Retourne les statistiques du cache."""
        total_size = sum(entry.size_bytes for entry in self.metadata.values())
        return {
            'entries': len(self.metadata),
            'size_mb': total_size / 1024 / 1024,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'predictor_trained': self.time_predictor is not None
        }
        
    def _compute_key(self, parameters: Dict[str, Any]) -> str:
        """Calcule la clé de cache pour des paramètres."""
        # Trier pour assurer la cohérence
        sorted_params = {
            k: parameters[k] for k in sorted(parameters.keys())
        }
        # Créer un hash SHA-256
        return hashlib.sha256(
            json.dumps(sorted_params).encode()
        ).hexdigest()
        
    def _compute_hash(self, file_path: str) -> str:
        """Calcule le hash d'un fichier."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for block in iter(lambda: f.read(65536), b''):
                sha256.update(block)
        return sha256.hexdigest()
        
    def _cleanup_if_needed(self, new_size: int):
        """Nettoie le cache si nécessaire."""
        current_size = sum(entry.size_bytes for entry in self.metadata.values())
        
        while (
            len(self.metadata) >= self.max_entries or
            current_size + new_size > self.max_size_bytes
        ):
            # Trouver l'entrée la moins utile
            if not self.metadata:
                break
                
            key_to_remove = min(
                self.metadata.keys(),
                key=lambda k: self._compute_utility(self.metadata[k])
            )
            
            # Supprimer l'entrée
            removed_size = self.metadata[key_to_remove].size_bytes
            self._remove_entry(key_to_remove)
            current_size -= removed_size
            
    def _compute_utility(self, entry: CacheEntry) -> float:
        """Calcule l'utilité d'une entrée de cache."""
        age = (datetime.now() - entry.creation_time).total_seconds()
        frequency = entry.access_count / max(1, age / 86400)  # accès par jour
        size_factor = 1.0 / (1 + entry.size_bytes / 1024 / 1024)  # pénalité taille
        
        return frequency * size_factor
        
    def _remove_entry(self, key: str):
        """Supprime une entrée du cache."""
        try:
            # Supprimer le fichier
            file_path = os.path.join(self.cache_dir, f"{key}.pkl")
            if os.path.exists(file_path):
                os.remove(file_path)
            
            # Supprimer les métadonnées
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
                
        except Exception as e:
            logger.error(f"Erreur lors de la suppression: {str(e)}")
            
    def _save_metadata(self):
        """Sauvegarde les métadonnées."""
        try:
            with open(os.path.join(self.meta_dir, "metadata.pkl"), 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            logger.error(f"Erreur de sauvegarde des métadonnées: {str(e)}")
            
    def _load_metadata(self):
        """Charge les métadonnées."""
        try:
            meta_path = os.path.join(self.meta_dir, "metadata.pkl")
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    self.metadata = pickle.load(f)
        except Exception as e:
            logger.error(f"Erreur de chargement des métadonnées: {str(e)}")
            self.metadata = {}
            
    def _extract_features(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Extrait les features pour la prédiction."""
        features = []
        
        # Ajouter les paramètres numériques
        for key in ['mesh_resolution', 'temperature', 'pressure']:
            features.append(float(parameters.get(key, 0)))
            
        # Encoder les paramètres catégoriels
        material_map = {'PLA': 0, 'ABS': 1, 'PETG': 2}
        material = parameters.get('material', 'PLA')
        features.append(material_map.get(material, -1))
        
        return np.array(features)
        
    def _basic_time_estimate(self, parameters: Dict[str, Any]) -> float:
        """Estimation basique du temps de calcul."""
        base_time = 1.0  # seconde
        
        # Facteurs multiplicateurs
        mesh_factor = parameters.get('mesh_resolution', 10000) / 10000
        material_factor = 1.2 if parameters.get('material') in ['ABS', 'PETG'] else 1.0
        
        return base_time * mesh_factor * material_factor
        
    def _update_predictor(
        self,
        parameters: Dict[str, Any],
        computation_time: float
    ):
        """Met à jour le prédicteur de temps."""
        try:
            # Collecter les données d'entraînement
            X, y = [], []
            for entry in self.metadata.values():
                try:
                    features = self._extract_features(entry.parameters)
                    X.append(features)
                    y.append(entry.computation_time)
                except Exception:
                    continue
                    
            if len(X) < 10:  # Pas assez de données
                return
                
            X = np.array(X)
            y = np.array(y)
            
            # Normaliser
            X_scaled = self.time_scaler.fit_transform(X)
            
            # Entraîner le modèle
            self.time_predictor = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
            self.time_predictor.fit(X_scaled, y)
            
            # Sauvegarder le modèle
            joblib.dump(
                self.time_predictor,
                os.path.join(self.meta_dir, "predictor.joblib")
            )
            joblib.dump(
                self.time_scaler,
                os.path.join(self.meta_dir, "scaler.joblib")
            )
            
        except Exception as e:
            logger.error(f"Erreur d'entraînement du prédicteur: {str(e)}")
            
    def _load_predictor(self):
        """Charge le prédicteur de temps."""
        try:
            predictor_path = os.path.join(self.meta_dir, "predictor.joblib")
            scaler_path = os.path.join(self.meta_dir, "scaler.joblib")
            
            if os.path.exists(predictor_path) and os.path.exists(scaler_path):
                self.time_predictor = joblib.load(predictor_path)
                self.time_scaler = joblib.load(scaler_path)
                
        except Exception as e:
            logger.error(f"Erreur de chargement du prédicteur: {str(e)}")
            self.time_predictor = None
