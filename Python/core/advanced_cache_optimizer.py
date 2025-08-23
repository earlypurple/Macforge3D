"""
Module d'optimisation avancée du cache pour MacForge3D.
Implémente des stratégies de cache adaptatif et intelligent.
"""

import os
import time
import hashlib
import threading
import numpy as np
import psutil
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import json
import pickle
import gzip
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Métriques de performance du cache."""
    hit_rate: float
    miss_rate: float
    size_mb: float
    items_count: int
    avg_access_time_ms: float
    compression_ratio: float
    memory_efficiency: float

@dataclass
class CacheItem:
    """Item du cache avec métadonnées."""
    key: str
    data: Any
    size_bytes: int
    access_count: int
    last_access: float
    creation_time: float
    ttl: Optional[float]
    priority: int = 1
    
class AdvancedCacheOptimizer:
    """Optimiseur de cache avancé avec stratégies adaptatifs."""
    
    def __init__(
        self,
        max_memory_mb: float = 1024,
        compression_enabled: bool = True,
        adaptive_sizing: bool = True
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_enabled = compression_enabled
        self.adaptive_sizing = adaptive_sizing
        
        # Cache principal
        self._cache: OrderedDict[str, CacheItem] = OrderedDict()
        self._cache_lock = threading.RLock()
        
        # Métriques et statistiques
        self._metrics = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'access_times': [],
            'size_history': []
        }
        
        # Configuration adaptative
        self._adaptive_config = {
            'memory_pressure_threshold': 0.8,
            'compression_threshold_kb': 10,
            'hot_data_threshold': 3,  # Accès minimum pour être "hot"
            'ttl_default': 3600,  # 1 heure
            'cleanup_interval': 300  # 5 minutes
        }
        
        # Prédicteur d'accès
        self._access_predictor = AccessPredictor()
        
        # Thread de nettoyage
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_running = True
        self._cleanup_thread.start()
        
    def put(
        self, 
        key: str, 
        data: Any, 
        ttl: Optional[float] = None,
        priority: int = 1,
        compress: Optional[bool] = None
    ) -> bool:
        """
        Stocke un élément dans le cache avec optimisations.
        
        Args:
            key: Clé du cache
            data: Données à stocker
            ttl: Temps de vie en secondes
            priority: Priorité (1-10, plus élevé = plus important)
            compress: Force la compression si True
            
        Returns:
            True si stocké avec succès
        """
        start_time = time.time()
        
        try:
            with self._cache_lock:
                # Calculer la taille des données
                data_size = self._calculate_size(data)
                
                # Décider de la compression
                should_compress = self._should_compress(data, data_size, compress)
                
                if should_compress:
                    data = self._compress_data(data)
                    compressed_size = self._calculate_size(data)
                    compression_ratio = data_size / max(compressed_size, 1)
                    self._metrics['compressions'] += 1
                    data_size = compressed_size
                else:
                    compression_ratio = 1.0
                
                # Vérifier l'espace disponible et faire de la place si nécessaire
                if not self._ensure_space(data_size):
                    return False
                
                # Créer l'item de cache
                current_time = time.time()
                cache_item = CacheItem(
                    key=key,
                    data=data,
                    size_bytes=data_size,
                    access_count=0,
                    last_access=current_time,
                    creation_time=current_time,
                    ttl=ttl or self._adaptive_config['ttl_default'],
                    priority=priority
                )
                
                # Stocker dans le cache
                if key in self._cache:
                    # Mise à jour d'un élément existant
                    old_item = self._cache[key]
                    self._cache[key] = cache_item
                    self._cache.move_to_end(key)
                else:
                    # Nouvel élément
                    self._cache[key] = cache_item
                
                # Mettre à jour les métriques
                access_time = (time.time() - start_time) * 1000
                self._update_access_metrics(access_time, True)
                
                # Entraîner le prédicteur
                self._access_predictor.record_access(key, current_time)
                
                return True
                
        except Exception as e:
            logger.error(f"Erreur lors du stockage en cache: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Récupère un élément du cache avec optimisations.
        
        Args:
            key: Clé du cache
            
        Returns:
            Données si trouvées, None sinon
        """
        start_time = time.time()
        
        try:
            with self._cache_lock:
                if key not in self._cache:
                    self._metrics['misses'] += 1
                    self._update_access_metrics((time.time() - start_time) * 1000, False)
                    return None
                
                cache_item = self._cache[key]
                
                # Vérifier TTL
                if self._is_expired(cache_item):
                    del self._cache[key]
                    self._metrics['misses'] += 1
                    self._update_access_metrics((time.time() - start_time) * 1000, False)
                    return None
                
                # Mettre à jour les statistiques d'accès
                cache_item.access_count += 1
                cache_item.last_access = time.time()
                
                # Déplacer vers la fin (LRU)
                self._cache.move_to_end(key)
                
                # Décompresser si nécessaire
                data = cache_item.data
                if self._is_compressed(data):
                    data = self._decompress_data(data)
                
                # Mettre à jour les métriques
                self._metrics['hits'] += 1
                access_time = (time.time() - start_time) * 1000
                self._update_access_metrics(access_time, True)
                
                # Enregistrer l'accès pour le prédicteur
                self._access_predictor.record_access(key, time.time())
                
                return data
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du cache: {e}")
            self._metrics['misses'] += 1
            return None
    
    def optimize_cache(self) -> Dict[str, Any]:
        """
        Optimise le cache en fonction des patterns d'utilisation.
        
        Returns:
            Statistiques d'optimisation
        """
        start_time = time.time()
        optimizations_applied = []
        
        try:
            with self._cache_lock:
                original_size = len(self._cache)
                original_memory = self._get_total_size()
                
                # 1. Nettoyer les éléments expirés
                expired_count = self._cleanup_expired()
                if expired_count > 0:
                    optimizations_applied.append(f"expired_cleanup: {expired_count} items")
                
                # 2. Optimiser la compression
                compression_savings = self._optimize_compression()
                if compression_savings > 0:
                    optimizations_applied.append(f"compression_optimized: {compression_savings} bytes saved")
                
                # 3. Réorganiser selon les patterns d'accès
                reorg_count = self._reorganize_by_access_patterns()
                if reorg_count > 0:
                    optimizations_applied.append(f"access_pattern_reorg: {reorg_count} items")
                
                # 4. Ajuster les TTL adaptatifs
                ttl_adjustments = self._adjust_adaptive_ttl()
                if ttl_adjustments > 0:
                    optimizations_applied.append(f"ttl_adjusted: {ttl_adjustments} items")
                
                # 5. Évictions préemptives basées sur prédictions
                preemptive_evictions = self._preemptive_eviction()
                if preemptive_evictions > 0:
                    optimizations_applied.append(f"preemptive_eviction: {preemptive_evictions} items")
                
                final_size = len(self._cache)
                final_memory = self._get_total_size()
                
                optimization_stats = {
                    'duration_ms': round((time.time() - start_time) * 1000, 2),
                    'items_before': original_size,
                    'items_after': final_size,
                    'items_removed': original_size - final_size,
                    'memory_before_mb': round(original_memory / (1024*1024), 2),
                    'memory_after_mb': round(final_memory / (1024*1024), 2),
                    'memory_saved_mb': round((original_memory - final_memory) / (1024*1024), 2),
                    'optimizations_applied': optimizations_applied,
                    'hit_rate': self._calculate_hit_rate(),
                    'compression_ratio': self._calculate_compression_ratio()
                }
                
                logger.info(f"Cache optimisé: {optimization_stats}")
                return optimization_stats
                
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation du cache: {e}")
            return {'error': str(e)}
    
    def _should_compress(self, data: Any, size_bytes: int, force: Optional[bool]) -> bool:
        """Détermine s'il faut compresser les données."""
        if force is not None:
            return force
            
        if not self.compression_enabled:
            return False
            
        # Compresser si taille > seuil et type approprié
        threshold = self._adaptive_config['compression_threshold_kb'] * 1024
        if size_bytes < threshold:
            return False
            
        # Vérifier le type de données
        compressible_types = (str, bytes, list, dict, np.ndarray)
        return isinstance(data, compressible_types)
    
    def _compress_data(self, data: Any) -> bytes:
        """Compresse les données."""
        serialized = pickle.dumps(data)
        compressed = gzip.compress(serialized, compresslevel=6)
        return b'COMPRESSED:' + compressed
    
    def _decompress_data(self, data: bytes) -> Any:
        """Décompresse les données."""
        if data.startswith(b'COMPRESSED:'):
            compressed_data = data[11:]  # Retirer le préfixe
            decompressed = gzip.decompress(compressed_data)
            return pickle.loads(decompressed)
        return data
    
    def _is_compressed(self, data: Any) -> bool:
        """Vérifie si les données sont compressées."""
        return isinstance(data, bytes) and data.startswith(b'COMPRESSED:')
    
    def _calculate_size(self, data: Any) -> int:
        """Calcule la taille des données en bytes."""
        try:
            if isinstance(data, bytes):
                return len(data)
            elif isinstance(data, str):
                return len(data.encode('utf-8'))
            elif isinstance(data, np.ndarray):
                return data.nbytes
            else:
                # Estimation pour les autres types
                return len(pickle.dumps(data))
        except:
            return 1024  # Taille par défaut si impossible à calculer
    
    def _cleanup_expired(self) -> int:
        """Nettoie les éléments expirés."""
        current_time = time.time()
        expired_keys = []
        
        for key, item in self._cache.items():
            if self._is_expired(item):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            
        return len(expired_keys)
    
    def _is_expired(self, item: CacheItem) -> bool:
        """Vérifie si un élément est expiré."""
        if item.ttl is None:
            return False
        return time.time() - item.creation_time > item.ttl
    
    def _ensure_space(self, required_bytes: int) -> bool:
        """S'assure qu'il y a assez d'espace pour les nouvelles données."""
        current_size = self._get_total_size()
        if current_size + required_bytes <= self.max_memory_bytes:
            return True
            
        # Calculer combien d'espace libérer
        space_to_free = (current_size + required_bytes) - self.max_memory_bytes
        return self._evict_items(space_to_free)
    
    def _evict_items(self, space_needed: int) -> bool:
        """Évince des éléments pour libérer de l'espace avec stratégie intelligente."""
        evicted_space = 0
        evicted_count = 0
        
        # Stratégie d'éviction ML-enhanced: LRU avec prédiction d'utilisation future
        items_by_score = []
        current_time = time.time()
        
        for key, item in self._cache.items():
            # Prédire la probabilité d'accès futur
            future_access_prob = self._access_predictor.predict_future_access(key, 3600)
            
            # Score composite avec plusieurs facteurs
            time_since_access = current_time - item.last_access
            access_frequency = self._access_predictor.get_access_frequency(key, 3600)
            size_penalty = np.log(max(item.size_bytes, 1)) / 20.0  # Pénalité pour les gros items
            
            # Score plus bas = plus susceptible d'être évincé
            eviction_score = (
                time_since_access * 0.35 +
                (1 / max(access_frequency, 0.001)) * 0.25 +
                (1 / max(item.priority, 1)) * 0.15 +
                (1 / max(item.access_count, 1)) * 0.10 +
                (1 / max(future_access_prob, 0.001)) * 0.10 +
                size_penalty * 0.05
            )
            
            items_by_score.append((eviction_score, key, item))
        
        # Trier par score d'éviction (descendant)
        items_by_score.sort(reverse=True)
        
        # Éviction en plusieurs passes pour optimiser
        # Passe 1: Évincer les items avec faible probabilité d'accès futur
        for score, key, item in items_by_score:
            if evicted_space >= space_needed:
                break
                
            future_prob = self._access_predictor.predict_future_access(key, 1800)  # 30 min
            if future_prob < 0.1:  # Très faible probabilité d'accès
                evicted_space += item.size_bytes
                evicted_count += 1
                del self._cache[key]
        
        # Passe 2: Si pas assez d'espace, éviction LRU classique
        if evicted_space < space_needed:
            for score, key, item in items_by_score:
                if evicted_space >= space_needed or key not in self._cache:
                    break
                    
                evicted_space += item.size_bytes
                evicted_count += 1
                del self._cache[key]
            
        self._metrics['evictions'] += evicted_count
        logger.debug(f"Évincé {evicted_count} éléments, libéré {evicted_space} bytes")
        
        return evicted_space >= space_needed
    
    def _get_total_size(self) -> int:
        """Calcule la taille totale du cache."""
        return sum(item.size_bytes for item in self._cache.values())
    
    def _update_access_metrics(self, access_time_ms: float, hit: bool):
        """Met à jour les métriques d'accès."""
        self._metrics['access_times'].append(access_time_ms)
        
        # Limiter l'historique des temps d'accès
        if len(self._metrics['access_times']) > 1000:
            self._metrics['access_times'] = self._metrics['access_times'][-500:]
    
    def _optimize_compression(self) -> int:
        """Optimise la compression des éléments existants."""
        bytes_saved = 0
        
        for key, item in list(self._cache.items()):
            if not self._is_compressed(item.data):
                original_size = item.size_bytes
                if self._should_compress(item.data, original_size, None):
                    compressed_data = self._compress_data(item.data)
                    new_size = self._calculate_size(compressed_data)
                    
                    if new_size < original_size * 0.8:  # Au moins 20% de compression
                        item.data = compressed_data
                        item.size_bytes = new_size
                        bytes_saved += (original_size - new_size)
                        
        return bytes_saved
    
    def _reorganize_by_access_patterns(self) -> int:
        """Réorganise le cache selon les patterns d'accès."""
        # Identifier les éléments "hot" et "cold"
        current_time = time.time()
        hot_items = []
        reorganized_count = 0
        
        for key, item in self._cache.items():
            frequency = self._access_predictor.get_access_frequency(key, 3600)
            if frequency > 0.1 or item.access_count >= self._adaptive_config['hot_data_threshold']:
                hot_items.append(key)
        
        # Déplacer les éléments hot vers la fin (plus récents dans LRU)
        for key in hot_items:
            if key in self._cache:
                self._cache.move_to_end(key)
                reorganized_count += 1
                
        return reorganized_count
    
    def _adjust_adaptive_ttl(self) -> int:
        """Ajuste adaptivement les TTL selon les patterns d'usage."""
        adjustments = 0
        current_time = time.time()
        
        for key, item in self._cache.items():
            frequency = self._access_predictor.get_access_frequency(key, 3600)
            predicted_next = self._access_predictor.predict_next_access(key)
            
            # Ajuster TTL basé sur la fréquence d'accès
            if frequency > 0.1:  # Accès fréquent
                new_ttl = max(3600, item.ttl * 1.5) if item.ttl else 3600
            elif frequency < 0.01:  # Accès rare
                new_ttl = min(1800, item.ttl * 0.5) if item.ttl else 1800
            else:
                continue
                
            if abs(new_ttl - (item.ttl or 0)) > 300:  # Changement significatif
                item.ttl = new_ttl
                adjustments += 1
                
        return adjustments
    
    def _preemptive_eviction(self) -> int:
        """Éviction préemptive basée sur les prédictions."""
        evictions = 0
        current_time = time.time()
        prediction_window = 1800  # 30 minutes
        
        for key, item in list(self._cache.items()):
            predicted_next = self._access_predictor.predict_next_access(key)
            
            if predicted_next and predicted_next > current_time + prediction_window:
                # Probablement pas d'accès dans les 30 prochaines minutes
                if item.access_count < 2 and item.priority < 5:
                    del self._cache[key]
                    evictions += 1
                    
        return evictions
    
    def _calculate_hit_rate(self) -> float:
        """Calcule le taux de hit du cache."""
        total_requests = self._metrics['hits'] + self._metrics['misses']
        if total_requests == 0:
            return 0.0
        return self._metrics['hits'] / total_requests
    
    def _calculate_compression_ratio(self) -> float:
        """Calcule le ratio de compression moyen."""
        if self._metrics['compressions'] == 0:
            return 1.0
            
        # Estimation basée sur les compressions effectuées
        return 1.3  # Ratio moyen estimé
    
    def _cleanup_loop(self):
        """Boucle de nettoyage en arrière-plan."""
        while self._cleanup_running:
            try:
                time.sleep(self._adaptive_config['cleanup_interval'])
                
                if not self._cleanup_running:
                    break
                    
                # Nettoyage périodique
                with self._cache_lock:
                    expired_count = self._cleanup_expired()
                    if expired_count > 0:
                        logger.debug(f"Nettoyage automatique: {expired_count} éléments expirés supprimés")
                        
                    # Optimisation adaptative légère
                    if len(self._cache) > 100:
                        self._reorganize_by_access_patterns()
                        
            except Exception as e:
                logger.error(f"Erreur dans la boucle de nettoyage: {e}")
    
    def get_metrics(self) -> CacheMetrics:
        """Retourne les métriques actuelles du cache."""
        with self._cache_lock:
            total_size = self._get_total_size()
            avg_access_time = np.mean(self._metrics['access_times']) if self._metrics['access_times'] else 0
            
            return CacheMetrics(
                hit_rate=self._calculate_hit_rate(),
                miss_rate=1 - self._calculate_hit_rate(),
                size_mb=total_size / (1024 * 1024),
                items_count=len(self._cache),
                avg_access_time_ms=avg_access_time,
                compression_ratio=self._calculate_compression_ratio(),
                memory_efficiency=total_size / max(self.max_memory_bytes, 1)
            )
    
    def shutdown(self):
        """Arrête l'optimiseur de cache."""
        self._cleanup_running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)

class AccessPredictor:
    """Prédicteur d'accès pour optimiser le cache."""
    
    def __init__(self):
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.prediction_window = 300  # 5 minutes
        
    def record_access(self, key: str, timestamp: float):
        """Enregistre un accès."""
        self.access_history[key].append(timestamp)
        
        # Limiter l'historique
        if len(self.access_history[key]) > 100:
            self.access_history[key] = self.access_history[key][-50:]
    
    def predict_next_access(self, key: str) -> Optional[float]:
        """Prédit le prochain accès pour une clé."""
        if key not in self.access_history or len(self.access_history[key]) < 2:
            return None
            
        accesses = self.access_history[key]
        intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
        
        if not intervals:
            return None
            
        # Prédiction simple basée sur la moyenne des intervalles
        avg_interval = np.mean(intervals)
        return accesses[-1] + avg_interval
    
    def get_access_frequency(self, key: str, window_seconds: float = 3600) -> float:
        """Calcule la fréquence d'accès dans une fenêtre temporelle."""
        if key not in self.access_history:
            return 0.0
            
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_accesses = [
            t for t in self.access_history[key] 
            if t >= cutoff_time
        ]
        
        return len(recent_accesses) / window_seconds
    
    def predict_future_access(self, key: str, prediction_window: float = 3600) -> float:
        """
        Prédit la probabilité d'accès futur basée sur les patterns historiques.
        
        Args:
            key: Clé du cache
            prediction_window: Fenêtre de prédiction en secondes
            
        Returns:
            Probabilité d'accès entre 0 et 1
        """
        if key not in self.access_history or len(self.access_history[key]) < 2:
            return 0.1  # Probabilité par défaut pour nouveaux items
            
        accesses = self.access_history[key]
        current_time = time.time()
        
        # Analyser les patterns temporels
        if len(accesses) >= 3:
            # Calculer les intervalles entre accès
            intervals = [accesses[i] - accesses[i-1] for i in range(1, len(accesses))]
            
            if intervals:
                # Moyenne et variance des intervalles
                avg_interval = np.mean(intervals)
                interval_variance = np.var(intervals)
                
                # Temps depuis dernier accès
                time_since_last = current_time - accesses[-1]
                
                # Prédiction basée sur la régularité des accès
                if avg_interval > 0:
                    # Si les accès sont réguliers (faible variance)
                    regularity_score = 1.0 / (1.0 + interval_variance / max(avg_interval, 1))
                    
                    # Probabilité basée sur quand le prochain accès est attendu
                    expected_next_access = accesses[-1] + avg_interval
                    time_to_expected = abs(expected_next_access - current_time)
                    
                    # Plus on s'approche du moment attendu, plus la probabilité augmente
                    temporal_score = max(0, 1.0 - (time_to_expected / prediction_window))
                    
                    # Fréquence récente
                    frequency_score = min(1.0, self.get_access_frequency(key, prediction_window * 2))
                    
                    # Score composite
                    probability = (
                        regularity_score * 0.4 +
                        temporal_score * 0.4 +
                        frequency_score * 0.2
                    )
                    
                    return min(1.0, max(0.0, probability))
        
        # Fallback: probabilité basée uniquement sur la fréquence récente
        frequency = self.get_access_frequency(key, prediction_window)
        return min(1.0, frequency * prediction_window / 10.0)  # Normalisation heuristique