"""
Système de cache avancé pour MacForge3D avec optimisations intelligentes.
"""

import os
import time
import json
import hashlib
import logging
import numpy as np
import torch
import trimesh
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import psutil
import gc
import weakref
import threading
import pickle
import lz4.frame  # Pour la compression rapide
from collections import OrderedDict, defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheStats:
    """Statistiques du cache."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 0.0
    access_patterns: Dict[str, int] = None
    
    def __post_init__(self):
        if self.access_patterns is None:
            self.access_patterns = defaultdict(int)

@dataclass
class CacheEntry:
    """Entrée de cache avec métadonnées avancées."""
    key: str
    data: Any
    creation_time: float
    last_access_time: float
    access_count: int
    size_bytes: int
    compression_level: int
    priority: float = 1.0
    expiry_time: Optional[float] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class IntelligentCache:
    """Cache intelligent avec gestion adaptative et prédiction d'accès."""
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        max_disk_gb: int = 10,
        cache_dir: Optional[Path] = None,
        compression_enabled: bool = True,
        adaptive_sizing: bool = True,
        prediction_enabled: bool = True
    ):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_gb * 1024 * 1024 * 1024
        self.cache_dir = cache_dir or Path.home() / ".macforge3d" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.compression_enabled = compression_enabled
        self.adaptive_sizing = adaptive_sizing
        self.prediction_enabled = prediction_enabled
        
        # Cache en mémoire avec LRU ordering
        self._memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._disk_index: Dict[str, Dict[str, Any]] = {}
        
        # Statistiques et monitoring
        self._stats = CacheStats()
        self._access_history: List[Tuple[str, float]] = []
        self._prediction_model: Optional[Dict[str, Any]] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background threads
        self._cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self._analytics_thread = threading.Thread(target=self._background_analytics, daemon=True)
        
        self._shutdown_event = threading.Event()
        self._cleanup_thread.start()
        self._analytics_thread.start()
        
        # Load existing disk cache index
        self._load_disk_index()
        
        logger.info(f"Cache intelligent initialisé: {max_memory_mb}MB RAM, {max_disk_gb}GB disk")
    
    def _generate_key(self, key_components: Union[str, List[Any]]) -> str:
        """Génère une clé de cache unique."""
        if isinstance(key_components, str):
            return hashlib.sha256(key_components.encode()).hexdigest()[:16]
        
        # Sérialise les composants pour créer une clé stable
        serialized = json.dumps(key_components, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    
    def _estimate_size(self, obj: Any) -> int:
        """Estime la taille d'un objet."""
        try:
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement()
            elif isinstance(obj, trimesh.Trimesh):
                return (obj.vertices.nbytes + obj.faces.nbytes + 
                       sum(getattr(attr, 'nbytes', 0) for attr in obj.__dict__.values()))
            elif isinstance(obj, (str, bytes)):
                return len(obj.encode('utf-8') if isinstance(obj, str) else obj)
            else:
                # Fallback: utilise pickle pour estimer
                return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            return 1024  # Estimation par défaut
    
    def _compress_data(self, data: Any, level: int = 3) -> bytes:
        """Compresse les données avec LZ4."""
        if not self.compression_enabled:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = lz4.frame.compress(pickled, compression_level=level)
        
        # Calculer le ratio de compression
        ratio = len(pickled) / len(compressed) if compressed else 1.0
        self._stats.compression_ratio = (self._stats.compression_ratio * 0.9 + ratio * 0.1)
        
        return compressed
    
    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Décompresse les données."""
        if not self.compression_enabled:
            return pickle.loads(compressed_data)
        
        try:
            pickled = lz4.frame.decompress(compressed_data)
            return pickle.loads(pickled)
        except:
            # Fallback: données non compressées
            return pickle.loads(compressed_data)
    
    def _calculate_priority(self, key: str, access_count: int, last_access: float) -> float:
        """Calcule la priorité d'une entrée de cache."""
        now = time.time()
        recency_factor = 1.0 / (1.0 + (now - last_access) / 3600)  # Décroissance horaire
        frequency_factor = min(access_count / 10.0, 1.0)  # Normalise la fréquence
        
        # Prédiction d'accès futur si disponible
        prediction_factor = 1.0
        if self.prediction_enabled and self._prediction_model:
            prediction_factor = self._predict_access_probability(key)
        
        return recency_factor * 0.4 + frequency_factor * 0.4 + prediction_factor * 0.2
    
    def _predict_access_probability(self, key: str) -> float:
        """Prédit la probabilité d'accès futur basée sur l'historique."""
        if not self._access_history:
            return 0.5
        
        # Analyse des patterns d'accès
        recent_accesses = [k for k, t in self._access_history[-100:] if time.time() - t < 3600]
        key_frequency = recent_accesses.count(key) / max(len(recent_accesses), 1)
        
        return min(key_frequency * 2, 1.0)
    
    def _evict_entries(self, target_size: int):
        """Évince les entrées les moins prioritaires."""
        with self._lock:
            current_size = sum(entry.size_bytes for entry in self._memory_cache.values())
            
            if current_size <= target_size:
                return
            
            # Trie par priorité (plus faible = éviction prioritaire)
            entries_by_priority = sorted(
                self._memory_cache.items(),
                key=lambda x: self._calculate_priority(x[0], x[1].access_count, x[1].last_access_time)
            )
            
            bytes_to_free = current_size - target_size
            freed_bytes = 0
            
            for key, entry in entries_by_priority:
                if freed_bytes >= bytes_to_free:
                    break
                
                # Sauvegarder sur disque si possible
                if entry.size_bytes < self.max_disk_bytes // 100:  # Max 1% du cache disque
                    self._save_to_disk(key, entry)
                
                del self._memory_cache[key]
                freed_bytes += entry.size_bytes
                self._stats.evictions += 1
            
            logger.debug(f"Éviction de {freed_bytes} bytes ({len(entries_by_priority)} entrées)")
    
    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Sauvegarde une entrée sur disque."""
        try:
            disk_path = self.cache_dir / f"{key}.cache"
            compressed_data = self._compress_data(entry.data, entry.compression_level)
            
            with open(disk_path, 'wb') as f:
                f.write(compressed_data)
            
            # Mettre à jour l'index disque
            self._disk_index[key] = {
                'path': str(disk_path),
                'size': len(compressed_data),
                'creation_time': entry.creation_time,
                'last_access': entry.last_access_time,
                'compression_level': entry.compression_level
            }
            
        except Exception as e:
            logger.warning(f"Erreur sauvegarde disque pour {key}: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[Any]:
        """Charge une entrée depuis le disque."""
        if key not in self._disk_index:
            return None
        
        try:
            disk_info = self._disk_index[key]
            disk_path = Path(disk_info['path'])
            
            if not disk_path.exists():
                del self._disk_index[key]
                return None
            
            with open(disk_path, 'rb') as f:
                compressed_data = f.read()
            
            data = self._decompress_data(compressed_data)
            
            # Mettre à jour le temps d'accès
            self._disk_index[key]['last_access'] = time.time()
            
            return data
            
        except Exception as e:
            logger.warning(f"Erreur chargement disque pour {key}: {e}")
            return None
    
    def _load_disk_index(self):
        """Charge l'index du cache disque."""
        index_path = self.cache_dir / "cache_index.json"
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    self._disk_index = json.load(f)
                logger.info(f"Index disque chargé: {len(self._disk_index)} entrées")
            except Exception as e:
                logger.warning(f"Erreur chargement index disque: {e}")
                self._disk_index = {}
    
    def _save_disk_index(self):
        """Sauvegarde l'index du cache disque."""
        index_path = self.cache_dir / "cache_index.json"
        try:
            with open(index_path, 'w') as f:
                json.dump(self._disk_index, f, indent=2)
        except Exception as e:
            logger.warning(f"Erreur sauvegarde index disque: {e}")
    
    def put(
        self,
        key: Union[str, List[Any]],
        data: Any,
        ttl_seconds: Optional[int] = None,
        compression_level: int = 3,
        priority: float = 1.0
    ) -> bool:
        """Stocke une entrée dans le cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            # Vérifier la mémoire disponible
            data_size = self._estimate_size(data)
            
            if data_size > self.max_memory_bytes:
                logger.warning(f"Objet trop volumineux pour le cache: {data_size} bytes")
                return False
            
            # Éviction si nécessaire
            if data_size > self.max_memory_bytes - sum(e.size_bytes for e in self._memory_cache.values()):
                self._evict_entries(self.max_memory_bytes - data_size)
            
            # Créer l'entrée
            now = time.time()
            entry = CacheEntry(
                key=cache_key,
                data=data,
                creation_time=now,
                last_access_time=now,
                access_count=1,
                size_bytes=data_size,
                compression_level=compression_level,
                priority=priority,
                expiry_time=now + ttl_seconds if ttl_seconds else None
            )
            
            self._memory_cache[cache_key] = entry
            self._stats.total_size_bytes += data_size
            
            return True
    
    def get(self, key: Union[str, List[Any]]) -> Optional[Any]:
        """Récupère une entrée du cache."""
        cache_key = self._generate_key(key)
        now = time.time()
        
        with self._lock:
            # Chercher en mémoire
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                
                # Vérifier l'expiration
                if entry.expiry_time and now > entry.expiry_time:
                    del self._memory_cache[cache_key]
                    self._stats.misses += 1
                    return None
                
                # Mettre à jour les statistiques d'accès
                entry.last_access_time = now
                entry.access_count += 1
                
                # Déplacer en fin de LRU
                self._memory_cache.move_to_end(cache_key)
                
                self._stats.hits += 1
                self._access_history.append((cache_key, now))
                return entry.data
            
            # Chercher sur disque
            disk_data = self._load_from_disk(cache_key)
            if disk_data is not None:
                # Remettre en mémoire
                self.put(cache_key, disk_data)
                self._stats.hits += 1
                self._access_history.append((cache_key, now))
                return disk_data
            
            self._stats.misses += 1
            return None
    
    def invalidate(self, key: Union[str, List[Any]]):
        """Invalide une entrée du cache."""
        cache_key = self._generate_key(key)
        
        with self._lock:
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            if cache_key in self._disk_index:
                disk_path = Path(self._disk_index[cache_key]['path'])
                if disk_path.exists():
                    disk_path.unlink()
                del self._disk_index[cache_key]
    
    def clear(self):
        """Vide complètement le cache."""
        with self._lock:
            self._memory_cache.clear()
            
            # Nettoyer le cache disque
            for key in list(self._disk_index.keys()):
                disk_path = Path(self._disk_index[key]['path'])
                if disk_path.exists():
                    disk_path.unlink()
            
            self._disk_index.clear()
            self._stats = CacheStats()
            
        logger.info("Cache entièrement vidé")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        with self._lock:
            total_requests = self._stats.hits + self._stats.misses
            hit_rate = self._stats.hits / total_requests if total_requests > 0 else 0
            
            return {
                "hit_rate": round(hit_rate * 100, 2),
                "total_requests": total_requests,
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "evictions": self._stats.evictions,
                "memory_entries": len(self._memory_cache),
                "disk_entries": len(self._disk_index),
                "total_size_mb": round(self._stats.total_size_bytes / (1024 * 1024), 2),
                "compression_ratio": round(self._stats.compression_ratio, 2),
                "memory_usage_percent": round(
                    sum(e.size_bytes for e in self._memory_cache.values()) / self.max_memory_bytes * 100, 2
                )
            }
    
    def _background_cleanup(self):
        """Nettoyage périodique en arrière-plan."""
        while not self._shutdown_event.wait(300):  # 5 minutes
            try:
                self._cleanup_expired_entries()
                self._optimize_disk_cache()
                self._save_disk_index()
            except Exception as e:
                logger.error(f"Erreur nettoyage arrière-plan: {e}")
    
    def _background_analytics(self):
        """Analyse des patterns d'accès en arrière-plan."""
        while not self._shutdown_event.wait(600):  # 10 minutes
            try:
                self._analyze_access_patterns()
                self._update_prediction_model()
            except Exception as e:
                logger.error(f"Erreur analytics arrière-plan: {e}")
    
    def _cleanup_expired_entries(self):
        """Nettoie les entrées expirées."""
        now = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self._memory_cache.items():
                if entry.expiry_time and now > entry.expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._memory_cache[key]
        
        if expired_keys:
            logger.debug(f"Nettoyage de {len(expired_keys)} entrées expirées")
    
    def _optimize_disk_cache(self):
        """Optimise le cache disque."""
        # Supprimer les fichiers orphelins
        cache_files = set(self.cache_dir.glob("*.cache"))
        indexed_files = {Path(info['path']) for info in self._disk_index.values()}
        
        orphan_files = cache_files - indexed_files
        for orphan in orphan_files:
            try:
                orphan.unlink()
            except Exception:
                pass
        
        # Supprimer les entrées d'index sans fichier
        missing_keys = []
        for key, info in self._disk_index.items():
            if not Path(info['path']).exists():
                missing_keys.append(key)
        
        for key in missing_keys:
            del self._disk_index[key]
    
    def _analyze_access_patterns(self):
        """Analyse les patterns d'accès pour optimisation."""
        if len(self._access_history) < 10:
            return
        
        # Garder seulement les 1000 derniers accès
        if len(self._access_history) > 1000:
            self._access_history = self._access_history[-1000:]
        
        # Analyser les patterns de fréquence
        access_counts = defaultdict(int)
        for key, _ in self._access_history[-100:]:
            access_counts[key] += 1
        
        self._stats.access_patterns = dict(access_counts)
    
    def _update_prediction_model(self):
        """Met à jour le modèle de prédiction d'accès."""
        if not self.prediction_enabled or len(self._access_history) < 50:
            return
        
        # Modèle simple basé sur la fréquence et la récence
        model = {}
        now = time.time()
        
        for key, access_time in self._access_history[-200:]:
            if key not in model:
                model[key] = {'count': 0, 'last_access': 0, 'avg_interval': 0}
            
            model[key]['count'] += 1
            model[key]['last_access'] = max(model[key]['last_access'], access_time)
        
        self._prediction_model = model
    
    def __del__(self):
        """Nettoyage lors de la destruction."""
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()
        if hasattr(self, '_disk_index'):
            self._save_disk_index()

# Instance globale du cache
_global_cache = None

def get_global_cache() -> IntelligentCache:
    """Retourne l'instance globale du cache."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCache()
    return _global_cache

def cache_result(ttl_seconds: Optional[int] = None, compression_level: int = 3):
    """Décorateur pour mettre en cache les résultats de fonction."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Générer une clé basée sur la fonction et ses arguments
            key_components = [func.__name__, str(args), json.dumps(kwargs, sort_keys=True, default=str)]
            cache_key = tuple(key_components)
            
            cache = get_global_cache()
            
            # Chercher dans le cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Calculer et stocker
            result = func(*args, **kwargs)
            cache.put(cache_key, result, ttl_seconds=ttl_seconds, compression_level=compression_level)
            
            return result
        
        wrapper.__wrapped__ = func
        return wrapper
    return decorator