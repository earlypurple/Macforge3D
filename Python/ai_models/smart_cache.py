"""
Module de cache intelligent avec gestion de la mémoire et du stockage.
"""

import os
import json
import time
import hashlib
import logging
import tempfile
import threading
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import mmap
import zstandard
import lz4.frame
import blosc2
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration du cache."""
    max_memory_size: int = 8 * 1024 * 1024 * 1024  # 8 GB par défaut
    max_disk_size: int = 100 * 1024 * 1024 * 1024  # 100 GB par défaut
    compression_level: int = 3  # Niveau de compression (1-9)
    cache_dir: str = os.path.expanduser("~/.macforge3d/cache")
    use_memory_mapping: bool = True
    cleanup_interval: int = 3600  # Nettoyage toutes les heures
    compression_algorithm: str = "zstd"  # zstd, lz4, blosc2

class SmartCache:
    """Gestionnaire de cache intelligent avec plusieurs niveaux."""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._memory_cache: Dict[str, Any] = {}
        self._mmap_cache = {}
        self._disk_cache_path = Path(self.config.cache_dir)
        self._disk_cache_path.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._memory_usage = 0
        self._disk_usage = 0
        
        # Démarrer le thread de nettoyage
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
        
        # Pool de threads pour les opérations async
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
    def _get_key(self, data: Any) -> str:
        """Génère une clé unique pour les données."""
        if isinstance(data, (str, bytes)):
            data_bytes = data.encode() if isinstance(data, str) else data
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = json.dumps(data, sort_keys=True).encode()
            
        return hashlib.sha256(data_bytes).hexdigest()
        
    def _compress_data(self, data: bytes) -> bytes:
        """Compresse les données avec l'algorithme choisi."""
        if self.config.compression_algorithm == "zstd":
            cctx = zstandard.ZstdCompressor(level=self.config.compression_level)
            return cctx.compress(data)
        elif self.config.compression_algorithm == "lz4":
            return lz4.frame.compress(
                data,
                compression_level=self.config.compression_level
            )
        elif self.config.compression_algorithm == "blosc2":
            return blosc2.compress(
                data,
                clevel=self.config.compression_level,
                typesize=8
            )
        else:
            raise ValueError(f"Algorithme inconnu: {self.config.compression_algorithm}")
            
    def _decompress_data(self, data: bytes) -> bytes:
        """Décompresse les données."""
        if self.config.compression_algorithm == "zstd":
            dctx = zstandard.ZstdDecompressor()
            return dctx.decompress(data)
        elif self.config.compression_algorithm == "lz4":
            return lz4.frame.decompress(data)
        elif self.config.compression_algorithm == "blosc2":
            return blosc2.decompress(data)
        else:
            raise ValueError(f"Algorithme inconnu: {self.config.compression_algorithm}")
            
    def _store_in_memory(self, key: str, data: Any) -> bool:
        """Stocke les données en mémoire si possible."""
        data_size = self._get_data_size(data)
        
        with self._lock:
            if self._memory_usage + data_size <= self.config.max_memory_size:
                self._memory_cache[key] = data
                self._memory_usage += data_size
                return True
                
        return False
        
    def _store_in_mmap(self, key: str, data: Any) -> bool:
        """Stocke les données dans un fichier mappé en mémoire."""
        if not self.config.use_memory_mapping:
            return False
            
        try:
            mmap_path = self._disk_cache_path / f"{key}.mmap"
            # Écrire les données dans un fichier
            data_bytes = (
                data.tobytes() if isinstance(data, np.ndarray)
                else str(data).encode()
            )
            with open(mmap_path, 'wb') as f:
                f.write(data_bytes)
            
            # Créer le mmap
            with open(mmap_path, 'rb') as f:
                mmap_file = mmap.mmap(
                    f.fileno(),
                    0,
                    access=mmap.ACCESS_READ
                )
            self._mmap_cache[key] = mmap_file
            return True
        except Exception as e:
            logger.error(f"Erreur mmap: {e}")
            return False
            
    def _store_on_disk(self, key: str, data: Any) -> bool:
        """Stocke les données sur le disque."""
        try:
            data_bytes = (
                data.tobytes() if isinstance(data, np.ndarray)
                else json.dumps(data).encode()
            )
            compressed = self._compress_data(data_bytes)
            
            cache_file = self._disk_cache_path / key
            with open(cache_file, 'wb') as f:
                f.write(compressed)
                
            self._disk_usage += len(compressed)
            return True
        except Exception as e:
            logger.error(f"Erreur disque: {e}")
            return False
            
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Récupère les données depuis la mémoire."""
        return self._memory_cache.get(key)
        
    def _get_from_mmap(self, key: str) -> Optional[Any]:
        """Récupère les données depuis un fichier mappé."""
        if not self.config.use_memory_mapping:
            return None
            
        mmap_file = self._mmap_cache.get(key)
        if mmap_file:
            data = mmap_file.read()
            try:
                return np.frombuffer(data)
            except:
                return data.decode()
        return None
        
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Récupère les données depuis le disque."""
        cache_file = self._disk_cache_path / key
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'rb') as f:
                compressed = f.read()
            data_bytes = self._decompress_data(compressed)
            
            try:
                return json.loads(data_bytes)
            except:
                return np.frombuffer(data_bytes)
        except Exception as e:
            logger.error(f"Erreur lecture disque: {e}")
            return None
            
    def _get_data_size(self, data: Any) -> int:
        """Calcule la taille approximative des données en mémoire."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (bytes, str)):
            return len(data)
        else:
            return len(json.dumps(data).encode())
            
    def _periodic_cleanup(self):
        """Nettoie périodiquement le cache."""
        while True:
            time.sleep(self.config.cleanup_interval)
            self.cleanup()
            
    def cleanup(self):
        """Nettoie le cache en supprimant les éléments les moins utilisés."""
        with self._lock:
            # Nettoyer la mémoire
            if self._memory_usage > self.config.max_memory_size:
                items = sorted(
                    self._memory_cache.items(),
                    key=lambda x: self._get_data_size(x[1])
                )
                while (
                    self._memory_usage > self.config.max_memory_size * 0.8
                    and items
                ):
                    key, data = items.pop()
                    self._memory_usage -= self._get_data_size(data)
                    del self._memory_cache[key]
                    
            # Nettoyer le disque
            if self._disk_usage > self.config.max_disk_size:
                cache_files = sorted(
                    self._disk_cache_path.glob("*"),
                    key=lambda x: x.stat().st_mtime
                )
                while (
                    self._disk_usage > self.config.max_disk_size * 0.8
                    and cache_files
                ):
                    oldest = cache_files.pop(0)
                    size = oldest.stat().st_size
                    oldest.unlink()
                    self._disk_usage -= size
                    
    def put(
        self,
        data: Any,
        key: Optional[str] = None,
        force_disk: bool = False
    ) -> str:
        """
        Stocke des données dans le cache.
        
        Args:
            data: Les données à stocker
            key: Clé optionnelle (générée si non fournie)
            force_disk: Force le stockage sur disque
            
        Returns:
            La clé des données
        """
        key = key or self._get_key(data)
        
        if not force_disk:
            # Essayer la mémoire d'abord
            if self._store_in_memory(key, data):
                return key
                
            # Puis le mmap
            if self._store_in_mmap(key, data):
                return key
                
        # Finalement le disque
        if self._store_on_disk(key, data):
            return key
            
        raise RuntimeError("Impossible de stocker les données")
        
    def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Récupère des données du cache.
        
        Args:
            key: La clé des données
            default: Valeur par défaut si non trouvé
            
        Returns:
            Les données ou la valeur par défaut
        """
        # Chercher dans la mémoire
        data = self._get_from_memory(key)
        if data is not None:
            return data
            
        # Chercher dans le mmap
        data = self._get_from_mmap(key)
        if data is not None:
            return data
            
        # Chercher sur le disque
        data = self._get_from_disk(key)
        if data is not None:
            # Mettre en cache mémoire si possible
            self._store_in_memory(key, data)
            return data
            
        return default
        
    def invalidate(self, key: str):
        """Invalide une entrée du cache."""
        with self._lock:
            if key in self._memory_cache:
                self._memory_usage -= self._get_data_size(self._memory_cache[key])
                del self._memory_cache[key]
                
            if key in self._mmap_cache:
                self._mmap_cache[key].close()
                del self._mmap_cache[key]
                
            cache_file = self._disk_cache_path / key
            if cache_file.exists():
                self._disk_usage -= cache_file.stat().st_size
                cache_file.unlink()
                
    def clear(self):
        """Vide complètement le cache."""
        with self._lock:
            self._memory_cache.clear()
            self._memory_usage = 0
            
            for mmap_file in self._mmap_cache.values():
                mmap_file.close()
            self._mmap_cache.clear()
            
            for cache_file in self._disk_cache_path.glob("*"):
                cache_file.unlink()
            self._disk_usage = 0
            
    @property
    def stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "memory_usage": self._memory_usage,
            "memory_limit": self.config.max_memory_size,
            "disk_usage": self._disk_usage,
            "disk_limit": self.config.max_disk_size,
            "memory_items": len(self._memory_cache),
            "mmap_items": len(self._mmap_cache),
            "compression_algo": self.config.compression_algorithm,
            "compression_level": self.config.compression_level
        }
        
    def optimize(self):
        """Optimise le cache en ajustant les paramètres."""
        # Analyser l'utilisation
        mem_ratio = self._memory_usage / self.config.max_memory_size
        disk_ratio = self._disk_usage / self.config.max_disk_size
        
        # Ajuster la compression
        if disk_ratio > 0.9:
            self.config.compression_level = min(9, self.config.compression_level + 1)
        elif disk_ratio < 0.5:
            self.config.compression_level = max(1, self.config.compression_level - 1)
            
        # Ajuster l'utilisation du mmap
        if mem_ratio > 0.9:
            self.config.use_memory_mapping = True
        elif mem_ratio < 0.5:
            self.config.use_memory_mapping = False
            
    def preload(self, keys: list[str]):
        """Précharge des données en mémoire."""
        def _load(key):
            data = self.get(key)
            if data is not None:
                self._store_in_memory(key, data)
                
        # Charger en parallèle
        self._thread_pool.map(_load, keys)
