"""
Module de gestion de cache et d'optimisation des performances pour MacForge3D.
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
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
import weakref
from functools import lru_cache
import tempfile
import mmap
import threading
import queue

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration du système de cache."""
    max_memory_usage: float = 0.75  # Pourcentage maximum de RAM utilisable
    max_cache_size: int = 1024 * 1024 * 1024  # 1 GB par défaut
    cache_dir: Path = Path(tempfile.gettempdir()) / "macforge3d_cache"
    enable_memory_mapping: bool = True
    enable_gpu_caching: bool = True
    cleanup_interval: int = 300  # 5 minutes
    max_items_per_category: int = 1000

class MemoryManager:
    """Gestionnaire de mémoire intelligent."""
    
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total
        self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self._memory_usage = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        
    def check_memory_usage(self) -> float:
        """Vérifie l'utilisation actuelle de la mémoire."""
        return psutil.virtual_memory().percent / 100.0
        
    def estimate_size(self, obj: Any) -> int:
        """Estime la taille en mémoire d'un objet."""
        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, trimesh.Trimesh):
            return (
                obj.vertices.nbytes +
                obj.faces.nbytes +
                sum(arr.nbytes for arr in obj.vertex_attributes.values())
            )
        return 0
        
    def can_allocate(self, size: int) -> bool:
        """Vérifie si une allocation est possible."""
        with self._lock:
            current_usage = self.check_memory_usage()
            estimated_usage = current_usage + (size / self.total_memory)
            return estimated_usage < 0.9  # Garde 10% de marge
            
    def register_allocation(self, obj: Any):
        """Enregistre une allocation."""
        with self._lock:
            self._memory_usage[obj] = self.estimate_size(obj)
            
    def cleanup(self):
        """Force le nettoyage de la mémoire."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class CacheManager:
    """Gestionnaire de cache intelligent."""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.memory_manager = MemoryManager()
        self.cache_dir = self.config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache: Dict[str, weakref.ref] = {}
        self._disk_cache_index: Dict[str, Dict[str, Any]] = {}
        self._mmap_files: Dict[str, mmap.mmap] = {}
        
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            daemon=True
        )
        self._cleanup_thread.start()
        
        # File d'attente pour les opérations asynchrones
        self._cache_queue = queue.Queue()
        self._worker_thread = threading.Thread(
            target=self._process_cache_queue,
            daemon=True
        )
        self._worker_thread.start()
        
    def _compute_key(self, data: Any, category: str) -> str:
        """Calcule une clé de cache unique."""
        if isinstance(data, (str, bytes)):
            hash_input = data
        elif isinstance(data, np.ndarray):
            hash_input = data.tobytes()
        elif isinstance(data, trimesh.Trimesh):
            hash_input = data.vertices.tobytes() + data.faces.tobytes()
        else:
            hash_input = str(data).encode()
            
        return f"{category}_{hashlib.sha256(hash_input).hexdigest()}"
        
    def _save_to_disk(self, key: str, data: Any, metadata: Dict[str, Any]):
        """Sauvegarde les données sur le disque."""
        file_path = self.cache_dir / f"{key}.cache"
        
        if isinstance(data, trimesh.Trimesh):
            # Sauvegarder en format binaire optimisé
            cached_data = {
                "vertices": data.vertices,
                "faces": data.faces,
                "attributes": data.vertex_attributes
            }
            np.savez_compressed(file_path, **cached_data)
        elif isinstance(data, np.ndarray):
            np.save(file_path, data)
        else:
            with open(file_path, 'wb') as f:
                np.save(f, data)
                
        # Sauvegarder les métadonnées
        meta_path = self.cache_dir / f"{key}.meta"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f)
            
    def _load_from_disk(self, key: str) -> Tuple[Any, Dict[str, Any]]:
        """Charge les données depuis le disque."""
        file_path = self.cache_dir / f"{key}.cache"
        meta_path = self.cache_dir / f"{key}.meta"
        
        if not file_path.exists() or not meta_path.exists():
            raise KeyError(f"Cache non trouvé pour la clé: {key}")
            
        # Charger les métadonnées
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
            
        # Charger les données
        if metadata.get("type") == "mesh":
            with np.load(file_path) as data:
                mesh = trimesh.Trimesh(
                    vertices=data["vertices"],
                    faces=data["faces"]
                )
                for key, value in data["attributes"].items():
                    mesh.vertex_attributes[key] = value
                return mesh, metadata
        else:
            return np.load(file_path), metadata
            
    def get(
        self,
        data: Any,
        category: str,
        generator: Optional[callable] = None
    ) -> Any:
        """
        Récupère un élément du cache ou le génère.
        
        Args:
            data: Données pour générer la clé
            category: Catégorie de cache
            generator: Fonction pour générer les données si non trouvées
            
        Returns:
            Les données cachées ou nouvellement générées
        """
        key = self._compute_key(data, category)
        
        # Vérifier le cache mémoire
        if key in self._memory_cache:
            cached = self._memory_cache[key]()
            if cached is not None:
                return cached
                
        # Vérifier le cache disque
        try:
            result, metadata = self._load_from_disk(key)
            
            # Mettre en cache mémoire si possible
            if self.memory_manager.can_allocate(self.memory_manager.estimate_size(result)):
                self._memory_cache[key] = weakref.ref(result)
                
            return result
            
        except (KeyError, FileNotFoundError):
            if generator is None:
                raise KeyError(f"Données non trouvées pour: {key}")
                
            # Générer nouvelles données
            result = generator()
            
            # Sauvegarder en cache
            self.put(result, category)
            
            return result
            
    def put(self, data: Any, category: str):
        """
        Met des données en cache.
        
        Args:
            data: Données à mettre en cache
            category: Catégorie de cache
        """
        key = self._compute_key(data, category)
        
        # Préparer les métadonnées
        metadata = {
            "timestamp": time.time(),
            "category": category,
            "size": self.memory_manager.estimate_size(data),
            "type": "mesh" if isinstance(data, trimesh.Trimesh) else "array"
        }
        
        # Mettre en cache mémoire si possible
        if self.memory_manager.can_allocate(metadata["size"]):
            self._memory_cache[key] = weakref.ref(data)
            
        # Ajouter à la file d'attente pour sauvegarde disque
        self._cache_queue.put((key, data, metadata))
        
    def _process_cache_queue(self):
        """Traite la file d'attente des opérations de cache."""
        while True:
            try:
                key, data, metadata = self._cache_queue.get(timeout=1)
                self._save_to_disk(key, data, metadata)
                self._disk_cache_index[key] = metadata
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Erreur de cache: {e}")
                
    def _periodic_cleanup(self):
        """Nettoie périodiquement le cache."""
        while True:
            try:
                # Attendre l'intervalle configuré
                time.sleep(self.config.cleanup_interval)
                
                # Nettoyer la mémoire
                self.memory_manager.cleanup()
                
                # Vérifier et nettoyer le cache disque
                current_time = time.time()
                total_size = 0
                items_by_category: Dict[str, int] = {}
                
                for key, metadata in list(self._disk_cache_index.items()):
                    category = metadata["category"]
                    items_by_category[category] = items_by_category.get(category, 0) + 1
                    
                    # Supprimer si trop d'items dans la catégorie
                    if items_by_category[category] > self.config.max_items_per_category:
                        file_path = self.cache_dir / f"{key}.cache"
                        meta_path = self.cache_dir / f"{key}.meta"
                        
                        try:
                            file_path.unlink()
                            meta_path.unlink()
                            del self._disk_cache_index[key]
                        except FileNotFoundError:
                            pass
                        
                    else:
                        total_size += metadata["size"]
                        
                # Nettoyer si le cache dépasse la taille maximale
                if total_size > self.config.max_cache_size:
                    # Trier par timestamp et supprimer les plus anciens
                    sorted_items = sorted(
                        self._disk_cache_index.items(),
                        key=lambda x: x[1]["timestamp"]
                    )
                    
                    for key, _ in sorted_items:
                        if total_size <= self.config.max_cache_size:
                            break
                            
                        file_path = self.cache_dir / f"{key}.cache"
                        meta_path = self.cache_dir / f"{key}.meta"
                        
                        try:
                            file_path.unlink()
                            meta_path.unlink()
                            total_size -= self._disk_cache_index[key]["size"]
                            del self._disk_cache_index[key]
                        except FileNotFoundError:
                            pass
                            
            except Exception as e:
                logger.error(f"Erreur de nettoyage: {e}")
                
    def clear(self, category: Optional[str] = None):
        """
        Vide le cache.
        
        Args:
            category: Si spécifié, vide uniquement cette catégorie
        """
        with self._lock:
            if category:
                # Supprimer par catégorie
                keys_to_remove = [
                    key for key, meta in self._disk_cache_index.items()
                    if meta["category"] == category
                ]
                
                for key in keys_to_remove:
                    file_path = self.cache_dir / f"{key}.cache"
                    meta_path = self.cache_dir / f"{key}.meta"
                    
                    try:
                        file_path.unlink()
                        meta_path.unlink()
                        del self._disk_cache_index[key]
                    except FileNotFoundError:
                        pass
                        
            else:
                # Vider tout le cache
                for file in self.cache_dir.glob("*"):
                    try:
                        file.unlink()
                    except FileNotFoundError:
                        pass
                        
                self._disk_cache_index.clear()
                self._memory_cache.clear()
                self.memory_manager.cleanup()

class PerformanceOptimizer:
    """Optimiseur de performances pour le traitement 3D."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.thread_pool = ThreadPoolExecutor()
        self.process_pool = ProcessPoolExecutor()
        
    def optimize_mesh(
        self,
        mesh: trimesh.Trimesh,
        level: str = "medium"
    ) -> trimesh.Trimesh:
        """
        Optimise un maillage pour les performances.
        
        Args:
            mesh: Maillage à optimiser
            level: Niveau d'optimisation ("low", "medium", "high")
            
        Returns:
            Maillage optimisé
        """
        # Vérifier le cache
        cache_key = (mesh.vertices.tobytes(), mesh.faces.tobytes(), level)
        
        try:
            return self.cache_manager.get(cache_key, "mesh_optimization")
        except KeyError:
            pass
            
        # Optimisation progressive selon le niveau
        if level == "low":
            # Simplification légère
            target_faces = len(mesh.faces) * 0.8
        elif level == "medium":
            # Simplification moyenne
            target_faces = len(mesh.faces) * 0.6
        else:  # high
            # Simplification agressive
            target_faces = len(mesh.faces) * 0.4
            
        # Optimiser
        optimized = mesh.copy()
        
        # Simplification du maillage
        optimized = optimized.simplify_quadratic_decimation(target_faces)
        
        # Supprimer les sommets inutilisés
        optimized.remove_unreferenced_vertices()
        
        # Fusionner les sommets proches
        optimized.merge_vertices(digits_precision=8)
        
        # Optimiser les normales
        optimized.fix_normals()
        
        # Mettre en cache
        self.cache_manager.put(optimized, "mesh_optimization")
        
        return optimized
        
    def parallel_process_mesh(
        self,
        mesh: trimesh.Trimesh,
        process_fn: callable,
        chunk_size: Optional[int] = None
    ) -> trimesh.Trimesh:
        """
        Traite un maillage en parallèle.
        
        Args:
            mesh: Maillage à traiter
            process_fn: Fonction de traitement
            chunk_size: Taille des chunks pour le traitement parallèle
            
        Returns:
            Maillage traité
        """
        if chunk_size is None:
            # Calculer une taille de chunk optimale
            num_cores = os.cpu_count() or 4
            chunk_size = max(len(mesh.vertices) // (num_cores * 2), 1000)
            
        # Diviser les vertices en chunks
        vertex_chunks = np.array_split(mesh.vertices, len(mesh.vertices) // chunk_size)
        
        # Traiter en parallèle
        processed_chunks = list(
            self.thread_pool.map(process_fn, vertex_chunks)
        )
        
        # Recombiner
        mesh.vertices = np.concatenate(processed_chunks)
        
        return mesh
        
    def batch_process_meshes(
        self,
        meshes: List[trimesh.Trimesh],
        process_fn: callable
    ) -> List[trimesh.Trimesh]:
        """
        Traite plusieurs maillages en parallèle.
        
        Args:
            meshes: Liste de maillages
            process_fn: Fonction de traitement
            
        Returns:
            Liste des maillages traités
        """
        return list(self.process_pool.map(process_fn, meshes))
