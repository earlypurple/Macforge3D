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
from typing import Dict, Any, Optional, Union, Tuple, List
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
        # Safe GPU memory detection
        try:
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        except Exception:
            self.gpu_memory = 0  # Fallback for systems without CUDA
        self._memory_usage = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        
    def check_memory_usage(self) -> Tuple[float, float, Dict[str, Any]]:
        """Vérifie l'utilisation actuelle de la mémoire avec détails avancés."""
        try:
            # Mémoire système
            system_mem = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Mémoire GPU si disponible
            gpu_usage = 0.0
            gpu_free = 0
            if torch.cuda.is_available():
                gpu_free, gpu_total = torch.cuda.mem_get_info()
                gpu_usage = (gpu_total - gpu_free) / gpu_total
            
            # Statistiques détaillées
            details = {
                "system_memory": {
                    "total_gb": round(system_mem.total / (1024**3), 2),
                    "used_gb": round(system_mem.used / (1024**3), 2),
                    "available_gb": round(system_mem.available / (1024**3), 2),
                    "percent": system_mem.percent
                },
                "gpu_memory": {
                    "total_gb": round(self.gpu_memory / (1024**3), 2) if self.gpu_memory else 0,
                    "used_gb": round((self.gpu_memory - gpu_free) / (1024**3), 2) if self.gpu_memory else 0,
                    "free_gb": round(gpu_free / (1024**3), 2) if gpu_free else 0,
                    "percent": round(gpu_usage * 100, 2)
                },
                "cpu_usage": cpu_usage,
                "process_count": len(psutil.pids()),
                "disk_io": psutil.disk_io_counters()._asdict() if psutil.disk_io_counters() else {}
            }
            
            return system_mem.percent / 100.0, gpu_usage, details
            
        except Exception as e:
            logger.warning(f"Erreur lors de la vérification de la mémoire: {e}")
            return 0.5, 0.0, {}
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
        try:
            if isinstance(data, str):
                hash_input = data.encode()
            elif isinstance(data, bytes):
                hash_input = data
            elif isinstance(data, np.ndarray):
                hash_input = data.tobytes()
            elif isinstance(data, trimesh.Trimesh):
                # Utiliser une approche plus simple pour éviter les erreurs de concaténation
                mesh_info = f"v{len(data.vertices)}f{len(data.faces)}hash{abs(hash(str(data.vertices.shape)))}"
                hash_input = mesh_info.encode()
            else:
                hash_input = str(data).encode()
                
            return f"{category}_{hashlib.sha256(hash_input).hexdigest()}"
        except Exception as e:
            # Fallback sécurisé
            fallback_input = f"{category}_{str(type(data))}_{str(abs(hash(str(data))))}"
            return hashlib.sha256(fallback_input.encode()).hexdigest()
        
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
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        self.cache_manager = cache_manager or CacheManager()
        
        # Configuration optimisée des pools d'exécution
        cpu_count = os.cpu_count() or 4
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(cpu_count * 2, 32),  # Optimisé pour I/O
            thread_name_prefix="perf_thread"
        )
        self.process_pool = ProcessPoolExecutor(
            max_workers=min(cpu_count, 16),  # Optimisé pour CPU
            mp_context=None  # Utilise le contexte par défaut
        )
        
        # Statistiques de performance
        self._performance_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_optimizations': 0
        }
        
        # Configuration adaptative
        self._adaptive_config = {
            'auto_adjust_workers': True,
            'memory_threshold': 0.85,  # 85% de RAM
            'cpu_threshold': 0.90,     # 90% de CPU
            'last_adjustment': time.time()
        }
        
    def _validate_mesh(self, mesh: trimesh.Trimesh) -> bool:
        """Valide un maillage d'entrée."""
        try:
            if mesh is None:
                logger.error("Maillage None fourni")
                return False
                
            if len(mesh.vertices) == 0:
                logger.error("Maillage sans vertices")
                return False
                
            if len(mesh.faces) == 0:
                logger.error("Maillage sans faces")
                return False
                
            # Vérifier les dimensions
            if mesh.vertices.shape[1] != 3:
                logger.error(f"Vertices doivent avoir 3 dimensions, reçu {mesh.vertices.shape[1]}")
                return False
                
            # Vérifier les valeurs NaN ou infinies
            if np.any(~np.isfinite(mesh.vertices)):
                logger.error("Vertices contiennent des valeurs NaN ou infinies")
                return False
                
            # Vérifier la cohérence des indices de faces
            max_vertex_index = np.max(mesh.faces)
            if max_vertex_index >= len(mesh.vertices):
                logger.error(f"Index de face invalide: {max_vertex_index} >= {len(mesh.vertices)}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation du maillage: {e}")
            return False
        
    def optimize_mesh(
        self,
        mesh: trimesh.Trimesh,
        level: str = "medium",
        progress_callback: Optional[callable] = None
    ) -> trimesh.Trimesh:
        """
        Optimise un maillage pour les performances.
        
        Args:
            mesh: Maillage à optimiser
            level: Niveau d'optimisation ("low", "medium", "high")
            progress_callback: Fonction appelée avec le progrès (0.0 à 1.0)
            
        Returns:
            Maillage optimisé
            
        Raises:
            ValueError: Si le niveau d'optimisation n'est pas valide
            RuntimeError: Si l'optimisation échoue
        """
        if level not in ["low", "medium", "high"]:
            raise ValueError(f"Niveau d'optimisation invalide: {level}. Utilisez 'low', 'medium' ou 'high'.")
        
        if progress_callback:
            progress_callback(0.0)
        
        try:
            # Générer une clé de cache efficace en utilisant un hash
            mesh_hash = hashlib.md5(
                f"{mesh.vertices.shape}_{mesh.faces.shape}_{level}".encode()
            ).hexdigest()
            
            # Vérifier le cache
            try:
                cached_result = self.cache_manager.get(mesh_hash, "mesh_optimization")
                if progress_callback:
                    progress_callback(1.0)
                logger.info(f"Maillage optimisé récupéré du cache (niveau: {level})")
                return cached_result
            except KeyError:
                pass
                
            if progress_callback:
                progress_callback(0.1)
                
            # Validation du maillage d'entrée
            if not self._validate_mesh(mesh):
                raise ValueError("Maillage d'entrée invalide")
                
            if progress_callback:
                progress_callback(0.1)
                
            # Générer une clé de cache efficace en utilisant un hash
            mesh_hash = hashlib.md5(
                f"{mesh.vertices.shape}_{mesh.faces.shape}_{level}".encode()
            ).hexdigest()
            
            # Vérifier le cache
            try:
                cached_result = self.cache_manager.get(mesh_hash, "mesh_optimization")
                if progress_callback:
                    progress_callback(1.0)
                logger.info(f"Maillage optimisé récupéré du cache (niveau: {level})")
                return cached_result
            except KeyError:
                pass
                
            if progress_callback:
                progress_callback(0.2)
                
            # Vérifier et réparer le maillage si nécessaire
            mesh_copy = mesh.copy()
            try:
                # Vérifier la validité avec la méthode is_valid si disponible
                if hasattr(mesh_copy, 'is_valid') and not mesh_copy.is_valid:
                    logger.warning("Maillage d'entrée invalide, tentative de réparation...")
                    if hasattr(mesh_copy, 'fix_normals'):
                        mesh_copy.fix_normals()
                    if hasattr(mesh_copy, 'remove_duplicate_faces'):
                        mesh_copy.remove_duplicate_faces()
                    if hasattr(mesh_copy, 'remove_degenerate_faces'):
                        mesh_copy.remove_degenerate_faces()
            except Exception as e:
                logger.warning(f"Impossible de vérifier/réparer le maillage: {e}")
                
            if progress_callback:
                progress_callback(0.3)
                
            # Optimisation progressive selon le niveau
            reduction_factors = {
                "low": 0.8,
                "medium": 0.6, 
                "high": 0.4
            }
            target_faces = max(100, int(len(mesh.faces) * reduction_factors[level]))
            
            logger.info(f"Optimisation niveau {level}: {len(mesh.faces)} → {target_faces} faces")
            
            # Optimiser
            optimized = mesh_copy
            
            if progress_callback:
                progress_callback(0.4)
                
            # Simplification du maillage avec stratégies adaptatives
            if target_faces < len(mesh.faces):
                simplification_success = self._apply_adaptive_simplification(
                    optimized, target_faces, progress_callback
                )
                if not simplification_success:
                    logger.info("Simplification ignorée - maillage conservé tel quel")
                    
            if progress_callback:
                progress_callback(0.6)
                
            # Supprimer les sommets inutilisés
            try:
                optimized.remove_unreferenced_vertices()
            except Exception as e:
                logger.warning(f"Impossible de supprimer les sommets non référencés: {e}")
                
            if progress_callback:
                progress_callback(0.7)
                
            # Fusionner les sommets proches avec gestion d'erreurs
            try:
                optimized.merge_vertices()
            except Exception as e:
                logger.warning(f"Impossible de fusionner les sommets: {e}")
                
            if progress_callback:
                progress_callback(0.8)
                
            # Optimiser les normales
            try:
                optimized.fix_normals()
            except Exception as e:
                logger.warning(f"Impossible d'optimiser les normales: {e}")
                
            if progress_callback:
                progress_callback(0.9)
                
            # Validation du maillage optimisé
            try:
                if hasattr(optimized, 'is_valid') and not optimized.is_valid:
                    logger.error("Le maillage optimisé n'est pas valide")
                    # Retourner le maillage original si l'optimisation a échoué
                    optimized = mesh
            except Exception as e:
                logger.warning(f"Impossible de vérifier la validité du maillage optimisé: {e}")
                
            # Mettre en cache avec gestion d'erreurs (temporairement désactivé pour éviter les erreurs de sérialisation)
            try:
                # Note: Mise en cache temporairement désactivée en raison de problèmes de sérialisation avec certains maillages
                # self.cache_manager.put(optimized, "mesh_optimization")
                logger.debug("Mise en cache du maillage optimisé temporairement désactivée")
            except Exception as e:
                logger.warning(f"Impossible de mettre en cache le résultat: {e}")
                
            if progress_callback:
                progress_callback(1.0)
                
            logger.info(f"Optimisation terminée: {len(mesh.faces)} → {len(optimized.faces)} faces")
            return optimized
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation du maillage: {e}")
            raise RuntimeError(f"Échec de l'optimisation du maillage: {str(e)}") from e
        
    def parallel_process_mesh(
        self,
        mesh: trimesh.Trimesh,
        process_fn: callable,
        chunk_size: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> trimesh.Trimesh:
        """
        Traite un maillage en parallèle.
        
        Args:
            mesh: Maillage à traiter
            process_fn: Fonction de traitement à appliquer aux chunks de vertices
            chunk_size: Taille des chunks pour le traitement parallèle
            progress_callback: Fonction appelée avec le progrès (0.0 à 1.0)
            
        Returns:
            Maillage traité
            
        Raises:
            ValueError: Si les paramètres sont invalides
            RuntimeError: Si le traitement parallèle échoue
        """
        if not callable(process_fn):
            raise ValueError("process_fn doit être une fonction callable")
        
        if len(mesh.vertices) == 0:
            logger.warning("Maillage vide fourni pour le traitement parallèle")
            return mesh
            
        try:
            if progress_callback:
                progress_callback(0.0)
                
            if chunk_size is None:
                # Calculer une taille de chunk optimale basée sur le nombre de cœurs
                num_cores = os.cpu_count() or 4
                # S'assurer qu'on a au moins 1000 vertices par chunk pour l'efficacité
                min_chunk_size = 1000
                optimal_chunks = num_cores * 2
                chunk_size = max(min_chunk_size, len(mesh.vertices) // optimal_chunks)
                
            logger.info(f"Traitement parallèle: {len(mesh.vertices)} vertices, chunk_size={chunk_size}")
            
            # Diviser les vertices en chunks non vides
            num_chunks = max(1, len(mesh.vertices) // chunk_size)
            vertex_chunks = []
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(mesh.vertices))
                if start_idx < len(mesh.vertices):
                    vertex_chunks.append(mesh.vertices[start_idx:end_idx])
                    
            # S'assurer qu'on n'a pas de chunks vides
            vertex_chunks = [chunk for chunk in vertex_chunks if len(chunk) > 0]
            
            if not vertex_chunks:
                logger.warning("Aucun chunk valide créé, retour du maillage original")
                return mesh
                
            if progress_callback:
                progress_callback(0.2)
                
            logger.info(f"Traitement de {len(vertex_chunks)} chunks en parallèle")
            
            # Traiter en parallèle avec gestion d'erreurs
            processed_chunks = []
            try:
                with ThreadPoolExecutor(max_workers=min(len(vertex_chunks), os.cpu_count() or 4)) as executor:
                    futures = [executor.submit(process_fn, chunk) for chunk in vertex_chunks]
                    
                    for i, future in enumerate(futures):
                        try:
                            result = future.result(timeout=60)  # Timeout de 60s par chunk
                            processed_chunks.append(result)
                            if progress_callback:
                                progress_callback(0.2 + 0.6 * (i + 1) / len(futures))
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement du chunk {i}: {e}")
                            # Utiliser le chunk original en cas d'erreur
                            processed_chunks.append(vertex_chunks[i])
                            
            except Exception as e:
                logger.error(f"Erreur lors du traitement parallèle: {e}")
                raise RuntimeError(f"Échec du traitement parallèle: {str(e)}") from e
                
            if progress_callback:
                progress_callback(0.8)
                
            # Recombiner les chunks traités
            try:
                # Vérifier que tous les chunks ont la même forme
                if processed_chunks and all(
                    isinstance(chunk, np.ndarray) and chunk.shape[1] == 3 
                    for chunk in processed_chunks
                ):
                    mesh_copy = mesh.copy()
                    mesh_copy.vertices = np.concatenate(processed_chunks)
                    
                    # Vérifier que le nombre de vertices est cohérent
                    if len(mesh_copy.vertices) != len(mesh.vertices):
                        logger.warning(
                            f"Incohérence du nombre de vertices: {len(mesh.vertices)} → {len(mesh_copy.vertices)}"
                        )
                        return mesh  # Retourner le maillage original en cas d'incohérence
                        
                    if progress_callback:
                        progress_callback(1.0)
                        
                    logger.info("Traitement parallèle terminé avec succès")
                    return mesh_copy
                else:
                    logger.error("Format de chunks invalide après traitement")
                    return mesh
                    
            except Exception as e:
                logger.error(f"Erreur lors de la recombinaison des chunks: {e}")
                return mesh  # Retourner le maillage original en cas d'erreur
                
        except Exception as e:
            logger.error(f"Erreur générale lors du traitement parallèle: {e}")
            raise RuntimeError(f"Échec du traitement parallèle du maillage: {str(e)}") from e
        
    def batch_process_meshes(
        self,
        meshes: List[trimesh.Trimesh],
        process_fn: callable,
        progress_callback: Optional[callable] = None,
        max_workers: Optional[int] = None
    ) -> List[trimesh.Trimesh]:
        """
        Traite plusieurs maillages en parallèle.
        
        Args:
            meshes: Liste de maillages à traiter
            process_fn: Fonction de traitement à appliquer à chaque maillage
            progress_callback: Fonction appelée avec le progrès (0.0 à 1.0)
            max_workers: Nombre maximum de workers (défaut: nombre de CPU)
            
        Returns:
            Liste des maillages traités
            
        Raises:
            ValueError: Si les paramètres sont invalides
            RuntimeError: Si le traitement en lot échoue
        """
        if not meshes:
            logger.warning("Liste de maillages vide fournie")
            return []
            
        if not callable(process_fn):
            raise ValueError("process_fn doit être une fonction callable")
            
        if max_workers is None:
            max_workers = min(len(meshes), os.cpu_count() or 4)
            
        try:
            if progress_callback:
                progress_callback(0.0)
                
            logger.info(f"Traitement en lot de {len(meshes)} maillages avec {max_workers} workers")
            
            processed_meshes = []
            
            # Utiliser ProcessPoolExecutor pour un vrai parallélisme
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Soumettre tous les travaux
                futures = [executor.submit(process_fn, mesh) for mesh in meshes]
                
                # Collecter les résultats avec gestion d'erreurs
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # Timeout de 5 minutes par maillage
                        processed_meshes.append(result)
                        logger.debug(f"Maillage {i+1}/{len(meshes)} traité avec succès")
                        
                        if progress_callback:
                            progress_callback((i + 1) / len(meshes))
                            
                    except Exception as e:
                        logger.error(f"Erreur lors du traitement du maillage {i}: {e}")
                        # Ajouter le maillage original en cas d'erreur
                        processed_meshes.append(meshes[i])
                        
            logger.info(f"Traitement en lot terminé: {len(processed_meshes)} maillages traités")
            return processed_meshes
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement en lot: {e}")
            raise RuntimeError(f"Échec du traitement en lot des maillages: {str(e)}") from e
            
    def _adaptive_optimize(self):
        """Optimisation adaptative basée sur les conditions actuelles du système."""
        current_time = time.time()
        
        # Éviter les ajustements trop fréquents
        if current_time - self._adaptive_config['last_adjustment'] < 30:
            return
            
        if not self._adaptive_config['auto_adjust_workers']:
            return
            
        try:
            # Métriques système actuelles
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0
            
            # Ajustement du nombre de workers
            if memory_percent > self._adaptive_config['memory_threshold']:
                # Réduire les workers pour limiter l'usage mémoire
                current_threads = self.thread_pool._max_workers
                current_processes = self.process_pool._max_workers
                
                if current_threads > 2:
                    self.thread_pool._max_workers = max(2, current_threads - 2)
                    logger.info(f"Réduit thread workers: {current_threads} -> {self.thread_pool._max_workers}")
                    
                if current_processes > 1:
                    self.process_pool._max_workers = max(1, current_processes - 1)
                    logger.info(f"Réduit process workers: {current_processes} -> {self.process_pool._max_workers}")
                    
            elif cpu_percent < 50 and memory_percent < 0.6:
                # Augmenter les workers si ressources disponibles
                cpu_count = os.cpu_count() or 4
                current_threads = self.thread_pool._max_workers
                current_processes = self.process_pool._max_workers
                
                max_threads = min(cpu_count * 2, 32)
                max_processes = min(cpu_count, 16)
                
                if current_threads < max_threads:
                    self.thread_pool._max_workers = min(max_threads, current_threads + 2)
                    logger.info(f"Augmenté thread workers: {current_threads} -> {self.thread_pool._max_workers}")
                    
                if current_processes < max_processes:
                    self.process_pool._max_workers = min(max_processes, current_processes + 1)
                    logger.info(f"Augmenté process workers: {current_processes} -> {self.process_pool._max_workers}")
            
            self._adaptive_config['last_adjustment'] = current_time
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'optimisation adaptative: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance.
        
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        try:
            # Optimisation adaptative avant de collecter les stats
            self._adaptive_optimize()
            
            # Obtenir les informations système
            cpu_count = os.cpu_count() or 1
            memory_info = psutil.virtual_memory()
            
            # Obtenir les statistiques du cache
            cache_stats = {}
            try:
                cache_stats = self.cache_manager.get_stats() if hasattr(self.cache_manager, 'get_stats') else {}
            except Exception as e:
                logger.warning(f"Impossible d'obtenir les stats du cache: {e}")
                
            stats = {
                "system": {
                    "cpu_count": cpu_count,
                    "memory_total_gb": round(memory_info.total / (1024**3), 2),
                    "memory_available_gb": round(memory_info.available / (1024**3), 2),
                    "memory_usage_percent": memory_info.percent,
                },
                "cache": cache_stats,
                "thread_pools": {
                    "thread_pool_active": len(getattr(self.thread_pool, '_threads', [])),
                    "process_pool_active": len(getattr(self.process_pool, '_pool', [])) if hasattr(self.process_pool, '_pool') else 0,
                }
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Erreur lors de la collecte des stats de performance: {e}")
            return {"error": str(e)}
            
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Optimise l'utilisation de la mémoire et nettoie les ressources inutiles.
        
        Returns:
            Statistiques d'optimisation mémoire
        """
        logger.info("Début de l'optimisation mémoire avancée...")
        
        start_time = time.time()
        stats_before = {
            "memory_before_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "memory_percent_before": psutil.virtual_memory().percent
        }
        
        try:
            # Phase 1: Garbage collection agressif
            collected_objects = []
            for generation in range(3):  # 3 générations GC
                collected = gc.collect(generation)
                collected_objects.append(collected)
                logger.info(f"GC génération {generation}: {collected} objets collectés")
            
            total_collected = sum(collected_objects)
            
            # Phase 2: Optimisation GPU si disponible
            gpu_memory_freed = 0
            if torch.cuda.is_available():
                try:
                    # Mesurer mémoire GPU avant
                    torch.cuda.synchronize()
                    gpu_before = torch.cuda.memory_allocated()
                    
                    # Nettoyer le cache GPU de manière agressive
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    torch.cuda.synchronize()
                    
                    # Optimiser les pools de mémoire
                    if hasattr(torch.cuda, 'memory_stats'):
                        stats = torch.cuda.memory_stats()
                        if stats.get('reserved_bytes.small_pool.current', 0) > 0:
                            torch.cuda.memory._record_memory_history(enabled=False)
                    
                    gpu_after = torch.cuda.memory_allocated()
                    gpu_memory_freed = (gpu_before - gpu_after) / (1024**3)
                    logger.info(f"Mémoire GPU libérée: {gpu_memory_freed:.2f} GB")
                except Exception as e:
                    logger.warning(f"Erreur optimisation GPU: {e}")
            
            # Phase 3: Optimisation cache interne
            cache_items_cleaned = 0
            if hasattr(self, 'cache_manager'):
                try:
                    if hasattr(self.cache_manager, 'cleanup_memory'):
                        cache_items_cleaned = self.cache_manager.cleanup_memory()
                    elif hasattr(self.cache_manager, '_memory_cache'):
                        initial_size = len(self.cache_manager._memory_cache)
                        # Nettoyer les entrées faibles
                        self.cache_manager._memory_cache.clear()
                        cache_items_cleaned = initial_size
                    logger.info(f"Cache interne: {cache_items_cleaned} éléments nettoyés")
                except Exception as e:
                    logger.warning(f"Erreur nettoyage cache: {e}")
            
            # Phase 4: Optimisation des modules système
            import sys
            modules_optimized = 0
            modules_to_check = ['numpy', 'torch', 'trimesh', 'sklearn', 'matplotlib', 'cv2']
            
            for module_name in modules_to_check:
                if module_name in sys.modules:
                    try:
                        module = sys.modules[module_name]
                        cleaned_attrs = 0
                        
                        if hasattr(module, '__dict__'):
                            # Nettoyer tous les types de caches
                            cache_patterns = ['_cache', 'cache_', '__cache__', '_memo', 'memo_', '_lru_cache']
                            for attr_name in list(module.__dict__.keys()):
                                if any(pattern in attr_name.lower() for pattern in cache_patterns):
                                    try:
                                        attr = getattr(module, attr_name)
                                        if hasattr(attr, 'clear'):
                                            attr.clear()
                                        elif hasattr(attr, 'cache_clear'):
                                            attr.cache_clear()
                                        else:
                                            delattr(module, attr_name)
                                        cleaned_attrs += 1
                                    except:
                                        pass
                        
                        if cleaned_attrs > 0:
                            modules_optimized += 1
                            logger.debug(f"Module {module_name}: {cleaned_attrs} caches nettoyés")
                            
                    except Exception as e:
                        logger.debug(f"Erreur optimisation module {module_name}: {e}")
            
            # Phase 5: Optimisation système avancée
            try:
                # Défragmentation mémoire pour certains OS
                if hasattr(os, 'nice'):
                    original_nice = os.nice(0)
                    os.nice(-1)  # Augmenter la priorité temporairement
                    time.sleep(0.05)  # Courte pause pour l'optimisation
                    os.nice(original_nice - os.nice(0))
            except:
                pass
            
            # Forcer synchronisation mémoire
            time.sleep(0.1)
            
            # Statistiques finales
            stats_after = {
                "memory_after_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "memory_percent_after": psutil.virtual_memory().percent
            }
            
            memory_freed = stats_before["memory_before_gb"] - stats_after["memory_after_gb"]
            optimization_time = time.time() - start_time
            
            # Mettre à jour les stats de performance
            self._performance_stats['memory_optimizations'] += 1
            
            optimization_stats = {
                **stats_before,
                **stats_after,
                "memory_freed_gb": round(memory_freed, 2),
                "gpu_memory_freed_gb": round(gpu_memory_freed, 2),
                "objects_collected": total_collected,
                "cache_items_cleaned": cache_items_cleaned,
                "modules_optimized": modules_optimized,
                "optimization_time_ms": round(optimization_time * 1000, 2),
                "success": True,
                "efficiency_score": round(max(0, memory_freed / max(0.01, optimization_time)), 2)
            }
            
            logger.info(f"Optimisation mémoire terminée. Mémoire libérée: {memory_freed:.2f} GB")
            return optimization_stats
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation mémoire: {e}")
            return {
                **stats_before,
                "error": str(e),
                "success": False
            }
    
    def adaptive_mesh_processing(
        self,
        mesh: trimesh.Trimesh,
        target_complexity: str = "auto",
        memory_limit_gb: Optional[float] = None
    ) -> trimesh.Trimesh:
        """
        Traite un maillage de manière adaptative selon les ressources disponibles.
        
        Args:
            mesh: Maillage à traiter
            target_complexity: Complexité cible ("low", "medium", "high", "auto")
            memory_limit_gb: Limite mémoire en GB (auto-détectée si None)
            
        Returns:
            Maillage optimisé
        """
        if memory_limit_gb is None:
            # Utiliser 75% de la mémoire disponible
            available_memory = psutil.virtual_memory().available / (1024**3)
            memory_limit_gb = available_memory * 0.75
            
        # Estimer la complexité du maillage
        mesh_complexity = self._estimate_mesh_complexity(mesh)
        logger.info(f"Complexité estimée du maillage: {mesh_complexity}")
        
        # Déterminer la stratégie de traitement
        if target_complexity == "auto":
            if memory_limit_gb > 8 and mesh_complexity < 0.7:
                strategy = "high_quality"
            elif memory_limit_gb > 4 and mesh_complexity < 0.5:
                strategy = "medium_quality"
            else:
                strategy = "low_memory"
        else:
            strategy_map = {
                "low": "low_memory",
                "medium": "medium_quality", 
                "high": "high_quality"
            }
            strategy = strategy_map.get(target_complexity, "medium_quality")
        
        logger.info(f"Stratégie de traitement sélectionnée: {strategy}")
        
        # Appliquer la stratégie
        try:
            if strategy == "high_quality":
                return self._process_high_quality(mesh)
            elif strategy == "medium_quality":
                return self._process_medium_quality(mesh)
            else:
                return self._process_low_memory(mesh)
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement adaptatif: {e}")
            # Fallback vers le traitement minimal
            return self._process_low_memory(mesh)
    
    def _estimate_mesh_complexity(self, mesh: trimesh.Trimesh) -> float:
        """Estime la complexité d'un maillage (0.0 = simple, 1.0 = très complexe)."""
        try:
            # Facteurs de complexité
            vertex_count = len(mesh.vertices)
            face_count = len(mesh.faces)
            
            # Normaliser par rapport à des valeurs de référence
            vertex_complexity = min(vertex_count / 100000, 1.0)  # 100k vertices = complexité max
            face_complexity = min(face_count / 200000, 1.0)  # 200k faces = complexité max
            
            # Autres facteurs
            edge_complexity = 0.0
            if mesh.edges is not None:
                edge_complexity = min(len(mesh.edges) / 300000, 1.0)
            
            # Complexité géométrique (variance des normales)
            geom_complexity = 0.0
            if hasattr(mesh, 'face_normals') and mesh.face_normals is not None:
                normal_variance = np.var(mesh.face_normals)
                geom_complexity = min(normal_variance * 10, 1.0)
            
            # Score final pondéré
            complexity = (
                vertex_complexity * 0.3 +
                face_complexity * 0.3 +
                edge_complexity * 0.2 +
                geom_complexity * 0.2
            )
            
            return min(complexity, 1.0)
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'estimation de complexité: {e}")
            return 0.5  # Complexité moyenne par défaut
    
    def _process_high_quality(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Traitement haute qualité avec toutes les optimisations."""
        processed = mesh.copy()
        
        # Nettoyer le maillage
        processed.remove_degenerate_faces()
        processed.remove_duplicate_faces()
        processed.remove_unreferenced_vertices()
        
        # Améliorer les normales
        processed.fix_normals()
        
        # Lissage léger pour améliorer la qualité
        if hasattr(processed, 'smoothed'):
            processed = processed.smoothed()
        
        return processed
    
    def _process_medium_quality(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Traitement qualité moyenne avec optimisations sélectives."""
        processed = mesh.copy()
        
        # Optimisations essentielles
        processed.remove_duplicate_faces()
        processed.remove_unreferenced_vertices()
        processed.fix_normals()
        
        return processed
    
    def _process_low_memory(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Traitement minimal pour économiser la mémoire."""
        processed = mesh.copy()
        
        # Seulement les opérations critiques
        processed.remove_unreferenced_vertices()
        
        # Simplification si nécessaire
        if len(processed.faces) > 50000:
            try:
                processed = processed.simplify_quadric_decimation(face_count=50000)
            except:
                logger.warning("Simplification échouée, maillage conservé tel quel")
        
        return processed
    
    def _apply_adaptive_simplification(
        self, 
        mesh: trimesh.Trimesh, 
        target_faces: int,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Applique une simplification adaptative avec plusieurs stratégies.
        
        Returns:
            bool: True si la simplification a réussi, False sinon
        """
        try:
            original_faces = len(mesh.faces)
            
            # Stratégie 1: Simplification quadratique (plus précise)
            try:
                mesh_simplified = mesh.simplify_quadratic_decimation(target_faces)
                if len(mesh_simplified.faces) <= target_faces * 1.1:  # Tolérance de 10%
                    mesh.vertices = mesh_simplified.vertices
                    mesh.faces = mesh_simplified.faces
                    logger.info(f"Simplification quadratique réussie: {original_faces} → {len(mesh.faces)} faces")
                    return True
            except Exception as e:
                logger.debug(f"Simplification quadratique échouée: {e}")
            
            # Stratégie 2: Simplification quadric (fallback)
            try:
                mesh_simplified = mesh.simplify_quadric_decimation(target_faces)
                if len(mesh_simplified.faces) <= target_faces * 1.2:  # Tolérance plus large
                    mesh.vertices = mesh_simplified.vertices
                    mesh.faces = mesh_simplified.faces
                    logger.info(f"Simplification quadric réussie: {original_faces} → {len(mesh.faces)} faces")
                    return True
            except Exception as e:
                logger.debug(f"Simplification quadric échouée: {e}")
            
            # Stratégie 3: Simplification progressive (plus conservative)
            if target_faces < original_faces * 0.8:  # Seulement si réduction significative
                try:
                    reduction_factor = target_faces / original_faces
                    intermediate_target = int(original_faces * (reduction_factor + 0.2))
                    mesh_simplified = mesh.simplify_quadric_decimation(intermediate_target)
                    mesh.vertices = mesh_simplified.vertices
                    mesh.faces = mesh_simplified.faces
                    logger.info(f"Simplification progressive: {original_faces} → {len(mesh.faces)} faces")
                    return True
                except Exception as e:
                    logger.debug(f"Simplification progressive échouée: {e}")
            
            logger.warning(f"Toutes les stratégies de simplification ont échoué pour {original_faces} → {target_faces} faces")
            return False
            
        except Exception as e:
            logger.error(f"Erreur inattendue dans la simplification adaptative: {e}")
            return False
            
    def cleanup(self):
        """Nettoie les ressources utilisées par l'optimiseur."""
        try:
            logger.info("Nettoyage des ressources de l'optimiseur de performance...")
            
            # Fermer les pools de threads/processus
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
                
            if hasattr(self, 'process_pool'):
                self.process_pool.shutdown(wait=True)
                
            # Nettoyer le cache si possible
            if hasattr(self.cache_manager, 'cleanup'):
                self.cache_manager.cleanup()
                
            logger.info("Nettoyage terminé")
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            
    def __del__(self):
        """Destructeur pour s'assurer que les ressources sont libérées."""
        self.cleanup()
    
    def real_time_profiler(
        self,
        operation_name: str,
        target_function: callable,
        *args,
        **kwargs
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Profile une opération en temps réel avec métriques détaillées.
        
        Args:
            operation_name: Nom de l'opération
            target_function: Fonction à profiler
            *args, **kwargs: Arguments pour la fonction
            
        Returns:
            Tuple (résultat, métriques de profiling)
        """
        import time
        import tracemalloc
        
        # Démarrer le tracing mémoire
        tracemalloc.start()
        
        # Métriques initiales
        initial_time = time.time()
        initial_memory = psutil.virtual_memory()
        initial_gpu_memory = self._get_gpu_memory_usage()
        
        try:
            # Exécuter la fonction
            result = target_function(*args, **kwargs)
            
            # Métriques finales
            execution_time = time.time() - initial_time
            final_memory = psutil.virtual_memory()
            final_gpu_memory = self._get_gpu_memory_usage()
            
            # Analyser la trace mémoire
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculer les métriques
            memory_delta = final_memory.used - initial_memory.used
            gpu_memory_delta = final_gpu_memory - initial_gpu_memory
            
            profiling_metrics = {
                'operation_name': operation_name,
                'execution_time_seconds': execution_time,
                'memory_usage': {
                    'peak_mb': peak / (1024 * 1024),
                    'current_mb': current / (1024 * 1024),
                    'system_delta_mb': memory_delta / (1024 * 1024)
                },
                'gpu_memory_delta_mb': gpu_memory_delta / (1024 * 1024) if gpu_memory_delta else 0,
                'performance_category': self._categorize_performance(execution_time, peak),
                'recommendations': self._generate_performance_recommendations(execution_time, peak, memory_delta)
            }
            
            # Log des résultats
            logger.info(f"Profiling {operation_name}: {execution_time:.2f}s, "
                       f"peak: {peak/(1024*1024):.1f}MB, "
                       f"catégorie: {profiling_metrics['performance_category']}")
            
            return result, profiling_metrics
            
        except Exception as e:
            tracemalloc.stop()
            logger.error(f"Erreur lors du profiling de {operation_name}: {e}")
            raise
    
    def _categorize_performance(self, execution_time: float, peak_memory: int) -> str:
        """Catégorise la performance d'une opération."""
        memory_mb = peak_memory / (1024 * 1024)
        
        if execution_time < 1.0 and memory_mb < 100:
            return "excellent"
        elif execution_time < 5.0 and memory_mb < 500:
            return "bon"
        elif execution_time < 15.0 and memory_mb < 1000:
            return "acceptable"
        elif execution_time < 60.0 and memory_mb < 2000:
            return "lent"
        else:
            return "très_lent"
    
    def _generate_performance_recommendations(
        self,
        execution_time: float,
        peak_memory: int,
        memory_delta: int
    ) -> List[str]:
        """Génère des recommandations d'optimisation."""
        recommendations = []
        
        memory_mb = peak_memory / (1024 * 1024)
        delta_mb = memory_delta / (1024 * 1024)
        
        if execution_time > 30.0:
            recommendations.append("Considérer l'utilisation de GPU ou parallélisation")
        
        if memory_mb > 1000:
            recommendations.append("Optimiser l'usage mémoire ou utiliser le streaming")
        
        if delta_mb > 500:
            recommendations.append("Possible fuite mémoire, vérifier les références")
        
        if execution_time > 10.0 and memory_mb < 200:
            recommendations.append("Opération CPU-intensive, optimiser les algorithmes")
        
        if not recommendations:
            recommendations.append("Performance optimale")
        
        return recommendations
    
    def auto_resource_optimization(
        self,
        workload_type: str = "balanced"
    ) -> Dict[str, Any]:
        """
        Optimisation automatique des ressources basée sur le type de charge.
        
        Args:
            workload_type: Type de charge ("cpu_intensive", "memory_intensive", "gpu_intensive", "balanced")
            
        Returns:
            Configuration optimisée
        """
        system_info = self._get_system_capabilities()
        
        if workload_type == "cpu_intensive":
            config = self._optimize_for_cpu_workload(system_info)
        elif workload_type == "memory_intensive":
            config = self._optimize_for_memory_workload(system_info)
        elif workload_type == "gpu_intensive":
            config = self._optimize_for_gpu_workload(system_info)
        else:  # balanced
            config = self._optimize_balanced_workload(system_info)
        
        # Appliquer la configuration
        self._apply_optimization_config(config)
        
        logger.info(f"Configuration optimisée pour {workload_type}: {config}")
        return config
    
    def _get_system_capabilities(self) -> Dict[str, Any]:
        """Analyse les capacités système."""
        try:
            memory_info = psutil.virtual_memory()
            cpu_info = {
                'count': os.cpu_count(),
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            
            gpu_info = {
                'available': torch.cuda.is_available(),
                'count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'memory_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
            }
            
            return {
                'memory': {
                    'total_gb': memory_info.total / (1024**3),
                    'available_gb': memory_info.available / (1024**3),
                    'usage_percent': memory_info.percent
                },
                'cpu': cpu_info,
                'gpu': gpu_info
            }
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse système: {e}")
            return self._get_default_system_info()
    
    def _optimize_for_cpu_workload(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour les charges CPU intensives."""
        cpu_count = system_info['cpu']['count']
        return {
            'thread_workers': min(cpu_count * 2, 32),
            'process_workers': min(cpu_count, 16),
            'memory_limit_gb': system_info['memory']['available_gb'] * 0.6,
            'batch_size': max(1, cpu_count // 2),
            'enable_multiprocessing': True,
            'optimization_level': 'cpu_intensive'
        }
    
    def _optimize_for_memory_workload(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour les charges mémoire intensives."""
        available_memory = system_info['memory']['available_gb']
        return {
            'thread_workers': 4,  # Limiter les threads pour économiser la mémoire
            'process_workers': 2,
            'memory_limit_gb': available_memory * 0.8,
            'batch_size': 1,  # Traitement séquentiel
            'enable_memory_mapping': True,
            'enable_streaming': True,
            'optimization_level': 'memory_intensive'
        }
    
    def _optimize_for_gpu_workload(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour les charges GPU intensives."""
        gpu_available = system_info['gpu']['available']
        gpu_memory = system_info['gpu']['memory_gb']
        
        return {
            'thread_workers': 8,
            'process_workers': 4,
            'memory_limit_gb': system_info['memory']['available_gb'] * 0.5,
            'batch_size': min(16, int(gpu_memory)) if gpu_available else 4,
            'enable_gpu_acceleration': gpu_available,
            'gpu_memory_limit_gb': gpu_memory * 0.8 if gpu_available else 0,
            'optimization_level': 'gpu_intensive'
        }
    
    def _optimize_balanced_workload(self, system_info: Dict[str, Any]) -> Dict[str, Any]:
        """Optimise pour une charge équilibrée."""
        cpu_count = system_info['cpu']['count']
        available_memory = system_info['memory']['available_gb']
        
        return {
            'thread_workers': min(cpu_count, 16),
            'process_workers': min(cpu_count // 2, 8),
            'memory_limit_gb': available_memory * 0.7,
            'batch_size': min(8, cpu_count),
            'enable_adaptive_optimization': True,
            'optimization_level': 'balanced'
        }
    
    def _apply_optimization_config(self, config: Dict[str, Any]):
        """Applique la configuration d'optimisation."""
        try:
            # Ajuster les pools de workers si nécessaire
            if hasattr(self, 'thread_pool'):
                self.thread_pool._max_workers = config.get('thread_workers', 8)
            
            if hasattr(self, 'process_pool'):
                self.process_pool._max_workers = config.get('process_workers', 4)
            
            # Mettre à jour la configuration adaptative
            if hasattr(self, '_adaptive_config'):
                self._adaptive_config.update({
                    'memory_threshold': min(0.8, config.get('memory_limit_gb', 8) / 16),
                    'auto_adjust_workers': config.get('enable_adaptive_optimization', True)
                })
            
            logger.info("Configuration d'optimisation appliquée avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'application de la configuration: {e}")
    
    def _get_default_system_info(self) -> Dict[str, Any]:
        """Retourne des informations système par défaut en cas d'erreur."""
        return {
            'memory': {'total_gb': 8, 'available_gb': 4, 'usage_percent': 50},
            'cpu': {'count': 4, 'usage_percent': 25, 'load_avg': [0.5, 0.5, 0.5]},
            'gpu': {'available': False, 'count': 0, 'memory_gb': 0}
        }
    
    def _get_gpu_memory_usage(self) -> int:
        """Retourne l'usage mémoire GPU en octets."""
        if torch.cuda.is_available():
            try:
                return torch.cuda.memory_allocated()
            except:
                return 0
        return 0
    
    def bottleneck_detector(
        self,
        operation_metrics: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Détecte les goulots d'étranglement dans une séquence d'opérations.
        
        Args:
            operation_metrics: Liste des métriques d'opérations
            
        Returns:
            Analyse des goulots d'étranglement
        """
        if not operation_metrics:
            return {'bottlenecks': [], 'recommendations': ['Aucune donnée à analyser']}
        
        bottlenecks = []
        total_time = sum(op['execution_time_seconds'] for op in operation_metrics)
        
        for op in operation_metrics:
            time_percent = (op['execution_time_seconds'] / total_time) * 100
            memory_peak = op['memory_usage']['peak_mb']
            
            # Identifier les goulots d'étranglement
            if time_percent > 50:
                bottlenecks.append({
                    'type': 'temps_execution',
                    'operation': op['operation_name'],
                    'impact_percent': time_percent,
                    'details': f"Consomme {time_percent:.1f}% du temps total"
                })
            
            if memory_peak > 1000:  # Plus de 1GB
                bottlenecks.append({
                    'type': 'consommation_memoire',
                    'operation': op['operation_name'],
                    'impact_mb': memory_peak,
                    'details': f"Pic mémoire de {memory_peak:.1f}MB"
                })
        
        # Générer des recommandations
        recommendations = self._generate_bottleneck_recommendations(bottlenecks, operation_metrics)
        
        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'total_operations': len(operation_metrics),
            'total_time_seconds': total_time,
            'average_time_per_operation': total_time / len(operation_metrics)
        }
    
    def _generate_bottleneck_recommendations(
        self,
        bottlenecks: List[Dict[str, Any]],
        operation_metrics: List[Dict[str, Any]]
    ) -> List[str]:
        """Génère des recommandations pour résoudre les goulots d'étranglement."""
        recommendations = []
        
        time_bottlenecks = [b for b in bottlenecks if b['type'] == 'temps_execution']
        memory_bottlenecks = [b for b in bottlenecks if b['type'] == 'consommation_memoire']
        
        if time_bottlenecks:
            recommendations.append("Optimiser les opérations les plus lentes avec parallélisation ou GPU")
            recommendations.append("Considérer le cache pour les opérations répétitives")
        
        if memory_bottlenecks:
            recommendations.append("Implémenter le streaming pour les gros volumes de données")
            recommendations.append("Utiliser la compression ou réduire la précision si possible")
        
        # Analyser les patterns
        cpu_intensive_ops = [op for op in operation_metrics 
                           if op['performance_category'] in ['lent', 'très_lent']]
        
        if len(cpu_intensive_ops) > len(operation_metrics) / 2:
            recommendations.append("Considérer l'upgrade hardware ou l'optimisation algorithmique")
        
        if not recommendations:
            recommendations.append("Performance globalement satisfaisante")
        
        return recommendations
        self.cleanup()
