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
                
            # Simplification du maillage avec gestion d'erreurs
            try:
                if target_faces < len(mesh.faces):
                    optimized = optimized.simplify_quadratic_decimation(target_faces)
                    logger.info(f"Simplification terminée: {len(optimized.faces)} faces")
            except Exception as e:
                logger.warning(f"Échec de la simplification quadratique: {e}")
                # Fallback vers une simplification plus basique
                try:
                    optimized = optimized.simplify_quadric_decimation(target_faces)
                except Exception as e2:
                    logger.warning(f"Échec de la simplification de fallback: {e2}")
                    # Continuer sans simplification
                    
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
                optimized.merge_vertices(digits_precision=8)
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
                
            # Mettre en cache avec gestion d'erreurs
            try:
                self.cache_manager.put(optimized, mesh_hash)
                logger.info("Maillage optimisé mis en cache")
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
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Retourne les statistiques de performance.
        
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        try:
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
        logger.info("Début de l'optimisation mémoire...")
        
        stats_before = {
            "memory_before_gb": round(psutil.virtual_memory().used / (1024**3), 2),
            "memory_percent_before": psutil.virtual_memory().percent
        }
        
        try:
            # Forcer le garbage collection
            collected = gc.collect()
            logger.info(f"Garbage collection: {collected} objets collectés")
            
            # Nettoyer les caches PyTorch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("Cache GPU PyTorch vidé")
            
            # Nettoyer le cache interne si disponible
            if hasattr(self, 'cache_manager') and hasattr(self.cache_manager, 'cleanup_memory'):
                self.cache_manager.cleanup_memory()
                logger.info("Cache interne nettoyé")
            
            # Optimiser les variables globales numpy/torch
            import sys
            modules_to_check = ['numpy', 'torch', 'trimesh']
            for module_name in modules_to_check:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    if hasattr(module, '__dict__'):
                        # Nettoyer les caches internes des modules
                        for attr_name in list(module.__dict__.keys()):
                            if attr_name.startswith('_cache') or attr_name.endswith('_cache'):
                                try:
                                    delattr(module, attr_name)
                                except:
                                    pass
            
            # Attendre que le nettoyage soit effectif
            time.sleep(0.1)
            
            stats_after = {
                "memory_after_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "memory_percent_after": psutil.virtual_memory().percent
            }
            
            memory_freed = stats_before["memory_before_gb"] - stats_after["memory_after_gb"]
            
            optimization_stats = {
                **stats_before,
                **stats_after,
                "memory_freed_gb": round(memory_freed, 2),
                "objects_collected": collected,
                "optimization_time": time.time(),
                "success": True
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
