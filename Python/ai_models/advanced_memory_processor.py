"""
Advanced memory-aware processor for handling large meshes efficiently.
Implements intelligent chunking, memory monitoring, and automatic fallback strategies.
"""

import time
import psutil
import gc
import logging
import numpy as np
import trimesh
from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import weakref
from pathlib import Path
import tempfile
import mmap
import pickle

try:
    from ..core.enhanced_exceptions import MemoryError as MacForgeMemoryError, PerformanceError
except ImportError:
    # Fallback pour les tests directs
    from core.enhanced_exceptions import MemoryError as MacForgeMemoryError, PerformanceError

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Configuration pour la gestion mémoire."""
    max_memory_percent: float = 75.0  # Pourcentage max de RAM utilisable
    chunk_memory_limit_mb: float = 512.0  # Limite mémoire par chunk en MB
    enable_disk_cache: bool = True
    cache_dir: Optional[Path] = None
    enable_compression: bool = True
    compression_level: int = 3
    monitoring_interval: float = 1.0  # Secondes entre les vérifications mémoire

@dataclass
class ProcessingStats:
    """Statistiques de traitement."""
    total_vertices: int
    total_faces: int
    chunks_processed: int
    memory_used_mb: float
    processing_time: float
    cache_hits: int
    cache_misses: int
    fallback_used: bool

class MemoryMonitor:
    """Moniteur de mémoire en temps réel."""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.monitoring = False
        self.peak_memory_mb = 0.0
        self.current_memory_mb = 0.0
        self._monitor_thread = None
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """Démarre la surveillance mémoire."""
        if not self.monitoring:
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            
    def stop_monitoring(self):
        """Arrête la surveillance mémoire."""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self):
        """Boucle de surveillance mémoire."""
        while self.monitoring:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                current_mb = memory_info.rss / (1024 * 1024)
                
                with self._lock:
                    self.current_memory_mb = current_mb
                    self.peak_memory_mb = max(self.peak_memory_mb, current_mb)
                
                # Vérifier si on dépasse la limite
                system_memory = psutil.virtual_memory()
                memory_percent = (memory_info.rss / system_memory.total) * 100
                
                if memory_percent > self.config.max_memory_percent:
                    logger.warning(f"Utilisation mémoire élevée: {memory_percent:.1f}%")
                    gc.collect()  # Forcer le garbage collection
                    
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Erreur surveillance mémoire: {e}")
                break
                
    def get_memory_stats(self) -> Dict[str, float]:
        """Retourne les statistiques mémoire."""
        with self._lock:
            return {
                "current_mb": self.current_memory_mb,
                "peak_mb": self.peak_memory_mb,
                "available_mb": psutil.virtual_memory().available / (1024 * 1024)
            }

class AdvancedMemoryProcessor:
    """Processeur avancé avec gestion mémoire intelligente."""
    
    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.monitor = MemoryMonitor(self.config)
        self.cache_dir = self.config.cache_dir or Path(tempfile.gettempdir()) / "macforge_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}
        
    def process_large_mesh(
        self,
        mesh: trimesh.Trimesh,
        process_fn: Callable,
        max_chunk_vertices: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> Tuple[trimesh.Trimesh, ProcessingStats]:
        """
        Traite un large maillage avec gestion mémoire intelligente.
        
        Args:
            mesh: Maillage à traiter
            process_fn: Fonction de traitement
            max_chunk_vertices: Taille max des chunks
            progress_callback: Callback de progression
            
        Returns:
            Tuple (maillage traité, statistiques)
        """
        start_time = time.time()
        self.monitor.start_monitoring()
        
        try:
            # Analyser la mémoire disponible
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            # Calculer la taille optimale des chunks
            if max_chunk_vertices is None:
                max_chunk_vertices = self._calculate_optimal_chunk_size(mesh, available_memory_mb)
            
            logger.info(f"Traitement maillage: {len(mesh.vertices)} vertices, chunk_size={max_chunk_vertices}")
            
            # Diviser le maillage en chunks
            chunks = self._create_mesh_chunks(mesh, max_chunk_vertices)
            
            if progress_callback:
                progress_callback(0.1)
            
            # Traiter les chunks
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                try:
                    # Vérifier la mémoire avant traitement
                    self._check_memory_before_processing()
                    
                    # Traiter le chunk
                    processed_chunk = self._process_chunk_with_cache(chunk, process_fn)
                    processed_chunks.append(processed_chunk)
                    
                    if progress_callback:
                        progress = 0.1 + 0.8 * (i + 1) / len(chunks)
                        progress_callback(progress)
                        
                except Exception as e:
                    logger.error(f"Erreur traitement chunk {i}: {e}")
                    # Utiliser une stratégie de fallback
                    fallback_chunk = self._fallback_process_chunk(chunk, process_fn)
                    processed_chunks.append(fallback_chunk)
            
            # Reconstituer le maillage
            result_mesh = self._merge_chunks(processed_chunks, mesh.faces)
            
            if progress_callback:
                progress_callback(1.0)
            
            # Créer les statistiques
            stats = ProcessingStats(
                total_vertices=len(mesh.vertices),
                total_faces=len(mesh.faces),
                chunks_processed=len(chunks),
                memory_used_mb=self.monitor.get_memory_stats()["peak_mb"],
                processing_time=time.time() - start_time,
                cache_hits=self._cache_stats["hits"],
                cache_misses=self._cache_stats["misses"],
                fallback_used=False  # TODO: tracker les fallbacks
            )
            
            return result_mesh, stats
            
        finally:
            self.monitor.stop_monitoring()
    
    def _calculate_optimal_chunk_size(self, mesh: trimesh.Trimesh, available_memory_mb: float) -> int:
        """Calcule la taille optimale des chunks basée sur la mémoire disponible."""
        
        # Estimer la mémoire nécessaire par vertex (en bytes)
        # Position (3 float32) + normales (3 float32) + overhead = ~32 bytes
        bytes_per_vertex = 32
        
        # Utiliser seulement une partie de la mémoire disponible pour être sûr
        usable_memory_mb = available_memory_mb * 0.3  # 30% de la mémoire disponible
        usable_memory_bytes = usable_memory_mb * 1024 * 1024
        
        # Calculer le nombre de vertices possibles
        max_vertices = int(usable_memory_bytes / bytes_per_vertex)
        
        # S'assurer qu'on a une taille minimale et maximale raisonnable
        min_chunk_size = 1000
        max_chunk_size = 50000
        
        optimal_size = max(min_chunk_size, min(max_vertices, max_chunk_size))
        
        logger.info(f"Taille de chunk calculée: {optimal_size} vertices (mémoire dispo: {available_memory_mb:.1f}MB)")
        
        return optimal_size
    
    def _create_mesh_chunks(self, mesh: trimesh.Trimesh, chunk_size: int) -> List[np.ndarray]:
        """Divise le maillage en chunks de taille appropriée."""
        vertices = mesh.vertices
        chunks = []
        
        for i in range(0, len(vertices), chunk_size):
            end_idx = min(i + chunk_size, len(vertices))
            chunk = vertices[i:end_idx].copy()
            chunks.append(chunk)
        
        logger.info(f"Créé {len(chunks)} chunks de vertices")
        return chunks
    
    def _check_memory_before_processing(self):
        """Vérifie la mémoire avant de traiter un chunk."""
        memory_stats = self.monitor.get_memory_stats()
        
        if memory_stats["available_mb"] < self.config.chunk_memory_limit_mb:
            logger.warning("Mémoire faible, nettoyage...")
            gc.collect()
            
            # Vérifier à nouveau
            memory_stats = self.monitor.get_memory_stats()
            if memory_stats["available_mb"] < self.config.chunk_memory_limit_mb / 2:
                raise MacForgeMemoryError(
                    f"Mémoire insuffisante: {memory_stats['available_mb']:.1f}MB disponible",
                    required_memory=self.config.chunk_memory_limit_mb / 1024
                )
    
    def _process_chunk_with_cache(self, chunk: np.ndarray, process_fn: Callable) -> np.ndarray:
        """Traite un chunk avec mise en cache."""
        
        # Créer une clé de cache basée sur le contenu
        cache_key = self._generate_cache_key(chunk, process_fn.__name__)
        
        # Vérifier le cache
        if cache_key in self._cache:
            self._cache_stats["hits"] += 1
            logger.debug(f"Cache hit pour chunk {cache_key[:8]}...")
            return self._cache[cache_key]
        
        # Cache miss, traiter le chunk
        self._cache_stats["misses"] += 1
        
        try:
            result = process_fn(chunk)
            
            # Mettre en cache si pas trop gros
            result_size_mb = result.nbytes / (1024 * 1024)
            if result_size_mb < self.config.chunk_memory_limit_mb / 4:
                self._cache[cache_key] = result.copy()
                
                # Limiter la taille du cache
                if len(self._cache) > 100:
                    # Supprimer les anciens éléments
                    keys_to_remove = list(self._cache.keys())[:20]
                    for key in keys_to_remove:
                        del self._cache[key]
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur traitement chunk: {e}")
            raise
    
    def _fallback_process_chunk(self, chunk: np.ndarray, process_fn: Callable) -> np.ndarray:
        """Stratégie de fallback pour traiter un chunk en cas d'erreur."""
        
        try:
            # Essayer de subdiviser le chunk
            if len(chunk) > 100:
                mid = len(chunk) // 2
                chunk1 = chunk[:mid]
                chunk2 = chunk[mid:]
                
                result1 = self._process_chunk_with_cache(chunk1, process_fn)
                result2 = self._process_chunk_with_cache(chunk2, process_fn)
                
                return np.vstack([result1, result2])
            else:
                # Pour les très petits chunks, retourner tel quel
                logger.warning("Chunk trop petit pour subdivision, retour sans traitement")
                return chunk
                
        except Exception as e:
            logger.error(f"Fallback échoué: {e}")
            return chunk  # Retourner le chunk original
    
    def _merge_chunks(self, chunks: List[np.ndarray], original_faces: np.ndarray) -> trimesh.Trimesh:
        """Reconstruit le maillage à partir des chunks traités."""
        
        if not chunks:
            raise ValueError("Aucun chunk à fusionner")
        
        # Concaténer tous les vertices
        all_vertices = np.vstack(chunks)
        
        # Créer le nouveau maillage
        # Note: on garde les faces originales car on n'a traité que les vertices
        result_mesh = trimesh.Trimesh(
            vertices=all_vertices,
            faces=original_faces,
            process=False
        )
        
        logger.info(f"Maillage reconstruit: {len(all_vertices)} vertices, {len(original_faces)} faces")
        
        return result_mesh
    
    def _generate_cache_key(self, data: np.ndarray, function_name: str) -> str:
        """Génère une clé de cache pour les données."""
        # Utiliser un hash simple basé sur la forme et quelques valeurs
        shape_str = str(data.shape)
        sample_values = data.flatten()[:10] if data.size > 10 else data.flatten()
        sample_str = str(sample_values.tolist())
        
        import hashlib
        key_content = f"{function_name}_{shape_str}_{sample_str}"
        return hashlib.md5(key_content.encode()).hexdigest()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du cache."""
        return {
            "cache_size": len(self._cache),
            "hits": self._cache_stats["hits"],
            "misses": self._cache_stats["misses"],
            "hit_rate": self._cache_stats["hits"] / max(1, self._cache_stats["hits"] + self._cache_stats["misses"])
        }
    
    def clear_cache(self):
        """Vide le cache pour libérer la mémoire."""
        self._cache.clear()
        self._cache_stats = {"hits": 0, "misses": 0}
        gc.collect()
        logger.info("Cache vidé")

# Fonctions utilitaires pour l'optimisation mémoire

def optimize_mesh_memory_usage(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Optimise l'utilisation mémoire d'un maillage."""
    
    # Convertir en types de données plus efficaces si possible
    vertices = mesh.vertices.astype(np.float32)  # float64 -> float32
    faces = mesh.faces.astype(np.uint32)  # Potentiellement plus efficace
    
    # Supprimer les vertices non utilisés
    unique_vertices, inverse_indices = np.unique(
        vertices.view(np.void),
        return_inverse=True
    )
    
    if len(unique_vertices) < len(vertices):
        logger.info(f"Suppression de {len(vertices) - len(unique_vertices)} vertices dupliqués")
        
        # Reconstruire le maillage avec les vertices uniques
        new_vertices = unique_vertices.view(np.float32).reshape(-1, 3)
        new_faces = inverse_indices[faces]
        
        return trimesh.Trimesh(vertices=new_vertices, faces=new_faces)
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def estimate_mesh_memory_usage(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Estime l'utilisation mémoire d'un maillage."""
    
    vertices_mb = mesh.vertices.nbytes / (1024 * 1024)
    faces_mb = mesh.faces.nbytes / (1024 * 1024)
    
    # Estimation des structures additionnelles
    overhead_mb = (vertices_mb + faces_mb) * 0.3  # ~30% d'overhead estimé
    
    return {
        "vertices_mb": vertices_mb,
        "faces_mb": faces_mb,
        "overhead_mb": overhead_mb,
        "total_mb": vertices_mb + faces_mb + overhead_mb
    }