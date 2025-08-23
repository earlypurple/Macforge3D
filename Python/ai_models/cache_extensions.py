"""
Module d'extension pour la compression et le profilage du cache.
"""

import zstandard as zstd
import lz4.frame
import blosc2
import time
import numpy as np
import torch
import psutil
import logging
import cProfile
import pstats
import io
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import GPUtil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration de la compression."""
    algorithm: str = "zstd"  # "zstd", "lz4", "blosc"
    level: int = 3  # 1-22 pour zstd, 1-16 pour lz4, 1-9 pour blosc
    enable_gpu_compression: bool = True
    min_size_for_compression: int = 1024  # 1KB
    compression_threads: int = 4

@dataclass
class ProfilingResult:
    """Résultat du profilage."""
    operation: str
    duration: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float]
    compression_ratio: Optional[float]
    cache_hits: int
    cache_misses: int

class CacheProfiler:
    """Profileur pour le cache et les opérations."""
    
    def __init__(self):
        self.profiles: Dict[str, List[ProfilingResult]] = {}
        self.operation_counters = Counter()
        self._profiler = cProfile.Profile()
        
    def start_operation(self, operation: str):
        """Démarre le profilage d'une opération."""
        self._profiler.enable()
        return time.time()
        
    def end_operation(
        self,
        operation: str,
        start_time: float,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> ProfilingResult:
        """
        Termine le profilage d'une opération.
        
        Args:
            operation: Nom de l'opération
            start_time: Temps de début
            additional_data: Données supplémentaires à inclure
            
        Returns:
            Résultat du profilage
        """
        self._profiler.disable()
        duration = time.time() - start_time
        
        # Mesurer l'utilisation des ressources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Mesurer l'utilisation GPU si disponible
        gpu_usage = None
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except Exception:
                pass
                
        # Créer le résultat
        result = ProfilingResult(
            operation=operation,
            duration=duration,
            memory_usage=memory_percent,
            cpu_usage=cpu_percent,
            gpu_usage=gpu_usage,
            compression_ratio=additional_data.get("compression_ratio"),
            cache_hits=additional_data.get("cache_hits", 0),
            cache_misses=additional_data.get("cache_misses", 0)
        )
        
        # Sauvegarder le profil
        if operation not in self.profiles:
            self.profiles[operation] = []
        self.profiles[operation].append(result)
        
        self.operation_counters[operation] += 1
        
        return result
        
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtient les statistiques de profilage.
        
        Args:
            operation: Si spécifié, retourne les stats pour cette opération
            
        Returns:
            Statistiques de profilage
        """
        if operation:
            if operation not in self.profiles:
                return {}
                
            profiles = self.profiles[operation]
        else:
            profiles = [p for ps in self.profiles.values() for p in ps]
            
        if not profiles:
            return {}
            
        # Calculer les statistiques
        durations = [p.duration for p in profiles]
        memory_usages = [p.memory_usage for p in profiles]
        cpu_usages = [p.cpu_usage for p in profiles]
        
        stats = {
            "count": len(profiles),
            "duration": {
                "mean": np.mean(durations),
                "std": np.std(durations),
                "min": np.min(durations),
                "max": np.max(durations)
            },
            "memory_usage": {
                "mean": np.mean(memory_usages),
                "max": np.max(memory_usages)
            },
            "cpu_usage": {
                "mean": np.mean(cpu_usages),
                "max": np.max(cpu_usages)
            }
        }
        
        # Ajouter les stats GPU si disponibles
        gpu_usages = [p.gpu_usage for p in profiles if p.gpu_usage is not None]
        if gpu_usages:
            stats["gpu_usage"] = {
                "mean": np.mean(gpu_usages),
                "max": np.max(gpu_usages)
            }
            
        # Ajouter les stats de compression si disponibles
        compression_ratios = [p.compression_ratio for p in profiles if p.compression_ratio is not None]
        if compression_ratios:
            stats["compression_ratio"] = {
                "mean": np.mean(compression_ratios),
                "best": np.max(compression_ratios)
            }
            
        # Ajouter les stats de cache
        cache_hits = sum(p.cache_hits for p in profiles)
        cache_misses = sum(p.cache_misses for p in profiles)
        total_requests = cache_hits + cache_misses
        
        if total_requests > 0:
            stats["cache"] = {
                "hit_ratio": cache_hits / total_requests,
                "hits": cache_hits,
                "misses": cache_misses
            }
            
        return stats
        
    def save_profile(self, path: Union[str, Path]):
        """Sauvegarde le profil dans un fichier."""
        path = Path(path)
        stats = {op: self.get_stats(op) for op in self.profiles.keys()}
        
        with open(path, 'w') as f:
            json.dump(stats, f, indent=2)
            
    def suggest_optimizations(self) -> List[str]:
        """
        Suggère des optimisations basées sur le profilage.
        
        Returns:
            Liste de suggestions d'optimisation
        """
        suggestions = []
        stats = self.get_stats()
        
        if not stats:
            return ["Pas assez de données pour faire des suggestions"]
            
        # Analyser l'utilisation de la mémoire
        if stats["memory_usage"]["max"] > 80:
            suggestions.append(
                "Utilisation mémoire élevée détectée. Considérer:"
                "\n- Augmenter la fréquence de nettoyage du cache"
                "\n- Réduire la taille maximale du cache"
                "\n- Utiliser la compression pour les grands objets"
            )
            
        # Analyser l'utilisation CPU
        if stats["cpu_usage"]["mean"] > 70:
            suggestions.append(
                "Utilisation CPU élevée détectée. Considérer:"
                "\n- Augmenter le niveau de mise en cache"
                "\n- Réduire la fréquence des optimisations"
                "\n- Utiliser plus de threads pour la compression"
            )
            
        # Analyser les performances du cache
        if "cache" in stats:
            if stats["cache"]["hit_ratio"] < 0.5:
                suggestions.append(
                    "Faible taux de succès du cache. Considérer:"
                    "\n- Augmenter la taille du cache"
                    "\n- Ajuster les stratégies de mise en cache"
                    "\n- Préchauffer le cache pour les opérations fréquentes"
                )
                
        return suggestions

class CacheCompressor:
    """Gestionnaire de compression pour le cache."""
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        self.config = config or CompressionConfig()
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.compression_threads
        )
        
        # Initialiser les compresseurs
        self.zstd_compressor = zstd.ZstdCompressor(
            level=self.config.level,
            threads=self.config.compression_threads
        )
        self.zstd_decompressor = zstd.ZstdDecompressor()
        
    def should_compress(self, data_size: int) -> bool:
        """Détermine si les données doivent être compressées."""
        return data_size >= self.config.min_size_for_compression
        
    def compress(self, data: bytes) -> Tuple[bytes, float]:
        """
        Compresse les données.
        
        Args:
            data: Données à compresser
            
        Returns:
            Tuple (données compressées, ratio de compression)
        """
        if not self.should_compress(len(data)):
            return data, 1.0
            
        if self.config.algorithm == "zstd":
            compressed = self.zstd_compressor.compress(data)
        elif self.config.algorithm == "lz4":
            compressed = lz4.frame.compress(
                data,
                compression_level=self.config.level
            )
        else:  # blosc
            compressed = blosc2.compress(
                data,
                clevel=self.config.level,
                typesize=8
            )
            
        ratio = len(data) / len(compressed)
        return compressed, ratio
        
    def decompress(self, data: bytes) -> bytes:
        """
        Décompresse les données.
        
        Args:
            data: Données à décompresser
            
        Returns:
            Données décompressées
        """
        try:
            if self.config.algorithm == "zstd":
                return self.zstd_decompressor.decompress(data)
            elif self.config.algorithm == "lz4":
                return lz4.frame.decompress(data)
            else:  # blosc
                return blosc2.decompress(data)
        except Exception:
            # Si la décompression échoue, les données n'étaient peut-être pas compressées
            return data
            
    def compress_torch_tensor(
        self,
        tensor: torch.Tensor
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compresse un tenseur PyTorch.
        
        Args:
            tensor: Tenseur à compresser
            
        Returns:
            Tuple (données compressées, métadonnées)
        """
        # Sauvegarder les métadonnées
        metadata = {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "device": str(tensor.device)
        }
        
        # Convertir en bytes
        buffer = io.BytesIO()
        torch.save(tensor.cpu(), buffer)
        data = buffer.getvalue()
        
        # Compresser
        compressed, ratio = self.compress(data)
        metadata["compression_ratio"] = ratio
        
        return compressed, metadata
        
    def decompress_torch_tensor(
        self,
        data: bytes,
        metadata: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Décompresse un tenseur PyTorch.
        
        Args:
            data: Données compressées
            metadata: Métadonnées du tenseur
            
        Returns:
            Tenseur décompressé
        """
        # Décompresser
        decompressed = self.decompress(data)
        
        # Charger le tenseur
        buffer = io.BytesIO(decompressed)
        tensor = torch.load(buffer)
        
        # Déplacer sur le bon device
        device = metadata.get("device", "cpu")
        return tensor.to(device)
        
    def compress_numpy_array(
        self,
        array: np.ndarray
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compresse un tableau NumPy.
        
        Args:
            array: Tableau à compresser
            
        Returns:
            Tuple (données compressées, métadonnées)
        """
        metadata = {
            "dtype": str(array.dtype),
            "shape": list(array.shape)
        }
        
        # Compresser
        compressed, ratio = self.compress(array.tobytes())
        metadata["compression_ratio"] = ratio
        
        return compressed, metadata
        
    def decompress_numpy_array(
        self,
        data: bytes,
        metadata: Dict[str, Any]
    ) -> np.ndarray:
        """
        Décompresse un tableau NumPy.
        
        Args:
            data: Données compressées
            metadata: Métadonnées du tableau
            
        Returns:
            Tableau décompressé
        """
        # Décompresser
        decompressed = self.decompress(data)
        
        # Reconstruire le tableau
        return np.frombuffer(
            decompressed,
            dtype=np.dtype(metadata["dtype"])
        ).reshape(metadata["shape"])
        
    def compress_mesh(
        self,
        mesh: Any
    ) -> Tuple[bytes, Dict[str, Any]]:
        """
        Compresse un maillage.
        
        Args:
            mesh: Maillage à compresser
            
        Returns:
            Tuple (données compressées, métadonnées)
        """
        # Sauvegarder les attributs importants
        metadata = {
            "vertices_shape": list(mesh.vertices.shape),
            "faces_shape": list(mesh.faces.shape)
        }
        
        # Compresser les vertices et faces séparément
        vertices_compressed, v_ratio = self.compress(mesh.vertices.tobytes())
        faces_compressed, f_ratio = self.compress(mesh.faces.tobytes())
        
        # Combiner les données compressées
        combined = len(vertices_compressed).to_bytes(8, 'big')
        combined += vertices_compressed
        combined += faces_compressed
        
        # Calculer le ratio moyen
        metadata["compression_ratio"] = (v_ratio + f_ratio) / 2
        
        return combined, metadata
        
    def decompress_mesh(
        self,
        data: bytes,
        metadata: Dict[str, Any]
    ) -> Any:
        """
        Décompresse un maillage.
        
        Args:
            data: Données compressées
            metadata: Métadonnées du maillage
            
        Returns:
            Maillage décompressé
        """
        # Séparer les vertices et faces
        vertices_size = int.from_bytes(data[:8], 'big')
        vertices_data = data[8:8+vertices_size]
        faces_data = data[8+vertices_size:]
        
        # Décompresser
        vertices = np.frombuffer(
            self.decompress(vertices_data),
            dtype=np.float32
        ).reshape(metadata["vertices_shape"])
        
        faces = np.frombuffer(
            self.decompress(faces_data),
            dtype=np.int32
        ).reshape(metadata["faces_shape"])
        
        # Créer le maillage
        import trimesh
        return trimesh.Trimesh(vertices=vertices, faces=faces)
