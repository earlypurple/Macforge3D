"""
Exemple d'utilisation des fonctionnalités avancées d'optimisation.
"""

import sys
import os
import time
import numpy as np
import trimesh
from pathlib import Path
import logging

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.cache_extensions import CacheCompressor, CompressionConfig, CacheProfiler
from ai_models.cluster_manager import ClusterManager, ClusterConfig
from ai_models.text_to_mesh import create_text_mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_advanced_features():
    """Démontre les fonctionnalités avancées."""

    # 1. Configuration
    compression_config = CompressionConfig(
        algorithm="zstd", level=3, enable_gpu_compression=True
    )

    cluster_config = ClusterConfig(
        num_workers=4, gpu_per_worker=0.25, enable_load_balancing=True
    )

    compressor = CacheCompressor(compression_config)
    profiler = CacheProfiler()
    cluster = ClusterManager(cluster_config)

    # 2. Test de compression
    print("\nTest de compression...")

    # Créer un texte 3D complexe
    text = "Compression"
    original_mesh = create_text_mesh(text, font_size=72, depth=10, centered=True)

    # Compresser
    start_time = time.time()
    compressed_data, metadata = compressor.compress_mesh(original_mesh)
    compression_time = time.time() - start_time

    print(
        f"Taille originale: {len(original_mesh.vertices.tobytes() + original_mesh.faces.tobytes()):,} bytes"
    )
    print(f"Taille compressée: {len(compressed_data):,} bytes")
    print(f"Ratio de compression: {metadata['compression_ratio']:.2f}x")
    print(f"Temps de compression: {compression_time:.2f} secondes")

    # Décompresser
    start_time = time.time()
    decompressed_mesh = compressor.decompress_mesh(compressed_data, metadata)
    decompression_time = time.time() - start_time

    print(f"Temps de décompression: {decompression_time:.2f} secondes")
    print(
        "Vérification de l'intégrité:",
        np.allclose(original_mesh.vertices, decompressed_mesh.vertices),
    )

    # 3. Test de traitement distribué
    print("\nTest de traitement distribué...")

    # Créer plusieurs variations
    texts = ["Test1", "Test2", "Test3", "Test4"]
    meshes = []

    for text in texts:
        mesh = create_text_mesh(text, font_size=72, depth=10)
        meshes.append(mesh)

    # Traitement par lots
    start_time = time.time()
    processed_meshes = cluster.process_batch(
        meshes, operation="simplify", params={"ratio": 0.5}
    )
    batch_time = time.time() - start_time

    print(f"Temps de traitement par lots: {batch_time:.2f} secondes")
    print(f"Maillages traités: {len(processed_meshes)}")

    # Sauvegarder les résultats
    for i, mesh in enumerate(processed_meshes):
        if mesh is not None:
            mesh.export(f"/tmp/processed_{i}.ply")

    # 4. Analyser les performances
    print("\nAnalyse des performances...")

    stats = cluster.get_stats()
    print("\nStatistiques du cluster:")
    print(f"- Tâches complétées: {stats['tasks_completed']}")
    print(f"- Tâches échouées: {stats['tasks_failed']}")
    if "avg_processing_time" in stats:
        print(f"- Temps moyen de traitement: {stats['avg_processing_time']:.2f}s")

    print("\nSuggestions d'optimisation:")
    suggestions = profiler.suggest_optimizations()
    for suggestion in suggestions:
        print(f"- {suggestion}")

    # 5. Sauvegarder le profil
    profiler.save_profile("/tmp/performance_profile.json")
    print("\nProfil de performance sauvegardé dans: /tmp/performance_profile.json")

    # Nettoyer
    cluster.shutdown()


if __name__ == "__main__":
    demo_advanced_features()
