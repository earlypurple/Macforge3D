"""
Exemple d'utilisation du système de cache et d'optimisation des performances.
"""

import sys
import os
import time
import numpy as np
import trimesh
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.performance_optimizer import CacheManager, PerformanceOptimizer, CacheConfig
from ai_models.text_to_mesh import create_text_mesh

def demo_optimization():
    """Démontre les optimisations de performances."""
    
    # Configuration du cache
    cache_config = CacheConfig(
        max_memory_usage=0.75,  # Utiliser jusqu'à 75% de la RAM
        max_cache_size=2 * 1024 * 1024 * 1024,  # 2 GB
        cache_dir=Path("/tmp/macforge3d_cache"),
        enable_memory_mapping=True,
        enable_gpu_caching=True
    )
    
    # Initialiser les gestionnaires
    cache_manager = CacheManager(cache_config)
    optimizer = PerformanceOptimizer(cache_manager)
    
    # Test 1: Création et mise en cache de texte 3D
    print("\nTest 1: Création et mise en cache de texte 3D")
    text = "Performance"
    
    start_time = time.time()
    mesh1 = create_text_mesh(text, font_size=72, depth=10)
    print(f"Première création: {time.time() - start_time:.2f} secondes")
    
    # Mettre en cache
    cache_manager.put(mesh1, "text_mesh")
    
    # Récupérer du cache
    start_time = time.time()
    mesh2 = cache_manager.get((text, 72, 10), "text_mesh", 
                            lambda: create_text_mesh(text, font_size=72, depth=10))
    print(f"Récupération du cache: {time.time() - start_time:.2f} secondes")
    
    # Test 2: Optimisation de maillage
    print("\nTest 2: Optimisation de maillage")
    print(f"Maillage original: {len(mesh1.vertices)} vertices, {len(mesh1.faces)} faces")
    
    start_time = time.time()
    optimized_low = optimizer.optimize_mesh(mesh1, level="low")
    print(f"\nOptimisation légère: {time.time() - start_time:.2f} secondes")
    print(f"Résultat: {len(optimized_low.vertices)} vertices, {len(optimized_low.faces)} faces")
    
    start_time = time.time()
    optimized_high = optimizer.optimize_mesh(mesh1, level="high")
    print(f"\nOptimisation agressive: {time.time() - start_time:.2f} secondes")
    print(f"Résultat: {len(optimized_high.vertices)} vertices, {len(optimized_high.faces)} faces")
    
    # Test 3: Traitement parallèle
    print("\nTest 3: Traitement parallèle")
    
    def process_vertices(vertices):
        """Exemple de traitement de vertices."""
        # Simuler un traitement coûteux
        time.sleep(0.1)
        return vertices * 1.1
    
    start_time = time.time()
    processed = optimizer.parallel_process_mesh(mesh1, process_vertices)
    print(f"Traitement parallèle: {time.time() - start_time:.2f} secondes")
    
    # Test 4: Traitement par lots
    print("\nTest 4: Traitement par lots")
    meshes = [mesh1.copy() for _ in range(5)]
    
    start_time = time.time()
    processed_meshes = optimizer.batch_process_meshes(meshes, lambda m: m.scale(1.2))
    print(f"Traitement par lots: {time.time() - start_time:.2f} secondes")
    
    # Sauvegarder les résultats
    optimized_low.export("/tmp/optimized_low.ply")
    optimized_high.export("/tmp/optimized_high.ply")
    processed.export("/tmp/processed.ply")
    
    print("\nTests terminés. Fichiers sauvegardés dans /tmp/")
    
if __name__ == "__main__":
    demo_optimization()
