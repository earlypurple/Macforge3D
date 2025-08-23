#!/usr/bin/env python3
"""
Script de validation des améliorations apportées à MacForge3D.
Teste les nouvelles fonctionnalités et optimisations.
"""

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path

# Ajouter le répertoire Python au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_performance_optimizer_improvements():
    """Test des améliorations de l'optimiseur de performance."""
    logger.info("=== Test Performance Optimizer ===")
    
    try:
        from ai_models.performance_optimizer import PerformanceOptimizer
        import trimesh
        
        # Créer un maillage test simple
        box = trimesh.creation.box(extents=[1, 1, 1])
        subdivided = box.subdivide()  # Augmenter la complexité
        
        logger.info(f"Maillage test créé: {len(subdivided.vertices)} vertices, {len(subdivided.faces)} faces")
        
        # Tester l'optimiseur
        optimizer = PerformanceOptimizer()
        
        start_time = time.time()
        optimized_mesh = optimizer.optimize_mesh(subdivided, level="medium")
        optimization_time = time.time() - start_time
        
        logger.info(f"Optimisation terminée en {optimization_time:.2f}s")
        logger.info(f"Résultat: {len(optimized_mesh.vertices)} vertices, {len(optimized_mesh.faces)} faces")
        
        # Vérifier que l'optimisation a eu un effet
        assert len(optimized_mesh.vertices) <= len(subdivided.vertices), "L'optimisation devrait réduire ou maintenir le nombre de vertices"
        assert len(optimized_mesh.faces) <= len(subdivided.faces), "L'optimisation devrait réduire ou maintenir le nombre de faces"
        
        logger.info("✅ Performance Optimizer: Test réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Performance Optimizer: Erreur - {e}")
        return False

def test_cache_optimizer_improvements():
    """Test des améliorations de l'optimiseur de cache."""
    logger.info("=== Test Cache Optimizer ===")
    
    try:
        from core.advanced_cache_optimizer import AdvancedCacheOptimizer
        
        # Créer un optimiseur de cache
        cache_optimizer = AdvancedCacheOptimizer(max_memory_mb=100)
        
        # Tester les opérations de base
        test_data = {"key1": np.random.random(100), "key2": "test_string", "key3": [1, 2, 3, 4, 5]}
        
        # Insérer des données
        for i, (key, data) in enumerate(test_data.items()):
            cache_optimizer.put(key, data, priority=i+1)
        
        # Récupérer les données
        for key in test_data.keys():
            retrieved_data = cache_optimizer.get(key)
            assert retrieved_data is not None, f"Données pour {key} non trouvées"
        
        # Tester l'optimisation
        optimization_result = cache_optimizer.optimize_cache()
        assert 'total_time_ms' in optimization_result, "Résultat d'optimisation invalide"
        
        # Tester la prédiction d'accès futur
        future_prob = cache_optimizer._access_predictor.predict_future_access("key1", 3600)
        assert 0 <= future_prob <= 1, "Probabilité d'accès futur invalide"
        
        logger.info("✅ Cache Optimizer: Test réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Cache Optimizer: Erreur - {e}")
        return False

def test_mesh_enhancer_improvements():
    """Test des améliorations de l'améliorateur de maillage."""
    logger.info("=== Test Mesh Enhancer ===")
    
    try:
        from ai_models.mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
        import torch
        import trimesh
        
        # Créer un maillage test
        sphere = trimesh.creation.icosphere(subdivisions=2)
        logger.info(f"Sphère test: {len(sphere.vertices)} vertices, {len(sphere.faces)} faces")
        
        # Convertir en tenseurs PyTorch
        vertices = torch.tensor(sphere.vertices, dtype=torch.float32)
        faces = torch.tensor(sphere.faces, dtype=torch.long)
        
        # Créer l'améliorateur
        config = MeshEnhancementConfig(device="cpu")  # Forcer CPU pour la compatibilité
        enhancer = MeshEnhancer(config)
        
        # Tester le lissage préservant les arêtes
        smoothed_vertices = enhancer.edge_preserving_smooth(
            vertices, faces, 
            iterations=2, 
            edge_threshold=0.3
        )
        
        assert smoothed_vertices.shape == vertices.shape, "Les dimensions doivent être préservées"
        
        # Vérifier que le lissage a eu un effet (changement des positions)
        vertex_diff = torch.norm(smoothed_vertices - vertices, dim=1).mean()
        assert vertex_diff > 0, "Le lissage devrait modifier les positions des vertices"
        
        logger.info(f"Lissage appliqué avec succès, changement moyen: {vertex_diff:.4f}")
        logger.info("✅ Mesh Enhancer: Test réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mesh Enhancer: Erreur - {e}")
        return False

def test_text_effects_improvements():
    """Test des améliorations des effets de texte."""
    logger.info("=== Test Text Effects ===")
    
    try:
        from ai_models.text_effects import TextEffects
        import trimesh
        
        # Créer un maillage test simple
        box = trimesh.creation.box(extents=[2, 1, 0.5])
        logger.info(f"Boîte test: {len(box.vertices)} vertices, {len(box.faces)} faces")
        
        # Créer le gestionnaire d'effets
        effects = TextEffects()
        
        # Tester le lissage amélioré
        smoothed_mesh = effects._apply_smoothing(box, iterations=3)
        
        assert isinstance(smoothed_mesh, trimesh.Trimesh), "Le résultat doit être un maillage Trimesh"
        assert len(smoothed_mesh.vertices) == len(box.vertices), "Le nombre de vertices doit être préservé"
        assert len(smoothed_mesh.faces) == len(box.faces), "Le nombre de faces doit être préservé"
        
        # Vérifier que le lissage a modifié les positions
        vertex_diff = np.linalg.norm(smoothed_mesh.vertices - box.vertices, axis=1).mean()
        logger.info(f"Changement moyen des vertices après lissage: {vertex_diff:.4f}")
        
        logger.info("✅ Text Effects: Test réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Text Effects: Erreur - {e}")
        return False

def test_global_orchestrator_improvements():
    """Test des améliorations de l'orchestrateur global."""
    logger.info("=== Test Global Performance Orchestrator ===")
    
    try:
        from core.global_performance_orchestrator import GlobalPerformanceOrchestrator
        
        # Créer l'orchestrateur
        orchestrator = GlobalPerformanceOrchestrator()
        
        # Tester la collecte de métriques
        metrics = orchestrator._collect_system_metrics()
        required_keys = ['cpu_usage', 'memory_usage', 'memory_available_gb']
        
        for key in required_keys:
            assert key in metrics, f"Métrique {key} manquante"
            assert isinstance(metrics[key], (int, float)), f"Métrique {key} doit être numérique"
        
        # Tester la détermination de séquence d'optimisation
        sequence = orchestrator._determine_optimization_sequence(metrics)
        assert isinstance(sequence, dict), "La séquence doit être un dictionnaire"
        assert len(sequence) > 0, "La séquence ne doit pas être vide"
        
        # Tester l'optimisation globale (version limitée pour les tests)
        start_time = time.time()
        results = orchestrator.optimize_all_modules()
        optimization_time = time.time() - start_time
        
        assert isinstance(results, dict), "Les résultats doivent être un dictionnaire"
        assert optimization_time < 10, "L'optimisation ne devrait pas prendre plus de 10 secondes"
        
        logger.info(f"Optimisation globale terminée en {optimization_time:.2f}s")
        logger.info("✅ Global Performance Orchestrator: Test réussi")
        return True
        
    except Exception as e:
        logger.error(f"❌ Global Performance Orchestrator: Erreur - {e}")
        return False

def main():
    """Fonction principale de validation."""
    logger.info("🚀 Démarrage des tests de validation des améliorations MacForge3D")
    logger.info("=" * 60)
    
    tests = [
        test_performance_optimizer_improvements,
        test_cache_optimizer_improvements,
        test_mesh_enhancer_improvements,
        test_text_effects_improvements,
        test_global_orchestrator_improvements
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_func in tests:
        test_name = test_func.__name__.replace('test_', '').replace('_improvements', '')
        logger.info(f"\n📋 Exécution du test: {test_name}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
        except Exception as e:
            logger.error(f"❌ Erreur critique dans {test_name}: {e}")
            results[test_name] = False
    
    # Rapport final
    logger.info("\n" + "=" * 60)
    logger.info("📊 RAPPORT FINAL DE VALIDATION")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        logger.info(f"{test_name:<30} : {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    logger.info(f"\n🎯 Taux de réussite: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 Validation RÉUSSIE! Les améliorations fonctionnent correctement.")
        return 0
    elif success_rate >= 60:
        logger.info("⚠️  Validation PARTIELLE. Quelques améliorations nécessitent attention.")
        return 1
    else:
        logger.info("🚨 Validation ÉCHOUÉE. Révision majeure nécessaire.")
        return 2

if __name__ == "__main__":
    sys.exit(main())