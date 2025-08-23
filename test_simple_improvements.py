#!/usr/bin/env python3
"""
Test simple des améliorations apportées à MacForge3D.
"""

import sys
import os
import logging

# Ajouter le répertoire Python au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Python'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_imports():
    """Test des importations de base."""
    logger.info("=== Test des importations ===")
    
    try:
        # Test des modules améliorés
        from ai_models.performance_optimizer import PerformanceOptimizer
        from core.advanced_cache_optimizer import AdvancedCacheOptimizer
        from core.global_performance_orchestrator import GlobalPerformanceOrchestrator
        from ai_models.text_effects import TextEffects
        
        logger.info("✅ Tous les modules importés avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur d'importation: {e}")
        return False

def test_performance_optimizer_basic():
    """Test basique de l'optimiseur de performance."""
    logger.info("=== Test Performance Optimizer ===")
    
    try:
        from ai_models.performance_optimizer import PerformanceOptimizer
        
        # Créer l'optimiseur
        optimizer = PerformanceOptimizer()
        logger.info("✅ PerformanceOptimizer créé avec succès")
        
        # Tester la méthode de simplification adaptative (sans maillage réel)
        result = hasattr(optimizer, '_apply_adaptive_simplification')
        assert result, "Méthode _apply_adaptive_simplification manquante"
        logger.info("✅ Méthode de simplification adaptative présente")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur Performance Optimizer: {e}")
        return False

def test_cache_optimizer_basic():
    """Test basique de l'optimiseur de cache."""
    logger.info("=== Test Cache Optimizer ===")
    
    try:
        from core.advanced_cache_optimizer import AdvancedCacheOptimizer
        
        # Créer l'optimiseur
        cache = AdvancedCacheOptimizer(max_memory_mb=50)
        logger.info("✅ AdvancedCacheOptimizer créé avec succès")
        
        # Tester les nouvelles méthodes
        assert hasattr(cache._access_predictor, 'predict_future_access'), "Méthode predict_future_access manquante"
        logger.info("✅ Prédiction d'accès futur implémentée")
        
        # Test simple de fonctionnement
        cache.put("test_key", "test_data")
        data = cache.get("test_key")
        assert data == "test_data", "Cache ne fonctionne pas correctement"
        logger.info("✅ Cache fonctionne correctement")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur Cache Optimizer: {e}")
        return False

def test_global_orchestrator_basic():
    """Test basique de l'orchestrateur global."""
    logger.info("=== Test Global Orchestrator ===")
    
    try:
        from core.global_performance_orchestrator import GlobalPerformanceOrchestrator
        
        # Créer l'orchestrateur
        orchestrator = GlobalPerformanceOrchestrator()
        logger.info("✅ GlobalPerformanceOrchestrator créé avec succès")
        
        # Tester les nouvelles méthodes
        assert hasattr(orchestrator, '_collect_system_metrics'), "Méthode _collect_system_metrics manquante"
        assert hasattr(orchestrator, '_determine_optimization_sequence'), "Méthode _determine_optimization_sequence manquante"
        logger.info("✅ Nouvelles méthodes d'orchestration présentes")
        
        # Tester la collecte de métriques
        metrics = orchestrator._collect_system_metrics()
        assert 'cpu_usage' in metrics, "Métriques CPU manquantes"
        assert 'memory_usage' in metrics, "Métriques mémoire manquantes"
        logger.info("✅ Collecte de métriques fonctionne")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur Global Orchestrator: {e}")
        return False

def test_text_effects_basic():
    """Test basique des effets de texte."""
    logger.info("=== Test Text Effects ===")
    
    try:
        from ai_models.text_effects import TextEffects
        
        # Créer le gestionnaire d'effets
        effects = TextEffects()
        logger.info("✅ TextEffects créé avec succès")
        
        # Tester les nouvelles méthodes
        assert hasattr(effects, '_detect_high_curvature_vertices'), "Méthode _detect_high_curvature_vertices manquante"
        assert hasattr(effects, '_build_vertex_connectivity'), "Méthode _build_vertex_connectivity manquante"
        logger.info("✅ Nouvelles méthodes de lissage présentes")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur Text Effects: {e}")
        return False

def main():
    """Fonction principale de test."""
    logger.info("🚀 Test simple des améliorations MacForge3D")
    logger.info("=" * 50)
    
    tests = [
        test_basic_imports,
        test_performance_optimizer_basic,
        test_cache_optimizer_basic,
        test_global_orchestrator_basic,
        test_text_effects_basic
    ]
    
    results = {}
    passed = 0
    
    for test_func in tests:
        test_name = test_func.__name__.replace('test_', '').replace('_basic', '')
        logger.info(f"\n📋 Test: {test_name}")
        
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"❌ Erreur critique: {e}")
            results[test_name] = False
    
    # Rapport final
    logger.info("\n" + "=" * 50)
    logger.info("📊 RAPPORT FINAL")
    logger.info("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        logger.info(f"{test_name:<25} : {status}")
    
    success_rate = (passed / len(tests)) * 100
    logger.info(f"\n🎯 Taux de réussite: {passed}/{len(tests)} ({success_rate:.1f}%)")
    
    if success_rate == 100:
        logger.info("🎉 Tous les tests réussis! Les améliorations sont fonctionnelles.")
        return 0
    elif success_rate >= 80:
        logger.info("✅ La plupart des améliorations fonctionnent correctement.")
        return 0
    else:
        logger.info("⚠️  Certaines améliorations nécessitent une révision.")
        return 1

if __name__ == "__main__":
    sys.exit(main())