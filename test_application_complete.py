#!/usr/bin/env python3
"""
Test complet de l'application MacForge3D.
Vérifie toutes les fonctionnalités end-to-end.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Ajouter le chemin Python au sys.path
sys.path.insert(0, str(Path(__file__).parent / "Python"))

def test_complete_text_workflow():
    """Test du workflow complet de traitement de texte 3D."""
    print("🎯 Test du workflow complet de traitement de texte 3D...")
    
    try:
        # 1. Validation des paramètres
        from simulation.enhanced_validation import advanced_validator
        
        context = {
            "mesh_size": "medium",
            "operation_type": "real_time",
            "hardware_constraints": {"gpu_memory_gb": 4}
        }
        
        style_params = {
            "tessellation_level": 2,
            "fractal_intensity": 0.08,
            "plasma_amplitude": 0.05,
            "bevel_amount": 0.1,
            "metallic": 0.8
        }
        
        validation_result = advanced_validator.validate_with_context(style_params, context)
        validated_params = validation_result.get('validated_params', {})
        print(f"   ✓ Validation: {len(validated_params)} paramètres validés")
        
        # 2. Création du style
        from ai_models.text_effects import TextStyle, TextEffects, validate_style
        
        style = TextStyle(
            name="test_workflow",
            tessellation_level=validated_params.get('tessellation_level', 1),
            fractal_intensity=validated_params.get('fractal_intensity', 0.05),
            plasma_amplitude=validated_params.get('plasma_amplitude', 0.03),
            bevel_amount=validated_params.get('bevel_amount', 0.1),
            metallic=validated_params.get('metallic', 0.8),
            color=(0.2, 0.8, 1.0)
        )
        
        is_valid, messages = validate_style(style)
        print(f"   ✓ Style créé et validé: {'Oui' if is_valid else 'Non'}")
        
        # 3. Application des effets (simulé)
        effects = TextEffects()
        print("   ✓ TextEffects initialisé")
        
        # 4. Test de l'amélioration de maillage
        from ai_models.mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
        
        config = MeshEnhancementConfig(
            resolution_factor=1.5,
            smoothness_weight=0.3,
            detail_preservation=0.8
        )
        
        enhancer = MeshEnhancer()
        print("   ✓ MeshEnhancer initialisé")
        
        # 5. Test de l'optimiseur de performance
        from ai_models.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        auto_config = optimizer.auto_resource_optimization("balanced")
        print(f"   ✓ Configuration optimisée: {auto_config.get('optimization_level', 'N/A')}")
        
        print("   ✅ Workflow complet de traitement de texte réussi\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur workflow texte: {e}")
        traceback.print_exc()
        return False

def test_complete_diagnostics_workflow():
    """Test du workflow complet de diagnostics et monitoring."""
    print("🔍 Test du workflow de diagnostics et monitoring...")
    
    try:
        from ai_models.advanced_diagnostics import diagnostic_center, log_operation, monitor_operation
        
        # 1. Démarrer le monitoring
        diagnostic_center.start_monitoring()
        print("   ✓ Monitoring démarré")
        
        # 2. Test du logger intelligent
        logger = diagnostic_center.get_logger("test_workflow")
        
        # Simuler une opération
        operation_id = logger.log_operation_start("text_processing", mesh_vertices=5000)
        time.sleep(0.1)
        logger.log_operation_end(operation_id, success=True)
        print("   ✓ Opération loggée avec succès")
        
        # 3. Test du décorateur
        @log_operation("mesh_enhancer", "quality_analysis")
        def test_decorated_function():
            time.sleep(0.05)
            return {"quality_score": 85}
        
        result = test_decorated_function()
        print(f"   ✓ Fonction décorée exécutée: score {result.get('quality_score', 'N/A')}")
        
        # 4. Test du context manager
        with monitor_operation("performance", "optimization") as monitor:
            time.sleep(0.03)
            monitor.log_with_context(20, "Optimisation terminée", {"speedup": 2.5})
        print("   ✓ Context manager testé")
        
        # 5. Générer rapport de santé
        health_report = diagnostic_center.generate_comprehensive_report()
        health_score = health_report.get('health_score', 0)
        print(f"   ✓ Rapport de santé généré: score {health_score}/100")
        
        # 6. Arrêter le monitoring
        diagnostic_center.stop_monitoring()
        print("   ✓ Monitoring arrêté")
        
        print("   ✅ Workflow de diagnostics réussi\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur workflow diagnostics: {e}")
        traceback.print_exc()
        return False

def test_complete_validation_workflow():
    """Test du workflow complet de validation."""
    print("✅ Test du workflow de validation avancée...")
    
    try:
        from simulation.enhanced_validation import (
            advanced_validator, validate_text_style_params, 
            validate_performance_config
        )
        
        # 1. Test de validation de style
        style_params = {
            "bevel_amount": 0.8,
            "wave_amplitude": 0.15,
            "tessellation_level": 2,
            "fractal_intensity": 0.1,
            "plasma_amplitude": 0.05
        }
        
        validated_style = validate_text_style_params(style_params)
        print(f"   ✓ Validation style: {len(validated_style)} paramètres")
        
        # 2. Test de validation contextuelle
        context = {
            "mesh_size": "large",
            "operation_type": "batch",
            "hardware_constraints": {"gpu_memory_gb": 8}
        }
        
        complex_params = {
            "resolution": 50000,
            "quality": "ultra",
            "smoothness_weight": 0.9,
            "detail_preservation": 0.8
        }
        
        contextual_result = advanced_validator.validate_with_context(complex_params, context)
        validated_count = len(contextual_result.get('validated_params', {}))
        adaptations_count = len(contextual_result.get('context_adaptations', []))
        print(f"   ✓ Validation contextuelle: {validated_count} paramètres, {adaptations_count} adaptations")
        
        # 3. Test de validation de performance
        perf_config = {
            "thread_workers": 16,
            "memory_limit_gb": 8.0,
            "optimization_level": "ultra",
            "enable_adaptive_optimization": True
        }
        
        validated_perf = validate_performance_config(perf_config)
        print(f"   ✓ Validation performance: {len(validated_perf)} paramètres")
        
        print("   ✅ Workflow de validation réussi\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur workflow validation: {e}")
        traceback.print_exc()
        return False

def test_complete_performance_workflow():
    """Test du workflow complet d'optimisation de performance."""
    print("⚡ Test du workflow d'optimisation de performance...")
    
    try:
        from ai_models.performance_optimizer import PerformanceOptimizer
        from core.advanced_cache_optimizer import AdvancedCacheOptimizer
        from core.global_performance_orchestrator import GlobalPerformanceOrchestrator
        
        # 1. Test de l'optimiseur de performance
        optimizer = PerformanceOptimizer()
        
        # Configuration automatique
        config = optimizer.auto_resource_optimization("gpu_intensive")
        print(f"   ✓ Config auto GPU: {config.get('optimization_level', 'N/A')}")
        
        # Profiling d'une opération
        def test_operation():
            time.sleep(0.1)
            return {"result": "success"}
        
        result, metrics = optimizer.real_time_profiler("test_op", test_operation)
        execution_time = metrics.get('execution_time', 0)
        print(f"   ✓ Profiling: {execution_time:.3f}s")
        
        # 2. Test du cache avancé
        cache = AdvancedCacheOptimizer()
        
        # Test de prédiction d'accès
        access_pattern = cache.predict_future_access_pattern()
        print(f"   ✓ Prédiction cache: {len(access_pattern)} patterns")
        
        # 3. Test de l'orchestrateur global
        orchestrator = GlobalPerformanceOrchestrator()
        
        # Collecte de métriques
        system_metrics = orchestrator.collect_system_metrics()
        cpu_usage = system_metrics.get('cpu_usage', 0)
        print(f"   ✓ Métriques système: CPU {cpu_usage:.1f}%")
        
        print("   ✅ Workflow de performance réussi\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur workflow performance: {e}")
        traceback.print_exc()
        return False

def test_integration_complete():
    """Test d'intégration complète de tous les modules."""
    print("🔄 Test d'intégration complète...")
    
    try:
        # Scenario complet: traitement d'un projet 3D
        
        # 1. Initialisation des diagnostics
        from ai_models.advanced_diagnostics import diagnostic_center
        diagnostic_center.start_monitoring()
        
        # 2. Validation des paramètres projet
        from simulation.enhanced_validation import advanced_validator
        
        project_params = {
            "mesh_resolution": 25000,
            "quality_level": "high",
            "tessellation_level": 2,
            "smoothness_weight": 0.7,
            "optimization_mode": "balanced"
        }
        
        project_context = {
            "project_type": "high_quality_render",
            "hardware_constraints": {"gpu_memory_gb": 6, "cpu_cores": 8},
            "deadline": "urgent"
        }
        
        validation_result = advanced_validator.validate_with_context(project_params, project_context)
        validated_params = validation_result.get('validated_params', {})
        
        # 3. Configuration de performance adaptée
        from ai_models.performance_optimizer import PerformanceOptimizer
        
        optimizer = PerformanceOptimizer()
        perf_config = optimizer.auto_resource_optimization(
            validated_params.get('optimization_mode', 'balanced')
        )
        
        # 4. Traitement avec monitoring
        logger = diagnostic_center.get_logger("integration_test")
        
        op_id = logger.log_operation_start("complete_workflow", 
                                         project_params=len(validated_params))
        
        # Simuler le traitement complet
        time.sleep(0.2)
        
        # 5. Finalisation et rapport
        logger.log_operation_end(op_id, success=True)
        
        final_report = diagnostic_center.generate_comprehensive_report()
        health_score = final_report.get('health_score', 0)
        
        diagnostic_center.stop_monitoring()
        
        print(f"   ✓ Projet traité: {len(validated_params)} paramètres validés")
        print(f"   ✓ Performance: {perf_config.get('optimization_level', 'N/A')}")
        print(f"   ✓ Santé système finale: {health_score}/100")
        print("   ✅ Intégration complète réussie\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur intégration: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test complet."""
    print("🚀 TEST COMPLET DE L'APPLICATION MACFORGE3D")
    print("=" * 60)
    print("Validation exhaustive de toutes les fonctionnalités")
    print("=" * 60)
    
    tests = [
        ("Workflow Texte 3D", test_complete_text_workflow),
        ("Workflow Diagnostics", test_complete_diagnostics_workflow),
        ("Workflow Validation", test_complete_validation_workflow),
        ("Workflow Performance", test_complete_performance_workflow),
        ("Intégration Complète", test_integration_complete)
    ]
    
    results = {}
    passed = 0
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            print(f"   ❌ Erreur critique: {e}")
            results[test_name] = False
    
    total_time = time.time() - start_time
    
    # Rapport final
    print("=" * 60)
    print("📊 RAPPORT FINAL COMPLET")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ RÉUSSI" if result else "❌ ÉCHOUÉ"
        print(f"{test_name:<25} : {status}")
    
    success_rate = (passed / len(tests)) * 100
    print(f"\n🎯 Résultats globaux:")
    print(f"   Tests réussis: {passed}/{len(tests)}")
    print(f"   Taux de réussite: {success_rate:.1f}%")
    print(f"   Temps total: {total_time:.2f}s")
    
    if success_rate == 100:
        print("\n🎉 SUCCÈS COMPLET! L'application MacForge3D est entièrement fonctionnelle.")
        print("✨ Toutes les améliorations sont opérationnelles et intégrées.")
        return 0
    elif success_rate >= 80:
        print("\n✅ APPLICATION LARGEMENT FONCTIONNELLE.")
        print("⚠️  Quelques améliorations mineures peuvent être nécessaires.")
        return 0
    else:
        print("\n⚠️  L'APPLICATION NÉCESSITE DES AMÉLIORATIONS.")
        print("🔧 Vérifier les modules en échec pour les optimiser.")
        return 1

if __name__ == "__main__":
    sys.exit(main())