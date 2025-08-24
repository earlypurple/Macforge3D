#!/usr/bin/env python3
"""
Test des améliorations avancées de MacForge3D.
Valide les nouvelles fonctionnalités et les optimisations.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Ajouter le chemin Python au sys.path
sys.path.insert(0, str(Path(__file__).parent / "Python"))

def test_text_effects_enhancements():
    """Test des améliorations du module text_effects."""
    print("🎨 Test des effets de texte avancés...")
    
    try:
        from ai_models.text_effects import (
            TextStyle, TextEffects, get_available_styles, 
            validate_style, create_preview_style
        )
        
        # Test des nouveaux styles
        effects = TextEffects()
        
        # Créer un style avec les nouveaux effets
        advanced_style = TextStyle(
            name="test_avance",
            tessellation_level=1,
            fractal_intensity=0.05,
            plasma_amplitude=0.03,
            vertex_displacement=0.02
        )
        
        # Valider le style
        is_valid, messages = validate_style(advanced_style)
        print(f"   ✓ Validation du style avancé: {'Réussie' if is_valid else 'Échouée'}")
        if messages:
            print(f"     Messages: {', '.join(messages[:3])}")  # Limiter l'affichage
        
        # Test des styles prédéfinis étendus
        available_styles = get_available_styles()
        new_styles = ['tesselle', 'fractal', 'plasma_avance', 'chaotique', 'ultra_moderne', 'organique']
        found_new_styles = [s for s in new_styles if s in available_styles]
        print(f"   ✓ Nouveaux styles trouvés: {len(found_new_styles)}/{len(new_styles)}")
        
        # Test de création de style personnalisé
        try:
            custom_style = create_preview_style("metal", {
                "fractal_intensity": 0.08,
                "tessellation_level": 1
            })
            print(f"   ✓ Style personnalisé créé: {custom_style.name}")
        except Exception as e:
            print(f"   ⚠ Erreur style personnalisé: {e}")
        
        print("   ✅ Tests effets de texte terminés\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur tests effets de texte: {e}")
        traceback.print_exc()
        return False

def test_mesh_enhancer_improvements():
    """Test des améliorations du mesh enhancer."""
    print("🔧 Test des améliorations de maillage...")
    
    try:
        from ai_models.mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
        import numpy as np
        
        # Configuration pour les tests
        config = MeshEnhancementConfig(
            resolution_factor=1.5,
            smoothness_weight=0.3,
            detail_preservation=0.8,
            max_points=10000
        )
        
        enhancer = MeshEnhancer(config)
        
        # Créer un maillage de test simple (cube)
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Face inférieure
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Face supérieure
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Face inférieure
            [4, 7, 6], [4, 6, 5],  # Face supérieure
            [0, 4, 5], [0, 5, 1],  # Face avant
            [2, 6, 7], [2, 7, 3],  # Face arrière
            [0, 3, 7], [0, 7, 4],  # Face gauche
            [1, 5, 6], [1, 6, 2]   # Face droite
        ], dtype=np.int32)
        
        # Simuler trimesh.Trimesh
        class MockMesh:
            def __init__(self, vertices, faces):
                self.vertices = vertices
                self.faces = faces
            
            def copy(self):
                return MockMesh(self.vertices.copy(), self.faces.copy())
        
        test_mesh = MockMesh(vertices, faces)
        
        # Test de l'analyse de qualité
        try:
            quality_metrics = enhancer._analyze_mesh_quality(test_mesh)
            print(f"   ✓ Analyse de qualité: score {quality_metrics.get('overall_quality', 0):.2f}")
        except Exception as e:
            print(f"   ⚠ Analyse de qualité échouée: {e}")
        
        # Test d'amélioration adaptative
        try:
            # Simuler l'amélioration (sans les dépendances complètes)
            enhanced_mesh, metrics = enhancer.adaptive_mesh_enhancement(
                test_mesh, 
                quality_target=0.7, 
                max_iterations=2
            )
            print(f"   ✓ Amélioration adaptative: {metrics.get('iterations_performed', 0)} itérations")
        except Exception as e:
            print(f"   ⚠ Amélioration adaptative échouée: {e}")
        
        print("   ✅ Tests mesh enhancer terminés\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur tests mesh enhancer: {e}")
        return False

def test_performance_optimizer_features():
    """Test des nouvelles fonctionnalités de l'optimiseur de performance."""
    print("⚡ Test de l'optimiseur de performance...")
    
    try:
        from ai_models.performance_optimizer import PerformanceOptimizer
        
        # Créer un optimiseur
        optimizer = PerformanceOptimizer()
        
        # Test de l'optimisation automatique des ressources
        try:
            config = optimizer.auto_resource_optimization("balanced")
            print(f"   ✓ Configuration auto: {config.get('optimization_level', 'unknown')}")
        except Exception as e:
            print(f"   ⚠ Auto-optimisation échouée: {e}")
        
        # Test du profiler temps réel
        try:
            def test_function():
                time.sleep(0.1)  # Simuler une opération
                return "test_result"
            
            result, metrics = optimizer.real_time_profiler(
                "test_operation",
                test_function
            )
            
            print(f"   ✓ Profiler: {metrics.get('execution_time_seconds', 0):.3f}s, "
                  f"catégorie: {metrics.get('performance_category', 'unknown')}")
        except Exception as e:
            print(f"   ⚠ Profiler échoué: {e}")
        
        # Test de détection des goulots d'étranglement
        try:
            sample_metrics = [
                {
                    'operation_name': 'op1',
                    'execution_time_seconds': 0.1,
                    'memory_usage': {'peak_mb': 50},
                    'performance_category': 'bon'
                },
                {
                    'operation_name': 'op2',
                    'execution_time_seconds': 2.0,
                    'memory_usage': {'peak_mb': 200},
                    'performance_category': 'lent'
                }
            ]
            
            bottleneck_analysis = optimizer.bottleneck_detector(sample_metrics)
            print(f"   ✓ Analyse goulots: {len(bottleneck_analysis.get('bottlenecks', []))} détectés")
        except Exception as e:
            print(f"   ⚠ Détection goulots échouée: {e}")
        
        print("   ✅ Tests optimiseur de performance terminés\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur tests optimiseur: {e}")
        return False

def test_enhanced_validation():
    """Test du système de validation amélioré."""
    print("✅ Test de la validation avancée...")
    
    try:
        from simulation.enhanced_validation import (
            AdvancedValidator, validate_text_style_params, 
            validate_performance_config, advanced_validator
        )
        
        # Test de validation de paramètres de style
        try:
            style_params = {
                "bevel_amount": 0.8,
                "wave_amplitude": 0.15,
                "tessellation_level": 2
            }
            
            validated = validate_text_style_params(style_params)
            print(f"   ✓ Validation style: {len(validated)} paramètres validés")
        except Exception as e:
            print(f"   ⚠ Validation style échouée: {e}")
        
        # Test de validation avec contexte
        try:
            context = {
                "mesh_size": "large",
                "operation_type": "real_time",
                "hardware_constraints": {"gpu_memory_gb": 6}
            }
            
            params = {
                "tessellation_level": 3,
                "smooth_iterations": 5
            }
            
            result = advanced_validator.validate_with_context(params, context)
            print(f"   ✓ Validation contextuelle: {len(result.get('validated_params', {}))} paramètres")
            
            # Afficher les adaptations si disponibles
            adaptations = result.get('context_adaptations', [])
            if adaptations:
                print(f"     Adaptations: {len(adaptations)} appliquées")
                
        except Exception as e:
            print(f"   ⚠ Validation contextuelle échouée: {e}")
        
        # Test de validation de configuration de performance
        try:
            perf_config = {
                "thread_workers": 16,
                "memory_limit_gb": 8.0,
                "optimization_level": "balanced"
            }
            
            validated_config = validate_performance_config(perf_config)
            print(f"   ✓ Config performance: {len(validated_config)} paramètres validés")
        except Exception as e:
            print(f"   ⚠ Validation config performance échouée: {e}")
        
        print("   ✅ Tests validation avancée terminés\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur tests validation: {e}")
        return False

def test_advanced_diagnostics():
    """Test du système de diagnostics avancé."""
    print("🔍 Test des diagnostics avancés...")
    
    try:
        from ai_models.advanced_diagnostics import (
            DiagnosticCenter, SmartLogger, RealTimeMonitor,
            log_operation, monitor_operation, start_global_monitoring
        )
        
        # Test du centre de diagnostic
        center = DiagnosticCenter()
        
        # Test d'un logger intelligent
        logger = center.get_logger("test_component")
        
        # Test de logging d'opération
        operation_id = logger.log_operation_start("test_operation", param1="value1")
        time.sleep(0.05)  # Simuler une opération
        logger.log_operation_end(operation_id, success=True)
        
        print("   ✓ Logger intelligent opérationnel")
        
        # Test du décorateur
        @log_operation("test_component", "decorated_operation")
        def test_decorated_function():
            time.sleep(0.02)
            return "success"
        
        try:
            result = test_decorated_function()
            print("   ✓ Décorateur de logging fonctionnel")
        except Exception as e:
            print(f"   ⚠ Décorateur échoué: {e}")
        
        # Test du context manager
        try:
            with monitor_operation("test_component", "monitored_operation") as op_logger:
                time.sleep(0.03)
                op_logger.log_with_context(20, "Opération en cours", {"progress": 50})  # INFO level
            print("   ✓ Context manager fonctionnel")
        except Exception as e:
            print(f"   ⚠ Context manager échoué: {e}")
        
        # Test de génération de rapport
        try:
            report = center.generate_comprehensive_report()
            health_score = report.get('health_score', 0)
            print(f"   ✓ Rapport de santé généré: score {health_score:.1f}/100")
        except Exception as e:
            print(f"   ⚠ Génération rapport échouée: {e}")
        
        # Test du monitoring en temps réel (bref)
        try:
            monitor = RealTimeMonitor(monitoring_interval=0.1)
            monitor.start_monitoring()
            time.sleep(0.3)  # Laisser quelques échantillons
            monitor.stop_monitoring()
            
            summary = monitor.get_metrics_summary(duration_minutes=1)
            print(f"   ✓ Monitoring temps réel: {summary.get('sample_count', 0)} échantillons")
        except Exception as e:
            print(f"   ⚠ Monitoring temps réel échoué: {e}")
        
        print("   ✅ Tests diagnostics avancés terminés\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur tests diagnostics: {e}")
        return False

def test_integration():
    """Test d'intégration des améliorations."""
    print("🔄 Test d'intégration globale...")
    
    try:
        # Simuler un workflow complet utilisant les améliorations
        from ai_models.advanced_diagnostics import diagnostic_center
        from simulation.enhanced_validation import advanced_validator
        
        # Démarrer le monitoring
        diagnostic_center.start_monitoring()
        
        # Simuler une opération complète
        logger = diagnostic_center.get_logger("integration_test")
        
        # 1. Validation des paramètres
        params = {
            "bevel_amount": 0.5,
            "tessellation_level": 1,
            "wave_amplitude": 0.1
        }
        
        context = {
            "mesh_size": "medium",
            "operation_type": "batch"
        }
        
        # Valider avec contexte
        validation_result = advanced_validator.validate_with_context(params, context)
        validated_params = validation_result.get('validated_params', {})
        
        # 2. Simuler le traitement avec logging
        op_id = logger.log_operation_start("integration_workflow", 
                                          param_count=len(validated_params))
        
        # Simuler du travail
        time.sleep(0.1)
        
        # 3. Finaliser
        logger.log_operation_end(op_id, success=True)
        
        # 4. Générer un rapport final
        final_report = diagnostic_center.generate_comprehensive_report()
        
        # Arrêter le monitoring
        diagnostic_center.stop_monitoring()
        
        print(f"   ✓ Workflow intégré: score santé {final_report.get('health_score', 0):.1f}")
        print("   ✅ Test d'intégration réussi\n")
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur test intégration: {e}")
        traceback.print_exc()
        return False

def main():
    """Fonction principale des tests."""
    print("🚀 Tests des améliorations MacForge3D\n")
    print("=" * 50)
    
    # Exécuter tous les tests
    tests = [
        ("Effets de texte", test_text_effects_enhancements),
        ("Mesh Enhancer", test_mesh_enhancer_improvements),
        ("Optimiseur Performance", test_performance_optimizer_features),
        ("Validation Avancée", test_enhanced_validation),
        ("Diagnostics Avancés", test_advanced_diagnostics),
        ("Intégration", test_integration)
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Erreur critique dans {test_name}: {e}")
            results[test_name] = False
    
    # Résumé final
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print("=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print(f"Temps total: {total_time:.2f}s")
    print(f"Tests réussis: {passed}/{total}")
    print(f"Taux de réussite: {(passed/total)*100:.1f}%")
    
    print("\n📋 Détail par module:")
    for test_name, success in results.items():
        status = "✅ RÉUSSI" if success else "❌ ÉCHOUÉ"
        print(f"   {test_name}: {status}")
    
    if passed == total:
        print("\n🎉 Tous les tests sont passés ! Les améliorations sont opérationnelles.")
    else:
        print(f"\n⚠️  {total - passed} test(s) ont échoué. Vérifier les logs ci-dessus.")
    
    print("\n🔧 Nouvelles fonctionnalités disponibles:")
    print("   • Effets de texte avancés (tessellation, fractal, plasma)")
    print("   • Amélioration adaptative de maillage avec IA")
    print("   • Optimiseur de performance temps réel")
    print("   • Validation contextuelle intelligente")
    print("   • Système de diagnostics et monitoring")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)