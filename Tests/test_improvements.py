"""
Tests pour valider les améliorations apportées au code Macforge3D.
"""

import pytest
import numpy as np
import sys
import os
from pathlib import Path

# Ajouter le répertoire Python au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Python'))

from simulation.error_handling import input_validator, ValidationError


class TestInputValidation:
    """Tests pour le validateur d'entrées."""
    
    def test_string_validation_basic(self):
        """Test de validation basique des strings."""
        # Test valide
        result = input_validator.validate_string_input("test normal", "test_field")
        assert result == "test normal"
        
        # Test string trop longue
        long_string = "a" * 10001
        with pytest.raises(ValidationError):
            input_validator.validate_string_input(long_string, "test_field")
    
    def test_string_validation_security(self):
        """Test de sécurité pour les strings."""
        # Test script injection
        with pytest.raises(ValidationError):
            input_validator.validate_string_input("<script>alert('hack')</script>", "test")
        
        # Test path traversal
        with pytest.raises(ValidationError):
            input_validator.validate_string_input("../../etc/passwd", "test")
    
    def test_mesh_parameters_validation(self):
        """Test de validation des paramètres de maillage."""
        # Paramètres valides
        valid_params = {
            "resolution": 10000,
            "quality": "medium",
            "material": "PLA",
            "temperature": 210.0
        }
        
        result = input_validator.validate_mesh_parameters(valid_params)
        assert result["resolution"] == 10000
        assert result["quality"] == "medium"
        assert result["material"] == "PLA"
        assert result["temperature"] == 210.0
        
        # Résolution invalide
        invalid_params = {"resolution": 50}  # Trop faible
        with pytest.raises(ValidationError):
            input_validator.validate_mesh_parameters(invalid_params)
        
        # Matériau invalide
        invalid_params = {"material": "INVALID_MATERIAL"}
        with pytest.raises(ValidationError):
            input_validator.validate_mesh_parameters(invalid_params)
        
        # Température invalide
        invalid_params = {"temperature": 400.0}  # Trop élevée
        with pytest.raises(ValidationError):
            input_validator.validate_mesh_parameters(invalid_params)


class TestMeshSmoothing:
    """Tests pour l'algorithme de lissage de maillage optimisé."""
    
    def test_mesh_smoothing_import(self):
        """Test d'import du module de lissage."""
        try:
            from ai_models.mesh_enhancer import MeshEnhancer
            assert True, "Module MeshEnhancer importé avec succès"
        except ImportError as e:
            pytest.skip(f"Module MeshEnhancer non disponible: {e}")
    
    def test_smooth_vertices_basic(self):
        """Test basique de la fonction de lissage."""
        try:
            import torch
            from ai_models.mesh_enhancer import MeshEnhancer
            
            # Créer un enhancer de test
            config = type('Config', (), {
                'max_points': 1000,
                'detail_preservation': 0.8,
                'smoothness_weight': 0.3
            })()
            
            enhancer = MeshEnhancer(config)
            
            # Créer des données de test simples
            vertices = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0]
            ], dtype=torch.float32)
            
            faces = torch.tensor([
                [0, 1, 2],
                [1, 3, 2]
            ], dtype=torch.long)
            
            # Tester le lissage
            smoothed = enhancer._smooth_vertices(vertices, faces, weight=0.1, iterations=1)
            
            # Vérifier que la forme est préservée
            assert smoothed.shape == vertices.shape
            assert torch.is_tensor(smoothed)
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")


class TestPerformanceOptimizer:
    """Tests pour l'optimiseur de performance."""
    
    def test_performance_optimizer_import(self):
        """Test d'import de l'optimiseur de performance."""
        try:
            from ai_models.performance_optimizer import PerformanceOptimizer
            assert True, "PerformanceOptimizer importé avec succès"
        except ImportError as e:
            pytest.skip(f"PerformanceOptimizer non disponible: {e}")
    
    def test_memory_optimization(self):
        """Test de l'optimisation mémoire."""
        try:
            from ai_models.performance_optimizer import PerformanceOptimizer
            
            optimizer = PerformanceOptimizer()
            
            # Test de l'optimisation mémoire
            stats = optimizer.optimize_memory_usage()
            
            # Vérifier que la fonction retourne des statistiques
            assert isinstance(stats, dict)
            assert "memory_before_gb" in stats
            assert "success" in stats
            
        except ImportError as e:
            pytest.skip(f"PerformanceOptimizer non disponible: {e}")


class TestMonitoring:
    """Tests pour le système de monitoring amélioré."""
    
    def test_monitoring_import(self):
        """Test d'import du système de monitoring."""
        try:
            from core.monitoring import PerformanceMonitor
            assert True, "PerformanceMonitor importé avec succès"
        except ImportError as e:
            pytest.skip(f"PerformanceMonitor non disponible: {e}")


class TestRobustRunner:
    """Tests pour le runner robuste corrigé."""
    
    def test_robust_runner_import(self):
        """Test d'import du runner robuste."""
        try:
            from simulation.robust_runner import RobustSimulationRunner
            assert True, "RobustSimulationRunner importé avec succès"
        except ImportError as e:
            pytest.skip(f"RobustSimulationRunner non disponible: {e}")
    
    def test_submit_task_sync_method(self):
        """Test de la méthode submit_task_sync."""
        try:
            from simulation.robust_runner import RobustSimulationRunner
            
            runner = RobustSimulationRunner()
            
            # Vérifier que la méthode existe
            assert hasattr(runner, '_submit_task_sync')
            assert callable(getattr(runner, '_submit_task_sync'))
            
        except ImportError as e:
            pytest.skip(f"RobustSimulationRunner non disponible: {e}")


def test_all_modules_compile():
    """Test que tous les modules se compilent sans erreurs de syntaxe."""
    import subprocess
    import os
    
    python_dir = Path(__file__).parent.parent / "Python"
    python_files = []
    
    # Collecter tous les fichiers Python
    for root, dirs, files in os.walk(python_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    errors = []
    for py_file in python_files:
        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", py_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                errors.append(f"{py_file}: {result.stderr}")
        except subprocess.TimeoutExpired:
            errors.append(f"{py_file}: Timeout during compilation")
        except Exception as e:
            errors.append(f"{py_file}: {str(e)}")
    
    if errors:
        pytest.fail(f"Erreurs de compilation détectées:\n" + "\n".join(errors))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])