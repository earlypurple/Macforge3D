"""
Tests complets pour les améliorations apportées à MacForge3D.
Couvre les nouvelles fonctionnalités et optimisations.
"""

import pytest
import numpy as np
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Ajouter le répertoire Python au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Python'))

# Tests des exceptions améliorées
class TestEnhancedExceptions:
    """Tests pour le système d'exceptions amélioré."""
    
    def test_macforge_exception_creation(self):
        """Test de création d'exception MacForge3D."""
        from core.enhanced_exceptions import MacForge3DException, ErrorSeverity, ErrorCategory
        
        exception = MacForge3DException(
            "Test exception",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MESH_PROCESSING
        )
        
        assert exception.message == "Test exception"
        assert exception.severity == ErrorSeverity.HIGH
        assert exception.category == ErrorCategory.MESH_PROCESSING
        assert exception.timestamp > 0
        
    def test_mesh_processing_error(self):
        """Test de l'erreur de traitement de maillage."""
        from core.enhanced_exceptions import MeshProcessingError
        
        mesh_info = {"vertices": 1000, "faces": 2000}
        error = MeshProcessingError("Maillage corrompu", mesh_info=mesh_info)
        
        assert "Maillage corrompu" in error.message
        assert error.parameters["mesh_info"] == mesh_info
        assert len(error.recovery_suggestions) > 0
        
    def test_memory_error(self):
        """Test de l'erreur mémoire."""
        from core.enhanced_exceptions import MemoryError
        
        error = MemoryError("Mémoire insuffisante", required_memory=2.0)
        
        assert error.parameters["required_memory_gb"] == 2.0
        assert "mémoire" in error.recovery_suggestions[0].lower()
        
    def test_exception_handler_decorator(self):
        """Test du décorateur de gestion d'exceptions."""
        from core.enhanced_exceptions import exception_handler
        
        @exception_handler(fallback_result="fallback")
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "fallback"

# Tests de l'analyse de qualité améliorée
class TestEnhancedMeshAnalysis:
    """Tests pour l'analyse de qualité de maillage améliorée."""
    
    def test_enhanced_mesh_quality_analysis(self):
        """Test de l'analyse de qualité avancée."""
        try:
            import trimesh
            from ai_models.mesh_processor import analyze_mesh_quality
            
            # Créer un maillage de test simple
            vertices = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            
            faces = np.array([
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
                [1, 2, 3]
            ])
            
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            analysis = analyze_mesh_quality(mesh)
            
            # Vérifier la structure de l'analyse
            assert "basic_stats" in analysis
            assert "geometric_quality" in analysis
            assert "topological_quality" in analysis
            assert "edge_quality" in analysis
            assert "vertex_quality" in analysis
            assert "overall_quality_score" in analysis
            
            # Vérifier les nouvelles métriques
            assert "aspect_ratio" in analysis["geometric_quality"]["bounding_box"]
            assert "bounding_sphere_radius" in analysis["geometric_quality"]
            assert "manifold_score" in analysis["topological_quality"]
            assert "length_distribution_score" in analysis["edge_quality"]
            assert "regularity_score" in analysis["vertex_quality"]
            
            # Le score global doit être entre 0 et 1
            assert 0 <= analysis["overall_quality_score"] <= 1
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")
    
    def test_triangle_quality_metrics(self):
        """Test des métriques de qualité des triangles."""
        try:
            import trimesh
            from ai_models.mesh_processor import analyze_mesh_quality
            
            # Créer un triangle équilatéral parfait
            vertices = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0.5, np.sqrt(3)/2, 0]
            ], dtype=np.float32)
            
            faces = np.array([[0, 1, 2]])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            analysis = analyze_mesh_quality(mesh)
            triangle_quality = analysis["geometric_quality"]["triangle_quality"]
            
            # Un triangle équilatéral devrait avoir un bon score d'aspect ratio
            assert triangle_quality["aspect_ratio"]["quality_score"] > 0.5
            assert triangle_quality["area_uniformity"] == 1.0  # Un seul triangle
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")

# Tests du processeur mémoire avancé
class TestAdvancedMemoryProcessor:
    """Tests pour le processeur mémoire avancé."""
    
    def test_memory_config_creation(self):
        """Test de création de configuration mémoire."""
        from ai_models.advanced_memory_processor import MemoryConfig
        
        config = MemoryConfig(
            max_memory_percent=80.0,
            chunk_memory_limit_mb=256.0
        )
        
        assert config.max_memory_percent == 80.0
        assert config.chunk_memory_limit_mb == 256.0
        assert config.enable_disk_cache is True
        
    def test_memory_monitor(self):
        """Test du moniteur mémoire."""
        from ai_models.advanced_memory_processor import MemoryMonitor, MemoryConfig
        
        config = MemoryConfig()
        monitor = MemoryMonitor(config)
        
        # Test des statistiques mémoire
        stats = monitor.get_memory_stats()
        assert "current_mb" in stats
        assert "peak_mb" in stats
        assert "available_mb" in stats
        assert stats["available_mb"] > 0
        
    def test_chunk_size_calculation(self):
        """Test du calcul de taille de chunk optimale."""
        from ai_models.advanced_memory_processor import AdvancedMemoryProcessor, MemoryConfig
        
        config = MemoryConfig()
        processor = AdvancedMemoryProcessor(config)
        
        # Créer un maillage de test
        try:
            import trimesh
            vertices = np.random.random((10000, 3)).astype(np.float32)
            faces = np.random.randint(0, 1000, (5000, 3))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Calculer la taille optimale
            chunk_size = processor._calculate_optimal_chunk_size(mesh, 1000.0)  # 1GB disponible
            
            assert chunk_size >= 1000  # Taille minimale
            assert chunk_size <= 50000  # Taille maximale
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")
    
    def test_cache_functionality(self):
        """Test de la fonctionnalité de cache."""
        from ai_models.advanced_memory_processor import AdvancedMemoryProcessor
        
        processor = AdvancedMemoryProcessor()
        
        # Fonction de test simple
        def test_process_function(chunk):
            return chunk * 2
        
        # Test du cache
        test_chunk = np.array([[1, 2, 3], [4, 5, 6]])
        
        result1 = processor._process_chunk_with_cache(test_chunk, test_process_function)
        result2 = processor._process_chunk_with_cache(test_chunk, test_process_function)
        
        # Les résultats devraient être identiques
        np.testing.assert_array_equal(result1, result2)
        
        # Vérifier les statistiques du cache
        stats = processor.get_cache_stats()
        assert stats["hits"] >= 1
        assert stats["cache_size"] >= 1

# Tests de validation améliorée
class TestEnhancedValidation:
    """Tests pour le système de validation amélioré."""
    
    def test_parameter_validation_rules(self):
        """Test des règles de validation des paramètres."""
        from simulation.enhanced_validation import EnhancedValidator, ParameterType
        
        validator = EnhancedValidator()
        
        # Test validation entier
        result = validator.validate_parameter("resolution", 10000)
        assert result.is_valid
        assert result.corrected_value == 10000
        
        # Test validation entier hors limites
        result = validator.validate_parameter("resolution", 50)
        assert not result.is_valid
        assert len(result.errors) > 0
        
    def test_auto_correction(self):
        """Test de l'auto-correction."""
        from simulation.enhanced_validation import EnhancedValidator
        
        validator = EnhancedValidator()
        
        # Test auto-correction température
        result = validator.validate_parameter("temperature", 400.0)  # Trop élevée
        assert not result.is_valid  # Original invalide
        assert result.corrected_value == 350.0  # Corrigé au maximum
        assert len(result.warnings) > 0
        
    def test_enum_validation(self):
        """Test de validation d'énumération."""
        from simulation.enhanced_validation import EnhancedValidator
        
        validator = EnhancedValidator()
        
        # Valeur valide
        result = validator.validate_parameter("quality", "high")
        assert result.is_valid
        
        # Valeur invalide
        result = validator.validate_parameter("quality", "super_ultra")
        assert not result.is_valid
        assert len(result.suggestions) > 0
        
    def test_path_validation(self):
        """Test de validation de chemin."""
        from simulation.enhanced_validation import EnhancedValidator
        
        validator = EnhancedValidator()
        
        # Créer un fichier temporaire pour le test
        with tempfile.NamedTemporaryFile(suffix=".obj", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Test fichier existant
            result = validator.validate_parameter("input_path", tmp_path)
            assert result.is_valid
            
            # Test fichier inexistant
            result = validator.validate_parameter("input_path", "/non/existent/file.obj")
            assert not result.is_valid
            
        finally:
            # Nettoyer le fichier temporaire
            os.unlink(tmp_path)
    
    def test_color_validation(self):
        """Test de validation de couleur."""
        from simulation.enhanced_validation import EnhancedValidator
        
        validator = EnhancedValidator()
        
        # Couleur hexadécimale valide
        result = validator.validate_parameter("color", "#FF0000")
        assert result.is_valid
        assert result.corrected_value == "#FF0000"
        
        # Nom de couleur valide
        result = validator.validate_parameter("color", "red")
        assert result.is_valid
        assert result.corrected_value == "red"
        
        # Couleur invalide
        result = validator.validate_parameter("color", "invalid_color")
        assert not result.is_valid
        assert len(result.suggestions) > 0
    
    def test_mesh_parameters_validation(self):
        """Test de validation des paramètres de maillage."""
        from simulation.enhanced_validation import input_validator, ValidationError
        
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

# Tests du lissage amélioré
class TestEnhancedSmoothing:
    """Tests pour les algorithmes de lissage améliorés."""
    
    def test_edge_preserving_smoothing(self):
        """Test du lissage préservant les arêtes."""
        try:
            import torch
            from ai_models.mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
            
            config = MeshEnhancementConfig(
                max_points=1000,
                detail_preservation=0.8,
                smoothness_weight=0.3
            )
            
            enhancer = MeshEnhancer(config)
            
            # Créer des données de test
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
            
            # Test du lissage préservant les arêtes
            smoothed = enhancer.edge_preserving_smooth(
                vertices, faces, iterations=2
            )
            
            # Vérifier que la forme est préservée
            assert smoothed.shape == vertices.shape
            assert torch.is_tensor(smoothed)
            
            # Les vertices ne devraient pas être exactement identiques après lissage
            assert not torch.allclose(vertices, smoothed, atol=1e-6)
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")
    
    def test_adaptive_weight_computation(self):
        """Test du calcul de poids adaptatif."""
        try:
            import torch
            from ai_models.mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
            
            config = MeshEnhancementConfig()
            enhancer = MeshEnhancer(config)
            
            vertices = torch.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0]
            ], dtype=torch.float32)
            
            neighbors = torch.tensor([
                [0.1, 0.0, 0.0],
                [0.9, 0.0, 0.0]
            ], dtype=torch.float32)
            
            weights = enhancer._compute_adaptive_weight(vertices, neighbors, 0.5)
            
            # Vérifier la forme des poids
            assert weights.shape == (2, 1)
            assert torch.all(weights >= 0)
            assert torch.all(weights <= 0.5)
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")

# Tests d'optimisation mémoire
class TestMemoryOptimization:
    """Tests pour les optimisations mémoire."""
    
    def test_mesh_memory_estimation(self):
        """Test de l'estimation d'utilisation mémoire."""
        try:
            import trimesh
            from ai_models.advanced_memory_processor import estimate_mesh_memory_usage
            
            # Créer un maillage de test
            vertices = np.random.random((1000, 3)).astype(np.float32)
            faces = np.random.randint(0, 1000, (500, 3))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            estimation = estimate_mesh_memory_usage(mesh)
            
            assert "vertices_mb" in estimation
            assert "faces_mb" in estimation
            assert "overhead_mb" in estimation
            assert "total_mb" in estimation
            
            # Les estimations devraient être positives
            assert estimation["vertices_mb"] > 0
            assert estimation["faces_mb"] > 0
            assert estimation["total_mb"] > 0
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")
    
    def test_mesh_memory_optimization(self):
        """Test de l'optimisation mémoire de maillage."""
        try:
            import trimesh
            from ai_models.advanced_memory_processor import optimize_mesh_memory_usage
            
            # Créer un maillage avec des vertices dupliqués
            vertices = np.array([
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],  # Dupliqué
                [1, 0, 0]   # Dupliqué
            ], dtype=np.float64)  # float64 pour tester la conversion
            
            faces = np.array([[0, 1, 2], [3, 4, 2]])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            optimized = optimize_mesh_memory_usage(mesh)
            
            # Le maillage optimisé devrait avoir moins de vertices
            assert len(optimized.vertices) <= len(mesh.vertices)
            
            # Les vertices devraient être en float32
            assert optimized.vertices.dtype == np.float32
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")

# Tests d'intégration
class TestIntegration:
    """Tests d'intégration pour les améliorations."""
    
    def test_complete_workflow(self):
        """Test d'un workflow complet avec les améliorations."""
        try:
            import trimesh
            from simulation.enhanced_validation import input_validator
            from ai_models.mesh_processor import analyze_mesh_quality
            from ai_models.advanced_memory_processor import estimate_mesh_memory_usage
            
            # Étape 1: Validation des paramètres
            params = {
                "resolution": 5000,
                "quality": "medium",
                "temperature": 210.0,
                "smoothness_weight": 0.3
            }
            
            validated_params = input_validator.validate_mesh_parameters(params)
            assert validated_params["resolution"] == 5000
            
            # Étape 2: Création d'un maillage de test
            vertices = np.random.random((1000, 3)).astype(np.float32)
            faces = np.random.randint(0, 1000, (500, 3))
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            # Étape 3: Analyse de qualité
            quality_analysis = analyze_mesh_quality(mesh)
            assert "overall_quality_score" in quality_analysis
            
            # Étape 4: Estimation mémoire
            memory_estimation = estimate_mesh_memory_usage(mesh)
            assert memory_estimation["total_mb"] > 0
            
            # Le workflow complet devrait s'exécuter sans erreur
            assert True
            
        except ImportError as e:
            pytest.skip(f"Dépendances non disponibles: {e}")

def test_all_modules_import():
    """Test que tous les nouveaux modules peuvent être importés."""
    try:
        from core.enhanced_exceptions import MacForge3DException
        from ai_models.advanced_memory_processor import AdvancedMemoryProcessor
        from simulation.enhanced_validation import EnhancedValidator
        
        assert True, "Tous les modules importés avec succès"
    except ImportError as e:
        pytest.fail(f"Échec d'import: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])