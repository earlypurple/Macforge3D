import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
from Python.simulation.robust_runner import RobustSimulationRunner, SimulationState
from Python.simulation.error_handling import ValidationError, SimulationError

class TestRobustSimulationRunner(unittest.TestCase):
    def setUp(self):
        """Initialisation avant chaque test."""
        self.runner = RobustSimulationRunner(
            max_workers=2,
            memory_threshold=90.0,
            timeout=60
        )
        self.valid_params = {
            "material": "PLA",
            "temperature": 200.0,
            "pressure": 1.0,
            "mesh_resolution": 10000
        }

    def test_init(self):
        """Test de l'initialisation."""
        self.assertEqual(self.runner.max_workers, 2)
        self.assertEqual(self.runner.memory_threshold, 90.0)
        self.assertEqual(self.runner.timeout, 60)
        self.assertIsInstance(self.runner.state, SimulationState)
        self.assertEqual(self.runner.state.status, "preparing")

    def test_validate_parameters_valid(self):
        """Test de la validation des paramètres valides."""
        try:
            self.runner._validate_parameters(self.valid_params)
        except Exception as e:
            self.fail(f"La validation a échoué avec des paramètres valides: {e}")

    def test_validate_parameters_missing(self):
        """Test de la validation avec paramètres manquants."""
        invalid_params = self.valid_params.copy()
        del invalid_params["material"]
        
        with self.assertRaises(ValidationError) as context:
            self.runner._validate_parameters(invalid_params)
        
        self.assertIn("material", str(context.exception))
        self.assertEqual(context.exception.code, "PARAM_MISSING")

    def test_validate_parameters_invalid_type(self):
        """Test de la validation avec types invalides."""
        invalid_params = self.valid_params.copy()
        invalid_params["temperature"] = "200"  # devrait être float
        
        with self.assertRaises(ValidationError) as context:
            self.runner._validate_parameters(invalid_params)
            
        self.assertIn("temperature", str(context.exception))
        self.assertEqual(context.exception.code, "PARAM_TYPE")

    def test_validate_parameters_out_of_range(self):
        """Test de la validation avec valeurs hors limites."""
        invalid_params = self.valid_params.copy()
        invalid_params["temperature"] = 1000.0
        
        with self.assertRaises(ValidationError) as context:
            self.runner._validate_parameters(invalid_params)
            
        self.assertIn("temperature", str(context.exception))
        self.assertEqual(context.exception.code, "PARAM_RANGE")

    @patch('psutil.Process')
    def test_check_resources_sufficient(self, mock_process):
        """Test de la vérification des ressources suffisantes."""
        mock_process.return_value.memory_percent.return_value = 50.0
        
        try:
            self.runner._check_resources()
        except Exception as e:
            self.fail(f"La vérification des ressources a échoué: {e}")
            
        self.assertEqual(self.runner.state.memory_usage, 50.0)

    @patch('psutil.Process')
    def test_check_resources_insufficient(self, mock_process):
        """Test de la vérification des ressources insuffisantes."""
        mock_process.return_value.memory_percent.return_value = 95.0
        
        with self.assertRaises(SimulationError) as context:
            self.runner._check_resources()
            
        self.assertEqual(context.exception.code, "RESOURCE_MEMORY")

    def test_prepare_simulation(self):
        """Test de la préparation de la simulation."""
        prepared = self.runner._prepare_simulation(self.valid_params)
        
        self.assertIn("thermal_conductivity", prepared)
        self.assertEqual(prepared["thermal_conductivity"], 0.13)  # PLA
        self.assertEqual(prepared["temperature"], 200.0)

    @patch('psutil.Process')
    def test_run_simulation_success(self, mock_process):
        """Test d'une simulation réussie."""
        mock_process.return_value.memory_percent.return_value = 50.0
        callback = Mock()
        
        # Simuler des résultats valides
        def mock_run_steps(*args):
            return {
                "mesh": {"vertices": [(0,0,0), (1,1,1)]},
                "thermal": {"temperature": [20.0, 25.0]},
                "structural": {"stress": [0.1, 0.2]}
            }
            
        with patch.object(
            self.runner,
            '_run_simulation_steps',
            side_effect=mock_run_steps
        ):
            results = self.runner.run_simulation(
                self.valid_params,
                callback
            )
            
        self.assertEqual(self.runner.state.status, "completed")
        self.assertIn("mesh", results)
        self.assertIn("thermal", results)
        self.assertIn("structural", results)

    def test_validate_results_valid(self):
        """Test de la validation des résultats valides."""
        valid_results = {
            "mesh": {"vertices": [(0,0,0), (1,1,1)]},
            "thermal": {"temperature": [20.0, 25.0]},
            "structural": {"stress": [0.1, 0.2]}
        }
        
        try:
            self.runner._validate_results(valid_results)
        except Exception as e:
            self.fail(f"La validation des résultats a échoué: {e}")

    def test_validate_results_missing_key(self):
        """Test de la validation avec résultats incomplets."""
        invalid_results = {
            "mesh": {"vertices": [(0,0,0), (1,1,1)]},
            "thermal": {"temperature": [20.0, 25.0]}
            # structural manquant
        }
        
        with self.assertRaises(ValidationError) as context:
            self.runner._validate_results(invalid_results)
            
        self.assertEqual(context.exception.code, "RESULT_MISSING")

    def test_validate_results_empty_mesh(self):
        """Test de la validation avec maillage vide."""
        invalid_results = {
            "mesh": {"vertices": []},
            "thermal": {"temperature": []},
            "structural": {"stress": []}
        }
        
        with self.assertRaises(ValidationError) as context:
            self.runner._validate_results(invalid_results)
            
        self.assertEqual(context.exception.code, "RESULT_EMPTY_MESH")

    def test_update_state(self):
        """Test de la mise à jour de l'état."""
        self.runner._update_state("running", "test", 0.5)
        
        self.assertEqual(self.runner.state.status, "running")
        self.assertEqual(self.runner.state.current_step, "test")
        self.assertEqual(self.runner.state.progress, 0.5)

    @patch('logging.error')
    def test_handle_simulation_error(self, mock_log):
        """Test de la gestion des erreurs."""
        error = SimulationError(
            "Test error",
            "TEST_ERROR",
            {"step": "test"}
        )
        
        self.runner._handle_simulation_error(error)
        
        self.assertEqual(self.runner.state.status, "failed")
        self.assertEqual(self.runner.state.error, error)
        mock_log.assert_called_once()

    def test_get_thermal_conductivity(self):
        """Test de l'obtention de la conductivité thermique."""
        self.assertEqual(
            self.runner._get_thermal_conductivity("PLA"),
            0.13
        )
        self.assertEqual(
            self.runner._get_thermal_conductivity("ABS"),
            0.17
        )
        self.assertEqual(
            self.runner._get_thermal_conductivity("UNKNOWN"),
            0.15  # valeur par défaut
        )

if __name__ == '__main__':
    unittest.main()
