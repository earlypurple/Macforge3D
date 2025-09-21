import unittest
import numpy as np
from pathlib import Path
import tempfile
from simulation.result_analyzer import ResultAnalyzer


class TestResultAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prépare les données de test."""
        cls.analyzer = ResultAnalyzer()

        # Résultats de test normaux
        cls.normal_results = {
            "max_stress": 30e6,  # 30 MPa
            "max_displacement": 0.1,  # 0.1 mm
            "min_safety_factor": 2.0,
            "max_temperature": 180.0,  # 180°C
            "cooling_rate": -3.0,  # °C/s
            "time_above_glass": 120.0,  # 120s
        }

        # Résultats de test anormaux
        cls.abnormal_results = {
            "max_stress": 100e6,  # 100 MPa (très élevé)
            "max_displacement": 1.0,  # 1 mm (élevé)
            "min_safety_factor": 0.8,  # < 1 (dangereux)
            "max_temperature": 250.0,  # 250°C (trop chaud)
            "cooling_rate": -15.0,  # °C/s (trop rapide)
            "time_above_glass": 600.0,  # 600s (trop long)
        }

        # Propriétés matériau de test
        cls.material_props = {
            "name": "PLA",
            "melting_point": 180,
            "glass_transition": 60,
            "yield_strength": 50e6,
        }

    def test_preprocess_results(self):
        """Test le prétraitement des résultats."""
        features = self.analyzer.preprocess_results(self.normal_results)

        self.assertEqual(features.shape, (1, 6))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_detect_anomalies(self):
        """Test la détection d'anomalies."""
        # Test avec résultats normaux
        features1 = self.analyzer.preprocess_results(self.normal_results)
        anomalies1 = self.analyzer.detect_anomalies(features1)
        self.assertEqual(len(anomalies1), 0)

        # Test avec résultats anormaux
        features2 = self.analyzer.preprocess_results(self.abnormal_results)
        anomalies2 = self.analyzer.detect_anomalies(features2)
        self.assertTrue(len(anomalies2) > 0)

    def test_analyze_results(self):
        """Test l'analyse complète des résultats."""
        # Test avec résultats normaux
        analysis1 = self.analyzer.analyze_results(
            self.normal_results, self.material_props
        )

        self.assertEqual(analysis1["risk_level"], "faible")
        self.assertFalse(analysis1["anomalies_detected"])

        # Test avec résultats anormaux
        analysis2 = self.analyzer.analyze_results(
            self.abnormal_results, self.material_props
        )

        self.assertEqual(analysis2["risk_level"], "élevé")
        self.assertTrue(analysis2["anomalies_detected"])
        self.assertTrue(len(analysis2["recommendations"]) > 0)

    def test_optimization_suggestions(self):
        """Test la génération de suggestions d'optimisation."""
        suggestions = self.analyzer.generate_optimization_suggestions(
            self.abnormal_results, self.material_props, "élevé"
        )

        self.assertTrue(len(suggestions) > 0)
        self.assertTrue(any("matériau" in s.lower() for s in suggestions))
        self.assertTrue(any("refroidissement" in s.lower() for s in suggestions))

    def test_model_save_load(self):
        """Test la sauvegarde et le chargement du modèle."""
        with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
            # Entraîner et sauvegarder
            training_data = [self.normal_results, self.abnormal_results]
            self.analyzer.train(training_data)
            save_success = self.analyzer.save_model(tmp.name)
            self.assertTrue(save_success)

            # Créer un nouvel analyseur et charger le modèle
            new_analyzer = ResultAnalyzer()
            load_success = new_analyzer.load_model(tmp.name)
            self.assertTrue(load_success)

            # Vérifier que le modèle chargé fonctionne
            features = new_analyzer.preprocess_results(self.normal_results)
            self.assertEqual(features.shape, (1, 6))

    def test_error_handling(self):
        """Test la gestion des erreurs."""
        # Test avec des données manquantes
        incomplete_results = {"max_stress": 30e6}
        analysis = self.analyzer.analyze_results(
            incomplete_results, self.material_props
        )
        self.assertNotIn("error", analysis)

        # Test avec des valeurs invalides
        invalid_results = {"max_stress": float("nan"), "max_displacement": float("inf")}
        analysis = self.analyzer.analyze_results(invalid_results, self.material_props)
        self.assertIn("error", analysis)

    def test_training(self):
        """Test l'entraînement du modèle."""
        # Générer des données d'entraînement
        training_data = []
        for i in range(100):
            if i < 80:
                # Données normales avec bruit
                data = {
                    k: v * (1 + np.random.normal(0, 0.1))
                    for k, v in self.normal_results.items()
                }
            else:
                # Données anormales avec bruit
                data = {
                    k: v * (1 + np.random.normal(0, 0.1))
                    for k, v in self.abnormal_results.items()
                }
            training_data.append(data)

        # Entraîner le modèle
        success = self.analyzer.train(training_data)
        self.assertTrue(success)

        # Vérifier que le modèle peut détecter les anomalies
        features = self.analyzer.preprocess_results(self.abnormal_results)
        anomalies = self.analyzer.detect_anomalies(features)
        self.assertTrue(len(anomalies) > 0)


if __name__ == "__main__":
    unittest.main()
