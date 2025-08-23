import unittest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtTest import QTest
from PyQt6.QtCore import Qt
import sys
from datetime import datetime
from simulation.result_viewer import ResultViewer, RiskIndicator, RecommendationItem

class TestResultViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialise l'application Qt pour les tests."""
        cls.app = QApplication(sys.argv)
        
    def setUp(self):
        """Prépare les widgets pour les tests."""
        self.viewer = ResultViewer()
        
        self.test_results = {
            "material": "PLA",
            "timestamp": datetime.now().isoformat(),
            "risk_level": "moyen",
            "anomalies_detected": True,
            "recommendations": [
                "Recommandation test 1",
                "Recommandation critique",
                "Recommandation test 3"
            ],
            "optimization_suggestions": [
                "Optimisation test 1",
                "Optimisation test 2"
            ],
            "temperature_curve": {
                "times": [0, 1, 2, 3],
                "max_temps": [200, 180, 160, 140],
                "min_temps": [180, 160, 140, 120]
            },
            "stress_distribution": [10, 20, 30, 40, 50] * 20,
            "yield_strength": 45
        }
        
    def test_initial_state(self):
        """Teste l'état initial du visualiseur."""
        self.assertEqual(self.viewer.material_label.text(), "Matériau: -")
        self.assertEqual(self.viewer.status_label.text(), "Statut: -")
        
    def test_update_results(self):
        """Teste la mise à jour avec de nouveaux résultats."""
        self.viewer.update_results(self.test_results)
        
        self.assertEqual(
            self.viewer.material_label.text(),
            self.test_results["material"]
        )
        self.assertEqual(
            self.viewer.status_label.text(),
            "Anomalies Détectées"
        )
        
    def test_risk_indicator(self):
        """Teste l'indicateur de risque."""
        indicator = RiskIndicator()
        
        # Test des différents niveaux de risque
        for level in ["faible", "moyen", "élevé"]:
            indicator.set_risk_level(level)
            self.assertEqual(indicator.risk_level, level)
            
    def test_recommendation_item(self):
        """Teste les éléments de recommandation."""
        # Test recommandation normale
        normal_rec = RecommendationItem("Test normal", False)
        self.assertIn("fff3e0", normal_rec.styleSheet())
        
        # Test recommandation critique
        critical_rec = RecommendationItem("Test critique", True)
        self.assertIn("ffebee", critical_rec.styleSheet())
        
    def test_graph_updates(self):
        """Teste la mise à jour des graphiques."""
        self.viewer.update_results(self.test_results)
        
        # Vérifier que les graphiques ont été mis à jour
        temp_plot_items = self.viewer.temp_graph.plotItem.items
        self.assertTrue(len(temp_plot_items) > 0)
        
        stress_plot_items = self.viewer.stress_graph.plotItem.items
        self.assertTrue(len(stress_plot_items) > 0)
        
    def test_recommendations_updates(self):
        """Teste la mise à jour des recommandations."""
        self.viewer.update_results(self.test_results)
        
        # Compter les widgets de recommandation
        rec_count = 0
        for i in range(self.viewer.recommendations_layout.count()):
            item = self.viewer.recommendations_layout.itemAt(i)
            if isinstance(item.widget(), RecommendationItem):
                rec_count += 1
                
        self.assertEqual(rec_count, len(self.test_results["recommendations"]))
        
    def test_optimization_updates(self):
        """Teste la mise à jour des optimisations."""
        self.viewer.update_results(self.test_results)
        
        # Compter les widgets d'optimisation
        opt_count = 0
        for i in range(self.viewer.optimizations_layout.count()):
            item = self.viewer.optimizations_layout.itemAt(i)
            if isinstance(item.widget(), RecommendationItem):
                opt_count += 1
                
        self.assertEqual(opt_count, len(self.test_results["optimization_suggestions"]))
        
    def test_analysis_request_signal(self):
        """Teste le signal de demande d'analyse."""
        signal_received = False
        
        def slot(data):
            nonlocal signal_received
            signal_received = True
            
        self.viewer.analysisRequested.connect(slot)
        
        # Simuler un clic sur le bouton d'analyse
        analyze_button = None
        for child in self.viewer.findChildren(QPushButton):
            if child.text() == "Nouvelle Analyse":
                analyze_button = child
                break
                
        self.assertIsNotNone(analyze_button)
        QTest.mouseClick(analyze_button, Qt.MouseButton.LeftButton)
        
        self.assertTrue(signal_received)
        
    def test_error_handling(self):
        """Teste la gestion des erreurs lors de la mise à jour."""
        # Test avec des données invalides
        invalid_results = {
            "material": "PLA",
            "timestamp": "invalid_date",
            "recommendations": None
        }
        
        try:
            self.viewer.update_results(invalid_results)
        except Exception as e:
            self.fail(f"La mise à jour a levé une exception: {str(e)}")
            
    def test_clear_functions(self):
        """Teste les fonctions de nettoyage."""
        # Ajouter des données
        self.viewer.update_results(self.test_results)
        
        # Nettoyer les recommandations
        self.viewer.clear_recommendations()
        rec_count = sum(
            1 for i in range(self.viewer.recommendations_layout.count())
            if isinstance(
                self.viewer.recommendations_layout.itemAt(i).widget(),
                RecommendationItem
            )
        )
        self.assertEqual(rec_count, 0)
        
        # Nettoyer les optimisations
        self.viewer.clear_optimizations()
        opt_count = sum(
            1 for i in range(self.viewer.optimizations_layout.count())
            if isinstance(
                self.viewer.optimizations_layout.itemAt(i).widget(),
                RecommendationItem
            )
        )
        self.assertEqual(opt_count, 0)

if __name__ == '__main__':
    unittest.main()
