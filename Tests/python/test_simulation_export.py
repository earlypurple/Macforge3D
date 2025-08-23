import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from exporters.simulation_export import SimulationExporter

class TestSimulationExporter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prépare les données de test."""
        # Créer des données de maillage de test
        cls.mesh_data = {
            "triangles": [
                {
                    "v1": {"position": [0, 0, 0], "normal": [0, 0, 1]},
                    "v2": {"position": [1, 0, 0], "normal": [0, 0, 1]},
                    "v3": {"position": [0, 1, 0], "normal": [0, 0, 1]}
                }
            ]
        }
        
        # Créer des résultats de test
        cls.results = {
            "max_stress": 50.5,
            "max_displacement": 0.1,
            "min_safety_factor": 2.0,
            "material": "PLA",
            "timestamp": "2025-08-23T12:00:00Z",
            "recommendations": [
                "Le modèle semble structurellement sain",
                "La déformation maximale est acceptable"
            ]
        }
        
    def test_vtk_export(self):
        """Test l'export au format VTK."""
        with tempfile.NamedTemporaryFile(suffix='.vtp') as tmp:
            success = SimulationExporter.export_to_vtk(
                self.mesh_data,
                self.results,
                tmp.name
            )
            
            self.assertTrue(success)
            self.assertTrue(Path(tmp.name).exists())
            self.assertTrue(Path(tmp.name).stat().st_size > 0)
            
    def test_json_export(self):
        """Test l'export au format JSON."""
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            # Test sans maillage
            success1 = SimulationExporter.export_to_json(
                self.results,
                tmp.name
            )
            self.assertTrue(success1)
            
            # Vérifier le contenu
            with open(tmp.name) as f:
                data = json.load(f)
                self.assertIn("results", data)
                self.assertIn("recommendations", data)
                self.assertEqual(data["material"], "PLA")
                
            # Test avec maillage
            success2 = SimulationExporter.export_to_json(
                self.results,
                tmp.name,
                include_mesh=True,
                mesh_data=self.mesh_data
            )
            self.assertTrue(success2)
            
            # Vérifier le contenu avec maillage
            with open(tmp.name) as f:
                data = json.load(f)
                self.assertIn("mesh", data)
                self.assertEqual(len(data["mesh"]["triangles"]), 1)
                
    def test_report_export(self):
        """Test la génération de rapport HTML."""
        with tempfile.NamedTemporaryFile(suffix='.html') as tmp:
            success = SimulationExporter.export_report(
                self.results,
                tmp.name,
                title="Test Report"
            )
            
            self.assertTrue(success)
            self.assertTrue(Path(tmp.name).exists())
            
            # Vérifier le contenu HTML
            with open(tmp.name) as f:
                content = f.read()
                self.assertIn("Test Report", content)
                self.assertIn("PLA", content)
                self.assertIn("structurellement sain", content)
                
    def test_error_handling(self):
        """Test la gestion des erreurs."""
        # Test avec un chemin invalide
        success = SimulationExporter.export_to_json(
            self.results,
            "/nonexistent/path/file.json"
        )
        self.assertFalse(success)
        
        # Test avec des données invalides
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            success = SimulationExporter.export_to_json(
                {"invalid": float('nan')},  # JSON ne supporte pas NaN
                tmp.name
            )
            self.assertFalse(success)
            
    def test_large_data(self):
        """Test avec un grand jeu de données."""
        # Créer un grand maillage
        num_triangles = 1000
        large_mesh = {
            "triangles": [
                {
                    "v1": {"position": [i, 0, 0], "normal": [0, 0, 1]},
                    "v2": {"position": [i+1, 0, 0], "normal": [0, 0, 1]},
                    "v3": {"position": [i, 1, 0], "normal": [0, 0, 1]}
                }
                for i in range(num_triangles)
            ]
        }
        
        # Test l'export VTK avec beaucoup de données
        with tempfile.NamedTemporaryFile(suffix='.vtp') as tmp:
            success = SimulationExporter.export_to_vtk(
                large_mesh,
                self.results,
                tmp.name
            )
            self.assertTrue(success)
            self.assertTrue(Path(tmp.name).stat().st_size > 0)

if __name__ == '__main__':
    unittest.main()
