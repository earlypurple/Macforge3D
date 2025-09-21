import unittest
import numpy as np
from pathlib import Path
from Python.simulation.fem_analysis import (
    MaterialProperties,
    FEMAnalysis,
    analyze_model,
)
from Python.simulation.thermal_sim import (
    ThermalProperties,
    ThermalSimulation,
    simulate_thermal,
)


class TestMaterialProperties(unittest.TestCase):
    def test_material_properties_creation(self):
        """Test la création des propriétés matériaux."""
        material = MaterialProperties(
            young_modulus=3.5e9,
            poisson_ratio=0.36,
            density=1240,
            yield_strength=50e6,
            material_name="TEST",
        )

        self.assertEqual(material.material_name, "TEST")
        self.assertEqual(material.young_modulus, 3.5e9)
        self.assertEqual(material.poisson_ratio, 0.36)
        self.assertEqual(material.density, 1240)
        self.assertEqual(material.yield_strength, 50e6)

    def test_default_materials(self):
        """Test l'obtention des matériaux par défaut."""
        materials = MaterialProperties.get_default_materials()

        self.assertIn("PLA", materials)
        self.assertIn("ABS", materials)
        self.assertIn("PETG", materials)

        pla = materials["PLA"]
        self.assertEqual(pla.material_name, "PLA")
        self.assertEqual(pla.young_modulus, 3.5e9)


class TestFEMAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prépare les fichiers de test."""
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)

        # Chemins des fichiers de test
        cls.test_mesh = cls.test_data_dir / "test_cube.stl"
        cls.invalid_mesh = cls.test_data_dir / "invalid.stl"
        cls.nonexistent_mesh = cls.test_data_dir / "nonexistent.stl"

        # Créer un fichier STL invalide pour les tests d'erreur
        with open(cls.invalid_mesh, "w") as f:
            f.write("Invalid STL content")

    def test_fem_analysis_creation(self):
        """Test la création d'une instance FEMAnalysis."""
        material = MaterialProperties.get_default_materials()["PLA"]
        analysis = FEMAnalysis(str(self.test_mesh), material)

        self.assertEqual(analysis.mesh_path, self.test_mesh)
        self.assertEqual(analysis.material, material)
        self.assertIsNone(analysis.mesh)
        self.assertIsNone(analysis.solution)

    def test_analyze_model(self):
        """Test l'analyse complète d'un modèle."""
        result = analyze_model(str(self.test_mesh), material_name="PLA")

        self.assertIn("max_stress", result)
        self.assertIn("max_displacement", result)
        self.assertIn("min_safety_factor", result)
        self.assertIn("recommendations", result)

    def test_invalid_material(self):
        """Test avec un matériau invalide."""
        result = analyze_model(str(self.test_mesh), material_name="INVALID_MATERIAL")

        self.assertIn("error", result)
        self.assertIn("non supporté", result["error"])

    def test_nonexistent_file(self):
        """Test avec un fichier inexistant."""
        result = analyze_model(str(self.nonexistent_mesh), material_name="PLA")

        self.assertIn("error", result)
        self.assertIn("chargement du maillage", result["error"])

    def test_invalid_mesh(self):
        """Test avec un maillage invalide."""
        result = analyze_model(str(self.invalid_mesh), material_name="PLA")

        self.assertIn("error", result)

    def test_extreme_forces(self):
        """Test avec des forces extrêmes."""
        result = analyze_model(
            str(self.test_mesh),
            material_name="PLA",
            forces=[(0, 0, 0, 1e6, 1e6, 1e6)],  # Forces très grandes
        )

        self.assertIn("recommendations", result)
        self.assertTrue(
            any("renforcer" in r.lower() for r in result["recommendations"])
        )


class TestThermalProperties(unittest.TestCase):
    def test_thermal_properties_creation(self):
        """Test la création des propriétés thermiques."""
        material = ThermalProperties(
            thermal_conductivity=0.13,
            specific_heat=1800,
            density=1240,
            melting_point=180,
            glass_transition=60,
            material_name="TEST",
        )

        self.assertEqual(material.material_name, "TEST")
        self.assertEqual(material.thermal_conductivity, 0.13)
        self.assertEqual(material.specific_heat, 1800)
        self.assertEqual(material.density, 1240)
        self.assertEqual(material.melting_point, 180)
        self.assertEqual(material.glass_transition, 60)

    def test_default_thermal_materials(self):
        """Test l'obtention des matériaux thermiques par défaut."""
        materials = ThermalProperties.get_default_materials()

        self.assertIn("PLA", materials)
        self.assertIn("ABS", materials)
        self.assertIn("PETG", materials)

        pla = materials["PLA"]
        self.assertEqual(pla.material_name, "PLA")
        self.assertEqual(pla.thermal_conductivity, 0.13)


class TestThermalSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Prépare les fichiers de test."""
        cls.test_data_dir = Path(__file__).parent / "test_data"
        cls.test_data_dir.mkdir(exist_ok=True)

        # Utiliser le même cube de test que pour FEM
        cls.test_mesh = cls.test_data_dir / "test_cube.stl"

    def test_thermal_simulation_creation(self):
        """Test la création d'une instance ThermalSimulation."""
        material = ThermalProperties.get_default_materials()["PLA"]
        simulation = ThermalSimulation(
            str(self.test_mesh), material, initial_temp=200.0, ambient_temp=25.0
        )

        self.assertEqual(simulation.mesh_path, self.test_mesh)
        self.assertEqual(simulation.material, material)
        self.assertEqual(simulation.initial_temp, 200.0)
        self.assertEqual(simulation.ambient_temp, 25.0)

    def test_simulate_thermal(self):
        """Test la simulation thermique complète."""
        result = simulate_thermal(
            str(self.test_mesh),
            material_name="PLA",
            initial_temp=200.0,
            ambient_temp=25.0,
            simulation_time=60.0,  # Simulation courte pour les tests
            time_steps=10,
        )

        self.assertIn("max_temperature", result)
        self.assertIn("final_temperature", result)
        self.assertIn("cooling_rate", result)
        self.assertIn("time_above_glass_transition", result)
        self.assertIn("temperature_curve", result)
        self.assertIn("recommendations", result)

    def test_invalid_initial_temperature(self):
        """Test avec une température initiale invalide."""
        result = simulate_thermal(
            str(self.test_mesh),
            material_name="PLA",
            initial_temp=-50.0,  # Température impossible
        )

        self.assertIn("error", result)

    def test_extreme_temperature(self):
        """Test avec une température très élevée."""
        material = ThermalProperties.get_default_materials()["PLA"]
        result = simulate_thermal(
            str(self.test_mesh),
            material_name="PLA",
            initial_temp=material.melting_point
            + 50,  # Bien au-dessus du point de fusion
        )

        self.assertIn("recommendations", result)
        self.assertTrue(any("fusion" in r.lower() for r in result["recommendations"]))

    def test_rapid_cooling(self):
        """Test avec un refroidissement rapide."""
        result = simulate_thermal(
            str(self.test_mesh),
            material_name="PLA",
            initial_temp=200.0,
            ambient_temp=0.0,  # Refroidissement très rapide
            simulation_time=10.0,
        )

        self.assertIn("recommendations", result)
        self.assertTrue(
            any("refroidissement" in r.lower() for r in result["recommendations"])
        )

    def test_time_step_limits(self):
        """Test les limites des pas de temps."""
        # Test avec très peu de pas de temps
        result1 = simulate_thermal(str(self.test_mesh), time_steps=2)  # Trop peu de pas
        self.assertIn("error", result1)

        # Test avec beaucoup de pas de temps
        result2 = simulate_thermal(
            str(self.test_mesh),
            time_steps=1000,  # Beaucoup de pas
            simulation_time=10.0,
        )
        self.assertNotIn("error", result2)

    def test_temperature_curve_consistency(self):
        """Test la cohérence des courbes de température."""
        result = simulate_thermal(
            str(self.test_mesh),
            material_name="PLA",
            initial_temp=200.0,
            ambient_temp=25.0,
            simulation_time=60.0,
            time_steps=10,
        )

        curve = result["temperature_curve"]
        times = curve["times"]
        max_temps = curve["max_temps"]
        min_temps = curve["min_temps"]

        # Vérifier que les températures diminuent
        self.assertTrue(all(t1 >= t2 for t1, t2 in zip(max_temps[:-1], max_temps[1:])))

        # Vérifier que max_temp ≥ min_temp
        self.assertTrue(all(tmax >= tmin for tmax, tmin in zip(max_temps, min_temps)))


if __name__ == "__main__":
    unittest.main()
