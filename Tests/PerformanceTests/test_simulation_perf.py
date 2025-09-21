import pytest
import time
import psutil
import numpy as np
from Python.simulation.robust_runner import RobustSimulationRunner


def get_memory_usage():
    """Retourne l'utilisation mémoire en MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def generate_large_mesh(size):
    """Génère un grand maillage pour les tests."""
    vertices = np.random.rand(size, 3)
    faces = np.random.randint(0, size, (size // 3, 3))
    return {"vertices": vertices, "faces": faces}


@pytest.mark.performance
class TestSimulationPerformance:
    @pytest.fixture(autouse=True)
    def setup_runner(self):
        self.runner = RobustSimulationRunner(
            max_workers=4, memory_threshold=90.0, timeout=300
        )

    def generate_test_params(self, mesh_size):
        """Génère des paramètres de test avec une taille de maillage spécifique."""
        return {
            "material": "PLA",
            "temperature": 200.0,
            "pressure": 1.0,
            "mesh_resolution": mesh_size,
        }

    @pytest.mark.parametrize("mesh_size", [1000, 10000, 100000])
    def test_simulation_scaling(self, mesh_size):
        """Test de la mise à l'échelle des performances avec différentes tailles de maillage."""
        params = self.generate_test_params(mesh_size)

        start_time = time.time()
        start_mem = get_memory_usage()

        try:
            results = self.runner.run_simulation(params)

            end_time = time.time()
            end_mem = get_memory_usage()

            duration = end_time - start_time
            mem_used = end_mem - start_mem

            # Vérification des performances
            expected_duration = (mesh_size / 1000) * 0.1  # 0.1s par 1000 vertices
            assert (
                duration <= expected_duration
            ), f"Simulation trop lente: {duration:.2f}s > {expected_duration:.2f}s"

            # Vérification de l'utilisation mémoire
            expected_mem = (mesh_size * 24) / 1024 / 1024  # 24 bytes par vertex
            assert (
                mem_used <= expected_mem * 1.5
            ), f"Utilisation mémoire excessive: {mem_used:.1f}MB > {expected_mem * 1.5:.1f}MB"

        except Exception as e:
            pytest.fail(f"Erreur pendant le test de performance: {str(e)}")

    def test_concurrent_simulations(self):
        """Test des performances avec des simulations concurrentes."""
        n_sims = 4
        mesh_size = 5000
        params = self.generate_test_params(mesh_size)

        start_time = time.time()

        try:
            # Lancer plusieurs simulations en parallèle
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_sims) as executor:
                futures = [
                    executor.submit(self.runner.run_simulation, params)
                    for _ in range(n_sims)
                ]
                results = [f.result() for f in futures]

            end_time = time.time()
            duration = end_time - start_time

            # La durée totale devrait être proche de la durée d'une seule simulation
            single_sim_time = 0.1 * (mesh_size / 1000)
            assert (
                duration <= single_sim_time * 2
            ), "Les simulations parallèles sont trop lentes"

        except Exception as e:
            pytest.fail(f"Erreur pendant le test concurrent: {str(e)}")

    def test_memory_cleanup(self):
        """Test de la libération correcte de la mémoire."""
        mesh_size = 50000
        params = self.generate_test_params(mesh_size)

        initial_mem = get_memory_usage()

        # Exécuter plusieurs simulations séquentiellement
        for _ in range(5):
            self.runner.run_simulation(params)
            time.sleep(0.1)  # Permettre au GC de s'exécuter

        final_mem = get_memory_usage()
        mem_growth = final_mem - initial_mem

        # La croissance mémoire devrait être limitée
        assert (
            mem_growth <= 50
        ), f"Fuite mémoire détectée: {mem_growth:.1f}MB de croissance"

    @pytest.mark.slow
    def test_long_running_simulation(self):
        """Test de simulation longue durée."""
        mesh_size = 200000
        params = self.generate_test_params(mesh_size)

        start_time = time.time()
        start_mem = get_memory_usage()

        try:
            results = self.runner.run_simulation(params)

            end_time = time.time()
            end_mem = get_memory_usage()

            duration = end_time - start_time
            mem_peak = end_mem - start_mem

            # Vérifier la stabilité sur longue durée
            assert duration <= 60, f"Simulation trop longue: {duration:.1f}s"
            assert mem_peak <= 1024, f"Pic mémoire trop élevé: {mem_peak:.1f}MB"

        except Exception as e:
            pytest.fail(f"Erreur pendant la simulation longue durée: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--durations=0"])
