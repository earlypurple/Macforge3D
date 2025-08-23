import pytest
import psutil
import gc
import time
import numpy as np
from Python.simulation.robust_runner import RobustSimulationRunner

def get_memory_info():
    """Retourne les informations détaillées sur l'utilisation mémoire."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / 1024 / 1024,  # MB
        'vms': mem_info.vms / 1024 / 1024,  # MB
        'shared': mem_info.shared / 1024 / 1024,  # MB
        'data': mem_info.data / 1024 / 1024,  # MB
        'uss': process.memory_full_info().uss / 1024 / 1024  # MB
    }

class MemoryLeakDetector:
    def __init__(self, threshold_mb=1.0):
        self.threshold = threshold_mb
        self.baseline = None
        
    def start(self):
        """Établit la référence mémoire initiale."""
        gc.collect()
        self.baseline = get_memory_info()
        
    def check(self):
        """Vérifie la croissance mémoire depuis le début."""
        gc.collect()
        current = get_memory_info()
        growth = {
            k: current[k] - self.baseline[k]
            for k in self.baseline.keys()
        }
        return growth

@pytest.mark.memory
class TestMemoryUsage:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.runner = RobustSimulationRunner(
            max_workers=2,
            memory_threshold=90.0,
            timeout=60
        )
        self.detector = MemoryLeakDetector(threshold_mb=1.0)
        
    def generate_test_data(self, size):
        """Génère des données de test de taille spécifiée."""
        return {
            "material": "PLA",
            "temperature": 200.0,
            "pressure": 1.0,
            "mesh_resolution": size,
            "custom_data": np.random.rand(size, 3)
        }

    def test_memory_single_simulation(self):
        """Test de l'utilisation mémoire pour une simulation simple."""
        self.detector.start()
        
        # Exécuter une simulation
        params = self.generate_test_data(10000)
        self.runner.run_simulation(params)
        
        # Vérifier la croissance mémoire
        growth = self.detector.check()
        assert growth['rss'] <= 50, \
            f"Croissance mémoire excessive: {growth['rss']:.1f}MB"

    def test_memory_leak_multiple_simulations(self):
        """Test des fuites mémoire sur plusieurs simulations."""
        self.detector.start()
        initial_usage = get_memory_info()
        
        # Exécuter plusieurs simulations
        for i in range(5):
            params = self.generate_test_data(5000)
            self.runner.run_simulation(params)
            
            # Forcer le GC
            gc.collect()
            
            # Vérifier après chaque itération
            current_usage = get_memory_info()
            growth = current_usage['rss'] - initial_usage['rss']
            
            assert growth <= 10 * (i + 1), \
                f"Fuite mémoire détectée à l'itération {i}: {growth:.1f}MB"

    def test_memory_pressure(self):
        """Test du comportement sous pression mémoire."""
        large_data = []
        initial_mem = get_memory_info()['rss']
        
        try:
            # Créer une pression mémoire
            for _ in range(10):
                large_data.append(np.random.rand(100000, 3))
                
            # Exécuter une simulation sous pression
            params = self.generate_test_data(5000)
            self.runner.run_simulation(params)
            
            # Vérifier l'utilisation mémoire
            peak_mem = get_memory_info()['rss']
            mem_growth = peak_mem - initial_mem
            
            assert mem_growth <= 1000, \
                f"Utilisation mémoire excessive sous pression: {mem_growth:.1f}MB"
            
        finally:
            # Nettoyer
            large_data.clear()
            gc.collect()

    def test_memory_recovery(self):
        """Test de la récupération mémoire après erreur."""
        self.detector.start()
        
        try:
            # Provoquer une erreur
            invalid_params = self.generate_test_data(5000)
            invalid_params["temperature"] = "invalid"
            
            with pytest.raises(Exception):
                self.runner.run_simulation(invalid_params)
            
            # Vérifier la récupération mémoire
            time.sleep(0.1)  # Permettre au GC de s'exécuter
            gc.collect()
            
            growth = self.detector.check()
            assert growth['rss'] <= 1.0, \
                f"Fuite mémoire après erreur: {growth['rss']:.1f}MB"
            
        except Exception as e:
            pytest.fail(f"Erreur inattendue: {str(e)}")

    @pytest.mark.slow
    def test_memory_large_dataset(self):
        """Test avec un grand jeu de données."""
        self.detector.start()
        
        try:
            # Créer un grand jeu de données
            params = self.generate_test_data(100000)
            
            # Mesurer l'utilisation mémoire pendant la simulation
            peak_mem = 0
            
            def memory_callback(state):
                nonlocal peak_mem
                current = get_memory_info()['rss']
                peak_mem = max(peak_mem, current)
            
            # Exécuter la simulation avec surveillance
            self.runner.run_simulation(params, memory_callback)
            
            # Vérifier les limites mémoire
            assert peak_mem <= 1024, \
                f"Pic mémoire trop élevé: {peak_mem:.1f}MB"
            
            # Vérifier la libération
            gc.collect()
            final_growth = self.detector.check()
            assert final_growth['rss'] <= 10, \
                f"Mémoire non libérée: {final_growth['rss']:.1f}MB"
            
        except Exception as e:
            pytest.fail(f"Erreur pendant le test grand volume: {str(e)}")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--durations=0'])
