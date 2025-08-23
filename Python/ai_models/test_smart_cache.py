"""
Tests unitaires pour le système de cache intelligent.
"""

import os
import time
import shutil
import tempfile
import unittest
import numpy as np
from pathlib import Path
from ai_models.smart_cache import SmartCache, CacheConfig

class TestSmartCache(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig(
            max_memory_size=1024 * 1024,  # 1 MB
            max_disk_size=10 * 1024 * 1024,  # 10 MB
            cache_dir=self.temp_dir,
            cleanup_interval=1
        )
        self.cache = SmartCache(self.config)
        
    def tearDown(self):
        self.cache.clear()
        shutil.rmtree(self.temp_dir)
        
    def test_put_and_get_simple(self):
        data = "test_data"
        key = self.cache.put(data)
        
        self.assertEqual(self.cache.get(key), data)
        
    def test_put_and_get_numpy(self):
        data = np.random.rand(100, 100)
        key = self.cache.put(data)
        
        retrieved = self.cache.get(key)
        np.testing.assert_array_equal(retrieved, data)
        
    def test_memory_limit(self):
        # Créer des données plus grandes que la limite mémoire
        data = np.random.rand(500, 500)  # ~2 MB
        key = self.cache.put(data, force_disk=True)
        
        # Vérifier que les données sont sur le disque
        self.assertIsNone(self.cache._get_from_memory(key))
        retrieved = self.cache._get_from_disk(key)
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved.reshape(500, 500), data)
        
    def test_invalidate(self):
        data = "test_data"
        key = self.cache.put(data)
        
        self.cache.invalidate(key)
        self.assertIsNone(self.cache.get(key))
        
    def test_clear(self):
        keys = []
        for i in range(10):
            key = self.cache.put(f"data_{i}")
            keys.append(key)
            
        self.cache.clear()
        
        for key in keys:
            self.assertIsNone(self.cache.get(key))
            
    def test_cleanup(self):
        # Remplir le cache avec des données
        data = np.random.rand(250, 250)  # ~0.5 MB
        keys = []
        
        for i in range(5):  # ~10 MB total
            key = self.cache.put(data)
            keys.append(key)
            
        # Attendre le nettoyage
        time.sleep(2)
        
        # Vérifier que certaines données ont été supprimées
        self.assertLess(
            self.cache._disk_usage,
            self.config.max_disk_size
        )
        
    def test_compression(self):
        # Tester différents algorithmes
        algorithms = ["zstd", "lz4", "blosc2"]
        
        for algo in algorithms:
            self.config.compression_algorithm = algo
            cache = SmartCache(self.config)
            
            data = np.random.rand(100, 100)
            key = cache.put(data)
            
            retrieved = cache.get(key)
            np.testing.assert_array_equal(retrieved, data)
            
    def test_preload(self):
        # Créer plusieurs entrées
        keys = []
        for i in range(5):
            key = self.cache.put(f"data_{i}")
            keys.append(key)
            
        # Précharger en mémoire
        self.cache.preload(keys)
        
        # Vérifier que tout est en mémoire
        for key in keys:
            self.assertIsNotNone(self.cache._get_from_memory(key))
            
    def test_stats(self):
        stats = self.cache.stats
        
        self.assertIn("memory_usage", stats)
        self.assertIn("disk_usage", stats)
        self.assertIn("compression_algo", stats)
        self.assertIn("compression_level", stats)
        
    def test_optimize(self):
        # Remplir le cache
        data = np.random.rand(500, 500)
        for i in range(5):
            self.cache.put(data)
            
        # Optimiser
        initial_level = self.cache.config.compression_level
        self.cache.optimize()
        
        # Vérifier les changements
        self.assertNotEqual(
            initial_level,
            self.cache.config.compression_level
        )

if __name__ == '__main__':
    unittest.main()
