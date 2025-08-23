"""
Tests unitaires pour le pipeline optimisé.
"""

import os
import torch
import unittest
import numpy as np
from typing import Dict, List
from pathlib import Path
import asyncio
from ai_models.optimized_pipeline import (
    OptimizedPipeline,
    PipelineConfig,
    CacheConfig
)

class SimpleModel(torch.nn.Module):
    """Modèle simple pour les tests."""
    
    def __init__(self, input_size=10, output_size=5):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, output_size)
        
    def forward(self, x):
        return self.linear(x)

def run_async_test(coro):
    """Helper pour exécuter un test asynchrone."""
    return asyncio.run(coro)

def async_test(f):
    """Décorateur pour les tests asynchrones."""
    def wrapper(*args, **kwargs):
        return run_async_test(f(*args, **kwargs))
    return wrapper

class TestOptimizedPipeline(unittest.TestCase):
    def setUp(self):
        self.config = PipelineConfig(
            batch_size=2,
            num_workers=2,
            device="cpu",  # Use CPU for testing
            optimize_memory=True,
            use_parallel=False,  # Disable parallel for testing
            use_cache=True
        )
        self.pipeline = OptimizedPipeline(self.config)
        
    def test_model_optimization(self):
        model = SimpleModel()
        optimized = self.pipeline._optimize_model(model)
        
        self.assertTrue(
            isinstance(optimized, torch.jit.RecursiveScriptModule)
        )
        
    def test_cache_usage(self):
        # Créer des données de test
        inputs = torch.randn(4, 10)
        model = SimpleModel()
        
        # Premier passage
        outputs1 = model(inputs)
        self.pipeline._cache_result("test_model", inputs, outputs1)
        
        # Deuxième passage depuis le cache
        cached = self.pipeline._get_cached_result("test_model", inputs)
        self.assertIsNotNone(cached)
        
        # Vérifier que les résultats sont identiques
        np.testing.assert_array_almost_equal(
            outputs1.detach().numpy(),
            cached
        )
        
    async def test_process_model(self):
        # Créer un modèle et des données
        model = SimpleModel()
        
        # Créer des données sans gradient
        with torch.no_grad():
            data = torch.randn(6, 10)  # 6 échantillons
        
        # Créer un dataset et dataloader
        dataset = torch.utils.data.TensorDataset(data)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False  # Pour garantir l'ordre
        )
        
        # Traiter avec le pipeline
        results = await self.pipeline.process_model(
            model,
            data_loader,
            "test"
        )
        
        # Vérifier les résultats
        total_outputs = torch.cat(results, dim=0)
        self.assertEqual(total_outputs.shape[0], 6)  # Nombre total d'échantillons
        self.assertEqual(total_outputs.shape[1], 5)  # Taille de sortie du modèle
        
    async def test_generate(self):
        # Créer des modèles de test avec des dimensions qui correspondent
        models = {
            "step1": SimpleModel(input_size=10, output_size=7),  # 10->7
            "step2": SimpleModel(input_size=7, output_size=5)    # 7->5
        }
        
        # Définir les étapes
        steps = ["step1", "step2"]
        
        # Créer un input tensor sans gradient
        with torch.no_grad():
            test_input = torch.randn(4, 10)  # Un batch de 4 échantillons
        
        # Générer
        results = await self.pipeline.generate(
            test_input,
            models,
            steps
        )
        
        # Vérifier les résultats
        self.assertIn("step1", results)
        self.assertIn("step2", results)
        self.assertIsInstance(results["step1"], torch.Tensor)
        self.assertIsInstance(results["step2"], torch.Tensor)
        
        # Vérifier les dimensions des résultats
        self.assertEqual(results["step1"].shape, (4, 7))  # Batch size x output_size step1
        self.assertEqual(results["step2"].shape, (4, 5))  # Batch size x output_size step2
        
    def test_memory_optimization(self):
        # Configurer le profiling
        self.pipeline.config.profile = True
        
        # Créer des données qui seront mises en cache
        data = torch.randn(100, 10)
        model_output = torch.randn(100, 5)
        
        # Mettre en cache plusieurs résultats
        for i in range(5):
            self.pipeline._cache_result(f"test_{i}", data, model_output)
            
        # Vérifier les stats initiales
        initial_stats = self.pipeline.profile_performance()
        self.assertIn("cache_stats", initial_stats)
        
        # Optimiser la mémoire
        self.pipeline.optimize_memory()
        
        # Vérifier les nouvelles stats
        final_stats = self.pipeline.profile_performance()
        self.assertIn("cache_stats", final_stats)
        
        # Les statistiques du cache devraient avoir changé
        initial_cache = initial_stats.get("cache_stats", {})
        final_cache = final_stats.get("cache_stats", {})
        
        # Au moins une des métriques devrait être différente
        self.assertGreater(
            len(set(final_cache.items()) ^ set(initial_cache.items())),
            0,
            "Les stats du cache devraient changer après optimisation"
        )
        
    def test_profile_performance(self):
        self.config.profile = True
        pipeline = OptimizedPipeline(self.config)
        
        stats = pipeline.profile_performance()
        
        self.assertIn("cache_stats", stats)
        self.assertIn("cuda_memory", stats)

def run_async_test(coro):
    return asyncio.run(coro)

# Wrapper pour les tests asynchrones
def async_test(f):
    def wrapper(*args, **kwargs):
        return run_async_test(f(*args, **kwargs))
    return wrapper

# Appliquer le wrapper aux tests asynchrones
TestOptimizedPipeline.test_process_model = async_test(
    TestOptimizedPipeline.test_process_model
)
TestOptimizedPipeline.test_generate = async_test(
    TestOptimizedPipeline.test_generate
)
