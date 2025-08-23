"""Tests pour le pipeline optimisé."""

import unittest
import torch
import torch.nn as nn
from ai_models.optimized_pipeline import OptimizedPipeline, PipelineConfig
import asyncio

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.linear(x)

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
        
    def test_tensor_cache_and_detach(self):
        """Test que les tenseurs sont correctement détachés et mis en cache."""
        model = SimpleModel()
        inputs = torch.randn(4, 10)
        
        # Premier passage
        result = asyncio.run(self.pipeline.process_model(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs),
                batch_size=2
            ),
            "test_model"
        ))
        
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertFalse(result[0].requires_grad)
        
        # Deuxième passage - devrait utiliser le cache
        result2 = asyncio.run(self.pipeline.process_model(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs),
                batch_size=2
            ),
            "test_model"
        ))
        
        # Vérifier que les résultats sont identiques
        torch.testing.assert_close(result[0], result2[0])
        
    def test_memory_optimization(self):
        """Test l'optimisation de la mémoire."""
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Créer quelques tenseurs
        tensors = [torch.randn(1000, 1000) for _ in range(10)]
        
        # Optimiser la mémoire
        self.pipeline.optimize_memory()
        
        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # La mémoire finale ne devrait pas être significativement plus élevée
        self.assertLess(final_memory - initial_memory, 1e6)  # Moins de 1MB de différence
        
    def test_batch_processing(self):
        """Test le traitement par batch."""
        model = SimpleModel()
        inputs = torch.randn(5, 10)  # 5 échantillons
        
        results = asyncio.run(self.pipeline.process_model(
            model,
            torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(inputs),
                batch_size=2  # Batch size de 2
            ),
            "test_batch"
        ))
        
        # Vérifier que nous avons le bon nombre de résultats
        total_outputs = torch.cat(results, dim=0)
        self.assertEqual(total_outputs.shape[0], 5)
        self.assertEqual(total_outputs.shape[1], 5)
        
    def test_error_handling(self):
        """Test la gestion des erreurs."""
        with self.assertRaises(ValueError):
            # Essayer de générer avec un modèle manquant
            asyncio.run(self.pipeline.generate(
                "test",
                {},  # Pas de modèles
                ["missing_model"]
            ))

if __name__ == '__main__':
    unittest.main()
