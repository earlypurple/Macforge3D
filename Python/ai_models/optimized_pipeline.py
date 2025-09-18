"""
Pipeline de génération optimisé avec parallélisation et caching intelligent.
"""

import os
import time
import asyncio
import logging
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import ray
from ai_models.smart_cache import SmartCache, CacheConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration du pipeline de génération."""

    batch_size: int = 4
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "float16"  # ou "float32"
    optimize_memory: bool = True
    use_parallel: bool = True
    use_cache: bool = True
    prefetch_size: int = 2
    profile: bool = False

    @property
    def cache_config(self) -> CacheConfig:
        """Configuration du cache par défaut."""
        return CacheConfig()


class OptimizedPipeline:
    """Pipeline de génération avec optimisations avancées."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.cache = SmartCache(self.config.cache_config)
        self._executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        self._setup_ray()

    def _setup_ray(self):
        """Configure Ray pour le traitement distribué."""
        if not ray.is_initialized() and self.config.use_parallel:
            ray.init(
                num_cpus=self.config.num_workers, include_dashboard=self.config.profile
            )

    def _optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimise un modèle pour l'inférence."""
        # Déterminer le dtype à utiliser
        dtype = torch.float16 if self.config.precision == "float16" else torch.float32

        # Convertir le modèle directement au bon dtype et device
        model = model.to(device=self.config.device, dtype=dtype)

        if self.config.optimize_memory:
            try:
                # Tenter l'optimisation JIT
                model = torch.jit.script(model)
                model = torch.jit.optimize_for_inference(model)

                # Reconvertir après JIT pour s'assurer du bon dtype
                model = model.to(dtype=dtype)
            except Exception as e:
                logger.warning(f"Échec de l'optimisation JIT: {e}")

        return model

    # Cette méthode sera utilisée avec ray.remote.remote lors de l'appel
    def _process_batch(
        self, model: torch.nn.Module, batch: torch.Tensor
    ) -> torch.Tensor:
        """Traite un batch de données en parallèle."""
        device = torch.device(self.config.device)
        batch = batch.to(device)
        model = model.to(device)

        with torch.cuda.amp.autocast(enabled=self.config.precision == "float16"):
            with torch.no_grad():
                output = model(batch)
                return output.detach().cpu()

    async def _prefetch_batch(
        self, data_loader: torch.utils.data.DataLoader
    ) -> List[Union[torch.Tensor, List[Any], Tuple[Any, ...]]]:
        """Précharge les prochains batchs."""
        futures = []
        for i, batch in enumerate(data_loader):
            if i >= self.config.prefetch_size:
                break
            if isinstance(batch, (list, tuple)):
                # Si c'est une séquence, convertir chaque élément
                futures.append(
                    self._executor.submit(
                        lambda x: [
                            (
                                item.to(self.config.device)
                                if isinstance(item, torch.Tensor)
                                else item
                            )
                            for item in x
                        ],
                        batch,
                    )
                )
            else:
                # Sinon, convertir directement le batch
                futures.append(
                    self._executor.submit(
                        lambda x: (
                            x.to(self.config.device)
                            if isinstance(x, torch.Tensor)
                            else x
                        ),
                        batch,
                    )
                )
        return [f.result() for f in futures]

    def _get_cached_result(
        self, model_name: str, inputs: Union[torch.Tensor, List, Tuple]
    ) -> Optional[torch.Tensor]:
        """Récupère un résultat du cache."""
        if not self.config.use_cache:
            return None

        # Extraire le tensor des inputs si c'est une liste/tuple
        if isinstance(inputs, (list, tuple)):
            input_tensor = inputs[0] if isinstance(inputs[0], torch.Tensor) else inputs
        else:
            input_tensor = inputs

        # Conversion sécurisée pour le hachage
        if isinstance(input_tensor, torch.Tensor):
            # Détacher et envoyer au CPU avant de convertir en numpy
            hash_value = hash(input_tensor.detach().cpu().numpy().tobytes())
        else:
            # Fallback pour les types non-tensors
            hash_value = hash(str(input_tensor))

        key = f"{model_name}_{hash_value}"
        return self.cache.get(key)

    def _cache_result(
        self,
        model_name: str,
        inputs: Union[torch.Tensor, List, Tuple],
        outputs: Union[torch.Tensor, List, Tuple],
    ):
        """Met en cache un résultat."""
        if not self.config.use_cache:
            return

        # Conversion sécurisée pour le hachage
        if isinstance(inputs, torch.Tensor):
            # Détacher et envoyer au CPU avant de convertir en numpy
            hash_value = hash(inputs.detach().cpu().numpy().tobytes())
        else:
            # Fallback pour les types non-tensors
            hash_value = hash(str(inputs))

        key = f"{model_name}_{hash_value}"

        # Conversion sécurisée pour le stockage
        if isinstance(outputs, torch.Tensor):
            self.cache.put(outputs.detach().cpu().numpy(), key)
        else:
            # Fallback pour les types non-tensors
            self.cache.put(str(outputs), key)

    async def process_model(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        model_name: str = "model",
    ) -> List[torch.Tensor]:
        """
        Traite des données avec un modèle optimisé.

        Args:
            model: Le modèle à utiliser
            data_loader: Le chargeur de données
            model_name: Nom du modèle pour le cache

        Returns:
            Liste des résultats
        """
        # Déterminer le dtype à utiliser
        dtype = torch.float16 if self.config.precision == "float16" else torch.float32

        # Convertir le modèle au bon dtype et device
        model = model.to(device=self.config.device, dtype=dtype)

        results = []

        # Traiter chaque batch
        for batch in data_loader:
            # Extraire le tensor du batch
            if isinstance(batch, (list, tuple)):
                input_tensor = batch[0] if isinstance(batch[0], torch.Tensor) else batch
            else:
                input_tensor = batch

            # Détacher les gradients et convertir au bon dtype/device
            with torch.no_grad():
                if isinstance(input_tensor, torch.Tensor):
                    input_tensor = input_tensor.detach().to(
                        device=self.config.device, dtype=dtype
                    )
                # Si ce n'est pas un tensor, nous devons le convertir ou l'ignorer
                else:
                    try:
                        input_tensor = torch.tensor(
                            input_tensor, device=self.config.device, dtype=dtype
                        )
                    except:
                        logger.warning(
                            f"Impossible de convertir les données en tensor: {type(input_tensor)}"
                        )

            # Vérifier le cache
            cached = self._get_cached_result(model_name, input_tensor)
            if cached is not None:
                results.append(
                    torch.from_numpy(cached).to(device=self.config.device, dtype=dtype)
                )
                continue

            # Traiter en parallèle
            if self.config.use_parallel:
                # Utilisation plus simple de ray.remote
                try:
                    # Définir une fonction pour le traitement en parallèle
                    @ray.remote
                    def remote_process(model_data, input_data, device, precision):
                        # Déplacer les données sur le bon appareil
                        input_data = input_data.to(device)
                        model_data = model_data.to(device)

                        # Traiter avec la précision appropriée
                        with torch.cuda.amp.autocast(enabled=precision == "float16"):
                            with torch.no_grad():
                                output = model_data(input_data)
                                return output.detach().cpu()

                    # Appeler la fonction distante
                    result_ref = remote_process.remote(
                        model, input_tensor, self.config.device, self.config.precision
                    )
                    # Attendre le résultat
                    result = await result_ref
                except Exception as e:
                    logger.error(f"Erreur lors du traitement parallèle: {e}")
                    # Fallback au traitement synchrone
                    with torch.no_grad():
                        result = model(input_tensor)
            else:
                with torch.no_grad():
                    result = model(input_tensor)

            # Mettre en cache
            self._cache_result(model_name, input_tensor, result)
            results.append(result)

        return results

    async def generate(
        self,
        prompt: Union[str, torch.Tensor],
        models: Dict[str, torch.nn.Module],
        steps: List[str],
    ) -> Dict[str, Any]:
        """
        Génère un résultat en utilisant plusieurs modèles en pipeline.

        Args:
            prompt: Le prompt initial
            models: Dictionnaire des modèles à utiliser
            steps: Liste des étapes du pipeline

        Returns:
            Résultats de la génération
        """
        results = {"prompt": prompt}
        current_input = prompt

        for step in steps:
            if step not in models:
                raise ValueError(f"Modèle non trouvé pour l'étape {step}")

            # Préparer les données
            if isinstance(current_input, torch.Tensor):
                dataset: torch.utils.data.Dataset = torch.utils.data.TensorDataset(
                    current_input
                )
            else:
                dataset = self._create_dataset(current_input)

            # Créer un DataLoader avec shuffle=False pour préserver l'ordre
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                shuffle=False,
            )

            # Traiter avec le modèle
            step_results = await self.process_model(models[step], data_loader, step)

            # Agréger les résultats et mettre à jour l'entrée pour la prochaine étape
            if step_results:
                current_input = torch.cat(step_results, dim=0)
            else:
                # Gérer le cas où aucun résultat n'est retourné
                logger.warning(f"Pas de résultats pour l'étape {step}")
                current_input = torch.tensor([])

            results[step] = current_input

        return results

    def _create_dataset(self, data: Any) -> torch.utils.data.Dataset:
        """Crée un dataset à partir des données."""
        if isinstance(data, str):
            # Pour le texte, convertir en tensor avec un embedding simple (pour le test)
            # Dans un cas réel, utilisez un vrai tokenizer/embedding
            text_tensor = torch.randn(1, 10)  # Simulation d'embedding
            return torch.utils.data.TensorDataset(text_tensor)
        elif isinstance(data, torch.Tensor):
            return torch.utils.data.TensorDataset(data)
        elif isinstance(data, (list, tuple)):
            return torch.utils.data.TensorDataset(torch.tensor(data))
        else:
            # Fallback pour tous les autres types de données
            return torch.utils.data.TensorDataset(torch.randn(1, 10))

    def _aggregate_results(self, results: List[torch.Tensor]) -> Optional[torch.Tensor]:
        """Agrège les résultats des batchs."""
        if not results:
            return None

        return torch.cat(results, dim=0)

    def optimize_memory(self):
        """Optimise l'utilisation de la mémoire."""
        if not self.config.optimize_memory:
            return

        # Nettoyer le cache CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Optimiser le cache
        self.cache.optimize()

        # Force collection des objets non utilisés
        import gc

        gc.collect()

        # Vérifier les fuites mémoire
        if self.config.profile:
            if torch.cuda.is_available():
                logger.info(
                    f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f}MB"
                )
                logger.info(
                    f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e6:.2f}MB"
                )

    def profile_performance(self) -> Dict[str, float]:
        """Profile les performances du pipeline."""
        if not self.config.profile:
            return {}

        stats = {
            "cache_stats": self.cache.stats,
            "cuda_memory": (
                {
                    "allocated": torch.cuda.memory_allocated(),
                    "cached": torch.cuda.memory_reserved(),
                }
                if torch.cuda.is_available()
                else {}
            ),
            "ray_stats": (
                ray.get_runtime_context().get_metrics()
                if self.config.use_parallel
                else {}
            ),
        }

        return stats


class TextDataset(torch.utils.data.Dataset):
    """Dataset pour les données textuelles."""

    def __init__(self, text: str):
        self.text = text
        # Ajouter tokenization si nécessaire

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> str:
        return self.text


class TensorDataset(torch.utils.data.Dataset):
    """Dataset pour les tenseurs."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __len__(self) -> int:
        return len(self.tensor)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensor[idx]
