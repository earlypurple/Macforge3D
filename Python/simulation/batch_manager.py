import os
import json
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
from .fem_analysis import FEMAnalysis
from .thermal_sim import ThermalSimulation

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SimulationBatch:
    """Classe pour stocker les informations d'un lot de simulation."""

    name: str
    description: str
    sim_type: str  # "fem" ou "thermal"
    parameters: Dict[str, Any]
    created_at: datetime
    results: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None


class BatchSimulationManager:
    """Gestionnaire de simulations par lots."""

    def __init__(
        self,
        output_dir: str,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        """
        Initialise le gestionnaire de simulations par lots.

        Args:
            output_dir: Répertoire de sortie pour les résultats
            max_workers: Nombre maximum de workers pour le ThreadPool
            progress_callback: Callback pour le suivi de la progression
        """
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.batches: Dict[str, SimulationBatch] = {}

        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)

        # Charger les batches existants
        self._load_batches()

    def _load_batches(self):
        """Charge les lots de simulation existants depuis le disque."""
        batch_file = os.path.join(self.output_dir, "batches.json")
        if os.path.exists(batch_file):
            try:
                with open(batch_file, "r") as f:
                    data = json.load(f)
                    for batch_data in data:
                        batch = SimulationBatch(
                            name=batch_data["name"],
                            description=batch_data["description"],
                            sim_type=batch_data["sim_type"],
                            parameters=batch_data["parameters"],
                            created_at=datetime.fromisoformat(batch_data["created_at"]),
                            results=batch_data.get("results"),
                            status=batch_data["status"],
                            error_message=batch_data.get("error_message"),
                        )
                        self.batches[batch.name] = batch
            except Exception as e:
                logger.error(f"Erreur lors du chargement des batches: {str(e)}")

    def _save_batches(self):
        """Sauvegarde les lots de simulation sur le disque."""
        batch_file = os.path.join(self.output_dir, "batches.json")
        try:
            data = []
            for batch in self.batches.values():
                batch_data = {
                    "name": batch.name,
                    "description": batch.description,
                    "sim_type": batch.sim_type,
                    "parameters": batch.parameters,
                    "created_at": batch.created_at.isoformat(),
                    "status": batch.status,
                    "results": batch.results,
                    "error_message": batch.error_message,
                }
                data.append(batch_data)

            with open(batch_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des batches: {str(e)}")

    def create_batch(
        self,
        name: str,
        description: str,
        sim_type: str,
        parameters: List[Dict[str, Any]],
    ) -> SimulationBatch:
        """
        Crée un nouveau lot de simulation.

        Args:
            name: Nom du lot
            description: Description du lot
            sim_type: Type de simulation ("fem" ou "thermal")
            parameters: Liste des paramètres pour chaque simulation

        Returns:
            Le lot de simulation créé
        """
        if name in self.batches:
            raise ValueError(f"Un lot nommé '{name}' existe déjà")

        batch = SimulationBatch(
            name=name,
            description=description,
            sim_type=sim_type,
            parameters=parameters,
            created_at=datetime.now(),
        )

        self.batches[name] = batch
        self._save_batches()

        return batch

    def run_batch(self, batch_name: str) -> Dict[str, Any]:
        """
        Exécute un lot de simulation.

        Args:
            batch_name: Nom du lot à exécuter

        Returns:
            Résultats des simulations
        """
        if batch_name not in self.batches:
            raise ValueError(f"Lot '{batch_name}' non trouvé")

        batch = self.batches[batch_name]
        batch.status = "running"
        batch.results = None
        batch.error_message = None
        self._save_batches()

        try:
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []

                # Soumettre les tâches
                for i, params in enumerate(batch.parameters):
                    if batch.sim_type == "fem":
                        future = executor.submit(
                            self._run_fem_simulation, params, i, len(batch.parameters)
                        )
                    else:  # thermal
                        future = executor.submit(
                            self._run_thermal_simulation,
                            params,
                            i,
                            len(batch.parameters),
                        )
                    futures.append(future)

                # Récupérer les résultats
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Erreur dans une simulation du lot: {str(e)}")
                        batch.error_message = str(e)

            # Mettre à jour le statut
            batch.status = "completed"
            if batch.error_message:
                batch.status = "failed"

            # Sauvegarder les résultats
            batch.results = {
                "simulations": results,
                "summary": self._compute_summary(results),
            }

            self._save_batches()
            return batch.results

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du lot: {str(e)}")
            batch.status = "failed"
            batch.error_message = str(e)
            self._save_batches()
            raise

    def _run_fem_simulation(
        self, params: Dict[str, Any], index: int, total: int
    ) -> Dict[str, Any]:
        """Exécute une simulation FEM."""
        sim = FEMAnalysis()

        # Créer un répertoire temporaire pour les fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_file = os.path.join(temp_dir, "mesh.ply")
            result_file = os.path.join(temp_dir, "results.json")

            # Configurer et exécuter la simulation
            sim.set_parameters(**params)
            result = sim.run_analysis()

            # Mettre à jour la progression
            if self.progress_callback:
                progress = (index + 1) / total * 100
                self.progress_callback("FEM simulation", progress)

            return result

    def _run_thermal_simulation(
        self, params: Dict[str, Any], index: int, total: int
    ) -> Dict[str, Any]:
        """Exécute une simulation thermique."""
        sim = ThermalSimulation()

        # Créer un répertoire temporaire pour les fichiers
        with tempfile.TemporaryDirectory() as temp_dir:
            mesh_file = os.path.join(temp_dir, "mesh.ply")
            result_file = os.path.join(temp_dir, "results.json")

            # Configurer et exécuter la simulation
            sim.set_parameters(**params)
            result = sim.run_simulation()

            # Mettre à jour la progression
            if self.progress_callback:
                progress = (index + 1) / total * 100
                self.progress_callback("Thermal simulation", progress)

            return result

    def _compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule un résumé des résultats."""
        summary = {
            "total_simulations": len(results),
            "successful_simulations": 0,
            "failed_simulations": 0,
            "average_values": {},
            "max_values": {},
            "min_values": {},
        }

        for result in results:
            if "error" not in result:
                summary["successful_simulations"] += 1

                # Calculer les statistiques pour chaque métrique
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        if key not in summary["average_values"]:
                            summary["average_values"][key] = []
                        summary["average_values"][key].append(value)

                        summary["max_values"][key] = max(
                            summary["max_values"].get(key, float("-inf")), value
                        )
                        summary["min_values"][key] = min(
                            summary["min_values"].get(key, float("inf")), value
                        )
            else:
                summary["failed_simulations"] += 1

        # Calculer les moyennes
        for key, values in summary["average_values"].items():
            summary["average_values"][key] = sum(values) / len(values)

        return summary

    def get_batch(self, name: str) -> Optional[SimulationBatch]:
        """Récupère un lot de simulation par son nom."""
        return self.batches.get(name)

    def list_batches(self) -> List[SimulationBatch]:
        """Liste tous les lots de simulation."""
        return list(self.batches.values())

    def delete_batch(self, name: str):
        """Supprime un lot de simulation."""
        if name in self.batches:
            del self.batches[name]
            self._save_batches()
        else:
            raise ValueError(f"Lot '{name}' non trouvé")
