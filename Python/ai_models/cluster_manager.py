"""
Module de distribution des tâches sur cluster.
"""

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("⚠️  Ray not available. Cluster functionality will be disabled.")

import torch
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration du cluster."""

    num_workers: int = 4
    gpu_per_worker: float = 0.25  # Fraction de GPU par worker
    cpu_per_worker: int = 2
    memory_per_worker: int = 4 * 1024 * 1024 * 1024  # 4GB
    timeout: float = 30.0
    retry_attempts: int = 3
    enable_load_balancing: bool = True
    scheduler_type: str = "dynamic"  # "static", "dynamic", "adaptive"


if RAY_AVAILABLE:
    @ray.remote
    class ClusterWorker:
        """Worker pour le traitement distribué."""

        def __init__(self, gpu_fraction: float = 0.25):
            self.gpu_fraction = gpu_fraction
            if torch.cuda.is_available():
                # Limiter l'utilisation GPU
                torch.cuda.set_per_process_memory_fraction(gpu_fraction)

    def process_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        operation: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Traite un maillage.

        Args:
            vertices: Vertices du maillage
            faces: Faces du maillage
            operation: Opération à effectuer
            params: Paramètres de l'opération

        Returns:
            Résultats du traitement
        """
        import trimesh

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

        if operation == "simplify":
            target_faces = int(len(faces) * params.get("ratio", 0.5))
            mesh = mesh.simplify_quadratic_decimation(target_faces)

        elif operation == "subdivide":
            mesh = mesh.subdivide(params.get("iterations", 1))

        elif operation == "smooth":
            mesh = mesh.smooth(params.get("iterations", 1))

        return {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "operation": operation,
            "success": True,
        }

    def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Traite un lot de tâches.

        Args:
            batch_data: Liste de tâches à traiter

        Returns:
            Résultats du traitement
        """
        results = []
        for task in batch_data:
            try:
                result = self.process_mesh(
                    task["vertices"],
                    task["faces"],
                    task["operation"],
                    task.get("params", {}),
                )
                results.append(result)
            except Exception as e:
                results.append(
                    {"success": False, "error": str(e), "operation": task["operation"]}
                )

        return results

else:
    # Fallback ClusterWorker quand Ray n'est pas disponible
    class ClusterWorker:
        """Worker local pour le traitement quand Ray n'est pas disponible."""

        def __init__(self, gpu_fraction: float = 0.25):
            self.gpu_fraction = gpu_fraction

        @staticmethod
        def remote(gpu_fraction: float = 0.25):
            """Méthode pour compatibilité avec l'API Ray."""
            return ClusterWorker(gpu_fraction)

        def process_mesh(self, vertices, faces, operation, params=None):
            # Même implémentation que dans la version Ray
            import trimesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            if params is None:
                params = {}

            if operation == "simplify":
                target_faces = int(len(faces) * params.get("ratio", 0.5))
                mesh = mesh.simplify_quadric_decimation(target_faces)

            elif operation == "subdivide":
                mesh = mesh.subdivide(params.get("iterations", 1))

            elif operation == "smooth":
                mesh = mesh.smoothed(params.get("iterations", 1))

            return {
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "operation": operation,
                "success": True,
            }

        def process_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            results = []
            for task in batch_data:
                try:
                    result = self.process_mesh(
                        task["vertices"],
                        task["faces"],
                        task["operation"],
                        task.get("params", {}),
                    )
                    results.append(result)
                except Exception as e:
                    results.append(
                        {"success": False, "error": str(e), "operation": task["operation"]}
                    )
            return results


class ClusterManager:
    """Gestionnaire de cluster pour le traitement distribué."""

    def __init__(self, config: Optional[ClusterConfig] = None):
        self.config = config or ClusterConfig()

        # Initialiser Ray si disponible et pas déjà fait
        if RAY_AVAILABLE and not ray.is_initialized():
            ray.init()

        # Créer les workers
        self.workers = [
            ClusterWorker.remote(self.config.gpu_per_worker)
            for _ in range(self.config.num_workers)
        ]

        # File d'attente des tâches
        self.task_queue = queue.Queue()

        # Thread pool pour le traitement asynchrone
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.num_workers)

        # Démarrer le scheduler
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler, daemon=True
        )
        self._scheduler_thread.start()

        # Statistiques
        self.stats = {"tasks_completed": 0, "tasks_failed": 0, "processing_time": []}

    def _run_scheduler(self):
        """Exécute le scheduler de tâches."""
        while True:
            try:
                # Collecter les tâches en attente
                batch = []
                try:
                    while len(batch) < self.config.num_workers:
                        task = self.task_queue.get_nowait()
                        batch.append(task)
                except queue.Empty:
                    if not batch:
                        time.sleep(0.1)
                        continue

                # Distribuer les tâches selon la stratégie
                if self.config.scheduler_type == "static":
                    # Distribution statique
                    chunk_size = len(batch) // len(self.workers)
                    for i, worker in enumerate(self.workers):
                        start = i * chunk_size
                        end = (
                            start + chunk_size
                            if i < len(self.workers) - 1
                            else len(batch)
                        )
                        if start < end:
                            self._process_batch_on_worker(worker, batch[start:end])

                elif self.config.scheduler_type == "dynamic":
                    # Distribution dynamique avec vol de tâches
                    remaining_tasks = list(batch)
                    while remaining_tasks:
                        for worker in self.workers:
                            if not remaining_tasks:
                                break
                            task = remaining_tasks.pop(0)
                            self._process_batch_on_worker(worker, [task])

                else:  # adaptive
                    # Distribution adaptative basée sur la charge
                    worker_loads = self._get_worker_loads()
                    sorted_workers = sorted(
                        enumerate(self.workers), key=lambda x: worker_loads[x[0]]
                    )

                    tasks_per_worker = {}
                    for task in batch:
                        # Assigner au worker le moins chargé
                        worker_idx, worker = sorted_workers[0]
                        if worker not in tasks_per_worker:
                            tasks_per_worker[worker] = []
                        tasks_per_worker[worker].append(task)

                        # Mettre à jour la charge estimée
                        worker_loads[worker_idx] += 1
                        sorted_workers.sort(key=lambda x: worker_loads[x[0]])

                    # Traiter les lots
                    for worker, tasks in tasks_per_worker.items():
                        self._process_batch_on_worker(worker, tasks)

            except Exception as e:
                logger.error(f"Erreur dans le scheduler: {e}")
                time.sleep(1)

    def _process_batch_on_worker(
        self, worker: ray.actor.ActorHandle, batch: List[Dict[str, Any]]
    ):
        """
        Traite un lot sur un worker.

        Args:
            worker: Worker Ray
            batch: Lot de tâches
        """
        try:
            start_time = time.time()

            # Traiter le lot
            future = worker.process_batch.remote(batch)

            # Attendre le résultat avec timeout
            results = ray.get(future, timeout=self.config.timeout)

            # Mettre à jour les stats
            duration = time.time() - start_time
            self.stats["processing_time"].append(duration)
            self.stats["tasks_completed"] += len(results)

            # Vérifier les erreurs
            for result in results:
                if not result.get("success", False):
                    self.stats["tasks_failed"] += 1
                    logger.error(f"Échec de tâche: {result.get('error')}")

        except Exception as e:
            logger.error(f"Erreur de traitement: {e}")
            self.stats["tasks_failed"] += len(batch)

    def _get_worker_loads(self) -> List[float]:
        """
        Obtient la charge de chaque worker.

        Returns:
            Liste des charges (0-1) pour chaque worker
        """
        try:
            # Obtenir les métriques Ray
            workers_info = ray.nodes()

            loads = []
            for worker_info in workers_info:
                # Calculer la charge basée sur CPU, GPU et mémoire
                cpu_load = worker_info["cpu"] / worker_info["cpu_total"]
                memory_load = worker_info["memory"] / worker_info["memory_total"]

                if "gpu" in worker_info:
                    gpu_load = worker_info["gpu"] / worker_info["gpu_total"]
                    load = (cpu_load + memory_load + gpu_load) / 3
                else:
                    load = (cpu_load + memory_load) / 2

                loads.append(load)

            return loads

        except Exception:
            # En cas d'erreur, retourner des charges égales
            return [0.5] * len(self.workers)

    def process_mesh(
        self, mesh: Any, operation: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Traite un maillage sur le cluster.

        Args:
            mesh: Maillage à traiter
            operation: Opération à effectuer
            params: Paramètres de l'opération

        Returns:
            Maillage traité
        """
        task = {
            "vertices": mesh.vertices,
            "faces": mesh.faces,
            "operation": operation,
            "params": params or {},
        }

        # Ajouter à la file d'attente
        self.task_queue.put(task)

        # Attendre le résultat
        while True:
            try:
                result = ray.get(self.task_queue.get(), timeout=self.config.timeout)
                if result["operation"] == operation:
                    if result["success"]:
                        import trimesh

                        return trimesh.Trimesh(
                            vertices=result["vertices"], faces=result["faces"]
                        )
                    else:
                        raise RuntimeError(
                            f"Échec du traitement: {result.get('error')}"
                        )
            except queue.Empty:
                continue

    def process_batch(
        self, meshes: List[Any], operation: str, params: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Traite un lot de maillages sur le cluster.

        Args:
            meshes: Liste de maillages
            operation: Opération à effectuer
            params: Paramètres de l'opération

        Returns:
            Liste des maillages traités
        """
        tasks = [
            {
                "vertices": mesh.vertices,
                "faces": mesh.faces,
                "operation": operation,
                "params": params or {},
            }
            for mesh in meshes
        ]

        # Ajouter les tâches à la file d'attente
        for task in tasks:
            self.task_queue.put(task)

        # Attendre tous les résultats
        results: List[trimesh.Trimesh] = []
        timeout = time.time() + self.config.timeout

        while len(results) < len(meshes) and time.time() < timeout:
            try:
                result = ray.get(self.task_queue.get(), timeout=1)
                if result["operation"] == operation:
                    if result["success"]:
                        import trimesh

                        mesh = trimesh.Trimesh(
                            vertices=result["vertices"], faces=result["faces"]
                        )
                        results.append(mesh)
                    else:
                        results.append(None)
                        logger.error(f"Échec du traitement: {result.get('error')}")
            except queue.Empty:
                continue

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Obtient les statistiques du cluster.

        Returns:
            Statistiques d'utilisation et de performance
        """
        stats = self.stats.copy()

        if self.stats["processing_time"]:
            stats["avg_processing_time"] = np.mean(self.stats["processing_time"])
            stats["max_processing_time"] = np.max(self.stats["processing_time"])

        # Ajouter les infos du cluster
        cluster_info = ray.nodes()
        stats["cluster"] = {
            "num_nodes": len(cluster_info),
            "total_cpus": sum(node["cpu_total"] for node in cluster_info),
            "total_gpus": sum(node.get("gpu_total", 0) for node in cluster_info),
            "total_memory": sum(node["memory_total"] for node in cluster_info),
        }

        return stats

    def shutdown(self):
        """Arrête le cluster."""
        ray.shutdown()
        self.thread_pool.shutdown()
