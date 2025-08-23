import numpy as np
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import uuid
import os
from .error_handling import error_handler, SimulationError, ValidationError
import traceback
from ..core.monitoring import PerformanceMonitor, SimulationMetrics
from ..core.resource_manager import (
    ResourceManager,
    TaskPriority,
    ResourceLimits
)
from ..core.smart_cache import SmartCache
from ..core.task_distributor import TaskDistributor, DistributedTask

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation des gestionnaires
performance_monitor = PerformanceMonitor()
performance_monitor.start()

resource_manager = ResourceManager(max_workers=4)

# Initialisation du cache
cache_dir = os.path.join(os.path.expanduser('~'), '.macforge3d', 'cache')
os.makedirs(cache_dir, exist_ok=True)
smart_cache = SmartCache(cache_dir, max_size_gb=20.0)

# Initialisation du distributeur de tâches
task_distributor = TaskDistributor()

@dataclass
class SimulationState:
    """État de la simulation."""
    timestamp: datetime
    status: str  # 'preparing', 'running', 'completed', 'failed'
    progress: float
    current_step: str
    memory_usage: float
    warnings: List[str]
    error: Optional[Exception] = None

class RobustSimulationRunner:
    """Exécuteur de simulation avec gestion d'erreurs robuste."""
    
    def __init__(
        self,
        max_workers: int = 4,
        memory_threshold: float = 90.0,
        timeout: int = 3600
    ):
        """
        Initialise l'exécuteur.
        
        Args:
            max_workers: Nombre maximum de workers
            memory_threshold: Seuil d'utilisation mémoire en %
            timeout: Timeout en secondes
        """
        self.max_workers = max_workers
        self.memory_threshold = memory_threshold
        self.timeout = timeout
        self.state = SimulationState(
            timestamp=datetime.now(),
            status="preparing",
            progress=0.0,
            current_step="initialisation",
            memory_usage=0.0,
            warnings=[]
        )
        
    def run_simulation(
        self,
        parameters: Dict[str, Any],
        callback: Optional[callable] = None,
        priority_level: int = 50,
        deadline: Optional[datetime] = None,
        use_cache: bool = True,
        distributed: bool = True,
        optimize_params: bool = True
    ) -> Dict[str, Any]:
        """
        Exécute une simulation avec gestion d'erreurs et ressources.
        
        Args:
            parameters: Paramètres de la simulation
            callback: Fonction de callback pour le progrès
            priority_level: Niveau de priorité (0-100)
            deadline: Date limite optionnelle
            use_cache: Utiliser le cache intelligent
            distributed: Utiliser l'exécution distribuée
            optimize_params: Optimiser les paramètres par ML
            
        Returns:
            Résultats de la simulation
        """
        simulation_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Optimiser les paramètres si demandé
        if optimize_params:
            try:
                optimized = self._optimize_parameters(parameters)
                if optimized.confidence > 0.7:  # Seuil de confiance
                    logger.info("Utilisation des paramètres optimisés")
                    parameters = optimized.parameters
            except Exception as e:
                logger.warning(f"Erreur d'optimisation: {str(e)}")
                
        # Vérifier le cache
        if use_cache:
            cached_result = smart_cache.get(parameters)
            if cached_result is not None:
                logger.info("Résultat trouvé dans le cache")
                return cached_result
        
        try:
            # Valider les paramètres
            self._validate_parameters(parameters)
            
            # Estimer les besoins en ressources
            resource_limits = self._estimate_resource_needs(parameters)
            
            # Calculer le coût des ressources
            cost_factor = self._calculate_cost_factor(parameters)
            
            # Créer la priorité
            priority = TaskPriority(
                level=priority_level,
                deadline=deadline,
                cost_factor=cost_factor
            )
            
            # Créer les métriques initiales
            metrics = SimulationMetrics(
                simulation_id=simulation_id,
                start_time=start_time,
                status="pending",
                progress=0.0,
                current_step="préparation",
                memory_usage=0.0,
                processing_time=0.0,
                vertices_processed=0,
                error_count=0
            )
            performance_monitor.add_simulation_metrics(metrics)
            
            # Wrapper pour le callback avec monitoring
            def monitored_callback(state):
                if callback:
                    callback(state)
                # Mettre à jour les métriques
                metrics.status = state.status
                metrics.progress = state.progress
                metrics.current_step = state.current_step
                metrics.memory_usage = state.memory_usage
                metrics.processing_time = (
                    datetime.now() - start_time
                ).total_seconds()
                performance_monitor.add_simulation_metrics(metrics)
            
            # Fonction principale de simulation
            def simulation_task():
                try:
                    # Préparer la simulation
                    self._update_state("running", "préparation")
                    prepared_params = self._prepare_simulation(parameters)
                    
                    # Exécuter la simulation
                    results = error_handler.run_with_recovery(
                        self._run_simulation_steps,
                        prepared_params,
                        monitored_callback
                    )
                    
                    # Valider les résultats
                    self._validate_results(results)
                    
                    return results
                    
                except Exception as e:
                    self._handle_simulation_error(e)
                    raise
            
            # Choisir l'exécution locale ou distribuée
            if distributed and task_distributor.get_worker_count() > 0:
                # Exécution distribuée
                dist_task = DistributedTask(
                    id=simulation_id,
                    name=f"Simulation_{simulation_id}",
                    worker_id=None,
                    status="pending",
                    parameters=parameters,
                    start_time=start_time,
                    end_time=None
                )
                
                # Soumettre au distributeur de manière synchrone
                try:
                    # Utiliser une approche synchrone pour la compatibilité
                    result = self._submit_task_sync(dist_task, deadline)
                    
                    # Mettre en cache
                    if use_cache and result:
                        computation_time = (
                            datetime.now() - start_time
                        ).total_seconds()
                        smart_cache.put(
                            parameters,
                            result,
                            computation_time
                        )
                        
                    return result
                except Exception as e:
                    logger.warning(f"Échec de l'exécution distribuée: {e}")
                    # Fallback vers l'exécution locale
                    distributed = False
                
            else:
                # Exécution locale
                task_id = resource_manager.submit_task(
                    name=f"Simulation_{simulation_id}",
                    function=simulation_task,
                    priority=priority,
                    resource_limits=resource_limits
                )
                
                # Attendre et surveiller la tâche
                while True:
                    task = resource_manager.get_task_status(task_id)
                    if not task:
                        raise SimulationError("Tâche perdue", "TASK_LOST")
                        
                    if task.status == "completed":
                        # Mettre à jour les métriques finales
                        metrics.status = "completed"
                        metrics.progress = 1.0
                        metrics.processing_time = (
                            datetime.now() - start_time
                        ).total_seconds()
                        
                        # Mettre en cache
                        if use_cache:
                            smart_cache.put(
                                parameters,
                                task.result,
                                metrics.processing_time
                            )
                            
                        if "mesh" in task.result:
                            metrics.vertices_processed = len(
                                task.result["mesh"].get("vertices", [])
                            )
                        performance_monitor.add_simulation_metrics(metrics)
                        
                        # Mettre à jour l'optimiseur
                        self._update_optimizer(
                            parameters,
                            task.result,
                            metrics
                        )
                        
                        return task.result
                        
                    elif task.status == "failed":
                        # Propager l'erreur
                        metrics.status = "failed"
                        metrics.error_count += 1
                        performance_monitor.add_simulation_metrics(metrics)
                        raise task.error or SimulationError(
                            "Échec de la simulation",
                            "TASK_FAILED"
                        )
                        
                    # Vérifier le deadline
                    if deadline and datetime.now() > deadline:
                        resource_manager.cancel_task(task_id)
                        raise SimulationError(
                            "Deadline dépassé",
                            "DEADLINE_EXCEEDED"
                        )
                        
                    time.sleep(0.1)
                
        except Exception as e:
            self._handle_simulation_error(e)
            # Mettre à jour les métriques d'erreur
            metrics.status = "failed"
            metrics.error_count += 1
            metrics.processing_time = (
                datetime.now() - start_time
            ).total_seconds()
            performance_monitor.add_simulation_metrics(metrics)
            raise
            
    def _estimate_resource_needs(self, parameters: Dict[str, Any]) -> ResourceLimits:
        """Estime les besoins en ressources d'une simulation."""
        # Estimation basée sur la résolution du maillage
        mesh_size = parameters.get('mesh_resolution', 10000)
        
        # Formules empiriques
        cpu_percent = min(25 + (mesh_size / 10000) * 10, 90)
        memory_gb = max(1.0, (mesh_size * 24) / (1024 * 1024))  # 24 bytes par vertex
        
        # GPU si nécessaire
        gpu_memory = None
        if mesh_size > 100000:
            gpu_memory = max(1.0, memory_gb * 0.5)  # 50% de la RAM
            
        return ResourceLimits(
            max_cpu_percent=cpu_percent,
            max_memory_gb=memory_gb,
            max_gpu_memory_gb=gpu_memory,
            io_priority=0
        )
        
    def _calculate_cost_factor(self, parameters: Dict[str, Any]) -> float:
        """Calcule le facteur de coût d'une simulation."""
        base_cost = 1.0
        
        # Facteurs multiplicateurs
        if parameters.get('mesh_resolution', 0) > 100000:
            base_cost *= 1.5
            
        if parameters.get('material') in ['ABS', 'PETG']:
            base_cost *= 1.2
            
        return base_cost
        
    def _optimize_parameters(self, parameters: Dict[str, Any]) -> OptimizationResult:
        """Optimise les paramètres de simulation."""
        # Définir l'espace des paramètres
        parameter_space = {
            'mesh_resolution': (int, (1000, 1000000)),
            'temperature': (float, (150.0, 300.0)),
            'pressure': (float, (0.1, 10.0)),
            'material': (str, ['PLA', 'ABS', 'PETG'])
        }
        
        # Définir les objectifs
        goals = [
            OptimizationGoal(
                metric_name='quality_score',
                direction='maximize',
                weight=1.0,
                constraint_min=0.7
            ),
            OptimizationGoal(
                metric_name='processing_time',
                direction='minimize',
                weight=0.5,
                constraint_max=3600
            ),
            OptimizationGoal(
                metric_name='memory_usage',
                direction='minimize',
                weight=0.3,
                constraint_max=16000  # MB
            )
        ]
        
        # Créer ou récupérer l'optimiseur
        if not hasattr(self, '_ml_optimizer'):
            history_file = os.path.join(
                os.path.expanduser('~'),
                '.macforge3d',
                'optimization_history.csv'
            )
            self._ml_optimizer = MLOptimizer(
                parameter_space,
                goals,
                history_file
            )
            
        # Optimiser
        return self._ml_optimizer.optimize(
            n_trials=100,
            timeout=60
        )
        
    def _update_optimizer(
        self,
        parameters: Dict[str, Any],
        results: Dict[str, Any],
        metrics: SimulationMetrics
    ):
        """Met à jour l'optimiseur avec les résultats."""
        if not hasattr(self, '_ml_optimizer'):
            return
            
        # Calculer les métriques pour l'optimisation
        optimization_metrics = {
            'quality_score': self._calculate_quality_score(results),
            'processing_time': metrics.processing_time,
            'memory_usage': metrics.memory_usage
        }
        
        # Mettre à jour l'historique
        self._ml_optimizer.add_result(
            parameters,
            optimization_metrics
        )
        
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calcule un score de qualité pour les résultats."""
        score = 0.0
        
        try:
            # Vérifier la géométrie
            if "mesh" in results:
                mesh = results["mesh"]
                n_vertices = len(mesh.get("vertices", []))
                n_faces = len(mesh.get("faces", []))
                
                if n_vertices > 0 and n_faces > 0:
                    # Ratio vertices/faces optimal
                    ratio = n_faces / n_vertices
                    score += min(1.0, ratio / 2.0)
                    
                    # Complexité du maillage
                    complexity = np.log10(n_vertices) / 6.0  # max pour 1M vertices
                    score += min(1.0, complexity)
                    
            # Vérifier l'analyse thermique
            if "thermal" in results:
                thermal = results["thermal"]
                if "temperature" in thermal:
                    temps = thermal["temperature"]
                    if isinstance(temps, (list, np.ndarray)) and len(temps) > 0:
                        # Variation de température
                        temp_range = np.ptp(temps)
                        score += min(1.0, temp_range / 100.0)
                        
            # Vérifier l'analyse structurelle
            if "structural" in results:
                structural = results["structural"]
                if "stress" in structural:
                    stress = structural["stress"]
                    if isinstance(stress, (list, np.ndarray)) and len(stress) > 0:
                        # Niveau de stress moyen
                        mean_stress = np.mean(stress)
                        score += 1.0 - min(1.0, mean_stress / 100.0)
                        
            # Normaliser le score final
            return score / 4.0  # 4 composantes maximum
            
        except Exception as e:
            logger.warning(f"Erreur calcul score qualité: {str(e)}")
            return 0.0
            
    def _validate_parameters(self, parameters: Dict[str, Any]):
        """Valide les paramètres de simulation."""
        required_params = {
            "material": (str, ["PLA", "ABS", "PETG"]),
            "temperature": (float, (150.0, 300.0)),
            "pressure": (float, (0.1, 10.0)),
            "mesh_resolution": (int, (1000, 1000000))
        }
        
        for param, (param_type, constraints) in required_params.items():
            if param not in parameters:
                raise ValidationError(
                    f"Paramètre manquant: {param}",
                    "PARAM_MISSING",
                    {"parameter": param}
                )
                
            value = parameters[param]
            if not isinstance(value, param_type):
                raise ValidationError(
                    f"Type invalide pour {param}: {type(value)}",
                    "PARAM_TYPE",
                    {"parameter": param, "expected": param_type.__name__}
                )
                
            if isinstance(constraints, list):
                if value not in constraints:
                    raise ValidationError(
                        f"Valeur invalide pour {param}: {value}",
                        "PARAM_VALUE",
                        {"parameter": param, "allowed": constraints}
                    )
            elif isinstance(constraints, tuple):
                min_val, max_val = constraints
                if not min_val <= value <= max_val:
                    raise ValidationError(
                        f"Valeur hors limites pour {param}: {value}",
                        "PARAM_RANGE",
                        {
                            "parameter": param,
                            "min": min_val,
                            "max": max_val
                        }
                    )
                    
    def _check_resources(self):
        """Vérifie les ressources disponibles."""
        import psutil
        
        # Vérifier la mémoire
        memory_usage = psutil.Process().memory_percent()
        if memory_usage > self.memory_threshold:
            raise SimulationError(
                "Mémoire insuffisante",
                "RESOURCE_MEMORY",
                {"memory_usage": memory_usage}
            )
            
        # Vérifier le CPU
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 90:
            self.state.warnings.append(
                f"Charge CPU élevée: {cpu_usage}%"
            )
            
        # Mettre à jour l'état
        self.state.memory_usage = memory_usage
        
    def _prepare_simulation(
        self,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prépare les paramètres de simulation."""
        try:
            # Copier les paramètres
            prepared = parameters.copy()
            
            # Normaliser les valeurs
            if "temperature" in prepared:
                prepared["temperature"] = float(prepared["temperature"])
            if "pressure" in prepared:
                prepared["pressure"] = float(prepared["pressure"])
                
            # Ajouter des paramètres calculés
            prepared["thermal_conductivity"] = self._get_thermal_conductivity(
                prepared.get("material", "PLA")
            )
            
            return prepared
            
        except Exception as e:
            raise SimulationError(
                "Erreur de préparation",
                "PREP_ERROR",
                {"original_error": str(e)}
            )
            
    def _run_simulation_steps(
        self,
        parameters: Dict[str, Any],
        callback: Optional[callable]
    ) -> Dict[str, Any]:
        """Exécute les étapes de simulation."""
        results = {}
        total_steps = 4
        
        try:
            # 1. Préparation du maillage
            self._update_state("running", "maillage", 0.25)
            results["mesh"] = self._run_mesh_generation(parameters)
            if callback:
                callback(self.state)
                
            # 2. Analyse thermique
            self._update_state("running", "thermique", 0.50)
            results["thermal"] = self._run_thermal_analysis(
                parameters,
                results["mesh"]
            )
            if callback:
                callback(self.state)
                
            # 3. Analyse structurelle
            self._update_state("running", "structurelle", 0.75)
            results["structural"] = self._run_structural_analysis(
                parameters,
                results["mesh"]
            )
            if callback:
                callback(self.state)
                
            # 4. Post-traitement
            self._update_state("running", "post-traitement", 0.90)
            results = self._post_process_results(results)
            if callback:
                callback(self.state)
                
            return results
            
        except Exception as e:
            # Enrichir l'erreur avec le contexte
            raise SimulationError(
                f"Erreur pendant {self.state.current_step}",
                "SIM_STEP_ERROR",
                {
                    "step": self.state.current_step,
                    "progress": self.state.progress,
                    "original_error": str(e)
                }
            )
            
    def _validate_results(self, results: Dict[str, Any]):
        """Valide les résultats de simulation."""
        required_keys = ["mesh", "thermal", "structural"]
        
        # Vérifier la présence des clés requises
        for key in required_keys:
            if key not in results:
                raise ValidationError(
                    f"Résultat manquant: {key}",
                    "RESULT_MISSING",
                    {"missing_key": key}
                )
                
        # Vérifier la cohérence des résultats
        mesh_data = results["mesh"]
        if len(mesh_data.get("vertices", [])) == 0:
            raise ValidationError(
                "Maillage vide",
                "RESULT_EMPTY_MESH"
            )
            
        # Vérifier les valeurs aberrantes
        thermal_data = results["thermal"]
        if "temperature" in thermal_data:
            temps = np.array(thermal_data["temperature"])
            if np.any(temps < 0) or np.any(temps > 1000):
                self.state.warnings.append(
                    "Températures potentiellement aberrantes détectées"
                )
                
    def _update_state(
        self,
        status: str,
        step: str,
        progress: Optional[float] = None
    ):
        """Met à jour l'état de la simulation."""
        self.state.timestamp = datetime.now()
        self.state.status = status
        self.state.current_step = step
        if progress is not None:
            self.state.progress = progress
            
    def _handle_simulation_error(self, error: Exception):
        """Gère une erreur de simulation."""
        self.state.status = "failed"
        self.state.error = error
        
        # Log détaillé
        logger.error(
            "Erreur de simulation:\n"
            f"État: {self.state.status}\n"
            f"Étape: {self.state.current_step}\n"
            f"Progrès: {self.state.progress:.1%}\n"
            f"Erreur: {str(error)}\n"
            f"Traceback:\n{''.join(traceback.format_tb(error.__traceback__))}"
        )
        
    def _submit_task_sync(self, task: DistributedTask, deadline: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Soumet une tâche de manière synchrone au distributeur de tâches.
        
        Args:
            task: Tâche à soumettre
            deadline: Date limite optionnelle
            
        Returns:
            Résultats de la tâche
        """
        import threading
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        
        result = None
        error = None
        
        def run_task():
            nonlocal result, error
            try:
                # Simuler l'exécution de la tâche localement
                # En attendant l'implémentation complète du distributeur
                logger.info(f"Exécution locale de la tâche {task.id}")
                
                # Préparer la simulation localement
                prepared_params = self._prepare_simulation(task.parameters)
                
                # Exécuter les étapes de simulation
                def dummy_callback(state):
                    pass
                    
                result = self._run_simulation_steps(prepared_params, dummy_callback)
                
            except Exception as e:
                error = e
        
        # Calculer le timeout
        timeout_seconds = 300  # 5 minutes par défaut
        if deadline:
            timeout_seconds = (deadline - datetime.now()).total_seconds()
            timeout_seconds = max(30, min(timeout_seconds, 1800))  # Entre 30s et 30min
        
        # Exécuter avec timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_task)
            try:
                future.result(timeout=timeout_seconds)
                if error:
                    raise error
                return result
            except TimeoutError:
                logger.error(f"Timeout lors de l'exécution de la tâche {task.id}")
                raise SimulationError("Timeout lors de l'exécution distribuée")
        
    @staticmethod
    def _get_thermal_conductivity(material: str) -> float:
        """Retourne la conductivité thermique d'un matériau."""
        conductivities = {
            "PLA": 0.13,
            "ABS": 0.17,
            "PETG": 0.20
        }
        return conductivities.get(material, 0.15)  # Valeur par défaut
