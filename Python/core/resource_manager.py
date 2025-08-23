"""
Gestionnaire de ressources intelligent pour MacForge3D.
Gère l'allocation des ressources, la priorisation et l'optimisation.
"""

import threading
import queue
import time
import logging
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import GPUtil
from ..simulation.error_handling import SimulationError

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Limites de ressources pour une tâche."""
    max_cpu_percent: float = 80.0
    max_memory_gb: float = 8.0
    max_gpu_memory_gb: Optional[float] = None
    io_priority: int = 0  # 0 = normal, -1 = bas, 1 = haut

@dataclass
class TaskPriority:
    """Priorité d'une tâche."""
    level: int  # 0-100, plus élevé = plus prioritaire
    deadline: Optional[datetime] = None
    cost_factor: float = 1.0  # multiplicateur de coût des ressources

@dataclass
class Task:
    """Tâche à exécuter avec ses métadonnées."""
    id: str
    name: str
    function: Callable
    args: tuple
    kwargs: Dict
    priority: TaskPriority
    resource_limits: ResourceLimits
    status: str = "pending"
    progress: float = 0.0
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metrics: Dict = field(default_factory=dict)

class ResourceManager:
    """Gestionnaire de ressources avec optimisation."""
    
    def __init__(
        self,
        max_workers: int = 4,
        optimization_interval: int = 60
    ):
        self.max_workers = max_workers
        self.optimization_interval = optimization_interval
        
        # Files d'attente par priorité
        self.queues = {
            'high': queue.PriorityQueue(),
            'normal': queue.PriorityQueue(),
            'low': queue.PriorityQueue()
        }
        
        # Tâches en cours
        self.running_tasks: Dict[str, Task] = {}
        
        # Verrou pour la synchronisation
        self.lock = threading.Lock()
        
        # Pool d'exécution
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Métriques et historique
        self.metrics_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
        
        # Démarrer l'optimiseur
        self.running = True
        self.optimizer_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimizer_thread.start()
        
    def submit_task(
        self,
        name: str,
        function: Callable,
        priority: TaskPriority,
        resource_limits: Optional[ResourceLimits] = None,
        *args,
        **kwargs
    ) -> str:
        """Soumet une nouvelle tâche."""
        # Créer l'ID de tâche
        task_id = f"task_{int(time.time())}_{hash(function.__name__)}"
        
        # Créer la tâche
        task = Task(
            id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            resource_limits=resource_limits or ResourceLimits()
        )
        
        # Déterminer la file appropriée
        queue_name = self._get_queue_name(priority.level)
        
        # Ajouter à la file
        self.queues[queue_name].put((-priority.level, task))
        logger.info(f"Tâche {task_id} soumise avec priorité {priority.level}")
        
        # Tenter de démarrer des tâches
        self._process_queues()
        
        return task_id
        
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Récupère le statut d'une tâche."""
        return self.running_tasks.get(task_id)
        
    def cancel_task(self, task_id: str) -> bool:
        """Annule une tâche."""
        with self.lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                task.status = "cancelled"
                return True
        return False
        
    def _get_queue_name(self, priority: int) -> str:
        """Détermine la file d'attente basée sur la priorité."""
        if priority >= 70:
            return 'high'
        elif priority >= 30:
            return 'normal'
        return 'low'
        
    def _process_queues(self):
        """Traite les files d'attente selon les ressources disponibles."""
        with self.lock:
            # Vérifier les ressources disponibles
            if not self._has_available_resources():
                return
                
            # Traiter les files par ordre de priorité
            for queue_name in ['high', 'normal', 'low']:
                queue = self.queues[queue_name]
                
                while not queue.empty():
                    # Vérifier à nouveau les ressources
                    if not self._has_available_resources():
                        return
                        
                    # Récupérer la prochaine tâche
                    _, task = queue.get()
                    
                    # Vérifier si la tâche peut s'exécuter
                    if self._can_run_task(task):
                        self._start_task(task)
                    else:
                        # Remettre dans la file si impossible
                        queue.put((-task.priority.level, task))
                        break
                        
    def _has_available_resources(self) -> bool:
        """Vérifie si des ressources sont disponibles."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return (
            cpu_percent < 90 and
            memory_percent < 90 and
            len(self.running_tasks) < self.max_workers
        )
        
    def _can_run_task(self, task: Task) -> bool:
        """Vérifie si une tâche peut s'exécuter avec les ressources actuelles."""
        # Vérifier CPU
        cpu_available = 100 - psutil.cpu_percent()
        if cpu_available < task.resource_limits.max_cpu_percent:
            return False
            
        # Vérifier mémoire
        memory = psutil.virtual_memory()
        memory_available_gb = memory.available / (1024**3)
        if memory_available_gb < task.resource_limits.max_memory_gb:
            return False
            
        # Vérifier GPU si nécessaire
        if task.resource_limits.max_gpu_memory_gb:
            gpus = GPUtil.getGPUs()
            if not gpus:
                return False
            gpu = gpus[0]
            if gpu.memoryFree < task.resource_limits.max_gpu_memory_gb:
                return False
                
        return True
        
    def _start_task(self, task: Task):
        """Démarre l'exécution d'une tâche."""
        task.start_time = datetime.now()
        task.status = "running"
        
        # Wrapper pour capturer les métriques
        def task_wrapper():
            try:
                # Configurer la priorité I/O
                if task.resource_limits.io_priority != 0:
                    import os
                    import psutil
                    process = psutil.Process()
                    if task.resource_limits.io_priority > 0:
                        os.nice(10)
                    else:
                        os.nice(-10)
                
                # Exécuter la tâche
                start_metrics = self._get_current_metrics()
                result = task.function(*task.args, **task.kwargs)
                end_metrics = self._get_current_metrics()
                
                # Enregistrer les métriques
                task.metrics = {
                    'start': start_metrics,
                    'end': end_metrics,
                    'duration': (datetime.now() - task.start_time).total_seconds()
                }
                
                # Mettre à jour le statut
                task.result = result
                task.status = "completed"
                task.end_time = datetime.now()
                
            except Exception as e:
                task.error = e
                task.status = "failed"
                task.end_time = datetime.now()
                logger.error(f"Erreur dans la tâche {task.id}: {str(e)}")
                
            finally:
                # Nettoyer
                with self.lock:
                    if task.id in self.running_tasks:
                        del self.running_tasks[task.id]
                
                # Traiter d'autres tâches
                self._process_queues()
        
        # Soumettre au pool
        self.executor.submit(task_wrapper)
        self.running_tasks[task.id] = task
        
    def _optimization_loop(self):
        """Boucle d'optimisation des ressources."""
        while self.running:
            try:
                # Collecter les métriques
                current_metrics = self._get_current_metrics()
                self.metrics_history.append(current_metrics)
                
                # Optimiser les paramètres
                self._optimize_parameters()
                
                # Attendre
                time.sleep(self.optimization_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle d'optimisation: {str(e)}")
                time.sleep(self.optimization_interval)
                
    def _get_current_metrics(self) -> Dict:
        """Récupère les métriques système actuelles."""
        cpu = psutil.cpu_percent(percpu=True)
        memory = psutil.virtual_memory()
        io = psutil.disk_io_counters()
        
        metrics = {
            'timestamp': datetime.now(),
            'cpu': {
                'total': np.mean(cpu),
                'per_cpu': cpu
            },
            'memory': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent
            },
            'io': {
                'read_bytes': io.read_bytes,
                'write_bytes': io.write_bytes
            },
            'tasks': {
                'running': len(self.running_tasks),
                'queued': sum(q.qsize() for q in self.queues.values())
            }
        }
        
        # Ajouter métriques GPU si disponible
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics['gpu'] = {
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'utilization': gpu.load * 100
                }
        except Exception:
            pass
            
        return metrics
        
    def _optimize_parameters(self):
        """Optimise les paramètres basés sur l'historique."""
        if len(self.metrics_history) < 10:
            return
            
        try:
            # Analyser l'historique récent
            recent_metrics = self.metrics_history[-10:]
            cpu_trend = [m['cpu']['total'] for m in recent_metrics]
            memory_trend = [m['memory']['percent'] for m in recent_metrics]
            
            # Ajuster le nombre de workers
            cpu_mean = np.mean(cpu_trend)
            if cpu_mean > 85:
                self.max_workers = max(1, self.max_workers - 1)
            elif cpu_mean < 50 and self.max_workers < psutil.cpu_count():
                self.max_workers += 1
                
            # Enregistrer l'optimisation
            self.optimization_history.append({
                'timestamp': datetime.now(),
                'metrics': {
                    'cpu_mean': cpu_mean,
                    'memory_mean': np.mean(memory_trend)
                },
                'changes': {
                    'max_workers': self.max_workers
                }
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            
    def shutdown(self):
        """Arrête le gestionnaire proprement."""
        self.running = False
        self.optimizer_thread.join()
        self.executor.shutdown(wait=True)
