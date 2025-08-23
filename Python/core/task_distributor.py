"""
Système de distribution des tâches pour MacForge3D.
Permet l'exécution distribuée des simulations sur plusieurs machines.
"""

import asyncio
import aiohttp
import json
import logging
import socket
import time
import uuid
import zmq
import zmq.asyncio
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WorkerInfo:
    """Information sur un worker distant."""
    id: str
    hostname: str
    address: str
    port: int
    capabilities: Dict[str, Any]
    status: str = "idle"
    last_heartbeat: datetime = field(default_factory=datetime.now)
    current_tasks: Set[str] = field(default_factory=set)
    total_memory: float = 0.0  # GB
    available_memory: float = 0.0  # GB
    cpu_count: int = 0
    cpu_percent: float = 0.0
    gpu_available: bool = False
    gpu_memory: float = 0.0  # GB

@dataclass
class DistributedTask:
    """Tâche distribuée."""
    id: str
    name: str
    worker_id: Optional[str]
    status: str
    parameters: Dict
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    result: Any = None
    error: Optional[str] = None

class TaskDistributor:
    """Gestionnaire de distribution des tâches."""
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5555,
        discovery_port: int = 5556
    ):
        self.host = host
        self.port = port
        self.discovery_port = discovery_port
        
        # État
        self.workers: Dict[str, WorkerInfo] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.running = True
        
        # ZMQ context
        self.context = zmq.asyncio.Context()
        
        # Verrous
        self.workers_lock = threading.Lock()
        self.tasks_lock = threading.Lock()
        
        # Thread pool pour les tâches async
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Démarrer les services
        self.start_services()
        
    def start_services(self):
        """Démarre les services de distribution."""
        loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._start_all_services())
            loop.run_forever()
            
        self.service_thread = threading.Thread(target=run_loop)
        self.service_thread.daemon = True
        self.service_thread.start()
        
    async def _start_all_services(self):
        """Démarre tous les services async."""
        await asyncio.gather(
            self._start_task_server(),
            self._start_discovery_service(),
            self._start_heartbeat_monitor()
        )
        
    async def _start_task_server(self):
        """Démarre le serveur de tâches."""
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.port}")
        
        while self.running:
            try:
                message = await socket.recv_json()
                response = await self._handle_task_message(message)
                await socket.send_json(response)
            except Exception as e:
                logger.error(f"Erreur serveur tâches: {str(e)}")
                
    async def _start_discovery_service(self):
        """Démarre le service de découverte des workers."""
        socket = self.context.socket(zmq.REP)
        socket.bind(f"tcp://{self.host}:{self.discovery_port}")
        
        while self.running:
            try:
                message = await socket.recv_json()
                response = await self._handle_discovery_message(message)
                await socket.send_json(response)
            except Exception as e:
                logger.error(f"Erreur service découverte: {str(e)}")
                
    async def _start_heartbeat_monitor(self):
        """Surveille les heartbeats des workers."""
        while self.running:
            try:
                self._check_workers_health()
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Erreur moniteur heartbeat: {str(e)}")
                
    async def _handle_task_message(self, message: Dict) -> Dict:
        """Gère les messages liés aux tâches."""
        try:
            action = message.get('action')
            
            if action == 'submit':
                return await self._handle_task_submission(message)
            elif action == 'status':
                return await self._handle_task_status(message)
            elif action == 'result':
                return await self._handle_task_result(message)
            else:
                return {'status': 'error', 'message': 'Action inconnue'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_discovery_message(self, message: Dict) -> Dict:
        """Gère les messages de découverte."""
        try:
            action = message.get('action')
            
            if action == 'register':
                return await self._handle_worker_registration(message)
            elif action == 'heartbeat':
                return await self._handle_worker_heartbeat(message)
            elif action == 'capabilities':
                return await self._handle_worker_capabilities(message)
            else:
                return {'status': 'error', 'message': 'Action inconnue'}
                
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    async def _handle_task_submission(self, message: Dict) -> Dict:
        """Gère la soumission d'une nouvelle tâche."""
        task_id = str(uuid.uuid4())
        task = DistributedTask(
            id=task_id,
            name=message.get('name', 'unknown'),
            worker_id=None,
            status='pending',
            parameters=message.get('parameters', {}),
            start_time=None,
            end_time=None
        )
        
        with self.tasks_lock:
            self.tasks[task_id] = task
            
        # Assigner la tâche
        worker_id = await self._assign_task(task)
        if worker_id:
            return {
                'status': 'success',
                'task_id': task_id,
                'worker_id': worker_id
            }
        else:
            return {
                'status': 'error',
                'message': 'Aucun worker disponible'
            }
            
    async def _handle_task_status(self, message: Dict) -> Dict:
        """Gère les demandes de status de tâche."""
        task_id = message.get('task_id')
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            
        if task:
            return {
                'status': 'success',
                'task': {
                    'id': task.id,
                    'status': task.status,
                    'worker_id': task.worker_id,
                    'error': task.error
                }
            }
        else:
            return {'status': 'error', 'message': 'Tâche non trouvée'}
            
    async def _handle_task_result(self, message: Dict) -> Dict:
        """Gère les résultats de tâche."""
        task_id = message.get('task_id')
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            
            if task:
                task.status = message.get('status', 'completed')
                task.result = message.get('result')
                task.error = message.get('error')
                task.end_time = datetime.now()
                
                # Libérer le worker
                if task.worker_id:
                    with self.workers_lock:
                        worker = self.workers.get(task.worker_id)
                        if worker:
                            worker.current_tasks.remove(task_id)
                            worker.status = 'idle'
                            
                return {'status': 'success'}
            else:
                return {'status': 'error', 'message': 'Tâche non trouvée'}
                
    async def _handle_worker_registration(self, message: Dict) -> Dict:
        """Gère l'enregistrement d'un nouveau worker."""
        worker_id = message.get('worker_id', str(uuid.uuid4()))
        worker = WorkerInfo(
            id=worker_id,
            hostname=message.get('hostname', 'unknown'),
            address=message.get('address'),
            port=message.get('port'),
            capabilities=message.get('capabilities', {}),
            total_memory=message.get('total_memory', 0.0),
            cpu_count=message.get('cpu_count', 0),
            gpu_available=message.get('gpu_available', False),
            gpu_memory=message.get('gpu_memory', 0.0)
        )
        
        with self.workers_lock:
            self.workers[worker_id] = worker
            
        return {
            'status': 'success',
            'worker_id': worker_id,
            'message': 'Enregistrement réussi'
        }
        
    async def _handle_worker_heartbeat(self, message: Dict) -> Dict:
        """Gère les heartbeats des workers."""
        worker_id = message.get('worker_id')
        with self.workers_lock:
            worker = self.workers.get(worker_id)
            
            if worker:
                worker.last_heartbeat = datetime.now()
                worker.available_memory = message.get('available_memory', 0.0)
                worker.cpu_percent = message.get('cpu_percent', 0.0)
                return {'status': 'success'}
            else:
                return {'status': 'error', 'message': 'Worker inconnu'}
                
    async def _handle_worker_capabilities(self, message: Dict) -> Dict:
        """Gère la mise à jour des capacités d'un worker."""
        worker_id = message.get('worker_id')
        with self.workers_lock:
            worker = self.workers.get(worker_id)
            
            if worker:
                worker.capabilities = message.get('capabilities', {})
                return {'status': 'success'}
            else:
                return {'status': 'error', 'message': 'Worker inconnu'}
                
    def _check_workers_health(self):
        """Vérifie la santé des workers."""
        now = datetime.now()
        with self.workers_lock:
            dead_workers = []
            
            for worker_id, worker in self.workers.items():
                if (now - worker.last_heartbeat).total_seconds() > 30:
                    dead_workers.append(worker_id)
                    
            # Retirer les workers morts
            for worker_id in dead_workers:
                worker = self.workers.pop(worker_id)
                logger.warning(f"Worker {worker_id} ne répond plus")
                
                # Réassigner les tâches
                self._reassign_tasks(worker_id)
                
    def _reassign_tasks(self, worker_id: str):
        """Réassigne les tâches d'un worker mort."""
        with self.tasks_lock:
            for task in self.tasks.values():
                if task.worker_id == worker_id and task.status == 'running':
                    task.status = 'pending'
                    task.worker_id = None
                    asyncio.create_task(self._assign_task(task))
                    
    async def _assign_task(self, task: DistributedTask) -> Optional[str]:
        """Assigne une tâche à un worker."""
        with self.workers_lock:
            # Trouver le meilleur worker
            best_worker = None
            min_load = float('inf')
            
            for worker in self.workers.values():
                if worker.status == 'idle':
                    # Vérifier les capacités
                    if not self._check_worker_capabilities(worker, task):
                        continue
                        
                    # Calculer la charge
                    load = len(worker.current_tasks) + worker.cpu_percent / 100
                    
                    if load < min_load:
                        best_worker = worker
                        min_load = load
                        
            if best_worker:
                # Assigner la tâche
                task.worker_id = best_worker.id
                task.status = 'assigned'
                best_worker.current_tasks.add(task.id)
                best_worker.status = 'busy'
                
                # Envoyer la tâche au worker
                try:
                    await self._send_task_to_worker(task, best_worker)
                    return best_worker.id
                except Exception as e:
                    logger.error(f"Erreur d'envoi de tâche: {str(e)}")
                    task.worker_id = None
                    task.status = 'pending'
                    best_worker.current_tasks.remove(task.id)
                    best_worker.status = 'idle'
                    return None
                    
        return None
        
    def _check_worker_capabilities(
        self,
        worker: WorkerInfo,
        task: DistributedTask
    ) -> bool:
        """Vérifie si un worker peut exécuter une tâche."""
        params = task.parameters
        
        # Vérifier la mémoire
        if params.get('memory_required', 0) > worker.available_memory:
            return False
            
        # Vérifier le GPU
        if params.get('gpu_required', False) and not worker.gpu_available:
            return False
            
        # Vérifier les capacités spécifiques
        required_caps = params.get('required_capabilities', {})
        for cap, value in required_caps.items():
            if cap not in worker.capabilities:
                return False
            if worker.capabilities[cap] < value:
                return False
                
        return True
        
    async def _send_task_to_worker(
        self,
        task: DistributedTask,
        worker: WorkerInfo
    ):
        """Envoie une tâche à un worker."""
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{worker.address}:{worker.port}")
        
        try:
            await socket.send_json({
                'action': 'execute',
                'task_id': task.id,
                'name': task.name,
                'parameters': task.parameters
            })
            
            response = await socket.recv_json()
            if response.get('status') != 'success':
                raise Exception(response.get('message', 'Erreur inconnue'))
                
        finally:
            socket.close()
            
    def shutdown(self):
        """Arrête proprement le distributeur."""
        self.running = False
        
        # Attendre la fin des services
        if hasattr(self, 'service_thread'):
            self.service_thread.join()
            
        # Fermer le context ZMQ
        self.context.term()
