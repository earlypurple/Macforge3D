"""
Module de monitoring et d'observabilité pour MacForge3D.
Fournit des métriques en temps réel sur les performances et l'état du système.
"""

import time
import psutil
import threading
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from collections import deque
import GPUtil

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """Métriques système instantanées."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used: float  # MB
    memory_available: float  # MB
    gpu_utilization: Optional[float]  # %
    gpu_memory_used: Optional[float]  # MB
    io_read_bytes: float  # MB/s
    io_write_bytes: float  # MB/s
    process_count: int

@dataclass
class SimulationMetrics:
    """Métriques spécifiques à une simulation."""
    simulation_id: str
    start_time: datetime
    status: str
    progress: float
    current_step: str
    memory_usage: float  # MB
    processing_time: float  # secondes
    vertices_processed: int
    error_count: int

@dataclass
class PerformanceAlert:
    """Alerte de performance."""
    timestamp: datetime
    severity: str  # 'info', 'warning', 'critical'
    message: str
    metrics: Dict
    context: Optional[Dict] = None

class MetricsBuffer:
    """Buffer circulaire pour stocker l'historique des métriques."""
    
    def __init__(self, max_size: int = 3600):  # 1 heure à 1 mesure/seconde
        self.system_metrics = deque(maxlen=max_size)
        self.simulation_metrics = deque(maxlen=max_size)
        self.alerts = deque(maxlen=1000)
        
    def add_system_metrics(self, metrics: SystemMetrics):
        self.system_metrics.append(metrics)
        
    def add_simulation_metrics(self, metrics: SimulationMetrics):
        self.simulation_metrics.append(metrics)
        
    def add_alert(self, alert: PerformanceAlert):
        self.alerts.append(alert)
        
    def get_system_metrics(self, seconds: int = 3600) -> List[SystemMetrics]:
        """Récupère les métriques système des dernières secondes."""
        now = datetime.now()
        return [
            m for m in self.system_metrics
            if (now - m.timestamp).total_seconds() <= seconds
        ]
        
    def get_simulation_metrics(
        self,
        simulation_id: Optional[str] = None
    ) -> List[SimulationMetrics]:
        """Récupère les métriques de simulation."""
        if simulation_id:
            return [
                m for m in self.simulation_metrics
                if m.simulation_id == simulation_id
            ]
        return list(self.simulation_metrics)
        
    def get_alerts(
        self,
        severity: Optional[str] = None,
        hours: int = 24
    ) -> List[PerformanceAlert]:
        """Récupère les alertes filtrées par sévérité et temps."""
        now = datetime.now()
        alerts = [
            a for a in self.alerts
            if (now - a.timestamp).total_seconds() <= hours * 3600
        ]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

class PerformanceMonitor:
    """Moniteur de performances temps réel."""
    
    def __init__(
        self,
        update_interval: float = 1.0,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        self.update_interval = update_interval
        self.alert_callbacks = alert_callbacks or []
        self.metrics_buffer = MetricsBuffer()
        self.running = False
        self.monitoring_thread = None
        
        # Seuils d'alerte
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_utilization': 95.0,
            'io_write_bytes': 100.0,  # MB/s
            'processing_time': 300.0  # secondes
        }
        
    def start(self):
        """Démarre le monitoring."""
        if self.running:
            return
            
        self.running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info("Monitoring démarré")
        
    def stop(self):
        """Arrête le monitoring."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
            logger.info("Monitoring arrêté")
            
    def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        last_io = psutil.disk_io_counters()
        last_time = time.time()
        
        while self.running:
            try:
                # Collecter les métriques système
                current_time = time.time()
                delta_time = current_time - last_time
                
                # CPU et mémoire
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # I/O
                io = psutil.disk_io_counters()
                io_read = (io.read_bytes - last_io.read_bytes) / delta_time / 1024 / 1024
                io_write = (io.write_bytes - last_io.write_bytes) / delta_time / 1024 / 1024
                
                # GPU si disponible
                gpu_metrics = self._get_gpu_metrics()
                
                # Créer les métriques
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    memory_used=memory.used / 1024 / 1024,
                    memory_available=memory.available / 1024 / 1024,
                    gpu_utilization=gpu_metrics.get('utilization'),
                    gpu_memory_used=gpu_metrics.get('memory_used'),
                    io_read_bytes=io_read,
                    io_write_bytes=io_write,
                    process_count=len(psutil.pids())
                )
                
                # Sauvegarder les métriques
                self.metrics_buffer.add_system_metrics(metrics)
                
                # Vérifier les seuils
                self._check_thresholds(metrics)
                
                # Mettre à jour les références
                last_io = io
                last_time = current_time
                
                # Attendre
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {str(e)}")
                time.sleep(self.update_interval)
                
    def _get_gpu_metrics(self) -> Dict:
        """Récupère les métriques GPU si disponible."""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Premier GPU
                return {
                    'utilization': gpu.load * 100,
                    'memory_used': gpu.memoryUsed
                }
        except Exception as e:
            logger.debug(f"Impossible d'obtenir les métriques GPU: {str(e)}")
        return {'utilization': None, 'memory_used': None}
        
    def _check_thresholds(self, metrics: SystemMetrics):
        """Vérifie les seuils et génère des alertes si nécessaire."""
        # CPU
        if metrics.cpu_percent > self.thresholds['cpu_percent']:
            self._create_alert(
                'warning',
                f"Utilisation CPU élevée: {metrics.cpu_percent:.1f}%",
                metrics
            )
            
        # Mémoire
        if metrics.memory_percent > self.thresholds['memory_percent']:
            self._create_alert(
                'critical',
                f"Mémoire critique: {metrics.memory_percent:.1f}%",
                metrics
            )
            
        # GPU
        if (metrics.gpu_utilization and 
            metrics.gpu_utilization > self.thresholds['gpu_utilization']):
            self._create_alert(
                'warning',
                f"Utilisation GPU élevée: {metrics.gpu_utilization:.1f}%",
                metrics
            )
            
        # I/O
        if metrics.io_write_bytes > self.thresholds['io_write_bytes']:
            self._create_alert(
                'info',
                f"I/O élevé: {metrics.io_write_bytes:.1f}MB/s",
                metrics
            )
            
    def _create_alert(
        self,
        severity: str,
        message: str,
        metrics: SystemMetrics,
        context: Optional[Dict] = None
    ):
        """Crée et enregistre une alerte."""
        alert = PerformanceAlert(
            timestamp=datetime.now(),
            severity=severity,
            message=message,
            metrics=metrics.__dict__,
            context=context
        )
        
        self.metrics_buffer.add_alert(alert)
        
        # Notifier les callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Erreur dans le callback d'alerte: {str(e)}")
                
    def add_simulation_metrics(self, metrics: SimulationMetrics):
        """Ajoute des métriques de simulation."""
        self.metrics_buffer.add_simulation_metrics(metrics)
        
        # Vérifier le temps de traitement
        if metrics.processing_time > self.thresholds['processing_time']:
            self._create_alert(
                'warning',
                f"Simulation longue: {metrics.processing_time:.1f}s",
                None,
                {'simulation_id': metrics.simulation_id}
            )
            
    def get_performance_report(self) -> Dict:
        """Génère un rapport de performance."""
        system_metrics = self.metrics_buffer.get_system_metrics(3600)
        
        if not system_metrics:
            return {}
            
        # Calculer les statistiques
        cpu_data = [m.cpu_percent for m in system_metrics]
        mem_data = [m.memory_percent for m in system_metrics]
        
        return {
            'period': '1h',
            'cpu': {
                'avg': np.mean(cpu_data),
                'max': np.max(cpu_data),
                'p95': np.percentile(cpu_data, 95),
                'min': np.min(cpu_data),
                'std': np.std(cpu_data)
            },
            'memory': {
                'avg': np.mean(mem_data),
                'max': np.max(mem_data),
                'p95': np.percentile(mem_data, 95),
                'min': np.min(mem_data),
                'std': np.std(mem_data)
            },
            'alerts': len(self.metrics_buffer.get_alerts(hours=1)),
            'simulations': len(self.metrics_buffer.get_simulation_metrics()),
            'performance_score': self._calculate_performance_score(cpu_data, mem_data)
        }
    
    def _calculate_performance_score(self, cpu_data: List[float], mem_data: List[float]) -> float:
        """
        Calcule un score de performance global (0-100).
        
        Args:
            cpu_data: Données d'utilisation CPU
            mem_data: Données d'utilisation mémoire
            
        Returns:
            Score de performance (100 = excellent, 0 = très mauvais)
        """
        try:
            if not cpu_data or not mem_data:
                return 50.0  # Score neutre si pas de données
            
            # Score CPU (plus c'est stable et modéré, mieux c'est)
            cpu_avg = np.mean(cpu_data)
            cpu_std = np.std(cpu_data)
            
            # Score optimal autour de 30-60% d'utilisation CPU
            if cpu_avg <= 30:
                cpu_score = 70 + (30 - cpu_avg)  # Bonus pour faible utilisation
            elif cpu_avg <= 60:
                cpu_score = 100 - (cpu_avg - 30) * 0.5  # Optimal
            elif cpu_avg <= 80:
                cpu_score = 85 - (cpu_avg - 60) * 2  # Dégradation modérée
            else:
                cpu_score = 45 - (cpu_avg - 80)  # Forte dégradation
            
            # Pénalité pour instabilité
            cpu_score -= cpu_std * 0.5
            
            # Score mémoire (similaire au CPU)
            mem_avg = np.mean(mem_data)
            mem_std = np.std(mem_data)
            
            if mem_avg <= 40:
                mem_score = 80 + (40 - mem_avg) * 0.5
            elif mem_avg <= 70:
                mem_score = 100 - (mem_avg - 40) * 0.67
            elif mem_avg <= 85:
                mem_score = 80 - (mem_avg - 70) * 2
            else:
                mem_score = 50 - (mem_avg - 85) * 2
            
            # Pénalité pour instabilité mémoire
            mem_score -= mem_std * 0.3
            
            # Score global pondéré
            performance_score = (cpu_score * 0.6 + mem_score * 0.4)
            
            # S'assurer que le score reste dans [0, 100]
            return max(0.0, min(100.0, performance_score))
            
        except Exception as e:
            logger.warning(f"Erreur lors du calcul du score de performance: {e}")
            return 50.0
    
    def get_optimization_suggestions(self) -> List[Dict[str, str]]:
        """
        Génère des suggestions d'optimisation basées sur les métriques actuelles.
        
        Returns:
            Liste de suggestions avec priorité et description
        """
        suggestions = []
        
        try:
            # Obtenir les métriques récentes
            recent_metrics = self.metrics_buffer.get_system_metrics(300)  # 5 dernières minutes
            
            if not recent_metrics:
                return [{"priority": "info", "suggestion": "Pas assez de données pour générer des suggestions"}]
            
            cpu_data = [m.cpu_percent for m in recent_metrics]
            mem_data = [m.memory_percent for m in recent_metrics]
            
            avg_cpu = np.mean(cpu_data)
            avg_mem = np.mean(mem_data)
            
            # Suggestions CPU
            if avg_cpu > 85:
                suggestions.append({
                    "priority": "high",
                    "category": "cpu",
                    "suggestion": f"Utilisation CPU très élevée ({avg_cpu:.1f}%). Réduisez le nombre de simulations parallèles ou fermez des applications."
                })
            elif avg_cpu > 70:
                suggestions.append({
                    "priority": "medium",
                    "category": "cpu", 
                    "suggestion": f"Utilisation CPU élevée ({avg_cpu:.1f}%). Surveillez les performances."
                })
            elif avg_cpu < 20:
                suggestions.append({
                    "priority": "low",
                    "category": "cpu",
                    "suggestion": f"Utilisation CPU faible ({avg_cpu:.1f}%). Vous pourriez augmenter la parallélisation."
                })
            
            # Suggestions mémoire
            if avg_mem > 90:
                suggestions.append({
                    "priority": "critical",
                    "category": "memory",
                    "suggestion": f"Mémoire critique ({avg_mem:.1f}%). Fermez des applications ou libérez de la mémoire immédiatement."
                })
            elif avg_mem > 80:
                suggestions.append({
                    "priority": "high",
                    "category": "memory",
                    "suggestion": f"Mémoire élevée ({avg_mem:.1f}%). Optimisez les paramètres ou réduisez la résolution des maillages."
                })
            elif avg_mem > 70:
                suggestions.append({
                    "priority": "medium",
                    "category": "memory",
                    "suggestion": f"Utilisation mémoire modérée ({avg_mem:.1f}%). Surveillez l'évolution."
                })
            
            # Suggestions de stabilité
            cpu_std = np.std(cpu_data)
            if cpu_std > 20:
                suggestions.append({
                    "priority": "medium",
                    "category": "stability",
                    "suggestion": f"Utilisation CPU instable (écart-type: {cpu_std:.1f}). Vérifiez la régularité des charges de travail."
                })
            
            # Suggestions générales de performance
            performance_score = self._calculate_performance_score(cpu_data, mem_data)
            if performance_score < 50:
                suggestions.append({
                    "priority": "high",
                    "category": "performance",
                    "suggestion": f"Score de performance faible ({performance_score:.1f}/100). Optimisation recommandée."
                })
            elif performance_score > 85:
                suggestions.append({
                    "priority": "info",
                    "category": "performance", 
                    "suggestion": f"Excellentes performances ({performance_score:.1f}/100). Système bien optimisé."
                })
            
            # Si aucune suggestion spécifique
            if not suggestions:
                suggestions.append({
                    "priority": "info",
                    "category": "general",
                    "suggestion": "Système fonctionne normalement. Aucune optimisation urgente nécessaire."
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de suggestions: {e}")
            return [{"priority": "error", "suggestion": f"Erreur lors de l'analyse: {str(e)}"}]
