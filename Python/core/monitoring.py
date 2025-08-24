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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Analyseur de tendances pour les métriques de performance."""
    
    def __init__(self):
        self.window_size = 20
        
    def analyze_trend(self, values: List[float]) -> Dict[str, Any]:
        """Analyse la tendance d'une série de valeurs."""
        if len(values) < 3:
            return {"trend": "insufficient_data"}
            
        # Calcul de la tendance linéaire
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        # Classification de la tendance
        if abs(slope) < 0.1:
            trend_type = "stable"
        elif slope > 0:
            trend_type = "increasing"
        else:
            trend_type = "decreasing"
            
        return {
            "trend": trend_type,
            "slope": slope,
            "strength": abs(slope),
            "r_squared": np.corrcoef(x, values)[0, 1] ** 2
        }

class PerformancePredictor:
    """Prédicteur de performance basé sur l'historique."""
    
    def __init__(self):
        self.min_data_points = 10
        
    def predict_next_values(self, values: List[float], steps_ahead: int = 5) -> List[float]:
        """Prédit les prochaines valeurs basées sur la tendance."""
        if len(values) < self.min_data_points:
            return [values[-1]] * steps_ahead if values else [0] * steps_ahead
            
        # Prédiction basée sur la régression linéaire
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)
        
        predictions = []
        for i in range(1, steps_ahead + 1):
            next_x = len(values) + i
            prediction = slope * next_x + intercept
            predictions.append(max(0, prediction))  # Éviter les valeurs négatives
            
        return predictions

class SmartAlertManager:
    """Gestionnaire d'alertes intelligent avec réduction de bruit."""
    
    def __init__(self):
        self.alert_history = []
        self.suppression_time = 300  # 5 minutes
        
    def should_alert(self, alert_type: str, severity: str) -> bool:
        """Détermine si une alerte doit être envoyée."""
        now = datetime.now()
        
        # Vérifier les alertes récentes du même type
        recent_alerts = [
            a for a in self.alert_history
            if a['type'] == alert_type and 
            (now - a['timestamp']).total_seconds() < self.suppression_time
        ]
        
        # Supprimer les alertes si trop récentes
        return len(recent_alerts) == 0
        
    def record_alert(self, alert_type: str, severity: str):
        """Enregistre une alerte envoyée."""
        self.alert_history.append({
            'type': alert_type,
            'severity': severity,
            'timestamp': datetime.now()
        })
        
        # Nettoyer l'historique ancien
        cutoff = datetime.now() - timedelta(hours=1)
        self.alert_history = [
            a for a in self.alert_history
            if a['timestamp'] > cutoff
        ]

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
    """Moniteur de performances temps réel avec capacités avancées."""
    
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
        
        # Seuils d'alerte adaptatifs
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'gpu_utilization': 95.0,
            'io_write_bytes': 100.0,  # MB/s
            'processing_time': 300.0  # secondes
        }
        
        # Métriques avancées
        self._advanced_metrics = {
            'cpu_load_1min': [],
            'cpu_load_5min': [],
            'cpu_freq': [],
            'memory_swap': [],
            'network_io': [],
            'process_metrics': {},
            'thermal_state': [],
            'power_consumption': []
        }
        
        # Prédictions et tendances
        self._trend_analyzer = TrendAnalyzer()
        self._performance_predictor = PerformancePredictor()
        
        # Système d'alerte intelligent
        self._alert_manager = SmartAlertManager()
        
    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques avancées du système."""
        try:
            advanced = {}
            
            # Métriques CPU avancées
            cpu_stats = psutil.cpu_stats()
            cpu_freq = psutil.cpu_freq()
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            advanced['cpu'] = {
                'cores_logical': psutil.cpu_count(logical=True),
                'cores_physical': psutil.cpu_count(logical=False),
                'frequency_current': cpu_freq.current if cpu_freq else 0,
                'frequency_max': cpu_freq.max if cpu_freq else 0,
                'load_avg_1min': load_avg[0],
                'load_avg_5min': load_avg[1],
                'load_avg_15min': load_avg[2],
                'context_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'soft_interrupts': cpu_stats.soft_interrupts
            }
            
            # Métriques mémoire avancées
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            advanced['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'free_gb': round(memory.free / (1024**3), 2),
                'cached_gb': round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else 0,
                'buffers_gb': round(memory.buffers / (1024**3), 2) if hasattr(memory, 'buffers') else 0,
                'swap_total_gb': round(swap.total / (1024**3), 2),
                'swap_used_gb': round(swap.used / (1024**3), 2),
                'swap_free_gb': round(swap.free / (1024**3), 2)
            }
            
            # Métriques réseau
            net_io = psutil.net_io_counters()
            if net_io:
                advanced['network'] = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout
                }
            
            # Métriques disque avancées
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            advanced['disk'] = {
                'total_gb': round(disk_usage.total / (1024**3), 2),
                'used_gb': round(disk_usage.used / (1024**3), 2),
                'free_gb': round(disk_usage.free / (1024**3), 2),
                'read_count': disk_io.read_count if disk_io else 0,
                'write_count': disk_io.write_count if disk_io else 0,
                'read_bytes': disk_io.read_bytes if disk_io else 0,
                'write_bytes': disk_io.write_bytes if disk_io else 0
            }
            
            # Métriques processus
            process_count = len(psutil.pids())
            running_processes = len([p for p in psutil.process_iter(['status']) 
                                   if p.info['status'] == psutil.STATUS_RUNNING])
            
            advanced['processes'] = {
                'total_count': process_count,
                'running_count': running_processes,
                'sleeping_count': process_count - running_processes
            }
            
            # Métriques thermiques (si disponible)
            try:
                sensors = psutil.sensors_temperatures()
                if sensors:
                    temps = []
                    for name, entries in sensors.items():
                        for entry in entries:
                            temps.append(entry.current)
                    if temps:
                        advanced['thermal'] = {
                            'avg_temp': round(np.mean(temps), 1),
                            'max_temp': round(max(temps), 1),
                            'min_temp': round(min(temps), 1)
                        }
            except:
                pass
            
            return advanced
            
        except Exception as e:
            logger.warning(f"Erreur lors de la collecte des métriques avancées: {e}")
            return {}
    
    def predict_performance_trend(self, minutes_ahead: int = 15) -> Dict[str, Any]:
        """Prédit les tendances de performance pour les prochaines minutes."""
        try:
            if len(self.metrics_buffer.system_metrics) < 10:
                return {"error": "Pas assez de données historiques"}
            
            recent_metrics = self.metrics_buffer.system_metrics[-30:]  # 30 dernières mesures
            
            # Extraire les séries temporelles
            cpu_series = [m.cpu_percent for m in recent_metrics]
            memory_series = [m.memory_percent for m in recent_metrics]
            
            # Prédiction simple basée sur la tendance linéaire
            predictions = {}
            
            # Prédiction CPU
            if len(cpu_series) >= 5:
                cpu_trend = np.polyfit(range(len(cpu_series)), cpu_series, 1)[0]
                cpu_prediction = cpu_series[-1] + (cpu_trend * minutes_ahead)
                predictions['cpu'] = {
                    'current': round(cpu_series[-1], 1),
                    'predicted': round(max(0, min(100, cpu_prediction)), 1),
                    'trend': 'increasing' if cpu_trend > 0.1 else 'decreasing' if cpu_trend < -0.1 else 'stable',
                    'confidence': min(100, len(cpu_series) * 10)  # Confiance basée sur l'historique
                }
            
            # Prédiction mémoire
            if len(memory_series) >= 5:
                memory_trend = np.polyfit(range(len(memory_series)), memory_series, 1)[0]
                memory_prediction = memory_series[-1] + (memory_trend * minutes_ahead)
                predictions['memory'] = {
                    'current': round(memory_series[-1], 1),
                    'predicted': round(max(0, min(100, memory_prediction)), 1),
                    'trend': 'increasing' if memory_trend > 0.1 else 'decreasing' if memory_trend < -0.1 else 'stable',
                    'confidence': min(100, len(memory_series) * 10)
                }
            
            # Alertes prédictives
            alerts = []
            if 'cpu' in predictions and predictions['cpu']['predicted'] > 85:
                alerts.append({
                    'type': 'cpu_overload_predicted',
                    'severity': 'warning',
                    'message': f"Surcharge CPU prédite dans {minutes_ahead} minutes: {predictions['cpu']['predicted']}%"
                })
            
            if 'memory' in predictions and predictions['memory']['predicted'] > 90:
                alerts.append({
                    'type': 'memory_exhaustion_predicted',
                    'severity': 'critical',
                    'message': f"Épuisement mémoire prédit dans {minutes_ahead} minutes: {predictions['memory']['predicted']}%"
                })
            
            predictions['alerts'] = alerts
            predictions['prediction_time'] = datetime.now().isoformat()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return {"error": str(e)}
        
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

class SystemMonitor:
    """Simple system monitor for diagnostics."""
    
    def __init__(self):
        self.monitoring = False
        self.samples = []
        
    def monitor_duration(self, duration_seconds: int) -> Dict[str, Any]:
        """Monitor system for specified duration."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        samples = []
        
        while time.time() < end_time:
            try:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                sample = {
                    'timestamp': time.time(),
                    'cpu_usage': cpu_percent,
                    'memory_usage': memory.percent,
                    'memory_available': memory.available / (1024**3)  # GB
                }
                
                samples.append(sample)
                
            except Exception as e:
                logger.warning(f"Error collecting metrics: {e}")
                
        return {
            'duration': duration_seconds,
            'samples_collected': len(samples),
            'avg_cpu': np.mean([s['cpu_usage'] for s in samples]) if samples else 0,
            'avg_memory': np.mean([s['memory_usage'] for s in samples]) if samples else 0,
            'samples': samples
        }
