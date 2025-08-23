"""
Système de diagnostics avancé pour MacForge3D.
Fournit un monitoring intelligent et des outils de débogage.
"""

import os
import sys
import time
import json
import logging
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
import queue
import weakref

import psutil
import numpy as np

# Configuration du logging principal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticEvent:
    """Événement de diagnostic."""
    timestamp: datetime
    event_type: str
    component: str
    severity: str
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    gpu_memory: Optional[int] = None

@dataclass
class PerformanceMetrics:
    """Métriques de performance pour une opération."""
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time: Optional[float] = None
    cpu_usage_percent: float = 0.0
    memory_peak_mb: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    error_count: int = 0
    warning_count: int = 0

class RealTimeMonitor:
    """Moniteur temps réel pour les performances système et application."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_running = False
        self.monitor_thread = None
        self.callbacks = []
        self.metrics_history = []
        self.max_history_size = 1000
        
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Ajoute un callback pour recevoir les métriques en temps réel."""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """Démarre le monitoring en temps réel."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Monitoring temps réel démarré")
    
    def stop_monitoring(self):
        """Arrête le monitoring."""
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
        logger.info("Monitoring temps réel arrêté")
    
    def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        while self.is_running:
            try:
                metrics = self._collect_system_metrics()
                
                # Conserver l'historique
                self.metrics_history.append(metrics)
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history.pop(0)
                
                # Notifier les callbacks
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.warning(f"Erreur dans callback de monitoring: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de monitoring: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collecte les métriques système actuelles."""
        try:
            # Métriques CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Métriques mémoire
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # Métriques GPU si disponible
            gpu_memory_mb = 0
            gpu_utilization = 0
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024**2)
                    # Note: GPU utilization nécessite des outils NVIDIA spécifiques
            except ImportError:
                pass
            
            # Métriques disque
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            return {
                'timestamp': datetime.now(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
                },
                'memory': {
                    'percent': memory_percent,
                    'used_gb': memory_used_gb,
                    'total_gb': memory_total_gb,
                    'available_gb': memory.available / (1024**3)
                },
                'gpu': {
                    'memory_mb': gpu_memory_mb,
                    'utilization_percent': gpu_utilization
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_usage.free / (1024**3)
                }
            }
        except Exception as e:
            logger.warning(f"Erreur lors de la collecte des métriques: {e}")
            return {'timestamp': datetime.now(), 'error': str(e)}
    
    def get_metrics_summary(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """Retourne un résumé des métriques sur une période donnée."""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)
        recent_metrics = [m for m in self.metrics_history 
                         if m.get('timestamp', datetime.min) > cutoff_time]
        
        if not recent_metrics:
            return {'message': 'Aucune donnée récente disponible'}
        
        # Calculer les statistiques
        cpu_values = [m.get('cpu', {}).get('percent', 0) for m in recent_metrics]
        memory_values = [m.get('memory', {}).get('percent', 0) for m in recent_metrics]
        
        return {
            'period_minutes': duration_minutes,
            'sample_count': len(recent_metrics),
            'cpu': {
                'avg_percent': np.mean(cpu_values) if cpu_values else 0,
                'max_percent': np.max(cpu_values) if cpu_values else 0,
                'min_percent': np.min(cpu_values) if cpu_values else 0
            },
            'memory': {
                'avg_percent': np.mean(memory_values) if memory_values else 0,
                'max_percent': np.max(memory_values) if memory_values else 0,
                'min_percent': np.min(memory_values) if memory_values else 0
            },
            'alerts': self._generate_alerts(recent_metrics)
        }
    
    def _generate_alerts(self, metrics: List[Dict[str, Any]]) -> List[str]:
        """Génère des alertes basées sur les métriques."""
        alerts = []
        
        if metrics:
            avg_cpu = np.mean([m.get('cpu', {}).get('percent', 0) for m in metrics])
            avg_memory = np.mean([m.get('memory', {}).get('percent', 0) for m in metrics])
            
            if avg_cpu > 90:
                alerts.append(f"Usage CPU élevé: {avg_cpu:.1f}%")
            if avg_memory > 85:
                alerts.append(f"Usage mémoire élevé: {avg_memory:.1f}%")
            
            # Vérifier la disponibilité disque
            latest_disk = metrics[-1].get('disk', {}).get('percent', 0)
            if latest_disk > 90:
                alerts.append(f"Espace disque faible: {latest_disk:.1f}% utilisé")
        
        return alerts

class SmartLogger:
    """Logger intelligent avec classification automatique et suggestions."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(f"macforge3d.{component_name}")
        self.events = []
        self.performance_metrics = {}
        self.error_patterns = {}
        
    def log_operation_start(self, operation_name: str, **kwargs) -> str:
        """Démarre le logging d'une opération."""
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.now()
        )
        
        self.performance_metrics[operation_id] = metrics
        
        self.logger.info(f"Démarrage opération: {operation_name}", extra={
            'operation_id': operation_id,
            'component': self.component_name,
            **kwargs
        })
        
        return operation_id
    
    def log_operation_end(self, operation_id: str, success: bool = True, **kwargs):
        """Termine le logging d'une opération."""
        if operation_id not in self.performance_metrics:
            self.logger.warning(f"Opération inconnue: {operation_id}")
            return
        
        metrics = self.performance_metrics[operation_id]
        metrics.end_time = datetime.now()
        metrics.execution_time = (metrics.end_time - metrics.start_time).total_seconds()
        
        # Collecter les métriques système finales
        try:
            memory_info = psutil.virtual_memory()
            metrics.memory_peak_mb = memory_info.used / (1024**2)
        except:
            pass
        
        level = logging.INFO if success else logging.ERROR
        status = "réussie" if success else "échouée"
        
        self.logger.log(level, f"Opération {status}: {metrics.operation_name} "
                              f"({metrics.execution_time:.2f}s)", extra={
            'operation_id': operation_id,
            'execution_time': metrics.execution_time,
            'success': success,
            **kwargs
        })
    
    def log_with_context(self, level: int, message: str, context: Dict[str, Any] = None):
        """Log avec contexte enrichi."""
        context = context or {}
        
        # Enrichir avec des informations système si pertinent
        if level >= logging.WARNING:
            try:
                context.update({
                    'memory_usage_percent': psutil.virtual_memory().percent,
                    'cpu_usage_percent': psutil.cpu_percent(interval=None)
                })
            except:
                pass
        
        # Analyser le message pour détecter des patterns
        self._analyze_error_pattern(message, level)
        
        self.logger.log(level, message, extra={
            'component': self.component_name,
            'timestamp': datetime.now().isoformat(),
            **context
        })
        
        # Enregistrer l'événement pour analyse
        event = DiagnosticEvent(
            timestamp=datetime.now(),
            event_type='log',
            component=self.component_name,
            severity=logging.getLevelName(level),
            message=message,
            metadata=context
        )
        self.events.append(event)
    
    def _analyze_error_pattern(self, message: str, level: int):
        """Analyse les patterns d'erreurs pour identifier les problèmes récurrents."""
        if level < logging.WARNING:
            return
        
        # Extraire les mots-clés du message
        keywords = set(word.lower() for word in message.split() 
                      if len(word) > 3 and word.isalnum())
        
        for keyword in keywords:
            if keyword not in self.error_patterns:
                self.error_patterns[keyword] = 0
            self.error_patterns[keyword] += 1
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des erreurs et suggestions."""
        # Trier les patterns d'erreurs par fréquence
        sorted_patterns = sorted(self.error_patterns.items(), 
                               key=lambda x: x[1], reverse=True)
        
        suggestions = []
        
        # Générer des suggestions basées sur les patterns
        for pattern, count in sorted_patterns[:5]:  # Top 5
            if count >= 3:  # Si répété au moins 3 fois
                suggestion = self._generate_suggestion_for_pattern(pattern, count)
                if suggestion:
                    suggestions.append(suggestion)
        
        return {
            'component': self.component_name,
            'total_events': len(self.events),
            'error_patterns': dict(sorted_patterns[:10]),
            'suggestions': suggestions,
            'performance_summary': self._get_performance_summary()
        }
    
    def _generate_suggestion_for_pattern(self, pattern: str, count: int) -> Optional[str]:
        """Génère une suggestion basée sur un pattern d'erreur."""
        suggestions_map = {
            'memory': f"Erreurs mémoire fréquentes ({count}x) - considérer l'optimisation mémoire",
            'timeout': f"Timeouts récurrents ({count}x) - augmenter les délais ou optimiser",
            'gpu': f"Problèmes GPU ({count}x) - vérifier les drivers et la mémoire GPU",
            'file': f"Erreurs de fichier ({count}x) - vérifier les permissions et chemins",
            'network': f"Problèmes réseau ({count}x) - vérifier la connectivité",
            'validation': f"Erreurs de validation ({count}x) - revoir les paramètres d'entrée"
        }
        
        return suggestions_map.get(pattern)
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des performances."""
        if not self.performance_metrics:
            return {'message': 'Aucune métrique de performance disponible'}
        
        completed_ops = [m for m in self.performance_metrics.values() 
                        if m.execution_time is not None]
        
        if not completed_ops:
            return {'message': 'Aucune opération terminée'}
        
        execution_times = [op.execution_time for op in completed_ops]
        
        return {
            'total_operations': len(completed_ops),
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'min_execution_time': np.min(execution_times),
            'slowest_operations': self._get_slowest_operations(completed_ops, 3)
        }
    
    def _get_slowest_operations(self, operations: List[PerformanceMetrics], limit: int) -> List[Dict[str, Any]]:
        """Retourne les opérations les plus lentes."""
        sorted_ops = sorted(operations, key=lambda x: x.execution_time, reverse=True)
        
        return [{
            'name': op.operation_name,
            'execution_time': op.execution_time,
            'memory_peak_mb': op.memory_peak_mb
        } for op in sorted_ops[:limit]]

class DiagnosticCenter:
    """Centre de diagnostic central pour MacForge3D."""
    
    def __init__(self):
        self.loggers = {}
        self.monitor = RealTimeMonitor()
        self.global_events = []
        self.diagnostic_reports = []
        
        # Configurer les callbacks de monitoring
        self.monitor.add_callback(self._on_system_metrics)
    
    def get_logger(self, component_name: str) -> SmartLogger:
        """Récupère ou crée un logger pour un composant."""
        if component_name not in self.loggers:
            self.loggers[component_name] = SmartLogger(component_name)
        return self.loggers[component_name]
    
    def start_monitoring(self):
        """Démarre le monitoring global."""
        self.monitor.start_monitoring()
    
    def stop_monitoring(self):
        """Arrête le monitoring global."""
        self.monitor.stop_monitoring()
    
    def _on_system_metrics(self, metrics: Dict[str, Any]):
        """Callback pour traiter les métriques système."""
        # Détecter les anomalies
        alerts = self._detect_anomalies(metrics)
        
        if alerts:
            for alert in alerts:
                event = DiagnosticEvent(
                    timestamp=datetime.now(),
                    event_type='system_alert',
                    component='system',
                    severity='WARNING',
                    message=alert,
                    metadata=metrics
                )
                self.global_events.append(event)
    
    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[str]:
        """Détecte les anomalies dans les métriques système."""
        alerts = []
        
        cpu_percent = metrics.get('cpu', {}).get('percent', 0)
        memory_percent = metrics.get('memory', {}).get('percent', 0)
        
        if cpu_percent > 95:
            alerts.append(f"CPU critique: {cpu_percent:.1f}%")
        
        if memory_percent > 90:
            alerts.append(f"Mémoire critique: {memory_percent:.1f}%")
        
        return alerts
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Génère un rapport de diagnostic complet."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.monitor.get_metrics_summary(),
            'component_summaries': {},
            'global_recommendations': [],
            'health_score': 0
        }
        
        # Résumés par composant
        total_score = 0
        component_count = 0
        
        for component_name, logger in self.loggers.items():
            summary = logger.get_error_summary()
            report['component_summaries'][component_name] = summary
            
            # Calculer un score de santé pour le composant
            component_score = self._calculate_component_health_score(summary)
            total_score += component_score
            component_count += 1
        
        # Score de santé global
        if component_count > 0:
            report['health_score'] = total_score / component_count
        
        # Recommandations globales
        report['global_recommendations'] = self._generate_global_recommendations(report)
        
        # Sauvegarder le rapport
        self.diagnostic_reports.append(report)
        
        return report
    
    def _calculate_component_health_score(self, summary: Dict[str, Any]) -> float:
        """Calcule un score de santé pour un composant (0-100)."""
        base_score = 100.0
        
        # Pénalités pour les erreurs
        error_patterns = summary.get('error_patterns', {})
        total_errors = sum(error_patterns.values())
        
        if total_errors > 0:
            base_score -= min(50, total_errors * 5)  # Max 50 points de pénalité
        
        # Pénalités pour les performances
        perf_summary = summary.get('performance_summary', {})
        if 'avg_execution_time' in perf_summary:
            avg_time = perf_summary['avg_execution_time']
            if avg_time > 10:  # Plus de 10 secondes en moyenne
                base_score -= min(25, (avg_time - 10) * 2)
        
        return max(0, base_score)
    
    def _generate_global_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Génère des recommandations globales basées sur le rapport."""
        recommendations = []
        
        health_score = report.get('health_score', 100)
        system_status = report.get('system_status', {})
        
        if health_score < 70:
            recommendations.append("Score de santé faible - investigation nécessaire")
        
        # Analyser les alertes système
        alerts = system_status.get('alerts', [])
        if alerts:
            recommendations.append("Problèmes système détectés - optimisation recommandée")
        
        # Analyser les patterns d'erreurs communs
        all_patterns = {}
        for summary in report['component_summaries'].values():
            for pattern, count in summary.get('error_patterns', {}).items():
                all_patterns[pattern] = all_patterns.get(pattern, 0) + count
        
        if all_patterns:
            most_common = max(all_patterns.items(), key=lambda x: x[1])
            if most_common[1] >= 5:
                recommendations.append(f"Pattern d'erreur récurrent: {most_common[0]} - action requise")
        
        if not recommendations:
            recommendations.append("Système en bon état de fonctionnement")
        
        return recommendations

# Instance globale du centre de diagnostic
diagnostic_center = DiagnosticCenter()

# Décorateur pour le logging automatique des fonctions
def log_operation(component_name: str, operation_name: str = None):
    """Décorateur pour logger automatiquement les opérations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            logger = diagnostic_center.get_logger(component_name)
            
            operation_id = logger.log_operation_start(op_name, 
                function=func.__name__, 
                args_count=len(args),
                kwargs_keys=list(kwargs.keys())
            )
            
            try:
                result = func(*args, **kwargs)
                logger.log_operation_end(operation_id, success=True)
                return result
            except Exception as e:
                logger.log_operation_end(operation_id, success=False, error=str(e))
                logger.log_with_context(logging.ERROR, f"Erreur dans {op_name}: {e}", {
                    'exception_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                })
                raise
        
        return wrapper
    return decorator

# Context manager pour le monitoring d'opérations
@contextmanager
def monitor_operation(component_name: str, operation_name: str, **context):
    """Context manager pour monitorer une opération."""
    logger = diagnostic_center.get_logger(component_name)
    operation_id = logger.log_operation_start(operation_name, **context)
    
    try:
        yield logger
        logger.log_operation_end(operation_id, success=True)
    except Exception as e:
        logger.log_operation_end(operation_id, success=False, error=str(e))
        logger.log_with_context(logging.ERROR, f"Erreur dans {operation_name}: {e}")
        raise

def start_global_monitoring():
    """Démarre le monitoring global de MacForge3D."""
    diagnostic_center.start_monitoring()
    logger.info("Système de diagnostic MacForge3D démarré")

def stop_global_monitoring():
    """Arrête le monitoring global."""
    diagnostic_center.stop_monitoring()
    logger.info("Système de diagnostic MacForge3D arrêté")

def generate_health_report() -> Dict[str, Any]:
    """Génère un rapport de santé complet du système."""
    return diagnostic_center.generate_comprehensive_report()