"""
Système de logging et monitoring avancé pour MacForge3D.
"""

import os
import time
import json
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import psutil
import traceback
import functools
import weakref

# Configuration des niveaux de log personnalisés
logging.addLevelName(25, "SUCCESS")
logging.addLevelName(35, "PERFORMANCE")

def success(self, message, *args, **kwargs):
    if self.isEnabledFor(25):
        self._log(25, message, args, **kwargs)

def performance(self, message, *args, **kwargs):
    if self.isEnabledFor(35):
        self._log(35, message, args, **kwargs)

logging.Logger.success = success
logging.Logger.performance = performance

@dataclass
class PerformanceMetric:
    """Métrique de performance."""
    name: str
    value: float
    unit: str
    timestamp: float
    category: str = "general"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ErrorEvent:
    """Événement d'erreur."""
    timestamp: float
    level: str
    message: str
    module: str
    function: str
    traceback_info: Optional[str] = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.context is None:
            self.context = {}

class PerformanceMonitor:
    """Moniteur de performance en temps réel."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self._start_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Métriques système
        self._system_metrics_enabled = True
        self._system_monitor_thread = None
        self._stop_system_monitoring = threading.Event()
        
        self.start_system_monitoring()
    
    def start_timing(self, operation: str) -> str:
        """Démarre le chronométrage d'une opération."""
        timing_id = f"{operation}_{time.time()}"
        with self._lock:
            self._start_times[timing_id] = time.time()
        return timing_id
    
    def end_timing(self, timing_id: str, category: str = "timing") -> Optional[float]:
        """Termine le chronométrage et enregistre la métrique."""
        with self._lock:
            if timing_id in self._start_times:
                duration = time.time() - self._start_times[timing_id]
                del self._start_times[timing_id]
                
                operation = timing_id.split('_')[0]
                self.record_metric(f"{operation}_duration", duration, "seconds", category)
                return duration
        return None
    
    def record_metric(
        self, 
        name: str, 
        value: float, 
        unit: str = "", 
        category: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Enregistre une métrique."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            category=category,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._metrics[name].append(metric)
    
    def get_metrics(
        self, 
        name: Optional[str] = None,
        category: Optional[str] = None,
        since_seconds: Optional[int] = None
    ) -> List[PerformanceMetric]:
        """Récupère les métriques selon les critères."""
        now = time.time()
        cutoff = now - since_seconds if since_seconds else 0
        
        with self._lock:
            if name:
                metrics = list(self._metrics.get(name, []))
            else:
                metrics = []
                for metric_list in self._metrics.values():
                    metrics.extend(metric_list)
            
            # Filtrer par catégorie
            if category:
                metrics = [m for m in metrics if m.category == category]
            
            # Filtrer par temps
            if since_seconds:
                metrics = [m for m in metrics if m.timestamp >= cutoff]
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_statistics(
        self, 
        name: str, 
        since_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """Calcule les statistiques pour une métrique."""
        metrics = self.get_metrics(name, since_seconds=since_seconds)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "last": values[-1] if values else 0,
            "rate_per_minute": len(values) / ((since_seconds or 60) / 60) if since_seconds else 0
        }
    
    def start_system_monitoring(self, interval_seconds: int = 30):
        """Démarre le monitoring système en arrière-plan."""
        if self._system_monitor_thread and self._system_monitor_thread.is_alive():
            return
        
        self._stop_system_monitoring.clear()
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._system_monitor_thread.start()
    
    def stop_system_monitoring(self):
        """Arrête le monitoring système."""
        self._stop_system_monitoring.set()
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5)
    
    def _system_monitoring_loop(self, interval: int):
        """Boucle de monitoring système."""
        while not self._stop_system_monitoring.wait(interval):
            try:
                # Métriques CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_metric("system_cpu_percent", cpu_percent, "%", "system")
                
                # Métriques mémoire
                memory = psutil.virtual_memory()
                self.record_metric("system_memory_percent", memory.percent, "%", "system")
                self.record_metric("system_memory_available_gb", memory.available / (1024**3), "GB", "system")
                
                # Métriques disque
                disk = psutil.disk_usage('/')
                self.record_metric("system_disk_percent", (disk.used / disk.total) * 100, "%", "system")
                
                # Processus Python actuel
                process = psutil.Process()
                self.record_metric("process_memory_mb", process.memory_info().rss / (1024**2), "MB", "process")
                self.record_metric("process_cpu_percent", process.cpu_percent(), "%", "process")
                
                # GPU si disponible
                try:
                    import torch
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            memory_used = torch.cuda.memory_allocated(i) / (1024**3)
                            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            
                            self.record_metric(f"gpu_{i}_memory_used_gb", memory_used, "GB", "gpu")
                            self.record_metric(f"gpu_{i}_memory_percent", (memory_used/memory_total)*100, "%", "gpu")
                except ImportError:
                    pass
                    
            except Exception as e:
                logging.warning(f"Erreur monitoring système: {e}")

class MacForgeLogger:
    """Logger avancé pour MacForge3D avec monitoring intégré."""
    
    def __init__(
        self, 
        log_dir: Optional[Path] = None,
        max_log_files: int = 10,
        max_file_size_mb: int = 100,
        enable_performance_monitoring: bool = True
    ):
        self.log_dir = log_dir or Path.home() / ".macforge3d" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_log_files = max_log_files
        self.max_file_size_mb = max_file_size_mb
        
        # Performance monitor
        self.performance_monitor = PerformanceMonitor() if enable_performance_monitoring else None
        
        # Error tracking
        self._error_events: deque = deque(maxlen=1000)
        self._error_counts: Dict[str, int] = defaultdict(int)
        
        # Setup logging
        self._setup_logging()
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_start = time.time()
        
        self.logger = logging.getLogger("MacForge3D")
        self.logger.info(f"🚀 Session démarrée: {self.session_id}")
    
    def _setup_logging(self):
        """Configure le système de logging."""
        # Format des logs
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)8s | %(name)20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Handler pour fichier
        log_file = self.log_dir / f"macforge3d_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Handler pour console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s | %(name)s | %(message)s'
        ))
        console_handler.setLevel(logging.INFO)
        
        # Configuration du logger racine
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Handler pour erreurs avec tracking
        error_handler = ErrorTrackingHandler(self)
        error_handler.setLevel(logging.WARNING)
        root_logger.addHandler(error_handler)
        
        # Rotation des logs
        self._rotate_logs()
    
    def _rotate_logs(self):
        """Effectue la rotation des fichiers de logs."""
        log_files = sorted(self.log_dir.glob("macforge3d_*.log"))
        
        # Supprimer les anciens fichiers
        while len(log_files) > self.max_log_files:
            oldest = log_files.pop(0)
            oldest.unlink()
        
        # Vérifier la taille des fichiers
        for log_file in log_files:
            if log_file.stat().st_size > self.max_file_size_mb * 1024 * 1024:
                # Archiver le fichier
                archive_name = log_file.with_suffix('.log.archive')
                log_file.rename(archive_name)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Récupère un logger nommé."""
        return logging.getLogger(f"MacForge3D.{name}")
    
    def log_performance(self, operation: str, duration: float, **metadata):
        """Log une métrique de performance."""
        if self.performance_monitor:
            self.performance_monitor.record_metric(
                f"{operation}_performance",
                duration,
                "seconds",
                "performance",
                metadata
            )
        
        logger = self.get_logger("Performance")
        logger.performance(f"{operation} completed in {duration:.3f}s", extra=metadata)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Génère un résumé de la session."""
        session_duration = time.time() - self.session_start
        
        summary = {
            "session_id": self.session_id,
            "session_duration_minutes": session_duration / 60,
            "total_errors": len(self._error_events),
            "error_types": dict(self._error_counts),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3),
                "python_version": f"{psutil.boot_time()}",  # Placeholder
            }
        }
        
        if self.performance_monitor:
            # Statistiques de performance
            summary["performance"] = {}
            
            for category in ["timing", "system", "process", "gpu"]:
                metrics = self.performance_monitor.get_metrics(category=category, since_seconds=int(session_duration))
                if metrics:
                    metric_names = list(set(m.name for m in metrics))
                    summary["performance"][category] = {
                        name: self.performance_monitor.get_statistics(name, since_seconds=int(session_duration))
                        for name in metric_names
                    }
        
        return summary
    
    def save_session_report(self):
        """Sauvegarde un rapport de session."""
        summary = self.get_session_summary()
        
        report_file = self.log_dir / f"session_report_{self.session_id}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"📊 Rapport de session sauvegardé: {report_file}")
        return report_file

class ErrorTrackingHandler(logging.Handler):
    """Handler pour tracker les erreurs."""
    
    def __init__(self, macforge_logger: MacForgeLogger):
        super().__init__()
        self.macforge_logger = macforge_logger
    
    def emit(self, record: logging.LogRecord):
        """Enregistre un événement d'erreur."""
        try:
            error_event = ErrorEvent(
                timestamp=record.created,
                level=record.levelname,
                message=record.getMessage(),
                module=record.module,
                function=record.funcName,
                traceback_info=record.exc_text if hasattr(record, 'exc_text') else None,
                context=getattr(record, '__dict__', {})
            )
            
            self.macforge_logger._error_events.append(error_event)
            self.macforge_logger._error_counts[record.levelname] += 1
            
        except Exception:
            self.handleError(record)

def performance_timer(category: str = "timing"):
    """Décorateur pour mesurer automatiquement les performances."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger_instance = get_global_logger()
            operation = f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            timing_id = logger_instance.performance_monitor.start_timing(operation) if logger_instance.performance_monitor else None
            
            try:
                result = func(*args, **kwargs)
                
                duration = time.time() - start_time
                logger_instance.log_performance(operation, duration, category=category)
                
                if timing_id and logger_instance.performance_monitor:
                    logger_instance.performance_monitor.end_timing(timing_id, category)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger_instance.get_logger("Performance").error(
                    f"{operation} failed after {duration:.3f}s: {e}"
                )
                raise
                
        return wrapper
    return decorator

def log_function_calls(log_args: bool = False, log_result: bool = False):
    """Décorateur pour logger automatiquement les appels de fonction."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_global_logger().get_logger(func.__module__)
            
            log_msg = f"Calling {func.__name__}"
            if log_args:
                log_msg += f" with args={args}, kwargs={kwargs}"
            
            logger.debug(log_msg)
            
            try:
                result = func(*args, **kwargs)
                
                if log_result:
                    logger.debug(f"{func.__name__} returned: {result}")
                else:
                    logger.debug(f"{func.__name__} completed successfully")
                
                return result
                
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
                
        return wrapper
    return decorator

# Instance globale
_global_logger: Optional[MacForgeLogger] = None

def init_logging(
    log_dir: Optional[Path] = None,
    max_log_files: int = 10,
    max_file_size_mb: int = 100,
    enable_performance_monitoring: bool = True
) -> MacForgeLogger:
    """Initialise le système de logging global."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = MacForgeLogger(
            log_dir=log_dir,
            max_log_files=max_log_files,
            max_file_size_mb=max_file_size_mb,
            enable_performance_monitoring=enable_performance_monitoring
        )
    
    return _global_logger

def get_global_logger() -> MacForgeLogger:
    """Récupère l'instance globale du logger."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = init_logging()
    
    return _global_logger

def shutdown_logging():
    """Ferme proprement le système de logging."""
    global _global_logger
    
    if _global_logger:
        _global_logger.save_session_report()
        if _global_logger.performance_monitor:
            _global_logger.performance_monitor.stop_system_monitoring()
        
        _global_logger.logger.info("🏁 Session terminée")
        logging.shutdown()