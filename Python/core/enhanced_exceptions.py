"""
Enhanced exception handling system for MacForge3D.
Provides detailed error information and recovery suggestions.
"""

import traceback
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    """Niveaux de gravité des erreurs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Catégories d'erreurs."""
    MESH_PROCESSING = "mesh_processing"
    MEMORY = "memory"
    PERFORMANCE = "performance"
    VALIDATION = "validation"
    FILE_IO = "file_io"
    NETWORK = "network"
    COMPUTATION = "computation"
    CONFIGURATION = "configuration"

@dataclass
class ErrorContext:
    """Contexte détaillé d'une erreur."""
    timestamp: float
    function_name: str
    file_path: str
    line_number: int
    severity: ErrorSeverity
    category: ErrorCategory
    parameters: Dict[str, Any]
    system_state: Dict[str, Any]
    recovery_suggestions: List[str]
    
class MacForge3DException(Exception):
    """Exception de base pour MacForge3D avec contexte enrichi."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        parameters: Optional[Dict[str, Any]] = None,
        recovery_suggestions: Optional[List[str]] = None,
        original_exception: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.parameters = parameters or {}
        self.recovery_suggestions = recovery_suggestions or []
        self.original_exception = original_exception
        self.timestamp = time.time()
        
        # Capturer le contexte d'exécution
        frame = traceback.extract_stack()[-2]
        self.context = ErrorContext(
            timestamp=self.timestamp,
            function_name=frame.name,
            file_path=frame.filename,
            line_number=frame.lineno,
            severity=severity,
            category=category,
            parameters=self.parameters,
            system_state=self._capture_system_state(),
            recovery_suggestions=self.recovery_suggestions
        )
        
        # Logger l'erreur automatiquement
        self._log_error()
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture l'état du système au moment de l'erreur."""
        try:
            import psutil
            return {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "process_memory_mb": psutil.Process().memory_info().rss / (1024**2)
            }
        except:
            return {"capture_failed": True}
    
    def _log_error(self):
        """Log l'erreur avec les détails appropriés selon la gravité."""
        log_message = f"[{self.category.value.upper()}] {self.message}"
        
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_context": self.context})
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(log_message, extra={"error_context": self.context})
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message, extra={"error_context": self.context})
        else:
            logger.info(log_message, extra={"error_context": self.context})
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """Retourne les informations détaillées de l'erreur."""
        return {
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "timestamp": self.timestamp,
            "function": self.context.function_name,
            "file": self.context.file_path,
            "line": self.context.line_number,
            "parameters": self.parameters,
            "system_state": self.context.system_state,
            "recovery_suggestions": self.recovery_suggestions,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }

class MeshProcessingError(MacForge3DException):
    """Erreur de traitement de maillage."""
    
    def __init__(self, message: str, mesh_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MESH_PROCESSING,
            parameters={"mesh_info": mesh_info} if mesh_info else {},
            recovery_suggestions=[
                "Vérifiez l'intégrité du fichier de maillage",
                "Réduisez la complexité du maillage",
                "Utilisez une méthode de réparation alternative",
                "Contactez le support technique si le problème persiste"
            ],
            **kwargs
        )

class MemoryError(MacForge3DException):
    """Erreur de mémoire insuffisante."""
    
    def __init__(self, message: str, required_memory: Optional[float] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.MEMORY,
            parameters={"required_memory_gb": required_memory} if required_memory else {},
            recovery_suggestions=[
                "Libérez de la mémoire en fermant d'autres applications",
                "Réduisez la taille du maillage ou la résolution",
                "Utilisez le traitement par chunks",
                "Activez le mode de traitement économique",
                "Redémarrez l'application pour nettoyer la mémoire"
            ],
            **kwargs
        )

class ValidationError(MacForge3DException):
    """Erreur de validation des données d'entrée."""
    
    def __init__(self, message: str, invalid_params: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            parameters={"invalid_params": invalid_params} if invalid_params else {},
            recovery_suggestions=[
                "Vérifiez les paramètres d'entrée",
                "Consultez la documentation pour les valeurs valides",
                "Utilisez les paramètres par défaut recommandés",
                "Contactez le support pour assistance"
            ],
            **kwargs
        )

class PerformanceError(MacForge3DException):
    """Erreur de performance (timeouts, lenteur excessive)."""
    
    def __init__(self, message: str, performance_data: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.PERFORMANCE,
            parameters={"performance_data": performance_data} if performance_data else {},
            recovery_suggestions=[
                "Réduisez la complexité de l'opération",
                "Augmentez les ressources allouées",
                "Utilisez le mode de traitement parallèle",
                "Vérifiez les performances système",
                "Contactez le support si les performances restent dégradées"
            ],
            **kwargs
        )

class FileIOError(MacForge3DException):
    """Erreur d'entrée/sortie de fichier."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.FILE_IO,
            parameters={"file_path": file_path} if file_path else {},
            recovery_suggestions=[
                "Vérifiez l'existence et les permissions du fichier",
                "Vérifiez l'espace disque disponible",
                "Essayez un chemin de fichier différent",
                "Vérifiez le format du fichier",
                "Redémarrez l'application si nécessaire"
            ],
            **kwargs
        )

def handle_exception_gracefully(
    func_name: str,
    exception: Exception,
    parameters: Optional[Dict[str, Any]] = None,
    fallback_result: Any = None
) -> Any:
    """
    Gère une exception de manière élégante en fournissant un contexte détaillé.
    
    Args:
        func_name: Nom de la fonction où l'erreur s'est produite
        exception: Exception originale
        parameters: Paramètres de la fonction
        fallback_result: Résultat de fallback à retourner
        
    Returns:
        Résultat de fallback ou None
    """
    
    # Déterminer le type d'erreur et créer une exception enrichie
    if isinstance(exception, MemoryError):
        enhanced_exception = MemoryError(
            f"Erreur mémoire dans {func_name}: {str(exception)}",
            original_exception=exception,
            parameters=parameters
        )
    elif "mesh" in str(exception).lower() or "trimesh" in str(exception).lower():
        enhanced_exception = MeshProcessingError(
            f"Erreur traitement maillage dans {func_name}: {str(exception)}",
            original_exception=exception,
            parameters=parameters
        )
    elif "file" in str(exception).lower() or "path" in str(exception).lower():
        enhanced_exception = FileIOError(
            f"Erreur fichier dans {func_name}: {str(exception)}",
            original_exception=exception,
            parameters=parameters
        )
    else:
        enhanced_exception = MacForge3DException(
            f"Erreur dans {func_name}: {str(exception)}",
            original_exception=exception,
            parameters=parameters
        )
    
    # Logger les détails de l'erreur
    logger.error(f"Exception gérée: {enhanced_exception.get_detailed_info()}")
    
    return fallback_result

# Décorateur pour la gestion automatique des exceptions
def exception_handler(
    fallback_result: Any = None,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    category: ErrorCategory = ErrorCategory.COMPUTATION
):
    """
    Décorateur pour la gestion automatique des exceptions.
    
    Args:
        fallback_result: Résultat retourné en cas d'erreur
        severity: Gravité de l'erreur
        category: Catégorie de l'erreur
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return handle_exception_gracefully(
                    func.__name__,
                    e,
                    {"args": args, "kwargs": kwargs},
                    fallback_result
                )
        return wrapper
    return decorator