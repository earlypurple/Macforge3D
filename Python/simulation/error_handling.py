import logging
import traceback
import sys
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import gc
import psutil
import numpy as np
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Type générique pour le retour des fonctions
T = TypeVar('T')

@dataclass
class ErrorContext:
    """Contexte détaillé d'une erreur."""
    timestamp: datetime
    error_type: str
    error_message: str
    traceback: str
    memory_usage: float
    cpu_usage: float
    additional_info: Dict[str, Any]

class MacForgeError(Exception):
    """Classe de base pour les erreurs personnalisées de MacForge3D."""
    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.context = self._capture_context()
        
    def _capture_context(self) -> ErrorContext:
        """Capture le contexte de l'erreur."""
        return ErrorContext(
            timestamp=datetime.now(),
            error_type=self.__class__.__name__,
            error_message=str(self),
            traceback=''.join(traceback.format_tb(sys.exc_info()[2])),
            memory_usage=psutil.Process().memory_percent(),
            cpu_usage=psutil.Process().cpu_percent(),
            additional_info=self.details
        )
        
class SimulationError(MacForgeError):
    """Erreur lors de la simulation."""
    pass

class ModelingError(MacForgeError):
    """Erreur lors de la modélisation."""
    pass

class ExportError(MacForgeError):
    """Erreur lors de l'export."""
    pass

class ResourceError(MacForgeError):
    """Erreur de ressources (mémoire, CPU, etc)."""
    pass

class ValidationError(MacForgeError):
    """Erreur de validation des données."""
    pass

class ErrorHandler:
    """Gestionnaire d'erreurs centralisé."""
    
    def __init__(self):
        self.error_log_path = Path("logs/errors.log")
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Stratégies de récupération par type d'erreur
        self.recovery_strategies: Dict[
            Type[Exception],
            Callable[[Exception], Optional[Any]]
        ] = {
            MemoryError: self._handle_memory_error,
            TimeoutError: self._handle_timeout,
            ValueError: self._handle_validation_error,
            RuntimeError: self._handle_runtime_error
        }
        
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Gère une erreur et tente de la récupérer si possible.
        
        Args:
            error: L'erreur à gérer
            context: Contexte supplémentaire
            
        Returns:
            Résultat de la récupération si possible
        """
        try:
            # Enregistrer l'erreur
            self._log_error(error, context)
            
            # Tenter la récupération
            if type(error) in self.recovery_strategies:
                return self.recovery_strategies[type(error)](error)
                
            # Convertir en erreur MacForge si nécessaire
            if not isinstance(error, MacForgeError):
                error = self._convert_error(error)
                
            # Propager l'erreur enrichie
            raise error
            
        except Exception as e:
            logger.error(f"Erreur lors de la gestion d'erreur: {str(e)}")
            raise
            
    def run_with_recovery(
        self,
        func: Callable[..., T],
        *args,
        **kwargs
    ) -> T:
        """
        Exécute une fonction avec gestion d'erreur et récupération.
        
        Args:
            func: Fonction à exécuter
            args: Arguments positionnels
            kwargs: Arguments nommés
            
        Returns:
            Résultat de la fonction
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            result = self.handle_error(e)
            if result is not None:
                return result
            raise
            
    def _log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ):
        """Enregistre l'erreur dans le fichier de log."""
        with self.error_log_path.open('a') as f:
            timestamp = datetime.now().isoformat()
            error_type = type(error).__name__
            error_message = str(error)
            context_str = str(context) if context else "No context"
            
            f.write(f"""
[{timestamp}] {error_type}: {error_message}
Context: {context_str}
Traceback:
{''.join(traceback.format_tb(sys.exc_info()[2]))}
{'=' * 80}
""")
            
    def _convert_error(self, error: Exception) -> MacForgeError:
        """Convertit une erreur standard en erreur MacForge."""
        if isinstance(error, MemoryError):
            return ResourceError(
                "Mémoire insuffisante",
                "RESOURCE_MEMORY",
                {"memory_usage": psutil.Process().memory_percent()}
            )
        elif isinstance(error, TimeoutError):
            return ResourceError(
                "Opération trop longue",
                "RESOURCE_TIMEOUT",
                {"cpu_usage": psutil.Process().cpu_percent()}
            )
        elif isinstance(error, ValueError):
            return ValidationError(
                str(error),
                "VALIDATION_ERROR"
            )
        else:
            return MacForgeError(
                str(error),
                "UNKNOWN_ERROR"
            )
            
    def _handle_memory_error(self, error: Exception) -> Optional[Any]:
        """Gère une erreur de mémoire."""
        logger.warning("Tentative de récupération après erreur mémoire...")
        
        # Libérer la mémoire
        gc.collect()
        
        # Vérifier la mémoire disponible
        memory_usage = psutil.Process().memory_percent()
        if memory_usage > 90:
            raise ResourceError(
                "Mémoire critique",
                "RESOURCE_MEMORY_CRITICAL",
                {"memory_usage": memory_usage}
            )
            
        return None
        
    def _handle_timeout(self, error: Exception) -> Optional[Any]:
        """Gère une erreur de timeout."""
        logger.warning("Tentative de récupération après timeout...")
        
        # Vérifier la charge CPU
        cpu_usage = psutil.Process().cpu_percent()
        if cpu_usage > 90:
            raise ResourceError(
                "CPU surchargé",
                "RESOURCE_CPU_CRITICAL",
                {"cpu_usage": cpu_usage}
            )
            
        return None
        
    def _handle_validation_error(self, error: Exception) -> Optional[Any]:
        """Gère une erreur de validation."""
        logger.warning("Tentative de correction des données...")
        
        if isinstance(error, np.AxisError):
            # Tenter de corriger les dimensions
            return None
            
        return None
        
    def _handle_runtime_error(self, error: Exception) -> Optional[Any]:
        """Gère une erreur d'exécution."""
        logger.warning("Tentative de récupération runtime...")
        
        # Analyser le message d'erreur pour des cas spécifiques
        error_msg = str(error).lower()
        
        if "cuda" in error_msg:
            # Erreur GPU, basculer sur CPU
            return None
        elif "thread" in error_msg:
            # Erreur de thread, réduire la parallélisation
            return None
            
        return None

class InputValidator:
    """
    Validateur d'entrées avec contrôles de sécurité.
    """
    
    def __init__(self):
        self.max_string_length = 10000
        self.max_array_size = 1000000
        self.allowed_file_extensions = {'.stl', '.obj', '.ply', '.off', '.3mf'}
        self.dangerous_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'\.\./\.\.',  # Path traversal
            r'\\\\',  # UNC paths
            r'[;&|`$]',  # Command injection
        ]
        
    def validate_string_input(self, value: str, field_name: str = "input") -> str:
        """
        Valide et nettoie une entrée string.
        
        Args:
            value: Valeur à valider
            field_name: Nom du champ pour les messages d'erreur
            
        Returns:
            String validée et nettoyée
            
        Raises:
            ValidationError: Si l'entrée est invalide
        """
        if not isinstance(value, str):
            raise ValidationError(
                f"{field_name} doit être une chaîne de caractères",
                "VALIDATION_TYPE_ERROR"
            )
        
        # Vérifier la longueur
        if len(value) > self.max_string_length:
            raise ValidationError(
                f"{field_name} trop long (max {self.max_string_length} caractères)",
                "VALIDATION_LENGTH_ERROR"
            )
        
        # Vérifier les patterns dangereux
        import re
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                logger.warning(f"Pattern dangereux détecté dans {field_name}: {pattern}")
                raise ValidationError(
                    f"{field_name} contient du contenu potentiellement dangereux",
                    "VALIDATION_SECURITY_ERROR"
                )
        
        # Nettoyer les caractères de contrôle
        cleaned = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')
        
        return cleaned
    
    def validate_mesh_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valide les paramètres de maillage.
        
        Args:
            params: Paramètres à valider
            
        Returns:
            Paramètres validés
            
        Raises:
            ValidationError: Si les paramètres sont invalides
        """
        validated = {}
        
        # Validation des paramètres communs
        if 'resolution' in params:
            resolution = params['resolution']
            if not isinstance(resolution, (int, float)) or resolution < 100 or resolution > 1000000:
                raise ValidationError(
                    "Resolution doit être entre 100 et 1000000",
                    "VALIDATION_RANGE_ERROR"
                )
            validated['resolution'] = resolution
        
        if 'quality' in params:
            quality = params['quality']
            if quality not in ['low', 'medium', 'high', 'ultra']:
                raise ValidationError(
                    "Quality doit être 'low', 'medium', 'high' ou 'ultra'",
                    "VALIDATION_VALUE_ERROR"
                )
            validated['quality'] = quality
        
        if 'material' in params:
            material = self.validate_string_input(params['material'], 'material')
            allowed_materials = ['PLA', 'ABS', 'PETG', 'TPU', 'WOOD', 'METAL']
            if material.upper() not in allowed_materials:
                raise ValidationError(
                    f"Matériau non supporté: {material}",
                    "VALIDATION_VALUE_ERROR"
                )
            validated['material'] = material.upper()
        
        if 'temperature' in params:
            temp = params['temperature']
            if not isinstance(temp, (int, float)) or temp < 150.0 or temp > 350.0:
                raise ValidationError(
                    "Temperature doit être entre 150°C et 350°C",
                    "VALIDATION_RANGE_ERROR"
                )
            validated['temperature'] = temp
        
        # Copier les autres paramètres après validation de base
        for key, value in params.items():
            if key not in validated:
                if isinstance(value, str):
                    validated[key] = self.validate_string_input(value, key)
                elif isinstance(value, (int, float)):
                    # Validation numérique de base
                    if np.isnan(value) or np.isinf(value):
                        raise ValidationError(
                            f"{key} ne peut pas être NaN ou infini",
                            "VALIDATION_VALUE_ERROR"
                        )
                    validated[key] = value
                else:
                    validated[key] = value  # Passer tel quel pour les types complexes
        
        return validated

# Instance globale du validateur
input_validator = InputValidator()

# Instance globale du gestionnaire d'erreurs
error_handler = ErrorHandler()
