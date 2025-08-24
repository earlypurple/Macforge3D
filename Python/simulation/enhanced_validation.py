"""
Enhanced input validation system for MacForge3D.
Provides comprehensive validation with detailed error messages and auto-correction.
"""

import re
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import trimesh
from pathlib import Path

try:
    from ..core.enhanced_exceptions import ValidationError
except ImportError:
    # Fallback pour les tests directs
    from core.enhanced_exceptions import ValidationError

logger = logging.getLogger(__name__)

class ParameterType(Enum):
    """Types de paramètres supportés."""
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    MESH = "mesh"
    PATH = "path"
    COLOR = "color"
    ENUM = "enum"

@dataclass
class ValidationRule:
    """Règle de validation pour un paramètre."""
    name: str
    param_type: ParameterType
    required: bool = True
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: str = ""
    auto_correct: bool = False
    correction_strategy: Optional[Callable] = None
    
@dataclass
class ValidationResult:
    """Résultat de validation d'un paramètre."""
    is_valid: bool
    original_value: Any
    corrected_value: Any = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class EnhancedValidator:
    """Validateur avancé avec auto-correction et suggestions."""
    
    def __init__(self):
        self.rules = {}
        self._setup_default_rules()
        
    def _setup_default_rules(self):
        """Configure les règles de validation par défaut."""
        
        # Règles pour les paramètres de maillage
        self.add_rule(ValidationRule(
            name="resolution",
            param_type=ParameterType.INTEGER,
            min_value=100,
            max_value=1000000,
            description="Résolution du maillage (nombre de points)",
            auto_correct=True,
            correction_strategy=lambda x: max(100, min(1000000, int(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="quality",
            param_type=ParameterType.ENUM,
            allowed_values=["low", "medium", "high", "ultra"],
            description="Niveau de qualité du rendu",
            auto_correct=True,
            correction_strategy=lambda x: "medium" if x not in ["low", "medium", "high", "ultra"] else x
        ))
        
        self.add_rule(ValidationRule(
            name="material",
            param_type=ParameterType.ENUM,
            allowed_values=["PLA", "ABS", "PETG", "TPU", "WOOD", "METAL"],
            description="Type de matériau d'impression",
            auto_correct=True,
            correction_strategy=lambda x: "PLA" if x not in ["PLA", "ABS", "PETG", "TPU", "WOOD", "METAL"] else x
        ))
        
        self.add_rule(ValidationRule(
            name="temperature",
            param_type=ParameterType.FLOAT,
            min_value=150.0,
            max_value=350.0,
            description="Température d'extrusion (°C)",
            auto_correct=True,
            correction_strategy=lambda x: max(150.0, min(350.0, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="layer_height",
            param_type=ParameterType.FLOAT,
            min_value=0.05,
            max_value=0.5,
            description="Hauteur de couche (mm)",
            auto_correct=True,
            correction_strategy=lambda x: max(0.05, min(0.5, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="infill_percentage",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=100.0,
            description="Pourcentage de remplissage",
            auto_correct=True,
            correction_strategy=lambda x: max(0.0, min(100.0, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="smoothness_weight",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=1.0,
            description="Poids de lissage (0.0 = aucun, 1.0 = maximum)",
            auto_correct=True,
            correction_strategy=lambda x: max(0.0, min(1.0, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="detail_preservation",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=1.0,
            description="Niveau de préservation des détails",
            auto_correct=True,
            correction_strategy=lambda x: max(0.0, min(1.0, float(x)))
        ))
        
        # Règles pour les effets avancés
        self.add_rule(ValidationRule(
            name="fractal_intensity",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=2.0,
            description="Intensité de l'effet fractal",
            auto_correct=True,
            correction_strategy=lambda x: max(0.0, min(2.0, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="plasma_amplitude",
            param_type=ParameterType.FLOAT,
            min_value=0.0,
            max_value=5.0,
            description="Amplitude de l'effet plasma",
            auto_correct=True,
            correction_strategy=lambda x: max(0.0, min(5.0, float(x)))
        ))
        
        self.add_rule(ValidationRule(
            name="enable_adaptive_optimization",
            param_type=ParameterType.BOOLEAN,
            description="Active l'optimisation adaptative",
            auto_correct=True,
            correction_strategy=lambda x: bool(x)
        ))
        
        # Règles pour les chemins de fichiers
        self.add_rule(ValidationRule(
            name="input_path",
            param_type=ParameterType.PATH,
            description="Chemin vers le fichier d'entrée",
            pattern=r".*\.(obj|stl|ply|gltf|glb)$"
        ))
        
        self.add_rule(ValidationRule(
            name="output_path",
            param_type=ParameterType.PATH,
            description="Chemin vers le fichier de sortie",
            required=False
        ))
        
        # Règles pour les couleurs
        self.add_rule(ValidationRule(
            name="color",
            param_type=ParameterType.COLOR,
            description="Couleur au format #RRGGBB ou nom de couleur",
            pattern=r"^#[0-9A-Fa-f]{6}$|^(red|green|blue|white|black|yellow|cyan|magenta)$"
        ))
    
    def add_rule(self, rule: ValidationRule):
        """Ajoute une règle de validation."""
        self.rules[rule.name] = rule
        
    def validate_parameter(self, name: str, value: Any) -> ValidationResult:
        """Valide un paramètre individuel."""
        
        if name not in self.rules:
            return ValidationResult(
                is_valid=True,
                original_value=value,
                warnings=[f"Aucune règle de validation pour '{name}'"]
            )
        
        rule = self.rules[name]
        result = ValidationResult(is_valid=True, original_value=value)
        
        # Vérifier si requis
        if rule.required and (value is None or value == ""):
            result.is_valid = False
            result.errors.append(f"Paramètre '{name}' requis mais non fourni")
            return result
        
        if value is None:
            return result
        
        # Validation par type
        try:
            if rule.param_type == ParameterType.INTEGER:
                result = self._validate_integer(value, rule, result)
            elif rule.param_type == ParameterType.FLOAT:
                result = self._validate_float(value, rule, result)
            elif rule.param_type == ParameterType.STRING:
                result = self._validate_string(value, rule, result)
            elif rule.param_type == ParameterType.BOOLEAN:
                result = self._validate_boolean(value, rule, result)
            elif rule.param_type == ParameterType.ARRAY:
                result = self._validate_array(value, rule, result)
            elif rule.param_type == ParameterType.MESH:
                result = self._validate_mesh(value, rule, result)
            elif rule.param_type == ParameterType.PATH:
                result = self._validate_path(value, rule, result)
            elif rule.param_type == ParameterType.COLOR:
                result = self._validate_color(value, rule, result)
            elif rule.param_type == ParameterType.ENUM:
                result = self._validate_enum(value, rule, result)
                
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Erreur validation '{name}': {str(e)}")
        
        # Auto-correction si demandée et nécessaire
        if not result.is_valid and rule.auto_correct and rule.correction_strategy:
            try:
                corrected = rule.correction_strategy(value)
                result.corrected_value = corrected
                result.warnings.append(f"Valeur auto-corrigée: {value} -> {corrected}")
                result.suggestions.append(f"Utilisez {corrected} pour '{name}'")
                
                # Re-valider la valeur corrigée
                corrected_result = self.validate_parameter(name, corrected)
                if corrected_result.is_valid:
                    result.is_valid = True
                    result.errors.clear()
                    
            except Exception as e:
                result.warnings.append(f"Échec auto-correction: {str(e)}")
        
        return result
    
    def _validate_integer(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide une valeur entière."""
        try:
            int_value = int(value)
            result.corrected_value = int_value
            
            if rule.min_value is not None and int_value < rule.min_value:
                result.is_valid = False
                result.errors.append(f"Valeur {int_value} < minimum {rule.min_value}")
                
            if rule.max_value is not None and int_value > rule.max_value:
                result.is_valid = False
                result.errors.append(f"Valeur {int_value} > maximum {rule.max_value}")
                
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f"Impossible de convertir '{value}' en entier")
            
        return result
    
    def _validate_float(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide une valeur flottante."""
        try:
            float_value = float(value)
            result.corrected_value = float_value
            
            if rule.min_value is not None and float_value < rule.min_value:
                result.is_valid = False
                result.errors.append(f"Valeur {float_value} < minimum {rule.min_value}")
                
            if rule.max_value is not None and float_value > rule.max_value:
                result.is_valid = False
                result.errors.append(f"Valeur {float_value} > maximum {rule.max_value}")
                
        except (ValueError, TypeError):
            result.is_valid = False
            result.errors.append(f"Impossible de convertir '{value}' en nombre")
            
        return result
    
    def _validate_string(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide une chaîne de caractères."""
        str_value = str(value)
        result.corrected_value = str_value
        
        if rule.pattern:
            if not re.match(rule.pattern, str_value):
                result.is_valid = False
                result.errors.append(f"Format invalide pour '{str_value}' (pattern: {rule.pattern})")
                
        return result
    
    def _validate_boolean(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide une valeur booléenne."""
        if isinstance(value, bool):
            result.corrected_value = value
        elif isinstance(value, str):
            lower_val = value.lower()
            if lower_val in ["true", "yes", "1", "on"]:
                result.corrected_value = True
            elif lower_val in ["false", "no", "0", "off"]:
                result.corrected_value = False
            else:
                result.is_valid = False
                result.errors.append(f"Impossible de convertir '{value}' en booléen")
        else:
            result.corrected_value = bool(value)
            
        return result
    
    def _validate_array(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide un tableau."""
        if not isinstance(value, (list, tuple, np.ndarray)):
            result.is_valid = False
            result.errors.append(f"'{value}' n'est pas un tableau")
        else:
            result.corrected_value = list(value) if not isinstance(value, list) else value
            
        return result
    
    def _validate_mesh(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide un maillage."""
        if not isinstance(value, trimesh.Trimesh):
            result.is_valid = False
            result.errors.append(f"'{type(value)}' n'est pas un maillage trimesh.Trimesh")
        else:
            # Vérifications de base du maillage
            if len(value.vertices) == 0:
                result.is_valid = False
                result.errors.append("Maillage vide (aucun vertex)")
            elif len(value.faces) == 0:
                result.warnings.append("Maillage sans faces")
            
            if not value.is_valid:
                result.warnings.append("Maillage avec des problèmes de validité")
                
            result.corrected_value = value
            
        return result
    
    def _validate_path(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide un chemin de fichier."""
        try:
            path = Path(str(value))
            result.corrected_value = str(path)
            
            if rule.pattern:
                if not re.match(rule.pattern, str(path), re.IGNORECASE):
                    result.is_valid = False
                    result.errors.append(f"Extension de fichier non supportée: {path.suffix}")
                    result.suggestions.append("Extensions supportées: .obj, .stl, .ply, .gltf, .glb")
            
            # Vérifier l'existence pour les fichiers d'entrée
            if "input" in rule.name.lower():
                if not path.exists():
                    result.is_valid = False
                    result.errors.append(f"Fichier non trouvé: {path}")
                elif not path.is_file():
                    result.is_valid = False
                    result.errors.append(f"Le chemin n'est pas un fichier: {path}")
            
            # Vérifier l'accès en écriture pour les fichiers de sortie
            if "output" in rule.name.lower():
                parent_dir = path.parent
                if not parent_dir.exists():
                    result.warnings.append(f"Répertoire de sortie n'existe pas: {parent_dir}")
                    result.suggestions.append(f"Créer le répertoire: mkdir -p {parent_dir}")
                    
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Chemin invalide: {str(e)}")
            
        return result
    
    def _validate_color(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationResult:
        """Valide une couleur."""
        str_value = str(value)
        
        # Vérifier le format hexadécimal
        hex_pattern = r"^#[0-9A-Fa-f]{6}$"
        color_names = ["red", "green", "blue", "white", "black", "yellow", "cyan", "magenta"]
        
        if re.match(hex_pattern, str_value):
            result.corrected_value = str_value.upper()
        elif str_value.lower() in color_names:
            result.corrected_value = str_value.lower()
        else:
            result.is_valid = False
            result.errors.append(f"Format de couleur invalide: '{str_value}'")
            result.suggestions.append("Utilisez #RRGGBB ou un nom de couleur standard")
            
        return result
    
    def _validate_enum(self, value: Any, rule: ValidationRule, result: ValidationResult) -> ValidationRule:
        """Valide une valeur d'énumération."""
        if rule.allowed_values and value not in rule.allowed_values:
            result.is_valid = False
            result.errors.append(f"Valeur '{value}' non autorisée")
            result.suggestions.append(f"Valeurs autorisées: {', '.join(map(str, rule.allowed_values))}")
        else:
            result.corrected_value = value
            
        return result
    
    def validate_mesh_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Valide un ensemble de paramètres de maillage."""
        validated_params = {}
        errors = []
        warnings = []
        
        for name, value in params.items():
            result = self.validate_parameter(name, value)
            
            if result.is_valid:
                validated_params[name] = result.corrected_value if result.corrected_value is not None else value
            else:
                errors.extend([f"{name}: {error}" for error in result.errors])
                
            warnings.extend([f"{name}: {warning}" for warning in result.warnings])
        
        if errors:
            error_message = "Erreurs de validation: " + "; ".join(errors)
            if warnings:
                error_message += f"\nAvertissements: {'; '.join(warnings)}"
            raise ValidationError(error_message, invalid_params=params)
        
        if warnings:
            logger.warning("Avertissements de validation: " + "; ".join(warnings))
            
        return validated_params
    
    def get_parameter_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Retourne les informations sur un paramètre."""
        if name not in self.rules:
            return None
            
        rule = self.rules[name]
        return {
            "name": rule.name,
            "type": rule.param_type.value,
            "required": rule.required,
            "description": rule.description,
            "min_value": rule.min_value,
            "max_value": rule.max_value,
            "allowed_values": rule.allowed_values,
            "pattern": rule.pattern,
            "auto_correct": rule.auto_correct
        }
    
    def list_all_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Liste tous les paramètres supportés avec leurs règles."""
        return {name: self.get_parameter_info(name) for name in self.rules.keys()}

# Instance globale du validateur
input_validator = EnhancedValidator()

# Fonctions de convenance
def validate_mesh_quality_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les paramètres pour l'analyse de qualité de maillage."""
    return input_validator.validate_mesh_parameters(params)

def validate_smoothing_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les paramètres pour le lissage de maillage."""
    return input_validator.validate_mesh_parameters(params)

def validate_processing_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les paramètres pour le traitement de maillage."""
    return input_validator.validate_mesh_parameters(params)

# Nouvelles fonctions de validation avancées

def validate_text_style_params(style_params: Dict[str, Any]) -> Dict[str, Any]:
    """Valide les paramètres d'un style de texte."""
    validator = EnhancedValidator()
    
    # Ajouter des règles spécifiques pour les styles de texte
    validator.add_rule(ValidationRule(
        name="bevel_amount",
        param_type=ParameterType.FLOAT,
        min_value=0.0,
        max_value=1.0,
        description="Quantité de bevel (0-1)",
        auto_correct=True
    ))
    
    validator.add_rule(ValidationRule(
        name="wave_amplitude",
        param_type=ParameterType.FLOAT,
        min_value=0.0,
        max_value=2.0,
        description="Amplitude des vagues",
        auto_correct=True
    ))
    
    validator.add_rule(ValidationRule(
        name="tessellation_level",
        param_type=ParameterType.INTEGER,
        min_value=0,
        max_value=4,
        description="Niveau de tessellation (0-4)",
        auto_correct=True
    ))
    
    return validator.validate_mesh_parameters(style_params)

def validate_performance_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Valide une configuration de performance."""
    validator = EnhancedValidator()
    
    # Règles pour la configuration de performance
    validator.add_rule(ValidationRule(
        name="thread_workers",
        param_type=ParameterType.INTEGER,
        min_value=1,
        max_value=64,
        description="Nombre de workers de thread",
        auto_correct=True
    ))
    
    validator.add_rule(ValidationRule(
        name="memory_limit_gb",
        param_type=ParameterType.FLOAT,
        min_value=0.5,
        max_value=128.0,
        description="Limite mémoire en GB",
        auto_correct=True
    ))
    
    validator.add_rule(ValidationRule(
        name="optimization_level",
        param_type=ParameterType.ENUM,
        allowed_values=["cpu_intensive", "memory_intensive", "gpu_intensive", "balanced", "ultra"],
        description="Niveau d'optimisation",
        auto_correct=False
    ))
    
    return validator.validate_mesh_parameters(config)

class AdvancedValidator(EnhancedValidator):
    """Validateur avancé avec règles contextuelles et intelligence artificielle."""
    
    def __init__(self):
        super().__init__()
        self.context_history = []
        self.pattern_analyzer = PatternAnalyzer()
    
    def validate_with_context(
        self,
        params: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Valide avec prise en compte du contexte d'utilisation.
        
        Args:
            params: Paramètres à valider
            context: Contexte d'utilisation (mesh_size, operation_type, etc.)
            
        Returns:
            Paramètres validés et optimisés
        """
        context = context or {}
        
        # Analyser le contexte pour ajuster les règles
        adapted_rules = self._adapt_rules_to_context(context)
        
        # Valider avec les règles adaptées
        validated_params = {}
        recommendations = []
        
        for name, value in params.items():
            if name in adapted_rules:
                rule = adapted_rules[name]
                result = self._validate_with_adaptive_rule(value, rule, context)
            else:
                result = self.validate_parameter(name, value)
            
            if result.is_valid:
                validated_params[name] = result.corrected_value if result.corrected_value is not None else value
            
            recommendations.extend(result.suggestions)
        
        # Enregistrer dans l'historique pour l'apprentissage
        self.context_history.append({
            'params': params,
            'context': context,
            'validated': validated_params,
            'recommendations': recommendations
        })
        
        return {
            'validated_params': validated_params,
            'recommendations': recommendations,
            'context_adaptations': self._get_context_adaptations(context)
        }
    
    def _adapt_rules_to_context(self, context: Dict[str, Any]) -> Dict[str, ValidationRule]:
        """Adapte les règles de validation au contexte."""
        adapted_rules = {}
        
        mesh_size = context.get('mesh_size', 'medium')
        operation_type = context.get('operation_type', 'general')
        hardware_constraints = context.get('hardware_constraints', {})
        
        # Adapter les règles selon la taille du mesh
        if mesh_size == 'large':
            # Pour les gros meshes, réduire les limites gourmandes en ressources
            adapted_rules['tessellation_level'] = ValidationRule(
                name="tessellation_level",
                param_type=ParameterType.INTEGER,
                min_value=0,
                max_value=2,  # Réduire pour les gros meshes
                description="Niveau de tessellation adapté aux gros meshes",
                auto_correct=True
            )
        
        # Adapter selon le type d'opération
        if operation_type == 'real_time':
            adapted_rules['smooth_iterations'] = ValidationRule(
                name="smooth_iterations",
                param_type=ParameterType.INTEGER,
                min_value=0,
                max_value=3,  # Limiter pour le temps réel
                description="Itérations de lissage pour temps réel",
                auto_correct=True
            )
        
        # Adapter selon les contraintes matérielles
        gpu_memory = hardware_constraints.get('gpu_memory_gb', 8)
        if gpu_memory < 4:
            adapted_rules['batch_size'] = ValidationRule(
                name="batch_size",
                param_type=ParameterType.INTEGER,
                min_value=1,
                max_value=4,  # Réduire pour GPU faible
                description="Taille de batch adaptée au GPU",
                auto_correct=True
            )
        
        return adapted_rules
    
    def _validate_with_adaptive_rule(
        self,
        value: Any,
        rule: ValidationRule,
        context: Dict[str, Any]
    ) -> ValidationResult:
        """Valide avec une règle adaptative."""
        result = ValidationResult(is_valid=True, original_value=value)
        
        # Utiliser la validation de base en appliquant la règle
        if rule.param_type == ParameterType.INTEGER:
            base_result = self._validate_integer(value, rule, result)
        elif rule.param_type == ParameterType.FLOAT:
            base_result = self._validate_float(value, rule, result)
        elif rule.param_type == ParameterType.STRING:
            base_result = self._validate_string(value, rule, result)
        elif rule.param_type == ParameterType.BOOLEAN:
            base_result = self._validate_boolean(value, rule, result)
        elif rule.param_type == ParameterType.ENUM:
            base_result = self._validate_enum(value, rule, result)
        else:
            base_result = result
            
        result.is_valid = base_result.is_valid
        result.errors = base_result.errors
        result.warnings = base_result.warnings
        result.corrected_value = base_result.corrected_value
        
        # Ajouter des suggestions contextuelles
        context_suggestions = self._generate_context_suggestions(value, rule, context)
        result.suggestions.extend(context_suggestions)
        
        return result
    
    def _generate_context_suggestions(
        self,
        value: Any,
        rule: ValidationRule,
        context: Dict[str, Any]
    ) -> List[str]:
        """Génère des suggestions basées sur le contexte."""
        suggestions = []
        
        mesh_size = context.get('mesh_size', 'medium')
        hardware_constraints = context.get('hardware_constraints', {})
        
        if rule.name == 'tessellation_level' and mesh_size == 'large':
            if value > 2:
                suggestions.append("Niveau de tessellation élevé pour un gros mesh - considérer la réduction")
        
        if rule.name == 'memory_limit_gb':
            system_memory = hardware_constraints.get('system_memory_gb', 16)
            if value > system_memory * 0.8:
                suggestions.append(f"Limite mémoire proche du maximum système ({system_memory}GB)")
        
        return suggestions
    
    def _get_context_adaptations(self, context: Dict[str, Any]) -> List[str]:
        """Retourne les adaptations appliquées selon le contexte."""
        adaptations = []
        
        if context.get('mesh_size') == 'large':
            adaptations.append("Règles adaptées pour gros maillages")
        
        if context.get('operation_type') == 'real_time':
            adaptations.append("Optimisation pour traitement temps réel")
        
        gpu_memory = context.get('hardware_constraints', {}).get('gpu_memory_gb', 8)
        if gpu_memory < 4:
            adaptations.append("Contraintes adaptées pour GPU à mémoire limitée")
        
        return adaptations
    
    def analyze_validation_patterns(self) -> Dict[str, Any]:
        """Analyse les patterns de validation pour optimiser les règles."""
        if not self.context_history:
            return {'message': 'Pas assez de données d\'historique'}
        
        return self.pattern_analyzer.analyze_patterns(self.context_history)

class PatternAnalyzer:
    """Analyseur de patterns pour optimiser la validation."""
    
    def analyze_patterns(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse les patterns dans l'historique de validation."""
        if len(history) < 5:
            return {'message': 'Données insuffisantes pour l\'analyse'}
        
        # Analyser les paramètres les plus problématiques
        error_counts = {}
        context_patterns = {}
        
        for entry in history:
            context_type = self._categorize_context(entry['context'])
            if context_type not in context_patterns:
                context_patterns[context_type] = []
            context_patterns[context_type].append(entry)
        
        # Identifier les optimisations possibles
        optimizations = []
        
        for context_type, entries in context_patterns.items():
            if len(entries) >= 3:
                common_adaptations = self._find_common_adaptations(entries)
                if common_adaptations:
                    optimizations.append({
                        'context': context_type,
                        'suggested_defaults': common_adaptations
                    })
        
        return {
            'total_validations': len(history),
            'context_patterns': {k: len(v) for k, v in context_patterns.items()},
            'suggested_optimizations': optimizations,
            'analysis_quality': 'good' if len(history) > 20 else 'limited'
        }
    
    def _categorize_context(self, context: Dict[str, Any]) -> str:
        """Catégorise un contexte d'utilisation."""
        mesh_size = context.get('mesh_size', 'unknown')
        operation_type = context.get('operation_type', 'unknown')
        return f"{mesh_size}_{operation_type}"
    
    def _find_common_adaptations(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Trouve les adaptations communes dans un groupe d'entrées."""
        # Analyser les paramètres les plus fréquemment utilisés
        common_params = {}
        param_counts = {}
        
        for entry in entries:
            for param, value in entry['validated'].items():
                if param not in param_counts:
                    param_counts[param] = []
                param_counts[param].append(value)
        
        # Calculer les valeurs moyennes/modales
        for param, values in param_counts.items():
            if len(values) >= len(entries) * 0.7:  # 70% des cas
                if isinstance(values[0], (int, float)):
                    common_params[param] = sum(values) / len(values)
                else:
                    # Pour les valeurs non numériques, prendre la plus fréquente
                    from collections import Counter
                    common_params[param] = Counter(values).most_common(1)[0][0]
        
        return common_params

# Instance globale du validateur avancé
advanced_validator = AdvancedValidator()