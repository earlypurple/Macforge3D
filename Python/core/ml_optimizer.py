"""
Optimisation des paramètres de simulation par Machine Learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import joblib
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import cross_val_score
import optuna
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationGoal:
    """Objectif d'optimisation."""

    metric_name: str
    direction: str  # 'minimize' ou 'maximize'
    weight: float = 1.0
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None


@dataclass
class OptimizationResult:
    """Résultat d'une optimisation."""

    parameters: Dict[str, Any]
    predicted_metrics: Dict[str, float]
    confidence: float
    optimization_time: float
    model_performance: Dict[str, float]


class MLOptimizer:
    """Optimiseur de paramètres basé sur ML."""

    def __init__(
        self,
        parameter_space: Dict[str, Tuple],
        goals: List[OptimizationGoal],
        history_file: str,
    ):
        self.parameter_space = parameter_space
        self.goals = goals
        self.history_file = history_file

        # Modèles
        self.models = {}
        self.scalers = {}

        # Historique
        self.history = self._load_history()

        # Performance des modèles
        self.model_performance = {}

    def add_result(self, parameters: Dict[str, Any], metrics: Dict[str, float]):
        """Ajoute un résultat à l'historique."""
        entry = {"timestamp": datetime.now(), **parameters, **metrics}
        self.history = pd.concat([self.history, pd.DataFrame([entry])])
        self.history.to_csv(self.history_file, index=False)

        # Réentraîner les modèles
        self._train_models()

    def optimize(self, n_trials: int = 100, timeout: int = 600) -> OptimizationResult:
        """Optimise les paramètres."""
        if len(self.history) < 10:
            logger.warning("Pas assez de données pour l'optimisation")
            return self._get_default_parameters()

        try:
            # Créer l'étude Optuna
            study = optuna.create_study(
                direction="maximize", sampler=optuna.samplers.TPESampler()
            )

            # Lancer l'optimisation
            study.optimize(self._objective, n_trials=n_trials, timeout=timeout)

            # Récupérer les meilleurs paramètres
            best_params = study.best_params

            # Prédire les métriques
            predicted_metrics = {}
            for metric in self.goals:
                if metric.metric_name in self.models:
                    value = self._predict_metric(metric.metric_name, best_params)
                    predicted_metrics[metric.metric_name] = value

            # Calculer la confiance
            confidence = self._calculate_confidence(best_params)

            return OptimizationResult(
                parameters=best_params,
                predicted_metrics=predicted_metrics,
                confidence=confidence,
                optimization_time=study.duration,
                model_performance=self.model_performance,
            )

        except Exception as e:
            logger.error(f"Erreur d'optimisation: {str(e)}")
            return self._get_default_parameters()

    def _objective(self, trial):
        """Fonction objectif pour Optuna."""
        # Générer les paramètres
        params = {}
        for name, (param_type, bounds) in self.parameter_space.items():
            if param_type == float:
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])
            elif param_type == int:
                params[name] = trial.suggest_int(name, bounds[0], bounds[1])
            elif param_type == str:
                params[name] = trial.suggest_categorical(name, bounds)

        # Calculer le score
        score = 0.0
        for goal in self.goals:
            if goal.metric_name not in self.models:
                continue

            predicted = self._predict_metric(goal.metric_name, params)

            # Appliquer les contraintes
            if goal.constraint_min is not None and predicted < goal.constraint_min:
                return float("-inf")

            if goal.constraint_max is not None and predicted > goal.constraint_max:
                return float("-inf")

            # Calculer le score
            if goal.direction == "maximize":
                score += predicted * goal.weight
            else:
                score -= predicted * goal.weight

        return score

    def _train_models(self):
        """Entraîne les modèles de prédiction."""
        X = self._prepare_features(self.history)

        for goal in self.goals:
            metric = goal.metric_name
            if metric not in self.history.columns:
                continue

            y = self.history[metric].values

            # Normaliser
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[metric] = scaler

            # Sélectionner et entraîner le meilleur modèle
            models = {
                "rf": RandomForestRegressor(n_estimators=100, random_state=42),
                "gp": GaussianProcessRegressor(
                    kernel=C(1.0) * RBF([1.0] * X.shape[1]), random_state=42
                ),
            }

            best_score = float("-inf")
            best_model = None

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
                    score = np.mean(scores)

                    if score > best_score:
                        best_score = score
                        best_model = model

                except Exception as e:
                    logger.warning(f"Erreur validation croisée {name}: {str(e)}")

            if best_model is not None:
                # Entraîner le meilleur modèle
                best_model.fit(X_scaled, y)
                self.models[metric] = best_model
                self.model_performance[metric] = best_score

    def _predict_metric(self, metric: str, parameters: Dict[str, Any]) -> float:
        """Prédit une métrique pour des paramètres."""
        if metric not in self.models:
            return 0.0

        # Préparer les features
        X = self._prepare_features(pd.DataFrame([parameters]))
        X_scaled = self.scalers[metric].transform(X)

        # Prédire
        if isinstance(self.models[metric], GaussianProcessRegressor):
            pred, std = self.models[metric].predict(X_scaled, return_std=True)
            return float(pred[0])
        else:
            return float(self.models[metric].predict(X_scaled)[0])

    def _calculate_confidence(self, parameters: Dict[str, Any]) -> float:
        """Calcule la confiance dans les prédictions."""
        if not self.models:
            return 0.0

        # Distance aux données d'entraînement
        X = self._prepare_features(self.history)
        X_query = self._prepare_features(pd.DataFrame([parameters]))

        distances = []
        for metric, scaler in self.scalers.items():
            X_scaled = scaler.transform(X)
            X_query_scaled = scaler.transform(X_query)

            # Distance euclidienne minimale
            min_dist = np.min(np.linalg.norm(X_scaled - X_query_scaled, axis=1))
            distances.append(min_dist)

        # Performance des modèles
        performances = list(self.model_performance.values())

        # Combiner distance et performance
        avg_distance = np.mean(distances)
        avg_performance = np.mean(performances)

        # Confiance inversement proportionnelle à la distance
        # et proportionnelle à la performance
        confidence = avg_performance * np.exp(-avg_distance)

        return float(confidence)

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prépare les features pour les modèles."""
        features = []

        for name, (param_type, _) in self.parameter_space.items():
            if name not in df.columns:
                continue

            if param_type in [float, int]:
                features.append(df[name].values.reshape(-1, 1))
            elif param_type == str:
                # One-hot encoding
                dummies = pd.get_dummies(df[name], prefix=name)
                features.append(dummies.values)

        return np.hstack(features)

    def _load_history(self) -> pd.DataFrame:
        """Charge l'historique des simulations."""
        try:
            if os.path.exists(self.history_file):
                return pd.read_csv(self.history_file)
        except Exception as e:
            logger.error(f"Erreur chargement historique: {str(e)}")

        return pd.DataFrame()

    def _get_default_parameters(self) -> OptimizationResult:
        """Retourne des paramètres par défaut."""
        params = {}
        for name, (param_type, bounds) in self.parameter_space.items():
            if param_type in [float, int]:
                params[name] = sum(bounds) / 2
            elif param_type == str:
                params[name] = bounds[0]

        return OptimizationResult(
            parameters=params,
            predicted_metrics={},
            confidence=0.0,
            optimization_time=0.0,
            model_performance={},
        )
