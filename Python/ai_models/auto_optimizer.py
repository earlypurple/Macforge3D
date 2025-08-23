"""
Module d'optimisation automatique des paramètres par machine learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import optuna
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
import joblib
import wandb
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration pour l'optimisation."""
    target_metric: str = "processing_time"  # ou "memory_usage", "quality"
    n_trials: int = 100
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    use_wandb: bool = True
    wandb_project: str = "macforge3d-optimization"
    model_save_path: Path = Path("/tmp/macforge3d_models")

class PerformancePredictor(nn.Module):
    """Réseau de neurones pour prédire les performances."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class AutoOptimizer:
    """Optimiseur automatique des paramètres."""
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.param_importance = {}
        
        # Créer le dossier de sauvegarde
        self.config.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Initialiser W&B si activé
        if self.config.use_wandb:
            wandb.init(project=self.config.wandb_project)
            
    def collect_training_data(
        self,
        performance_logs: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prépare les données d'entraînement.
        
        Args:
            performance_logs: Logs de performance
            
        Returns:
            Features et labels pour l'entraînement
        """
        # Convertir en DataFrame
        df = pd.DataFrame(performance_logs)
        
        # Extraire les features
        feature_cols = [
            'cache_size', 'num_workers', 'compression_level',
            'gpu_fraction', 'batch_size'
        ]
        X = df[feature_cols].values
        
        # Extraire la métrique cible
        y = df[self.config.target_metric].values
        
        return X, y
        
    def train_predictor(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """
        Entraîne le modèle prédictif.
        
        Args:
            X: Features
            y: Labels
        """
        # Normaliser les données
        X_scaled = self.scaler.fit_transform(X)
        
        # Diviser en train/val
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2
        )
        
        # Convertir en tenseurs
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val)
        
        # Créer le modèle
        self.model = PerformancePredictor(X.shape[1])
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        criterion = nn.MSELoss()
        
        # Entraînement
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(self.config.n_epochs):
            self.model.train()
            
            # Train
            optimizer.zero_grad()
            y_pred = self.model(X_train)
            loss = criterion(y_pred, y_train.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                y_val_pred = self.model(X_val)
                val_loss = criterion(y_val_pred, y_val.unsqueeze(1))
                
            # Logging
            metrics = {
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
                'epoch': epoch
            }
            
            if self.config.use_wandb:
                wandb.log(metrics)
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(
                    self.model.state_dict(),
                    self.config.model_save_path / "best_model.pt"
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping à l'epoch {epoch}")
                    break
                    
        # Calculer l'importance des paramètres
        self._calculate_feature_importance(X_scaled, y)
        
    def _calculate_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> None:
        """Calcule l'importance des features."""
        feature_importance = {}
        base_pred = self.predict(X)
        base_mse = np.mean((base_pred - y) ** 2)
        
        for i in range(X.shape[1]):
            X_shuffled = X.copy()
            X_shuffled[:, i] = np.random.permutation(X_shuffled[:, i])
            shuffled_pred = self.predict(X_shuffled)
            shuffled_mse = np.mean((shuffled_pred - y) ** 2)
            
            # L'importance est la différence relative de MSE
            importance = (shuffled_mse - base_mse) / base_mse
            feature_importance[f"feature_{i}"] = importance
            
        self.param_importance = feature_importance
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait une prédiction."""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.numpy().flatten()
        
    def optimize_parameters(
        self,
        param_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Optimise les paramètres avec Optuna.
        
        Args:
            param_ranges: Plages de valeurs pour chaque paramètre
            
        Returns:
            Paramètres optimaux
        """
        def objective(trial):
            # Générer des paramètres
            params = {
                name: trial.suggest_float(name, low, high)
                for name, (low, high) in param_ranges.items()
            }
            
            # Convertir en features
            X = np.array([[v for v in params.values()]])
            
            # Prédire la performance
            prediction = self.predict(X)[0]
            
            return prediction
            
        # Créer l'étude Optuna
        study = optuna.create_study(
            direction="minimize" if self.config.target_metric != "quality" else "maximize"
        )
        
        # Optimiser
        study.optimize(objective, n_trials=self.config.n_trials)
        
        return study.best_params
        
    def save_optimizer(self, path: Path):
        """Sauvegarde l'optimiseur."""
        path = Path(path)
        
        # Sauvegarder le modèle
        torch.save(self.model.state_dict(), path / "model.pt")
        
        # Sauvegarder le scaler
        joblib.dump(self.scaler, path / "scaler.pkl")
        
        # Sauvegarder les méta-données
        metadata = {
            "param_importance": self.param_importance,
            "config": vars(self.config)
        }
        
        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
    def load_optimizer(self, path: Path):
        """Charge l'optimiseur."""
        path = Path(path)
        
        # Charger le modèle
        model_state = torch.load(path / "model.pt")
        self.model = PerformancePredictor(len(model_state))
        self.model.load_state_dict(model_state)
        
        # Charger le scaler
        self.scaler = joblib.load(path / "scaler.pkl")
        
        # Charger les méta-données
        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)
            self.param_importance = metadata["param_importance"]
            
    def suggest_improvements(
        self,
        current_params: Dict[str, float]
    ) -> List[str]:
        """
        Suggère des améliorations de paramètres.
        
        Args:
            current_params: Paramètres actuels
            
        Returns:
            Liste de suggestions
        """
        suggestions = []
        
        # Prédire la performance actuelle
        current_X = np.array([[v for v in current_params.values()]])
        current_perf = self.predict(current_X)[0]
        
        # Tester des variations
        for param_name, current_value in current_params.items():
            # Tester une augmentation
            test_params = current_params.copy()
            test_params[param_name] *= 1.2
            test_X = np.array([[v for v in test_params.values()]])
            increased_perf = self.predict(test_X)[0]
            
            # Tester une diminution
            test_params = current_params.copy()
            test_params[param_name] *= 0.8
            test_X = np.array([[v for v in test_params.values()]])
            decreased_perf = self.predict(test_X)[0]
            
            # Comparer les performances
            best_perf = min(current_perf, increased_perf, decreased_perf)
            if best_perf < current_perf:
                if best_perf == increased_perf:
                    suggestions.append(
                        f"Augmenter {param_name} de 20% pour améliorer les performances"
                    )
                else:
                    suggestions.append(
                        f"Réduire {param_name} de 20% pour améliorer les performances"
                    )
                    
        return suggestions
