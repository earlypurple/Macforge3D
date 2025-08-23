import numpy as np
from typing import Dict, Any, List, Tuple, Callable, Optional
import logging
from scipy.optimize import minimize, differential_evolution
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import joblib

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationOptimizer:
    """Optimiseur de paramètres de simulation utilisant différentes stratégies."""
    
    def __init__(
        self,
        simulation_runner: Callable[[Dict[str, Any]], Dict[str, Any]],
        parameters_space: Dict[str, Tuple[float, float]],
        objective: str,
        maximize: bool = False,
        method: str = "bayesian",
        n_jobs: int = -1
    ):
        """
        Initialise l'optimiseur.
        
        Args:
            simulation_runner: Fonction qui exécute la simulation
            parameters_space: Espace des paramètres {nom: (min, max)}
            objective: Métrique à optimiser
            maximize: True pour maximiser, False pour minimiser
            method: Méthode d'optimisation ('bayesian', 'differential_evolution', 'nelder-mead')
            n_jobs: Nombre de jobs parallèles (-1 pour utiliser tous les CPU)
        """
        self.simulation_runner = simulation_runner
        self.parameters_space = parameters_space
        self.objective = objective
        self.maximize = maximize
        self.method = method
        self.n_jobs = n_jobs
        
        # Historique des essais
        self.history: List[Dict[str, Any]] = []
        
        # Pour l'optimisation bayésienne
        self.gp = None
        if method == "bayesian":
            kernel = ConstantKernel(1.0) * RBF([1.0] * len(parameters_space))
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10,
                random_state=42
            )
            
    def optimize(
        self,
        n_trials: int = 50,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Lance l'optimisation.
        
        Args:
            n_trials: Nombre d'essais
            callback: Fonction appelée après chaque essai
            
        Returns:
            Meilleurs paramètres trouvés
        """
        logger.info(f"Démarrage de l'optimisation avec méthode: {self.method}")
        
        try:
            if self.method == "bayesian":
                return self._bayesian_optimization(n_trials, callback)
            elif self.method == "differential_evolution":
                return self._differential_evolution(callback)
            else:  # nelder-mead
                return self._nelder_mead(callback)
                
        except Exception as e:
            logger.error(f"Erreur lors de l'optimisation: {str(e)}")
            raise
            
    def _bayesian_optimization(
        self,
        n_trials: int,
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Optimisation bayésienne."""
        param_names = list(self.parameters_space.keys())
        bounds = np.array([self.parameters_space[p] for p in param_names])
        
        # Échantillonnage initial
        n_initial = min(5, n_trials)
        X_init = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_initial, len(param_names))
        )
        y_init = np.array([
            self._evaluate(dict(zip(param_names, x)), callback)
            for x in X_init
        ])
        
        X = X_init
        y = y_init
        
        for i in range(n_initial, n_trials):
            # Mettre à jour le modèle GP
            self.gp.fit(X, y)
            
            # Trouver le prochain point à évaluer
            next_x = self._propose_next_point(X, y, bounds, param_names)
            
            # Évaluer le point
            next_y = self._evaluate(dict(zip(param_names, next_x)), callback)
            
            # Mettre à jour les données
            X = np.vstack((X, next_x))
            y = np.append(y, next_y)
            
        # Retourner les meilleurs paramètres
        best_idx = np.argmax(y) if self.maximize else np.argmin(y)
        return dict(zip(param_names, X[best_idx]))
        
    def _differential_evolution(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Optimisation par évolution différentielle."""
        param_names = list(self.parameters_space.keys())
        bounds = [self.parameters_space[p] for p in param_names]
        
        result = differential_evolution(
            lambda x: self._evaluate(dict(zip(param_names, x)), callback),
            bounds,
            maxiter=100,
            popsize=15,
            workers=self.n_jobs
        )
        
        return dict(zip(param_names, result.x))
        
    def _nelder_mead(
        self,
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> Dict[str, Any]:
        """Optimisation par la méthode Nelder-Mead."""
        param_names = list(self.parameters_space.keys())
        x0 = np.array([
            np.mean(self.parameters_space[p])
            for p in param_names
        ])
        
        result = minimize(
            lambda x: self._evaluate(dict(zip(param_names, x)), callback),
            x0,
            method="Nelder-Mead",
            options={"maxiter": 200}
        )
        
        return dict(zip(param_names, result.x))
        
    def _evaluate(
        self,
        parameters: Dict[str, Any],
        callback: Optional[Callable[[Dict[str, Any]], None]]
    ) -> float:
        """Évalue un ensemble de paramètres."""
        try:
            # Exécuter la simulation
            results = self.simulation_runner(parameters)
            
            # Extraire la métrique objective
            value = self._extract_objective(results)
            
            # Enregistrer l'essai
            trial = {
                "parameters": parameters,
                "results": results,
                "objective_value": value
            }
            self.history.append(trial)
            
            # Appeler le callback si fourni
            if callback:
                callback(trial)
                
            return -value if self.maximize else value
            
        except Exception as e:
            logger.error(
                f"Erreur lors de l'évaluation des paramètres {parameters}: {str(e)}"
            )
            return float("inf")
            
    def _extract_objective(self, results: Dict[str, Any]) -> float:
        """Extrait la valeur objective des résultats."""
        keys = self.objective.split(".")
        value = results
        for key in keys:
            value = value[key]
        return float(value)
        
    def _propose_next_point(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: np.ndarray,
        param_names: List[str]
    ) -> np.ndarray:
        """Propose le prochain point à évaluer (optimisation bayésienne)."""
        def acquisition(x):
            x = x.reshape(1, -1)
            mean, std = self.gp.predict(x, return_std=True)
            
            if self.maximize:
                z = (mean - np.max(y)) / std
            else:
                z = (np.min(y) - mean) / std
                
            return norm.cdf(z)
            
        # Échantillonnage aléatoire + optimisation locale
        n_random = 1000
        X_random = np.random.uniform(
            bounds[:, 0],
            bounds[:, 1],
            size=(n_random, len(param_names))
        )
        acquisition_values = [acquisition(x) for x in X_random]
        
        # Sélectionner les meilleurs points et optimiser localement
        n_best = 5
        best_indices = np.argsort(acquisition_values)[-n_best:]
        best_points = X_random[best_indices]
        
        results = []
        for x0 in best_points:
            res = minimize(
                lambda x: -acquisition(x),
                x0,
                bounds=bounds,
                method="L-BFGS-B"
            )
            results.append((res.fun, res.x))
            
        # Retourner le meilleur point
        return min(results, key=lambda x: x[0])[1]
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Retourne un résumé de l'optimisation."""
        objective_values = [
            trial["objective_value"]
            for trial in self.history
        ]
        
        return {
            "best_value": max(objective_values) if self.maximize else min(objective_values),
            "n_trials": len(self.history),
            "history": self.history,
            "parameters_space": self.parameters_space,
            "method": self.method
        }
        
    def save_optimizer_state(self, filepath: str):
        """Sauvegarde l'état de l'optimiseur."""
        state = {
            "history": self.history,
            "parameters_space": self.parameters_space,
            "objective": self.objective,
            "maximize": self.maximize,
            "method": self.method
        }
        
        if self.method == "bayesian" and self.gp is not None:
            state["gp_state"] = self.gp
            
        joblib.dump(state, filepath)
        
    @classmethod
    def load_optimizer_state(
        cls,
        filepath: str,
        simulation_runner: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "SimulationOptimizer":
        """Charge un état d'optimiseur sauvegardé."""
        state = joblib.load(filepath)
        
        optimizer = cls(
            simulation_runner=simulation_runner,
            parameters_space=state["parameters_space"],
            objective=state["objective"],
            maximize=state["maximize"],
            method=state["method"]
        )
        
        optimizer.history = state["history"]
        if "gp_state" in state and optimizer.method == "bayesian":
            optimizer.gp = state["gp_state"]
            
        return optimizer
