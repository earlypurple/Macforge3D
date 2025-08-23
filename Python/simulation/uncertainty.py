import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from scipy import stats
from scipy.stats import norm, multivariate_normal
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UncertaintyAnalyzer:
    """Analyseur d'incertitudes pour les simulations."""
    
    def __init__(
        self,
        simulation_runner: Callable[[Dict[str, Any]], Dict[str, Any]],
        parameter_uncertainties: Dict[str, Tuple[float, float]]
    ):
        """
        Initialise l'analyseur.
        
        Args:
            simulation_runner: Fonction qui exécute la simulation
            parameter_uncertainties: Incertitudes des paramètres {nom: (moyenne, écart-type)}
        """
        self.simulation_runner = simulation_runner
        self.parameter_uncertainties = parameter_uncertainties
        self.samples: List[Dict[str, Any]] = []
        self.results: List[Dict[str, Any]] = []
        
    def run_monte_carlo(
        self,
        n_samples: int,
        metrics: List[str],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Exécute une analyse Monte Carlo.
        
        Args:
            n_samples: Nombre d'échantillons
            metrics: Liste des métriques à analyser
            callback: Fonction appelée après chaque simulation
            
        Returns:
            Résultats de l'analyse
        """
        logger.info(f"Démarrage de l'analyse Monte Carlo avec {n_samples} échantillons")
        
        try:
            # Générer les échantillons
            self.samples = self._generate_samples(n_samples)
            
            # Exécuter les simulations
            self.results = []
            for i, params in enumerate(self.samples):
                try:
                    result = self.simulation_runner(params)
                    self.results.append(result)
                    
                    if callback:
                        callback({
                            "sample_index": i,
                            "parameters": params,
                            "results": result
                        })
                        
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la simulation {i}: {str(e)}"
                    )
                    
            # Analyser les résultats
            analysis = self._analyze_results(metrics)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse Monte Carlo: {str(e)}")
            raise
            
    def _generate_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Génère des échantillons de paramètres."""
        param_names = list(self.parameter_uncertainties.keys())
        means = np.array([
            self.parameter_uncertainties[p][0]
            for p in param_names
        ])
        stds = np.array([
            self.parameter_uncertainties[p][1]
            for p in param_names
        ])
        
        # Générer les échantillons
        samples = np.random.normal(
            means,
            stds,
            size=(n_samples, len(param_names))
        )
        
        return [
            dict(zip(param_names, sample))
            for sample in samples
        ]
        
    def _analyze_results(self, metrics: List[str]) -> Dict[str, Any]:
        """Analyse les résultats des simulations."""
        analysis = {
            "n_samples": len(self.results),
            "metrics": {},
            "correlations": {},
            "sensitivity": {},
            "principal_components": {}
        }
        
        # Extraire les valeurs des métriques
        metric_values = {}
        for metric in metrics:
            values = []
            for result in self.results:
                try:
                    value = self._extract_metric(result, metric)
                    values.append(value)
                except:
                    continue
            metric_values[metric] = np.array(values)
            
        # Calculer les statistiques pour chaque métrique
        for metric, values in metric_values.items():
            analysis["metrics"][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "skewness": float(stats.skew(values)),
                "kurtosis": float(stats.kurtosis(values))
            }
            
            # Intervalle de confiance
            ci = stats.norm.interval(
                0.95,
                loc=np.mean(values),
                scale=stats.sem(values)
            )
            analysis["metrics"][metric]["ci_95"] = {
                "lower": float(ci[0]),
                "upper": float(ci[1])
            }
            
        # Calculer les corrélations entre les métriques
        correlation_matrix = np.zeros((len(metrics), len(metrics)))
        for i, metric1 in enumerate(metrics):
            for j, metric2 in enumerate(metrics):
                values1 = metric_values[metric1]
                values2 = metric_values[metric2]
                correlation_matrix[i, j] = np.corrcoef(values1, values2)[0, 1]
                
        analysis["correlations"] = {
            "matrix": correlation_matrix.tolist(),
            "metrics": metrics
        }
        
        # Analyse de sensibilité
        for metric in metrics:
            sensitivities = {}
            values = metric_values[metric]
            
            for param in self.parameter_uncertainties:
                param_values = np.array([
                    sample[param]
                    for sample in self.samples
                ])
                
                # Corrélation de Spearman
                correlation, p_value = stats.spearmanr(
                    param_values,
                    values
                )
                
                sensitivities[param] = {
                    "correlation": float(correlation),
                    "p_value": float(p_value)
                }
                
            analysis["sensitivity"][metric] = sensitivities
            
        # Analyse en composantes principales
        for metric in metrics:
            values = metric_values[metric].reshape(-1, 1)
            scaler = StandardScaler()
            values_scaled = scaler.fit_transform(values)
            
            pca = PCA()
            pca.fit(values_scaled)
            
            analysis["principal_components"][metric] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "components": pca.components_.tolist()
            }
            
        return analysis
        
    def _extract_metric(self, results: Dict[str, Any], metric: str) -> float:
        """Extrait une métrique des résultats."""
        keys = metric.split(".")
        value = results
        for key in keys:
            value = value[key]
        return float(value)
        
    def get_confidence_intervals(
        self,
        metric: str,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """Calcule les intervalles de confiance pour une métrique."""
        values = []
        for result in self.results:
            try:
                value = self._extract_metric(result, metric)
                values.append(value)
            except:
                continue
                
        values = np.array(values)
        
        # Intervalle de confiance paramétrique
        ci_param = stats.norm.interval(
            confidence_level,
            loc=np.mean(values),
            scale=stats.sem(values)
        )
        
        # Intervalle de confiance non paramétrique (bootstrap)
        n_bootstrap = 10000
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
            
        ci_bootstrap = np.percentile(
            bootstrap_means,
            [(1 - confidence_level) * 100 / 2, (1 + confidence_level) * 100 / 2]
        )
        
        return {
            "parametric": {
                "lower": float(ci_param[0]),
                "upper": float(ci_param[1])
            },
            "bootstrap": {
                "lower": float(ci_bootstrap[0]),
                "upper": float(ci_bootstrap[1])
            }
        }
        
    def save_analysis(self, filepath: str):
        """Sauvegarde les résultats de l'analyse."""
        state = {
            "parameter_uncertainties": self.parameter_uncertainties,
            "samples": self.samples,
            "results": self.results
        }
        joblib.dump(state, filepath)
        
    @classmethod
    def load_analysis(
        cls,
        filepath: str,
        simulation_runner: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "UncertaintyAnalyzer":
        """Charge une analyse sauvegardée."""
        state = joblib.load(filepath)
        
        analyzer = cls(
            simulation_runner=simulation_runner,
            parameter_uncertainties=state["parameter_uncertainties"]
        )
        
        analyzer.samples = state["samples"]
        analyzer.results = state["results"]
        
        return analyzer
