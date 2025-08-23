import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error
from scipy import stats

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationComparator:
    """Classe pour comparer les résultats de différentes simulations."""
    
    def __init__(self):
        """Initialise le comparateur de simulations."""
        pass
        
    def compare_results(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare deux ensembles de résultats de simulation.
        
        Args:
            results1: Premier ensemble de résultats
            results2: Deuxième ensemble de résultats
            metrics: Liste des métriques à comparer (optionnel)
            
        Returns:
            Dictionnaire contenant les résultats de la comparaison
        """
        try:
            # Si aucune métrique n'est spécifiée, comparer toutes les métriques communes
            if metrics is None:
                metrics = self._find_common_metrics(results1, results2)
                
            comparison = {
                "metrics": {},
                "statistical_tests": {},
                "differences": {},
                "correlations": {}
            }
            
            # Comparer chaque métrique
            for metric in metrics:
                metric_comparison = self._compare_metric(
                    self._extract_metric(results1, metric),
                    self._extract_metric(results2, metric)
                )
                comparison["metrics"][metric] = metric_comparison
                
            # Tests statistiques globaux
            comparison["statistical_tests"] = self._perform_statistical_tests(
                results1,
                results2,
                metrics
            )
            
            # Différences structurelles
            comparison["differences"] = self._find_structural_differences(
                results1,
                results2
            )
            
            # Corrélations entre les résultats
            comparison["correlations"] = self._compute_correlations(
                results1,
                results2,
                metrics
            )
            
            return comparison
            
        except Exception as e:
            logger.error(f"Erreur lors de la comparaison: {str(e)}")
            raise
            
    def compare_multiple_results(
        self,
        results_list: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare plusieurs ensembles de résultats.
        
        Args:
            results_list: Liste des résultats à comparer
            metrics: Liste des métriques à comparer (optionnel)
            
        Returns:
            Dictionnaire contenant les résultats de la comparaison
        """
        try:
            if len(results_list) < 2:
                raise ValueError("Au moins deux ensembles de résultats sont nécessaires")
                
            # Si aucune métrique n'est spécifiée, trouver les métriques communes
            if metrics is None:
                metrics = self._find_common_metrics_multiple(results_list)
                
            comparison = {
                "pairwise_comparisons": [],
                "global_statistics": {},
                "clustering": {},
                "trends": {}
            }
            
            # Comparaisons deux à deux
            for i in range(len(results_list)):
                for j in range(i + 1, len(results_list)):
                    pair_comparison = self.compare_results(
                        results_list[i],
                        results_list[j],
                        metrics
                    )
                    comparison["pairwise_comparisons"].append({
                        "pair": (i, j),
                        "results": pair_comparison
                    })
                    
            # Statistiques globales
            comparison["global_statistics"] = self._compute_global_statistics(
                results_list,
                metrics
            )
            
            # Clustering des résultats similaires
            comparison["clustering"] = self._cluster_results(
                results_list,
                metrics
            )
            
            # Analyse des tendances
            comparison["trends"] = self._analyze_trends(
                results_list,
                metrics
            )
            
            return comparison
            
        except Exception as e:
            logger.error(
                f"Erreur lors de la comparaison multiple: {str(e)}"
            )
            raise
            
    def _compare_metric(
        self,
        values1: np.ndarray,
        values2: np.ndarray
    ) -> Dict[str, Any]:
        """Compare deux ensembles de valeurs pour une métrique."""
        comparison = {
            "difference_mean": float(np.mean(values2 - values1)),
            "difference_std": float(np.std(values2 - values1)),
            "rmse": float(np.sqrt(mean_squared_error(values1, values2))),
            "correlation": float(np.corrcoef(values1, values2)[0, 1]),
            "relative_difference": float(
                np.mean(np.abs(values2 - values1) / np.abs(values1))
            )
        }
        
        # Test statistique
        t_stat, p_value = stats.ttest_ind(values1, values2)
        comparison["t_test"] = {
            "statistic": float(t_stat),
            "p_value": float(p_value)
        }
        
        return comparison
        
    def _find_common_metrics(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any]
    ) -> List[str]:
        """Trouve les métriques communes entre deux ensembles de résultats."""
        metrics1 = self._find_numeric_metrics(results1)
        metrics2 = self._find_numeric_metrics(results2)
        return list(set(metrics1) & set(metrics2))
        
    def _find_common_metrics_multiple(
        self,
        results_list: List[Dict[str, Any]]
    ) -> List[str]:
        """Trouve les métriques communes entre plusieurs ensembles de résultats."""
        all_metrics = [
            set(self._find_numeric_metrics(results))
            for results in results_list
        ]
        return list(set.intersection(*all_metrics))
        
    def _find_numeric_metrics(
        self,
        results: Dict[str, Any],
        parent_key: str = ""
    ) -> List[str]:
        """Trouve toutes les métriques numériques dans les résultats."""
        metrics = []
        for key, value in results.items():
            current_key = f"{parent_key}.{key}" if parent_key else key
            if isinstance(value, (int, float, np.ndarray)):
                metrics.append(current_key)
            elif isinstance(value, dict):
                metrics.extend(
                    self._find_numeric_metrics(value, current_key)
                )
        return metrics
        
    def _extract_metric(
        self,
        results: Dict[str, Any],
        metric: str
    ) -> np.ndarray:
        """Extrait les valeurs d'une métrique des résultats."""
        keys = metric.split(".")
        value = results
        for key in keys:
            value = value[key]
        return np.array(value)
        
    def _perform_statistical_tests(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Effectue des tests statistiques entre les résultats."""
        tests = {}
        
        for metric in metrics:
            values1 = self._extract_metric(results1, metric)
            values2 = self._extract_metric(results2, metric)
            
            # Test de normalité
            _, norm_p1 = stats.normaltest(values1)
            _, norm_p2 = stats.normaltest(values2)
            
            # Choisir le test approprié
            if norm_p1 > 0.05 and norm_p2 > 0.05:
                # Données normalement distribuées
                stat, p_value = stats.ttest_ind(values1, values2)
                test_name = "t-test"
            else:
                # Données non normalement distribuées
                stat, p_value = stats.mannwhitneyu(values1, values2)
                test_name = "mann-whitney"
                
            tests[metric] = {
                "test": test_name,
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < 0.05
            }
            
        return tests
        
    def _find_structural_differences(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Trouve les différences structurelles entre les résultats."""
        differences = {
            "missing_keys": {
                "results1": [],
                "results2": []
            },
            "type_differences": [],
            "shape_differences": []
        }
        
        self._compare_structures(
            results1,
            results2,
            differences,
            path=""
        )
        
        return differences
        
    def _compare_structures(
        self,
        d1: Dict[str, Any],
        d2: Dict[str, Any],
        differences: Dict[str, List],
        path: str
    ):
        """Compare récursivement les structures de deux dictionnaires."""
        keys1 = set(d1.keys())
        keys2 = set(d2.keys())
        
        # Clés manquantes
        for key in keys1 - keys2:
            differences["missing_keys"]["results2"].append(
                f"{path}.{key}" if path else key
            )
        for key in keys2 - keys1:
            differences["missing_keys"]["results1"].append(
                f"{path}.{key}" if path else key
            )
            
        # Comparer les valeurs communes
        for key in keys1 & keys2:
            current_path = f"{path}.{key}" if path else key
            value1 = d1[key]
            value2 = d2[key]
            
            if type(value1) != type(value2):
                differences["type_differences"].append({
                    "path": current_path,
                    "type1": str(type(value1)),
                    "type2": str(type(value2))
                })
            elif isinstance(value1, dict):
                self._compare_structures(
                    value1,
                    value2,
                    differences,
                    current_path
                )
            elif isinstance(value1, np.ndarray):
                if value1.shape != value2.shape:
                    differences["shape_differences"].append({
                        "path": current_path,
                        "shape1": value1.shape,
                        "shape2": value2.shape
                    })
                    
    def _compute_correlations(
        self,
        results1: Dict[str, Any],
        results2: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Calcule les corrélations entre les métriques."""
        correlations = {}
        
        for metric in metrics:
            values1 = self._extract_metric(results1, metric)
            values2 = self._extract_metric(results2, metric)
            
            # Corrélation de Pearson
            pearson_corr, pearson_p = stats.pearsonr(values1, values2)
            
            # Corrélation de Spearman
            spearman_corr, spearman_p = stats.spearmanr(values1, values2)
            
            correlations[metric] = {
                "pearson": {
                    "correlation": float(pearson_corr),
                    "p_value": float(pearson_p)
                },
                "spearman": {
                    "correlation": float(spearman_corr),
                    "p_value": float(spearman_p)
                }
            }
            
        return correlations
        
    def _compute_global_statistics(
        self,
        results_list: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Calcule des statistiques globales sur plusieurs résultats."""
        statistics = {}
        
        for metric in metrics:
            values = np.array([
                self._extract_metric(results, metric)
                for results in results_list
            ])
            
            statistics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "variance": float(np.var(values))
            }
            
        return statistics
        
    def _cluster_results(
        self,
        results_list: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Regroupe les résultats similaires."""
        # Exemple simple de clustering basé sur la distance euclidienne
        clusters = {}
        
        # À implémenter : algorithme de clustering plus sophistiqué
        return clusters
        
    def _analyze_trends(
        self,
        results_list: List[Dict[str, Any]],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Analyse les tendances dans les résultats."""
        trends = {}
        
        for metric in metrics:
            values = np.array([
                np.mean(self._extract_metric(results, metric))
                for results in results_list
            ])
            
            # Tendance linéaire
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                x,
                values
            )
            
            trends[metric] = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_squared": float(r_value ** 2),
                "p_value": float(p_value),
                "std_err": float(std_err)
            }
            
        return trends
