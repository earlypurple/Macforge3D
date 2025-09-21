"""
Script d'évaluation pour le MeshEnhancer.
"""

import torch
import numpy as np
import trimesh
from pathlib import Path
import logging
from typing import Dict, Any, List
from mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(
    original_mesh: trimesh.Trimesh, enhanced_mesh: trimesh.Trimesh
) -> Dict[str, float]:
    """Calcule les métriques d'évaluation."""
    metrics = {}

    # Distance de Hausdorff
    try:
        hausdorff = trimesh.comparison.mesh_to_mesh(original_mesh, enhanced_mesh)[0]
        metrics["hausdorff_distance"] = hausdorff
    except Exception as e:
        logger.warning(f"Erreur calcul Hausdorff: {e}")

    # Erreur quadratique moyenne des vertices
    try:
        mse = mean_squared_error(original_mesh.vertices, enhanced_mesh.vertices)
        metrics["vertex_mse"] = mse
    except Exception as e:
        logger.warning(f"Erreur calcul MSE: {e}")

    # Ratio de lissage
    try:
        original_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
            original_mesh, original_mesh.vertices
        )
        enhanced_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(
            enhanced_mesh, enhanced_mesh.vertices
        )

        smoothness_ratio = np.mean(np.abs(enhanced_curvature)) / np.mean(
            np.abs(original_curvature)
        )
        metrics["smoothness_ratio"] = smoothness_ratio
    except Exception as e:
        logger.warning(f"Erreur calcul lissage: {e}")

    # Préservation des détails
    try:
        original_detail = np.std(original_mesh.vertices, axis=0)
        enhanced_detail = np.std(enhanced_mesh.vertices, axis=0)

        detail_preservation = np.mean(enhanced_detail / original_detail)
        metrics["detail_preservation"] = detail_preservation
    except Exception as e:
        logger.warning(f"Erreur calcul préservation détails: {e}")

    return metrics


def plot_metrics(metrics_list: List[Dict[str, float]], output_dir: str):
    """Génère des graphiques pour les métriques."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Préparer les données
    metric_names = list(metrics_list[0].keys())
    values = {name: [] for name in metric_names}

    for metrics in metrics_list:
        for name in metric_names:
            if name in metrics:
                values[name].append(metrics[name])

    # Créer les graphiques
    for name in metric_names:
        plt.figure(figsize=(10, 6))
        plt.hist(values[name], bins=30)
        plt.title(f"Distribution of {name}")
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.savefig(output_dir / f"{name}_distribution.png")
        plt.close()

        # Statistiques
        data = np.array(values[name])
        stats = {
            "mean": np.mean(data),
            "std": np.std(data),
            "min": np.min(data),
            "max": np.max(data),
            "median": np.median(data),
        }

        with open(output_dir / f"{name}_stats.txt", "w") as f:
            for stat_name, value in stats.items():
                f.write(f"{stat_name}: {value:.6f}\n")


def evaluate_model(
    model: MeshEnhancer, test_dir: str, output_dir: str, device: str = "cuda"
):
    """Évalue le modèle sur un ensemble de test."""
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Charger les maillages de test
    test_meshes = []
    for mesh_path in test_dir.glob("*.ply"):
        try:
            mesh = trimesh.load(mesh_path)
            test_meshes.append((mesh_path.name, mesh))
        except Exception as e:
            logger.warning(f"Erreur chargement {mesh_path}: {e}")

    logger.info(f"Chargé {len(test_meshes)} maillages de test")

    # Évaluer chaque maillage
    all_metrics = []
    for name, mesh in test_meshes:
        logger.info(f"Traitement de {name}")

        try:
            # Améliorer le maillage
            enhanced_mesh = model.enhance_mesh(mesh)

            # Calculer les métriques
            metrics = compute_metrics(mesh, enhanced_mesh)
            all_metrics.append(metrics)

            # Sauvegarder le résultat
            enhanced_mesh.export(output_dir / f"enhanced_{name}")

            # Sauvegarder les métriques
            with open(output_dir / f"{name}_metrics.txt", "w") as f:
                for metric_name, value in metrics.items():
                    f.write(f"{metric_name}: {value:.6f}\n")

        except Exception as e:
            logger.error(f"Erreur traitement {name}: {e}")

    # Générer les graphiques
    plot_metrics(all_metrics, output_dir / "plots")

    # Calculer les moyennes
    mean_metrics = {}
    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics if metric in m]
        mean_metrics[metric] = np.mean(values)

    logger.info("Métriques moyennes:")
    for name, value in mean_metrics.items():
        logger.info(f"{name}: {value:.6f}")

    return mean_metrics


if __name__ == "__main__":
    # Configuration
    config = MeshEnhancementConfig(
        resolution_factor=1.5, smoothness_weight=0.3, detail_preservation=0.8
    )

    # Charger le modèle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeshEnhancer(config)
    model = model.to(device)

    # Charger les poids
    weights_path = "models/mesh_enhancer_best.pth"
    try:
        state_dict = torch.load(weights_path, map_location=device)
        model.encoder.load_state_dict(state_dict["encoder"])
        model.decoder.load_state_dict(state_dict["decoder"])
        logger.info("Modèle chargé avec succès")
    except Exception as e:
        logger.error(f"Erreur chargement modèle: {e}")
        exit(1)

    # Évaluer le modèle
    metrics = evaluate_model(model, "data/test", "results/evaluation", device=device)
