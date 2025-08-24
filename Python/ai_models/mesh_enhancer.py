"""
Module d'amélioration neuronale des maillages 3D.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Dict
import trimesh
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MeshEnhancementConfig:
    """Configuration pour l'amélioration des maillages."""
    resolution_factor: float = 2.0
    smoothness_weight: float = 0.5
    detail_preservation: float = 0.7
    max_points: int = 100000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class MeshEncoder(nn.Module):
    """Encodeur de maillage pour l'amélioration de la qualité."""
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        
        self.norm1 = nn.BatchNorm1d(64)
        self.norm2 = nn.BatchNorm1d(128)
        self.norm3 = nn.BatchNorm1d(256)
        
        self.linear1 = nn.Linear(256, latent_dim)
        self.linear2 = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        
        return x

class MeshDecoder(nn.Module):
    """Décodeur de maillage pour la génération de détails."""
    
    def __init__(self, latent_dim: int = 256, out_channels: int = 3):
        super().__init__()
        
        self.linear1 = nn.Linear(latent_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 1024)
        
        self.conv1 = nn.Conv1d(1024, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, out_channels, 1)
        
        self.norm1 = nn.BatchNorm1d(512)
        self.norm2 = nn.BatchNorm1d(256)
        
    def forward(self, x: torch.Tensor, num_points: int) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        
        x = x.unsqueeze(2).repeat(1, 1, num_points)
        
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv3(x)
        
        return x

class MeshEnhancer:
    """Classe principale pour l'amélioration des maillages avec gestion d'erreurs avancée."""
    
    def __init__(self, config: Optional[MeshEnhancementConfig] = None):
        self.config = config or MeshEnhancementConfig()
        
        # Initialiser les modèles
        self.encoder = MeshEncoder().to(self.config.device)
        self.decoder = MeshDecoder().to(self.config.device)
        
        # Charger les poids pré-entraînés si disponibles
        try:
            weights_path = "models/mesh_enhancer.pth"
            state_dict = torch.load(weights_path, map_location=self.config.device)
            self.encoder.load_state_dict(state_dict["encoder"])
            self.decoder.load_state_dict(state_dict["decoder"])
            logger.info("Modèles chargés avec succès")
        except Exception as e:
            logger.warning(f"Impossible de charger les poids pré-entraînés: {e}")
            
        logger.info(f"MeshEnhancer initialisé sur {self.config.device}")
            
    def _prepare_mesh(
        self,
        mesh: trimesh.Trimesh
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Prépare le maillage pour le traitement avec gestion d'erreurs."""
        try:
            # Normaliser les vertices
            vertices = mesh.vertices.copy()
            centroid = vertices.mean(axis=0)
            vertices = vertices - centroid
            
            # Calculer l'échelle de manière robuste
            scale = np.percentile(np.abs(vertices), 95)  # Utiliser le 95e percentile
            if scale == 0:
                scale = 1.0
                logger.warning("Échelle zéro détectée, utilisation de l'échelle par défaut")
            
            vertices = vertices / scale
            
            # Vérifier les valeurs aberrantes
            if np.any(np.abs(vertices) > 10):
                logger.warning("Valeurs aberrantes détectées après normalisation")
                vertices = np.clip(vertices, -10, 10)
            
            # Convertir en tenseurs
            vertices_tensor = torch.FloatTensor(vertices).to(self.config.device)
            faces_tensor = torch.LongTensor(mesh.faces).to(self.config.device)
            
            return vertices_tensor, faces_tensor, scale
            
        except Exception as e:
            logger.error(f"Erreur lors de la préparation du maillage: {e}")
            raise RuntimeError(f"Échec de la préparation du maillage: {str(e)}") from e
        
    def _compute_edge_lengths(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """Calcule les longueurs des arêtes."""
        v0 = torch.index_select(vertices, 0, faces[:, 0])
        v1 = torch.index_select(vertices, 0, faces[:, 1])
        v2 = torch.index_select(vertices, 0, faces[:, 2])
        
        edge1 = v1 - v0
        edge2 = v2 - v1
        edge3 = v0 - v2
        
        return torch.stack([
            torch.norm(edge1, dim=1),
            torch.norm(edge2, dim=1),
            torch.norm(edge3, dim=1)
        ], dim=1)
        
    def _compute_face_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> torch.Tensor:
        """Calcule les normales des faces."""
        v0 = torch.index_select(vertices, 0, faces[:, 0])
        v1 = torch.index_select(vertices, 0, faces[:, 1])
        v2 = torch.index_select(vertices, 0, faces[:, 2])
        
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = F.normalize(face_normals, dim=1)
        
        return face_normals
        
    def _compute_vertex_normals(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        face_normals: torch.Tensor
    ) -> torch.Tensor:
        """Calcule les normales des vertices."""
        num_vertices = vertices.shape[0]
        vertex_normals = torch.zeros_like(vertices)
        
        # Accumuler les normales des faces
        for i in range(3):
            vertex_normals.index_add_(
                0, faces[:, i],
                face_normals
            )
            
        # Normaliser
        vertex_normals = F.normalize(vertex_normals, dim=1)
        return vertex_normals
        
    def _subdivide_mesh(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Subdivise le maillage."""
        edge_vertices = {}
        new_vertices = [vertices]
        new_faces = []
        
        # Créer les nouveaux vertices
        for face in faces:
            face_vertices = []
            
            for i in range(3):
                v0 = min(face[i], face[(i+1)%3])
                v1 = max(face[i], face[(i+1)%3])
                edge = (v0.item(), v1.item())
                
                if edge not in edge_vertices:
                    # Créer un nouveau vertex au milieu de l'arête
                    mid = (vertices[v0] + vertices[v1]) / 2
                    edge_vertices[edge] = len(new_vertices)
                    new_vertices.append(mid)
                    
                face_vertices.append(edge_vertices[edge])
                
            # Créer les nouvelles faces
            v0, v1, v2 = face
            e0, e1, e2 = face_vertices
            
            new_faces.extend([
                [v0.item(), e0, e2],
                [e0, v1.item(), e1],
                [e2, e1, v2.item()],
                [e0, e1, e2]
            ])
            
        return (
            torch.stack(new_vertices),
            torch.tensor(new_faces, device=vertices.device)
        )
        
    def enhance_mesh(
        self,
        mesh: trimesh.Trimesh,
        return_original_scale: bool = True
    ) -> trimesh.Trimesh:
        """
        Améliore la qualité du maillage en utilisant le réseau neuronal.
        
        Args:
            mesh: Maillage à améliorer
            return_original_scale: Si True, retourne le maillage à l'échelle d'origine
            
        Returns:
            Maillage amélioré
        """
        # Préparer le maillage
        vertices, faces, scale = self._prepare_mesh(mesh)
        
        # Calculer les features
        edge_lengths = self._compute_edge_lengths(vertices, faces)
        face_normals = self._compute_face_normals(vertices, faces)
        vertex_normals = self._compute_vertex_normals(
            vertices, faces, face_normals
        )
        
        # Préparer l'input pour l'encodeur
        x = torch.cat([
            vertices,
            vertex_normals,
            torch.mean(edge_lengths[faces], dim=1)
        ], dim=1)
        x = x.transpose(1, 2)
        
        # Encoder
        with torch.no_grad():
            latent = self.encoder(x.unsqueeze(0))
            
            # Subdiviser si nécessaire
            if vertices.shape[0] < self.config.max_points:
                vertices, faces = self._subdivide_mesh(vertices, faces)
                
            # Decoder
            num_points = vertices.shape[0]
            output = self.decoder(latent, num_points)
            
            # Appliquer les déplacements
            displacement = output[0].transpose(0, 1)
            vertices = vertices + displacement * self.config.detail_preservation
            
            # Lisser si nécessaire
            if self.config.smoothness_weight > 0:
                vertices = self._smooth_vertices(
                    vertices, faces,
                    weight=self.config.smoothness_weight
                )
                
        # Créer le nouveau maillage
        if return_original_scale:
            vertices = vertices * scale
            
        enhanced_mesh = trimesh.Trimesh(
            vertices=vertices.cpu().numpy(),
            faces=faces.cpu().numpy()
        )
        
        return enhanced_mesh
        
    def _smooth_vertices(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        weight: float = 0.5,
        iterations: int = 1
    ) -> torch.Tensor:
        """
        Lisse les vertices du maillage de manière optimisée.
        
        Utilise une approche sparse pour éviter les gros besoins mémoire.
        """
        device = vertices.device
        n_vertices = vertices.shape[0]
        
        # Construire la liste d'adjacence de manière plus efficace
        edges = []
        for face in faces:
            for i in range(3):
                v0, v1 = face[i].item(), face[(i+1)%3].item()
                if v0 != v1:  # Éviter les auto-références
                    edges.append((min(v0, v1), max(v0, v1)))
        
        # Éliminer les doublons et créer un dictionnaire d'adjacence
        edges = list(set(edges))
        adjacency_dict = {}
        
        for v0, v1 in edges:
            if v0 not in adjacency_dict:
                adjacency_dict[v0] = []
            if v1 not in adjacency_dict:
                adjacency_dict[v1] = []
            adjacency_dict[v0].append(v1)
            adjacency_dict[v1].append(v0)
        
        # Appliquer le lissage de manière itérative
        smoothed = vertices.clone()
        
        for iteration in range(iterations):
            new_vertices = smoothed.clone()
            
            # Traitement vectorisé par batches pour l'efficacité
            batch_size = min(1000, n_vertices)
            
            for start_idx in range(0, n_vertices, batch_size):
                end_idx = min(start_idx + batch_size, n_vertices)
                batch_indices = range(start_idx, end_idx)
                
                # Calculer les moyennes des voisins pour ce batch
                neighbor_averages = torch.zeros((end_idx - start_idx, 3), device=device)
                
                for i, vertex_idx in enumerate(batch_indices):
                    if vertex_idx in adjacency_dict:
                        neighbors = adjacency_dict[vertex_idx]
                        if neighbors:
                            neighbor_positions = smoothed[neighbors]
                            neighbor_averages[i] = neighbor_positions.mean(dim=0)
                        else:
                            neighbor_averages[i] = smoothed[vertex_idx]
                    else:
                        neighbor_averages[i] = smoothed[vertex_idx]
                
                # Appliquer le lissage avec interpolation
                original_batch = vertices[start_idx:end_idx]
                smoothed_batch = smoothed[start_idx:end_idx]
                
                # Facteur de lissage adaptatif basé sur la courbure locale
                adaptive_weight = self._compute_adaptive_weight(
                    smoothed_batch, neighbor_averages, weight
                )
                
                new_vertices[start_idx:end_idx] = (
                    (1 - adaptive_weight) * original_batch + 
                    adaptive_weight * neighbor_averages
                )
            
            smoothed = new_vertices
            
        return smoothed
    
    def _compute_adaptive_weight(
        self, 
        vertices: torch.Tensor, 
        neighbors: torch.Tensor, 
        base_weight: float
    ) -> torch.Tensor:
        """
        Calcule un poids de lissage adaptatif basé sur la courbure locale.
        
        Args:
            vertices: Positions des vertices originaux
            neighbors: Moyennes des voisins
            base_weight: Poids de base
            
        Returns:
            Poids adaptatifs pour chaque vertex
        """
        # Calculer la distance aux voisins (mesure de courbure)
        distances = torch.norm(vertices - neighbors, dim=1)
        
        # Normaliser les distances
        max_dist = distances.max()
        if max_dist > 0:
            normalized_distances = distances / max_dist
        else:
            normalized_distances = torch.zeros_like(distances)
        
        # Poids adaptatif: moins de lissage dans les zones de haute courbure
        adaptive_weights = base_weight * (1 - 0.5 * normalized_distances)
        
        return adaptive_weights.unsqueeze(1)
    
    def edge_preserving_smooth(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        iterations: int = 3,
        lambda_factor: float = 0.5,
        mu_factor: float = -0.53,
        edge_threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Lissage préservant les arêtes utilisant l'algorithme de Taubin.
        
        Args:
            vertices: Positions des vertices
            faces: Faces du maillage
            iterations: Nombre d'itérations
            lambda_factor: Facteur de lissage positif
            mu_factor: Facteur de correction négatif
            edge_threshold: Seuil pour la détection d'arêtes
            
        Returns:
            Vertices lissés avec préservation des arêtes
        """
        device = vertices.device
        n_vertices = vertices.shape[0]
        result = vertices.clone()
        
        # Construire la matrice de connectivité
        adjacency_dict = self._build_adjacency_dict(faces)
        
        # Détecter les arêtes importantes
        edge_vertices = self._detect_edge_vertices(vertices, faces, adjacency_dict, edge_threshold)
        
        for iteration in range(iterations):
            # Étape 1: Lissage avec lambda
            smoothed = self._apply_laplacian_smoothing(
                result, adjacency_dict, lambda_factor, edge_vertices
            )
            
            # Étape 2: Correction avec mu (facteur négatif)
            result = self._apply_laplacian_smoothing(
                smoothed, adjacency_dict, mu_factor, edge_vertices
            )
        
        return result
    
    def _build_adjacency_dict(self, faces: torch.Tensor) -> Dict[int, List[int]]:
        """Construit un dictionnaire d'adjacence optimisé."""
        adjacency_dict = {}
        
        # Traitement vectorisé des faces
        edges = []
        for face in faces:
            for i in range(3):
                v0, v1 = face[i].item(), face[(i+1)%3].item()
                if v0 != v1:
                    edges.append((min(v0, v1), max(v0, v1)))
        
        # Éliminer les doublons
        edges = list(set(edges))
        
        # Construire le dictionnaire
        for v0, v1 in edges:
            if v0 not in adjacency_dict:
                adjacency_dict[v0] = []
            if v1 not in adjacency_dict:
                adjacency_dict[v1] = []
            adjacency_dict[v0].append(v1)
            adjacency_dict[v1].append(v0)
        
        return adjacency_dict
    
    def _detect_edge_vertices(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        adjacency_dict: Dict[int, List[int]],
        threshold: float
    ) -> torch.Tensor:
        """
        Détecte les vertices sur les arêtes importantes du maillage avec optimisation vectorielle.
        
        Args:
            vertices: Positions des vertices
            faces: Faces du maillage
            adjacency_dict: Dictionnaire d'adjacence
            threshold: Seuil de détection d'arêtes
            
        Returns:
            Masque booléen des vertices d'arêtes
        """
        n_vertices = vertices.shape[0]
        edge_mask = torch.zeros(n_vertices, dtype=torch.bool, device=vertices.device)
        
        # Créer un mapping vertex -> faces pour optimiser la recherche
        vertex_to_faces = {v: [] for v in range(n_vertices)}
        for face_idx, face in enumerate(faces):
            for vertex_idx in face:
                vertex_to_faces[vertex_idx.item()].append(face_idx)
        
        # Calculer toutes les normales de faces d'un coup
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = torch.nn.functional.normalize(face_normals, dim=1)
        
        # Analyser chaque vertex en batch pour l'efficacité
        batch_size = min(1000, n_vertices)
        
        for start_idx in range(0, n_vertices, batch_size):
            end_idx = min(start_idx + batch_size, n_vertices)
            
            for vertex_idx in range(start_idx, end_idx):
                if vertex_idx in adjacency_dict and len(adjacency_dict[vertex_idx]) >= 3:
                    adjacent_faces = vertex_to_faces[vertex_idx]
                    
                    if len(adjacent_faces) >= 2:
                        # Obtenir les normales des faces adjacentes
                        adjacent_normals = face_normals[adjacent_faces]
                        
                        # Calculer l'angle maximum entre les normales de manière vectorielle
                        if len(adjacent_normals) >= 2:
                            # Calculer tous les produits scalaires
                            dot_products = torch.mm(adjacent_normals, adjacent_normals.t())
                            dot_products = torch.clamp(dot_products, -1.0, 1.0)
                            
                            # Obtenir l'angle maximum (ignorer la diagonale)
                            mask = ~torch.eye(len(adjacent_normals), dtype=torch.bool, device=vertices.device)
                            if mask.sum() > 0:
                                min_dot = torch.min(dot_products[mask])
                                max_angle = torch.acos(min_dot)
                                
                                if max_angle > threshold:
                                    edge_mask[vertex_idx] = True
        
        return edge_mask
    
    def _apply_laplacian_smoothing(
        self,
        vertices: torch.Tensor,
        adjacency_dict: Dict[int, List[int]],
        weight: float,
        edge_vertices: torch.Tensor
    ) -> torch.Tensor:
        """
        Applique le lissage laplacien avec préservation des arêtes.
        
        Args:
            vertices: Positions des vertices
            adjacency_dict: Dictionnaire d'adjacence
            weight: Poids de lissage
            edge_vertices: Masque des vertices d'arêtes
            
        Returns:
            Vertices lissés
        """
        device = vertices.device
        n_vertices = vertices.shape[0]
        result = vertices.clone()
        
        # Traitement par batches pour l'efficacité
        batch_size = min(1000, n_vertices)
        
        for start_idx in range(0, n_vertices, batch_size):
            end_idx = min(start_idx + batch_size, n_vertices)
            
            for vertex_idx in range(start_idx, end_idx):
                if vertex_idx in adjacency_dict:
                    neighbors = adjacency_dict[vertex_idx]
                    if neighbors:
                        # Calculer la moyenne des voisins
                        neighbor_positions = vertices[neighbors]
                        laplacian = neighbor_positions.mean(dim=0) - vertices[vertex_idx]
                        
                        # Réduire le lissage sur les arêtes importantes
                        edge_weight = weight
                        if edge_vertices[vertex_idx]:
                            edge_weight *= 0.1  # Réduire fortement le lissage sur les arêtes
                        
                        result[vertex_idx] = vertices[vertex_idx] + edge_weight * laplacian
        
        return result
    
    def adaptive_mesh_enhancement(
        self,
        mesh: trimesh.Trimesh,
        quality_target: float = 0.8,
        max_iterations: int = 5
    ) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
        """
        Amélioration adaptative du maillage basée sur la qualité cible.
        
        Args:
            mesh: Maillage à améliorer
            quality_target: Qualité cible (0-1)
            max_iterations: Nombre maximum d'itérations
            
        Returns:
            Tuple (maillage amélioré, métriques)
        """
        current_mesh = mesh.copy()
        iteration_metrics = []
        
        for iteration in range(max_iterations):
            # Analyser la qualité actuelle
            quality_metrics = self._analyze_mesh_quality(current_mesh)
            current_quality = quality_metrics['overall_quality']
            
            iteration_metrics.append({
                'iteration': iteration,
                'quality': current_quality,
                'vertex_count': len(current_mesh.vertices),
                'face_count': len(current_mesh.faces)
            })
            
            # Vérifier si la qualité cible est atteinte
            if current_quality >= quality_target:
                logger.info(f"Qualité cible atteinte à l'itération {iteration}")
                break
            
            # Appliquer des améliorations ciblées
            if quality_metrics['edge_length_variance'] > 0.5:
                current_mesh = self._regularize_edge_lengths(current_mesh)
            
            if quality_metrics['face_aspect_ratio'] > 2.0:
                current_mesh = self._improve_face_quality(current_mesh)
            
            if quality_metrics['normal_consistency'] < 0.8:
                current_mesh = self._fix_normal_consistency(current_mesh)
            
            # Lissage léger pour améliorer la qualité générale
            vertices, faces, scale = self._prepare_mesh(current_mesh)
            smoothed_vertices = self._smooth_vertices(vertices, faces, weight=0.3, iterations=1)
            
            # Restaurer l'échelle
            smoothed_vertices = smoothed_vertices * scale
            current_mesh = trimesh.Trimesh(
                vertices=smoothed_vertices.cpu().numpy(),
                faces=current_mesh.faces
            )
        
        final_metrics = {
            'iterations_performed': len(iteration_metrics),
            'initial_quality': iteration_metrics[0]['quality'] if iteration_metrics else 0,
            'final_quality': iteration_metrics[-1]['quality'] if iteration_metrics else 0,
            'quality_improvement': (iteration_metrics[-1]['quality'] - iteration_metrics[0]['quality']) if len(iteration_metrics) > 1 else 0,
            'iteration_details': iteration_metrics
        }
        
        return current_mesh, final_metrics
    
    def _analyze_mesh_quality(self, mesh: trimesh.Trimesh) -> Dict[str, float]:
        """Analyse détaillée de la qualité du maillage."""
        try:
            # Métriques de base
            edge_lengths = np.array([np.linalg.norm(mesh.vertices[edge[1]] - mesh.vertices[edge[0]]) 
                                   for edge in mesh.edges])
            edge_length_variance = np.var(edge_lengths) / (np.mean(edge_lengths) ** 2) if len(edge_lengths) > 0 else 0
            
            # Ratio d'aspect des faces
            face_areas = mesh.area_faces
            face_perimeters = []
            for face in mesh.faces:
                perimeter = 0
                for i in range(3):
                    edge_vec = mesh.vertices[face[(i+1)%3]] - mesh.vertices[face[i]]
                    perimeter += np.linalg.norm(edge_vec)
                face_perimeters.append(perimeter)
            
            face_perimeters = np.array(face_perimeters)
            aspect_ratios = face_perimeters**2 / (4 * np.pi * face_areas + 1e-8)
            avg_aspect_ratio = np.mean(aspect_ratios)
            
            # Cohérence des normales
            try:
                normal_consistency = 1.0 - np.std(mesh.face_normals.flatten())
            except:
                normal_consistency = 0.5
            
            # Qualité globale (combinaison pondérée)
            overall_quality = (
                0.4 * max(0, 1 - edge_length_variance) +
                0.3 * max(0, 1 - (avg_aspect_ratio - 1) / 2) +
                0.3 * normal_consistency
            )
            
            return {
                'edge_length_variance': edge_length_variance,
                'face_aspect_ratio': avg_aspect_ratio,
                'normal_consistency': normal_consistency,
                'overall_quality': max(0, min(1, overall_quality))
            }
        except Exception as e:
            logger.warning(f"Erreur lors de l'analyse de qualité: {e}")
            return {
                'edge_length_variance': 1.0,
                'face_aspect_ratio': 3.0,
                'normal_consistency': 0.0,
                'overall_quality': 0.0
            }
    
    def gpu_accelerated_enhancement(
        self,
        mesh: trimesh.Trimesh,
        enhancement_strength: float = 0.5
    ) -> trimesh.Trimesh:
        """
        Amélioration accélérée par GPU pour les gros maillages.
        
        Args:
            mesh: Maillage à améliorer
            enhancement_strength: Force de l'amélioration (0-1)
            
        Returns:
            Maillage amélioré
        """
        if not torch.cuda.is_available():
            logger.warning("GPU non disponible, utilisation du CPU")
            return self.enhance_mesh(mesh)
        
        try:
            # Préparer les données sur GPU
            vertices, faces, scale = self._prepare_mesh(mesh)
            vertices = vertices.cuda()
            faces = faces.cuda()
            
            # Optimisations GPU
            with torch.cuda.amp.autocast():  # Utiliser la précision mixte
                # Lissage adaptatif
                smoothed_vertices = self._gpu_adaptive_smoothing(vertices, faces, enhancement_strength)
            
            # Retourner sur CPU et restaurer l'échelle
            enhanced_vertices = smoothed_vertices.cpu() * scale
            
            return trimesh.Trimesh(
                vertices=enhanced_vertices.numpy(),
                faces=mesh.faces
            )
            
        except Exception as e:
            logger.error(f"Erreur GPU, fallback CPU: {e}")
            return self.enhance_mesh(mesh)
    
    def _gpu_adaptive_smoothing(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        strength: float
    ) -> torch.Tensor:
        """Lissage adaptatif optimisé GPU."""
        # Construction rapide du graphe d'adjacence sur GPU
        edges = torch.stack([
            torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]),
            torch.cat([faces[:, [1, 0]], faces[:, [2, 1]], faces[:, [0, 2]]])
        ], dim=1).reshape(-1, 2)
        
        # Suppression des doublons
        edges_sorted = torch.sort(edges, dim=1)[0]
        edges_unique = torch.unique(edges_sorted, dim=0)
        
        # Lissage par convolution 1D optimisée
        smoothed = vertices.clone()
        
        for _ in range(2):  # Quelques itérations
            vertex_sums = torch.zeros_like(vertices)
            vertex_counts = torch.zeros(vertices.shape[0], device=vertices.device)
            
            # Accumulation rapide avec scatter_add
            vertex_sums.scatter_add_(0, edges_unique[:, 0:1].expand(-1, 3), vertices[edges_unique[:, 1]])
            vertex_sums.scatter_add_(0, edges_unique[:, 1:2].expand(-1, 3), vertices[edges_unique[:, 0]])
            
            vertex_counts.scatter_add_(0, edges_unique[:, 0], torch.ones_like(edges_unique[:, 0], dtype=torch.float32))
            vertex_counts.scatter_add_(0, edges_unique[:, 1], torch.ones_like(edges_unique[:, 1], dtype=torch.float32))
            
            # Moyennage avec protection contre division par zéro
            valid_mask = vertex_counts > 0
            neighbor_means = torch.zeros_like(vertices)
            neighbor_means[valid_mask] = vertex_sums[valid_mask] / vertex_counts[valid_mask, None]
            
            # Application du lissage adaptatif
            smoothed = (1 - strength) * smoothed + strength * neighbor_means
        
        return smoothed
    
    def quality_analysis_detailed(
        self,
        mesh: trimesh.Trimesh,
        return_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Analyse détaillée de la qualité du maillage avec recommandations.
        
        Args:
            mesh: Maillage à analyser
            return_recommendations: Si True, inclut des recommandations
            
        Returns:
            Rapport d'analyse détaillé
        """
        try:
            # Analyse de base
            basic_metrics = self._analyze_mesh_quality(mesh)
            
            # Analyses supplémentaires
            detailed_analysis = {
                'basic_metrics': basic_metrics,
                'vertex_analysis': self._analyze_vertices(mesh),
                'face_analysis': self._analyze_faces(mesh),
                'topology_analysis': self._analyze_topology(mesh),
                'geometric_analysis': self._analyze_geometry(mesh)
            }
            
            # Score global pondéré
            overall_score = self._calculate_overall_quality_score(detailed_analysis)
            detailed_analysis['overall_score'] = overall_score
            
            # Recommandations si demandées
            if return_recommendations:
                detailed_analysis['recommendations'] = self._generate_quality_recommendations(detailed_analysis)
            
            return detailed_analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse détaillée: {e}")
            return {'error': str(e), 'overall_score': 0.0}
    
    def edge_detection_advanced(
        self,
        mesh: trimesh.Trimesh,
        detection_method: str = "curvature",
        sensitivity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Détection avancée des arêtes importantes avec différentes méthodes.
        
        Args:
            mesh: Maillage à analyser
            detection_method: Méthode de détection ("curvature", "angle", "hybrid")
            sensitivity: Sensibilité de détection (0-1)
            
        Returns:
            Résultats de détection avec indices des arêtes importantes
        """
        try:
            vertices, faces, scale = self._prepare_mesh(mesh)
            
            # Construire le graphe d'adjacence
            adjacency_dict = {}
            for face in faces:
                for i in range(3):
                    v0, v1 = face[i].item(), face[(i+1)%3].item()
                    if v0 not in adjacency_dict:
                        adjacency_dict[v0] = []
                    if v1 not in adjacency_dict:
                        adjacency_dict[v1] = []
                    adjacency_dict[v0].append(v1)
                    adjacency_dict[v1].append(v0)
            
            # Méthodes de détection
            if detection_method == "curvature":
                edge_vertices = self._detect_edge_vertices_by_curvature(vertices, faces, adjacency_dict, sensitivity)
            elif detection_method == "angle":
                edge_vertices = self._detect_edge_vertices_by_angle(vertices, faces, adjacency_dict, sensitivity)
            else:  # hybrid
                edge_curvature = self._detect_edge_vertices_by_curvature(vertices, faces, adjacency_dict, sensitivity * 0.8)
                edge_angle = self._detect_edge_vertices_by_angle(vertices, faces, adjacency_dict, sensitivity * 1.2)
                edge_vertices = edge_curvature | edge_angle  # Union des deux méthodes
            
            # Statistiques
            edge_indices = [i for i, is_edge in enumerate(edge_vertices) if is_edge]
            edge_percentage = (len(edge_indices) / len(vertices)) * 100
            
            return {
                'method': detection_method,
                'sensitivity': sensitivity,
                'edge_vertices_indices': edge_indices,
                'edge_vertices_mask': edge_vertices.cpu().numpy().tolist(),
                'total_vertices': len(vertices),
                'edge_vertices_count': len(edge_indices),
                'edge_percentage': edge_percentage,
                'detection_quality': self._assess_edge_detection_quality(edge_percentage)
            }
            
        except Exception as e:
            logger.error(f"Erreur détection arêtes: {e}")
            return {'error': str(e)}
    
    def _detect_edge_vertices_by_curvature(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        adjacency_dict: Dict[int, List[int]],
        sensitivity: float
    ) -> torch.Tensor:
        """Détection d'arêtes basée sur la courbure locale."""
        n_vertices = vertices.shape[0]
        edge_mask = torch.zeros(n_vertices, dtype=torch.bool, device=vertices.device)
        
        # Calculer la courbure locale pour chaque vertex
        for vertex_idx in range(n_vertices):
            if vertex_idx in adjacency_dict:
                neighbors = adjacency_dict[vertex_idx]
                if len(neighbors) >= 3:
                    # Calculer le vecteur Laplacien
                    neighbor_positions = vertices[neighbors]
                    neighbor_center = neighbor_positions.mean(dim=0)
                    laplacian = neighbor_center - vertices[vertex_idx]
                    curvature = torch.norm(laplacian)
                    
                    # Seuil adaptatif basé sur la sensibilité
                    threshold = sensitivity * 0.1  # Ajuster selon les besoins
                    if curvature > threshold:
                        edge_mask[vertex_idx] = True
        
        return edge_mask
    
    def _detect_edge_vertices_by_angle(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        adjacency_dict: Dict[int, List[int]],
        sensitivity: float
    ) -> torch.Tensor:
        """Détection d'arêtes basée sur les angles entre faces adjacentes."""
        return self._detect_edge_vertices(vertices, faces, adjacency_dict, sensitivity * np.pi / 4)
    
    def _analyze_vertices(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyse les propriétés des vertices."""
        try:
            vertices = mesh.vertices
            
            # Statistiques de distribution
            centroid = np.mean(vertices, axis=0)
            distances_to_centroid = np.linalg.norm(vertices - centroid, axis=1)
            
            # Détection des vertices isolés
            vertex_connectivity = {}
            for face in mesh.faces:
                for vertex_idx in face:
                    if vertex_idx not in vertex_connectivity:
                        vertex_connectivity[vertex_idx] = set()
                    vertex_connectivity[vertex_idx].update(face)
                    vertex_connectivity[vertex_idx].discard(vertex_idx)
            
            isolated_vertices = [v for v, neighbors in vertex_connectivity.items() if len(neighbors) < 2]
            
            return {
                'count': len(vertices),
                'centroid': centroid.tolist(),
                'bounding_box_size': (np.max(vertices, axis=0) - np.min(vertices, axis=0)).tolist(),
                'avg_distance_to_centroid': np.mean(distances_to_centroid),
                'isolated_vertices_count': len(isolated_vertices),
                'connectivity_distribution': {
                    'min': min(len(neighbors) for neighbors in vertex_connectivity.values()) if vertex_connectivity else 0,
                    'max': max(len(neighbors) for neighbors in vertex_connectivity.values()) if vertex_connectivity else 0,
                    'avg': np.mean([len(neighbors) for neighbors in vertex_connectivity.values()]) if vertex_connectivity else 0
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_faces(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyse les propriétés des faces."""
        try:
            faces = mesh.faces
            vertices = mesh.vertices
            
            # Calcul des aires et périmètres
            face_areas = []
            face_perimeters = []
            aspect_ratios = []
            
            for face in faces:
                v0, v1, v2 = vertices[face]
                
                # Aire du triangle
                area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                face_areas.append(area)
                
                # Périmètre
                perimeter = (
                    np.linalg.norm(v1 - v0) +
                    np.linalg.norm(v2 - v1) +
                    np.linalg.norm(v0 - v2)
                )
                face_perimeters.append(perimeter)
                
                # Ratio d'aspect (périmètre² / aire)
                if area > 1e-10:
                    aspect_ratio = perimeter**2 / (4 * np.pi * area)
                    aspect_ratios.append(aspect_ratio)
            
            return {
                'count': len(faces),
                'area_stats': {
                    'min': min(face_areas) if face_areas else 0,
                    'max': max(face_areas) if face_areas else 0,
                    'mean': np.mean(face_areas) if face_areas else 0,
                    'std': np.std(face_areas) if face_areas else 0
                },
                'aspect_ratio_stats': {
                    'min': min(aspect_ratios) if aspect_ratios else 0,
                    'max': max(aspect_ratios) if aspect_ratios else 0,
                    'mean': np.mean(aspect_ratios) if aspect_ratios else 0,
                    'std': np.std(aspect_ratios) if aspect_ratios else 0
                },
                'degenerate_faces': len([r for r in aspect_ratios if r > 10])  # Faces très allongées
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_topology(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyse la topologie du maillage."""
        try:
            # Caractéristique d'Euler pour les surfaces fermées: V - E + F = 2
            V = len(mesh.vertices)
            F = len(mesh.faces)
            
            # Compter les arêtes uniques
            edges = set()
            for face in mesh.faces:
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    edges.add(edge)
            E = len(edges)
            
            euler_characteristic = V - E + F
            
            # Vérifier les arêtes non-manifold (partagées par plus de 2 faces)
            edge_face_count = {}
            for face_idx, face in enumerate(mesh.faces):
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i+1)%3]]))
                    if edge not in edge_face_count:
                        edge_face_count[edge] = 0
                    edge_face_count[edge] += 1
            
            non_manifold_edges = len([e for e, count in edge_face_count.items() if count > 2])
            boundary_edges = len([e for e, count in edge_face_count.items() if count == 1])
            
            return {
                'vertices': V,
                'edges': E,
                'faces': F,
                'euler_characteristic': euler_characteristic,
                'is_closed_surface': euler_characteristic == 2 and boundary_edges == 0,
                'non_manifold_edges': non_manifold_edges,
                'boundary_edges': boundary_edges,
                'genus': (2 - euler_characteristic) // 2 if euler_characteristic <= 2 else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_geometry(self, mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """Analyse les propriétés géométriques."""
        try:
            vertices = mesh.vertices
            
            # Volume et surface (approximatifs)
            try:
                volume = abs(mesh.volume) if hasattr(mesh, 'volume') else 0
                surface_area = mesh.area if hasattr(mesh, 'area') else sum(mesh.area_faces) if hasattr(mesh, 'area_faces') else 0
            except:
                volume = 0
                surface_area = 0
            
            # Boîte englobante
            min_bounds = np.min(vertices, axis=0)
            max_bounds = np.max(vertices, axis=0)
            bbox_size = max_bounds - min_bounds
            bbox_volume = np.prod(bbox_size)
            
            # Compacité (ratio volume/surface)
            compactness = (volume / (surface_area**1.5)) if surface_area > 0 else 0
            
            # Facteur de forme (ratio volume réel / volume boîte englobante)
            shape_factor = volume / bbox_volume if bbox_volume > 0 else 0
            
            return {
                'volume': volume,
                'surface_area': surface_area,
                'bounding_box': {
                    'min': min_bounds.tolist(),
                    'max': max_bounds.tolist(),
                    'size': bbox_size.tolist(),
                    'volume': bbox_volume
                },
                'compactness': compactness,
                'shape_factor': shape_factor,
                'volume_to_surface_ratio': volume / surface_area if surface_area > 0 else 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_overall_quality_score(self, analysis: Dict[str, Any]) -> float:
        """Calcule un score de qualité global pondéré."""
        try:
            score = 100.0  # Score de base
            
            # Pénalités basées sur l'analyse
            basic = analysis.get('basic_metrics', {})
            faces = analysis.get('face_analysis', {})
            topology = analysis.get('topology_analysis', {})
            
            # Pénalité pour la variance des longueurs d'arêtes
            edge_variance = basic.get('edge_length_variance', 0)
            score -= min(20, edge_variance * 40)
            
            # Pénalité pour les ratios d'aspect
            avg_aspect = faces.get('aspect_ratio_stats', {}).get('mean', 1)
            score -= min(15, (avg_aspect - 1) * 10)
            
            # Pénalité pour les faces dégénérées
            degenerate = faces.get('degenerate_faces', 0)
            total_faces = faces.get('count', 1)
            score -= min(25, (degenerate / total_faces) * 100)
            
            # Pénalité pour les problèmes topologiques
            non_manifold = topology.get('non_manifold_edges', 0)
            score -= min(20, non_manifold * 5)
            
            # Bonus pour surface fermée
            if topology.get('is_closed_surface', False):
                score += 5
            
            return max(0, min(100, score))
            
        except Exception as e:
            logger.warning(f"Erreur calcul score qualité: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Génère des recommandations basées sur l'analyse."""
        recommendations = []
        
        try:
            basic = analysis.get('basic_metrics', {})
            faces = analysis.get('face_analysis', {})
            topology = analysis.get('topology_analysis', {})
            vertices = analysis.get('vertex_analysis', {})
            
            # Recommandations basées sur les métriques
            if basic.get('edge_length_variance', 0) > 0.5:
                recommendations.append("Régulariser les longueurs d'arêtes avec _regularize_edge_lengths")
            
            avg_aspect = faces.get('aspect_ratio_stats', {}).get('mean', 1)
            if avg_aspect > 2.0:
                recommendations.append("Améliorer la qualité des faces avec _improve_face_quality")
            
            degenerate = faces.get('degenerate_faces', 0)
            if degenerate > 0:
                recommendations.append(f"Corriger {degenerate} face(s) dégénérée(s)")
            
            non_manifold = topology.get('non_manifold_edges', 0)
            if non_manifold > 0:
                recommendations.append(f"Réparer {non_manifold} arête(s) non-manifold")
            
            isolated = vertices.get('isolated_vertices_count', 0)
            if isolated > 0:
                recommendations.append(f"Supprimer {isolated} vertex(vertices) isolé(s)")
            
            if not topology.get('is_closed_surface', False):
                recommendations.append("Fermer la surface si nécessaire")
            
            overall_score = analysis.get('overall_score', 100)
            if overall_score < 70:
                recommendations.append("Score qualité faible - amélioration générale recommandée")
            
            if not recommendations:
                recommendations.append("Qualité de maillage satisfaisante")
            
        except Exception as e:
            recommendations.append(f"Erreur génération recommandations: {e}")
        
        return recommendations
    
    def _assess_edge_detection_quality(self, edge_percentage: float) -> str:
        """Évalue la qualité de la détection d'arêtes."""
        if edge_percentage < 5:
            return "Très faible - maillage très lisse"
        elif edge_percentage < 15:
            return "Faible - peu d'arêtes détectées"
        elif edge_percentage < 30:
            return "Normale - détection équilibrée"
        elif edge_percentage < 50:
            return "Élevée - maillage détaillé"
        else:
            return "Très élevée - maillage très complexe"
