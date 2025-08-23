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
        """Lisse les vertices du maillage."""
        adj = torch.zeros(
            (vertices.shape[0], vertices.shape[0]),
            device=vertices.device
        )
        
        # Construire la matrice d'adjacence
        for face in faces:
            for i in range(3):
                v0 = face[i]
                v1 = face[(i+1)%3]
                adj[v0, v1] = 1
                adj[v1, v0] = 1
                
        # Normaliser
        deg = adj.sum(dim=1, keepdim=True)
        adj = adj / deg
        
        # Appliquer le lissage
        smoothed = vertices
        for _ in range(iterations):
            neighbor_sum = torch.mm(adj, smoothed)
            smoothed = (1 - weight) * vertices + weight * neighbor_sum
            
        return smoothed
