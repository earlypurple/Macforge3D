"""
Module de gestion des styles et effets spéciaux pour le texte 3D.
"""

import trimesh
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TextStyle:
    """Configuration d'un style de texte."""
    name: str
    bevel_amount: float = 0.0
    bevel_segments: int = 3
    roughness: float = 0.0
    metallic: float = 0.0
    emission: float = 0.0
    transparency: float = 0.0
    subsurface: float = 0.0
    noise_scale: float = 0.0
    distortion: float = 0.0
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    texture_path: Optional[str] = None
    normal_map_path: Optional[str] = None
    
class TextEffects:
    """Gestionnaire d'effets pour le texte 3D."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def apply_style(
        self,
        mesh: trimesh.Trimesh,
        style: TextStyle
    ) -> trimesh.Trimesh:
        """Applique un style au maillage."""
        # Copie du maillage
        result = mesh.copy()
        
        # Appliquer les effets dans l'ordre
        if style.bevel_amount > 0:
            result = self._apply_bevel(
                result,
                style.bevel_amount,
                style.bevel_segments
            )
            
        if style.noise_scale > 0:
            result = self._apply_noise(
                result,
                style.noise_scale
            )
            
        if style.distortion > 0:
            result = self._apply_distortion(
                result,
                style.distortion
            )
            
        # Ajouter les attributs de matériau
        result.visual = trimesh.visual.TextureVisuals()
        
        if style.texture_path:
            result.visual.material.image = trimesh.load_image(
                style.texture_path
            )
            
        if style.normal_map_path:
            result.visual.material.normal_image = trimesh.load_image(
                style.normal_map_path
            )
            
        material = {
            "roughness": style.roughness,
            "metallic": style.metallic,
            "emission": style.emission,
            "transparency": style.transparency,
            "subsurface": style.subsurface,
            "color": style.color
        }
        
        result.visual.material.kwargs.update(material)
        
        return result
        
    def _apply_bevel(
        self,
        mesh: trimesh.Trimesh,
        amount: float,
        segments: int
    ) -> trimesh.Trimesh:
        """Applique un effet de bevel aux arêtes."""
        # Trouver les arêtes vives
        edges = mesh.edges_sorted
        normals = mesh.face_normals
        
        sharp_edges = []
        for edge in edges:
            # Trouver les faces adjacentes
            face_idx = mesh.edges_face[edge]
            if len(face_idx) == 2:  # Arête intérieure
                angle = np.arccos(np.dot(
                    normals[face_idx[0]],
                    normals[face_idx[1]]
                ))
                if np.abs(angle) > np.pi/6:  # Seuil d'angle
                    sharp_edges.append(edge)
                    
        # Créer les segments de bevel
        new_vertices = []
        new_faces = []
        
        for edge in sharp_edges:
            v1 = mesh.vertices[edge[0]]
            v2 = mesh.vertices[edge[1]]
            
            # Créer les points de contrôle
            direction = v2 - v1
            length = np.linalg.norm(direction)
            direction = direction / length
            
            for i in range(segments):
                t = (i + 1) / (segments + 1)
                offset = amount * np.sin(np.pi * t)
                
                point = v1 + t * length * direction
                normal = mesh.vertex_normals[edge[0]]
                point += normal * offset
                
                new_vertices.append(point)
                
        # Mettre à jour le maillage
        vertices = np.vstack((mesh.vertices, new_vertices))
        
        # Retrianguler avec les nouveaux points
        from scipy.spatial import Delaunay
        points_2d = vertices[:, :2]  # Projection 2D pour la triangulation
        tri = Delaunay(points_2d)
        faces = tri.simplices
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
        
    def _apply_noise(
        self,
        mesh: trimesh.Trimesh,
        scale: float
    ) -> trimesh.Trimesh:
        """Applique un bruit procédural au maillage."""
        # Générer du bruit de Perlin 3D
        from noise import snoise3
        
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals
        
        for i in range(len(vertices)):
            pos = vertices[i]
            noise = snoise3(
                pos[0] * scale,
                pos[1] * scale,
                pos[2] * scale
            )
            vertices[i] += normals[i] * noise * scale
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        
    def _apply_distortion(
        self,
        mesh: trimesh.Trimesh,
        amount: float
    ) -> trimesh.Trimesh:
        """Applique une distortion au maillage."""
        vertices = mesh.vertices.copy()
        
        # Calculer le centre et l'échelle
        center = np.mean(vertices, axis=0)
        scale = np.max(np.abs(vertices - center))
        
        # Appliquer la distortion
        for i in range(len(vertices)):
            pos = vertices[i]
            dir_to_center = center - pos
            dist = np.linalg.norm(dir_to_center)
            
            if dist > 0:
                # Facteur de distortion basé sur la distance
                factor = 1 + amount * (1 - dist/scale)
                vertices[i] = pos + dir_to_center * factor
                
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
        
# Styles prédéfinis
PREDEFINED_STYLES = {
    "standard": TextStyle(
        name="standard",
        bevel_amount=0.0,
        roughness=0.5,
        color=(0.8, 0.8, 0.8)
    ),
    
    "moderne": TextStyle(
        name="moderne",
        bevel_amount=0.1,
        bevel_segments=5,
        roughness=0.2,
        metallic=0.8,
        color=(0.9, 0.9, 0.9)
    ),
    
    "neon": TextStyle(
        name="neon",
        bevel_amount=0.05,
        emission=1.0,
        transparency=0.3,
        color=(0.0, 1.0, 1.0)
    ),
    
    "cristal": TextStyle(
        name="cristal",
        bevel_amount=0.15,
        bevel_segments=8,
        roughness=0.1,
        transparency=0.8,
        subsurface=0.5,
        color=(1.0, 1.0, 1.0)
    ),
    
    "or": TextStyle(
        name="or",
        bevel_amount=0.08,
        roughness=0.3,
        metallic=1.0,
        color=(1.0, 0.8, 0.0)
    ),
    
    "pierre": TextStyle(
        name="pierre",
        noise_scale=0.2,
        roughness=0.9,
        color=(0.6, 0.6, 0.6)
    ),
    
    "liquide": TextStyle(
        name="liquide",
        distortion=0.2,
        roughness=0.1,
        transparency=0.6,
        color=(0.0, 0.3, 1.0)
    ),
    
    "lave": TextStyle(
        name="lave",
        noise_scale=0.15,
        emission=0.8,
        roughness=0.7,
        color=(1.0, 0.3, 0.0)
    ),
    
    "bois": TextStyle(
        name="bois",
        bevel_amount=0.05,
        roughness=0.8,
        texture_path="textures/wood.jpg",
        normal_map_path="textures/wood_normal.jpg"
    ),
    
    "chrome": TextStyle(
        name="chrome",
        bevel_amount=0.1,
        roughness=0.1,
        metallic=1.0,
        color=(0.95, 0.95, 0.95)
    ),
    
    "holographique": TextStyle(
        name="holographique",
        bevel_amount=0.05,
        emission=0.5,
        transparency=0.4,
        texture_path="textures/holographic.jpg"
    )
}

def get_available_styles() -> List[str]:
    """Retourne la liste des styles disponibles."""
    return list(PREDEFINED_STYLES.keys())

def get_style(name: str) -> TextStyle:
    """Récupère un style prédéfini."""
    if name not in PREDEFINED_STYLES:
        raise ValueError(f"Style '{name}' non trouvé")
    return PREDEFINED_STYLES[name]
