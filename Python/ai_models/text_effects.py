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
    """Configuration d'un style de texte avec effets avancés."""
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
    
    # Nouveaux effets avancés
    wave_amplitude: float = 0.0
    wave_frequency: float = 1.0
    twist_angle: float = 0.0
    emboss_depth: float = 0.0
    smooth_iterations: int = 0
    
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
        
        # Apply effects in optimized order
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
        
        # Apply new enhanced effects
        if hasattr(style, 'wave_amplitude') and style.wave_amplitude > 0:
            result = self._apply_wave_effect(
                result,
                style.wave_amplitude,
                getattr(style, 'wave_frequency', 1.0)
            )
            
        if hasattr(style, 'twist_angle') and style.twist_angle > 0:
            result = self._apply_twist_effect(
                result,
                style.twist_angle
            )
            
        if hasattr(style, 'emboss_depth') and style.emboss_depth > 0:
            result = self._apply_emboss_effect(
                result,
                style.emboss_depth
            )
            
        # Enhanced material application
        result.visual = trimesh.visual.TextureVisuals()
        
        if style.texture_path and os.path.exists(style.texture_path):
            try:
                result.visual.material.image = trimesh.load_image(style.texture_path)
            except Exception as e:
                logger.warning(f"Failed to load texture: {e}")
            
        if style.normal_map_path and os.path.exists(style.normal_map_path):
            try:
                result.visual.material.normal_image = trimesh.load_image(style.normal_map_path)
            except Exception as e:
                logger.warning(f"Failed to load normal map: {e}")
            
        # Enhanced material properties
        material = {
            "roughness": max(0.0, min(1.0, style.roughness)),
            "metallic": max(0.0, min(1.0, style.metallic)),
            "emission": max(0.0, style.emission),
            "transparency": max(0.0, min(1.0, style.transparency)),
            "subsurface": max(0.0, min(1.0, style.subsurface)),
            "color": [max(0.0, min(1.0, c)) for c in style.color]
        }
        
        result.visual.material.kwargs.update(material)
        
        # Apply smoothing if needed
        if hasattr(style, 'smooth_iterations') and style.smooth_iterations > 0:
            result = self._apply_smoothing(result, style.smooth_iterations)
        
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
    
    def _apply_wave_effect(
        self,
        mesh: trimesh.Trimesh,
        amplitude: float,
        frequency: float = 1.0
    ) -> trimesh.Trimesh:
        """Applique un effet d'ondulation au maillage."""
        vertices = mesh.vertices.copy()
        
        # Calculer les bounds pour normaliser
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        size = max_bounds - min_bounds
        
        for i in range(len(vertices)):
            pos = vertices[i]
            # Normaliser la position
            norm_pos = (pos - min_bounds) / (size + 1e-8)
            
            # Appliquer l'ondulation sur l'axe Z basée sur X et Y
            wave = amplitude * np.sin(2 * np.pi * frequency * norm_pos[0]) * np.cos(2 * np.pi * frequency * norm_pos[1])
            vertices[i][2] += wave
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _apply_twist_effect(
        self,
        mesh: trimesh.Trimesh,
        angle: float
    ) -> trimesh.Trimesh:
        """Applique un effet de torsion au maillage."""
        vertices = mesh.vertices.copy()
        
        # Calculer les bounds
        min_z = np.min(vertices[:, 2])
        max_z = np.max(vertices[:, 2])
        height = max_z - min_z
        
        if height == 0:
            return mesh
            
        for i in range(len(vertices)):
            pos = vertices[i]
            # Calculer l'angle de rotation basé sur la hauteur
            normalized_height = (pos[2] - min_z) / height
            rotation_angle = angle * normalized_height
            
            # Appliquer la rotation autour de l'axe Z
            cos_a = np.cos(rotation_angle)
            sin_a = np.sin(rotation_angle)
            
            x_new = pos[0] * cos_a - pos[1] * sin_a
            y_new = pos[0] * sin_a + pos[1] * cos_a
            
            vertices[i][0] = x_new
            vertices[i][1] = y_new
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _apply_emboss_effect(
        self,
        mesh: trimesh.Trimesh,
        depth: float
    ) -> trimesh.Trimesh:
        """Applique un effet de relief/emboss au maillage."""
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals
        
        # Calculer la courbure locale pour chaque vertex
        from scipy.spatial import cKDTree
        
        tree = cKDTree(vertices)
        
        for i in range(len(vertices)):
            # Trouver les voisins proches
            distances, indices = tree.query(vertices[i], k=min(10, len(vertices)))
            
            if len(indices) > 1:
                # Calculer la courbure moyenne locale
                neighbor_positions = vertices[indices[1:]]  # Exclure le point lui-même
                center = np.mean(neighbor_positions, axis=0)
                
                # Direction vers le centre des voisins
                direction = center - vertices[i]
                direction_norm = np.linalg.norm(direction)
                
                if direction_norm > 0:
                    direction = direction / direction_norm
                    # Calculer le facteur d'emboss basé sur la normale
                    emboss_factor = np.dot(normals[i], direction) * depth
                    vertices[i] += normals[i] * emboss_factor
                    
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _apply_smoothing(
        self,
        mesh: trimesh.Trimesh,
        iterations: int
    ) -> trimesh.Trimesh:
        """Applique un lissage Laplacien au maillage."""
        try:
            # Utiliser le lissage intégré de trimesh
            smoothed = mesh.smoothed()
            return smoothed
        except:
            # Fallback vers un lissage manuel simple
            vertices = mesh.vertices.copy()
            
            for _ in range(iterations):
                new_vertices = vertices.copy()
                
                for i in range(len(vertices)):
                    # Trouver les faces adjacentes à ce vertex
                    adjacent_faces = mesh.faces[np.any(mesh.faces == i, axis=1)]
                    
                    if len(adjacent_faces) > 0:
                        # Trouver tous les vertices voisins
                        neighbor_indices = np.unique(adjacent_faces.flatten())
                        neighbor_indices = neighbor_indices[neighbor_indices != i]
                        
                        if len(neighbor_indices) > 0:
                            # Moyenne des positions des voisins
                            neighbor_center = np.mean(vertices[neighbor_indices], axis=0)
                            # Mélange avec la position actuelle
                            new_vertices[i] = 0.7 * vertices[i] + 0.3 * neighbor_center
                
                vertices = new_vertices
            
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
    ),
    
    "vagues": TextStyle(
        name="vagues",
        wave_amplitude=0.1,
        wave_frequency=2.0,
        roughness=0.3,
        color=(0.2, 0.6, 1.0)
    ),
    
    "torsade": TextStyle(
        name="torsade",
        twist_angle=0.5,
        bevel_amount=0.08,
        metallic=0.7,
        color=(0.8, 0.4, 0.1)
    ),
    
    "relief": TextStyle(
        name="relief",
        emboss_depth=0.15,
        bevel_amount=0.05,
        roughness=0.6,
        color=(0.7, 0.7, 0.5)
    ),
    
    "lisse": TextStyle(
        name="lisse",
        smooth_iterations=3,
        roughness=0.1,
        metallic=0.3,
        color=(0.9, 0.9, 0.9)
    ),
    
    "plasma": TextStyle(
        name="plasma",
        wave_amplitude=0.05,
        wave_frequency=3.0,
        emission=0.8,
        noise_scale=0.1,
        color=(1.0, 0.2, 0.8)
    ),
    
    "cristal_vivant": TextStyle(
        name="cristal_vivant",
        bevel_amount=0.12,
        twist_angle=0.2,
        transparency=0.7,
        emission=0.3,
        color=(0.8, 1.0, 0.9)
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
