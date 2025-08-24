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
    
    # Effets ultra-avancés ajoutés
    tessellation_level: int = 0
    fractal_intensity: float = 0.0
    fractal_scale: float = 1.0
    plasma_frequency: float = 1.0
    plasma_amplitude: float = 0.0
    vertex_displacement: float = 0.0
    edge_hardening: float = 0.0
    surface_roughening: float = 0.0
    
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
            
        # Apply ultra-advanced effects
        if hasattr(style, 'tessellation_level') and style.tessellation_level > 0:
            result = self._apply_tessellation(
                result,
                style.tessellation_level
            )
            
        if hasattr(style, 'fractal_intensity') and style.fractal_intensity > 0:
            result = self._apply_fractal_effect(
                result,
                style.fractal_intensity,
                getattr(style, 'fractal_scale', 1.0)
            )
            
        if hasattr(style, 'plasma_amplitude') and style.plasma_amplitude > 0:
            result = self._apply_plasma_effect(
                result,
                getattr(style, 'plasma_frequency', 1.0),
                style.plasma_amplitude
            )
            
        if hasattr(style, 'vertex_displacement') and style.vertex_displacement > 0:
            result = self._apply_vertex_displacement(
                result,
                style.vertex_displacement
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
    
    def _apply_tessellation(
        self,
        mesh: trimesh.Trimesh,
        level: int
    ) -> trimesh.Trimesh:
        """Applique une tessellation adaptative au maillage."""
        if level <= 0:
            return mesh
            
        result = mesh.copy()
        
        for _ in range(level):
            try:
                # Subdivision Catmull-Clark si disponible
                result = result.subdivide()
            except:
                # Fallback: subdivision simple
                result = self._simple_subdivide(result)
                
        return result
    
    def _simple_subdivide(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Subdivision simple pour fallback."""
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        new_vertices = []
        new_faces = []
        
        # Ajouter les vertices existants
        new_vertices.extend(vertices)
        vertex_count = len(vertices)
        
        # Pour chaque face, ajouter un point au centre et créer de nouvelles faces
        for face in faces:
            # Calculer le centre de la face
            center = np.mean(vertices[face], axis=0)
            new_vertices.append(center)
            center_idx = vertex_count
            vertex_count += 1
            
            # Créer de nouvelles faces triangulaires
            for i in range(3):
                new_face = [face[i], face[(i+1)%3], center_idx]
                new_faces.append(new_face)
        
        return trimesh.Trimesh(vertices=np.array(new_vertices), faces=np.array(new_faces))
    
    def _apply_fractal_effect(
        self,
        mesh: trimesh.Trimesh,
        intensity: float,
        scale: float = 1.0
    ) -> trimesh.Trimesh:
        """Applique un effet fractal au maillage."""
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals
        
        # Générer du bruit fractal multi-octave
        for i in range(len(vertices)):
            pos = vertices[i] * scale
            
            # Bruit fractal avec plusieurs octaves
            fractal_value = 0.0
            amplitude = intensity
            frequency = 1.0
            
            for octave in range(4):  # 4 octaves pour plus de détail
                noise_val = self._simple_noise(pos * frequency)
                fractal_value += noise_val * amplitude
                amplitude *= 0.5
                frequency *= 2.0
            
            # Appliquer le déplacement dans la direction de la normale
            vertices[i] += normals[i] * fractal_value
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _apply_plasma_effect(
        self,
        mesh: trimesh.Trimesh,
        frequency: float,
        amplitude: float
    ) -> trimesh.Trimesh:
        """Applique un effet plasma dynamique au maillage."""
        vertices = mesh.vertices.copy()
        
        # Calculer les bounds pour normaliser
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        size = max_bounds - min_bounds
        
        for i in range(len(vertices)):
            pos = vertices[i]
            # Normaliser la position
            norm_pos = (pos - min_bounds) / (size + 1e-8)
            
            # Effet plasma avec plusieurs fonctions sinusoïdales
            plasma = (
                np.sin(norm_pos[0] * frequency * np.pi) +
                np.sin(norm_pos[1] * frequency * np.pi) +
                np.sin((norm_pos[0] + norm_pos[1]) * frequency * 0.5 * np.pi) +
                np.sin(np.sqrt(norm_pos[0]**2 + norm_pos[1]**2) * frequency * np.pi)
            ) / 4.0
            
            # Appliquer l'effet sur l'axe Z
            vertices[i][2] += plasma * amplitude
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _apply_vertex_displacement(
        self,
        mesh: trimesh.Trimesh,
        displacement: float
    ) -> trimesh.Trimesh:
        """Applique un déplacement aléatoire contrôlé aux vertices."""
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals
        
        # Générer des déplacements aléatoires avec distribution gaussienne
        np.random.seed(42)  # Pour la reproductibilité
        random_displacements = np.random.normal(0, displacement, len(vertices))
        
        for i in range(len(vertices)):
            # Appliquer le déplacement dans la direction de la normale
            vertices[i] += normals[i] * random_displacements[i]
            
        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _simple_noise(self, pos: np.ndarray) -> float:
        """Fonction de bruit simple pour le fallback."""
        # Bruit pseudo-aléatoire basé sur la position
        x, y, z = pos
        return (
            np.sin(x * 12.9898 + y * 78.233 + z * 37.719) * 43758.5453 % 1.0 - 0.5
        ) * 2.0
    
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
        """Applique un lissage Laplacien amélioré avec préservation des arêtes."""
        try:
            # Essayer d'utiliser le lissage intégré de trimesh
            smoothed = mesh.smoothed()
            return smoothed
        except:
            # Fallback vers un lissage manuel amélioré
            vertices = mesh.vertices.copy()
            
            # Construire un graphe de connectivité optimisé
            connectivity = self._build_vertex_connectivity(mesh)
            
            # Détecter les arêtes importantes (points de courbure élevée)
            edge_vertices = self._detect_high_curvature_vertices(mesh, threshold=0.5)
            
            for iteration in range(iterations):
                new_vertices = vertices.copy()
                
                # Traitement par batch pour l'efficacité
                batch_size = min(1000, len(vertices))
                
                for start_idx in range(0, len(vertices), batch_size):
                    end_idx = min(start_idx + batch_size, len(vertices))
                    
                    for i in range(start_idx, end_idx):
                        if i in connectivity and len(connectivity[i]) > 0:
                            neighbor_indices = connectivity[i]
                            
                            if len(neighbor_indices) > 0:
                                # Calculer la moyenne pondérée des voisins
                                neighbor_positions = vertices[neighbor_indices]
                                neighbor_center = np.mean(neighbor_positions, axis=0)
                                
                                # Facteur de lissage adaptatif
                                # Préserver les arêtes importantes
                                if i in edge_vertices:
                                    smoothing_factor = 0.1  # Lissage minimal pour les arêtes
                                else:
                                    # Calculer la courbure locale pour adapter le lissage
                                    curvature = self._compute_local_curvature(vertices, i, neighbor_indices)
                                    smoothing_factor = min(0.5, 0.3 * (1.0 - curvature))
                                
                                # Appliquer le lissage
                                new_vertices[i] = (
                                    (1 - smoothing_factor) * vertices[i] + 
                                    smoothing_factor * neighbor_center
                                )
                
                vertices = new_vertices
            
            return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    
    def _build_vertex_connectivity(self, mesh: trimesh.Trimesh) -> Dict[int, List[int]]:
        """Construit un dictionnaire de connectivité vertex-to-vertex optimisé."""
        connectivity = {i: [] for i in range(len(mesh.vertices))}
        
        # Utiliser les arêtes pour construire la connectivité
        edges = mesh.edges_unique
        for edge in edges:
            v0, v1 = edge
            connectivity[v0].append(v1)
            connectivity[v1].append(v0)
        
        # Éliminer les doublons
        for vertex_id in connectivity:
            connectivity[vertex_id] = list(set(connectivity[vertex_id]))
        
        return connectivity
    
    def _detect_high_curvature_vertices(self, mesh: trimesh.Trimesh, threshold: float = 0.5) -> set:
        """Détecte les vertices avec une courbure élevée (arêtes importantes)."""
        high_curvature_vertices = set()
        
        try:
            # Utiliser la courbure gaussienne si disponible
            if hasattr(mesh, 'vertex_defects'):
                vertex_defects = mesh.vertex_defects
                mean_defect = np.mean(np.abs(vertex_defects))
                
                for i, defect in enumerate(vertex_defects):
                    if abs(defect) > mean_defect * threshold:
                        high_curvature_vertices.add(i)
        except:
            # Fallback: détecter basé sur l'angle entre faces adjacentes
            for i in range(len(mesh.vertices)):
                adjacent_faces = mesh.faces[np.any(mesh.faces == i, axis=1)]
                
                if len(adjacent_faces) >= 2:
                    # Calculer les normales des faces adjacentes
                    normals = []
                    for face in adjacent_faces:
                        v0, v1, v2 = mesh.vertices[face]
                        normal = np.cross(v1 - v0, v2 - v0)
                        normal = normal / np.linalg.norm(normal)
                        normals.append(normal)
                    
                    # Calculer l'angle maximum entre normales
                    max_angle = 0
                    for j in range(len(normals)):
                        for k in range(j + 1, len(normals)):
                            dot_product = np.clip(np.dot(normals[j], normals[k]), -1, 1)
                            angle = np.arccos(dot_product)
                            max_angle = max(max_angle, angle)
                    
                    if max_angle > threshold:
                        high_curvature_vertices.add(i)
        
        return high_curvature_vertices
    
    def _compute_local_curvature(self, vertices: np.ndarray, vertex_idx: int, neighbors: List[int]) -> float:
        """Calcule la courbure locale d'un vertex."""
        if len(neighbors) < 3:
            return 0.0
        
        vertex_pos = vertices[vertex_idx]
        neighbor_positions = vertices[neighbors]
        
        # Calculer la distance moyenne aux voisins
        distances = np.linalg.norm(neighbor_positions - vertex_pos, axis=1)
        mean_distance = np.mean(distances)
        
        # Calculer la variance comme mesure de courbure
        distance_variance = np.var(distances)
        
        # Normaliser la courbure
        curvature = min(1.0, distance_variance / max(mean_distance, 0.001))
        
        return curvature
        
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
    
    "metal": TextStyle(
        name="metal",
        bevel_amount=0.08,
        roughness=0.4,
        metallic=0.9,
        color=(0.7, 0.7, 0.7)
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
    ),
    
    # Nouveaux styles ultra-avancés
    "tesselle": TextStyle(
        name="tesselle",
        tessellation_level=2,
        bevel_amount=0.08,
        roughness=0.4,
        metallic=0.6,
        color=(0.6, 0.8, 0.9)
    ),
    
    "fractal": TextStyle(
        name="fractal",
        fractal_intensity=0.1,
        fractal_scale=2.0,
        roughness=0.8,
        color=(0.5, 0.3, 0.8)
    ),
    
    "plasma_avance": TextStyle(
        name="plasma_avance",
        plasma_frequency=2.5,
        plasma_amplitude=0.08,
        emission=0.6,
        transparency=0.3,
        color=(1.0, 0.3, 0.6)
    ),
    
    "chaotique": TextStyle(
        name="chaotique",
        vertex_displacement=0.05,
        noise_scale=0.15,
        roughness=0.9,
        color=(0.4, 0.2, 0.1)
    ),
    
    "ultra_moderne": TextStyle(
        name="ultra_moderne",
        tessellation_level=1,
        plasma_frequency=1.5,
        plasma_amplitude=0.03,
        metallic=0.9,
        roughness=0.2,
        emission=0.1,
        color=(0.1, 0.9, 1.0)
    ),
    
    "organique": TextStyle(
        name="organique",
        fractal_intensity=0.08,
        fractal_scale=1.5,
        wave_amplitude=0.04,
        wave_frequency=1.8,
        subsurface=0.3,
        color=(0.3, 0.8, 0.4)
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

def validate_style(style: TextStyle) -> Tuple[bool, List[str]]:
    """Valide un style et retourne les erreurs/avertissements."""
    errors = []
    warnings = []
    
    # Vérifications des valeurs
    if style.bevel_amount < 0 or style.bevel_amount > 1:
        errors.append("bevel_amount doit être entre 0 et 1")
    
    if style.bevel_segments < 1 or style.bevel_segments > 10:
        warnings.append("bevel_segments recommandé entre 1 et 10")
    
    if style.roughness < 0 or style.roughness > 1:
        errors.append("roughness doit être entre 0 et 1")
    
    if style.metallic < 0 or style.metallic > 1:
        errors.append("metallic doit être entre 0 et 1")
    
    if style.emission < 0:
        errors.append("emission doit être >= 0")
    
    if style.transparency < 0 or style.transparency > 1:
        errors.append("transparency doit être entre 0 et 1")
    
    if style.subsurface < 0 or style.subsurface > 1:
        errors.append("subsurface doit être entre 0 et 1")
    
    # Vérifications des couleurs
    for i, c in enumerate(style.color):
        if c < 0 or c > 1:
            errors.append(f"color[{i}] doit être entre 0 et 1")
    
    # Vérifications des nouveaux paramètres
    if hasattr(style, 'tessellation_level') and style.tessellation_level > 3:
        warnings.append("tessellation_level > 3 peut causer des problèmes de performance")
    
    if hasattr(style, 'fractal_intensity') and style.fractal_intensity > 1:
        warnings.append("fractal_intensity > 1 peut créer des déformations extrêmes")
    
    return len(errors) == 0, errors + warnings

def create_preview_style(base_style: str, modifications: Dict[str, Any]) -> TextStyle:
    """Crée un style personnalisé basé sur un style existant."""
    if base_style not in PREDEFINED_STYLES:
        raise ValueError(f"Style de base '{base_style}' non trouvé")
    
    base = PREDEFINED_STYLES[base_style]
    
    # Créer une copie du style de base
    import copy
    new_style = copy.deepcopy(base)
    new_style.name = f"{base_style}_personnalise"
    
    # Appliquer les modifications
    for key, value in modifications.items():
        if hasattr(new_style, key):
            setattr(new_style, key, value)
        else:
            logger.warning(f"Attribut '{key}' non reconnu pour TextStyle")
    
    # Valider le nouveau style
    is_valid, messages = validate_style(new_style)
    if not is_valid:
        logger.error(f"Style personnalisé invalide: {messages}")
        raise ValueError(f"Style personnalisé invalide: {messages}")
    elif messages:
        logger.warning(f"Avertissements pour le style personnalisé: {messages}")
    
    return new_style
