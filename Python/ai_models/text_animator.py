"""
Module d'animation pour les textes 3D.
"""

import numpy as np
import trimesh
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
from enum import Enum
import quaternion  # numpy-quaternion
from scipy.spatial.transform import Slerp, Rotation
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnimationType(Enum):
    """Types d'animations disponibles."""
    MORPHING = "morphing"
    ROTATION = "rotation"
    TRANSLATION = "translation"
    SCALE = "scale"
    WAVE = "wave"
    BOUNCE = "bounce"
    FADE = "fade"
    CUSTOM = "custom"

@dataclass
class AnimationKeyframe:
    """Point clé d'une animation."""
    time: float
    value: np.ndarray
    easing: str = "linear"

@dataclass
class AnimationClip:
    """Séquence d'animation."""
    type: AnimationType
    keyframes: List[AnimationKeyframe]
    duration: float
    loop: bool = False
    reverse: bool = False
    
class TextAnimator:
    """Gestionnaire d'animations pour le texte 3D."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.animations: Dict[str, AnimationClip] = {}
        self.current_time = 0.0
        
    def add_animation(
        self,
        name: str,
        animation: AnimationClip
    ):
        """Ajoute une animation."""
        self.animations[name] = animation
        
    def remove_animation(self, name: str):
        """Supprime une animation."""
        if name in self.animations:
            del self.animations[name]
            
    @staticmethod
    def create_rotation_animation(
        duration: float,
        axis: np.ndarray = np.array([0, 1, 0]),
        angle_range: Tuple[float, float] = (0, 2*np.pi),
        num_keyframes: int = 60,
        loop: bool = True
    ) -> AnimationClip:
        """Crée une animation de rotation."""
        keyframes = []
        times = np.linspace(0, duration, num_keyframes)
        angles = np.linspace(angle_range[0], angle_range[1], num_keyframes)
        
        for t, angle in zip(times, angles):
            rot = Rotation.from_rotvec(axis * angle)
            keyframes.append(AnimationKeyframe(
                time=t,
                value=rot.as_matrix().flatten()
            ))
            
        return AnimationClip(
            type=AnimationType.ROTATION,
            keyframes=keyframes,
            duration=duration,
            loop=loop
        )
        
    @staticmethod
    def create_wave_animation(
        duration: float,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        direction: np.ndarray = np.array([0, 1, 0]),
        num_keyframes: int = 60,
        loop: bool = True
    ) -> AnimationClip:
        """Crée une animation d'onde."""
        keyframes = []
        times = np.linspace(0, duration, num_keyframes)
        
        for t in times:
            offset = direction * amplitude * np.sin(2*np.pi*frequency*t/duration)
            keyframes.append(AnimationKeyframe(
                time=t,
                value=offset
            ))
            
        return AnimationClip(
            type=AnimationType.WAVE,
            keyframes=keyframes,
            duration=duration,
            loop=loop
        )
        
    @staticmethod
    def create_bounce_animation(
        duration: float,
        height: float = 1.0,
        num_bounces: int = 3,
        damping: float = 0.5,
        num_keyframes: int = 60,
        loop: bool = True
    ) -> AnimationClip:
        """Crée une animation de rebond."""
        keyframes = []
        times = np.linspace(0, duration, num_keyframes)
        
        for t in times:
            phase = (t/duration) * (2*np.pi*num_bounces)
            amplitude = height * np.exp(-damping*t/duration)
            y = amplitude * np.abs(np.sin(phase))
            
            keyframes.append(AnimationKeyframe(
                time=t,
                value=np.array([0, y, 0])
            ))
            
        return AnimationClip(
            type=AnimationType.BOUNCE,
            keyframes=keyframes,
            duration=duration,
            loop=loop
        )
        
    def _interpolate_keyframes(
        self,
        keyframes: List[AnimationKeyframe],
        current_time: float,
        loop: bool
    ) -> np.ndarray:
        """Interpole entre les keyframes."""
        if not keyframes:
            return np.array([0, 0, 0])
            
        # Gérer le bouclage
        if loop:
            current_time = current_time % keyframes[-1].time
            
        # Trouver les keyframes encadrants
        next_idx = 0
        for i, kf in enumerate(keyframes):
            if kf.time > current_time:
                next_idx = i
                break
                
        if next_idx == 0:
            return keyframes[0].value
            
        prev_idx = next_idx - 1
        prev_kf = keyframes[prev_idx]
        next_kf = keyframes[next_idx]
        
        # Calculer le facteur d'interpolation
        t = (current_time - prev_kf.time) / (next_kf.time - prev_kf.time)
        
        # Appliquer l'easing
        if prev_kf.easing == "ease-in":
            t = t * t
        elif prev_kf.easing == "ease-out":
            t = 1 - (1 - t) * (1 - t)
        elif prev_kf.easing == "ease-in-out":
            t = 0.5 * (1 - np.cos(np.pi * t))
            
        # Interpoler les valeurs
        return (1 - t) * prev_kf.value + t * next_kf.value
        
    def apply_animations(
        self,
        mesh: trimesh.Trimesh,
        delta_time: float
    ) -> trimesh.Trimesh:
        """
        Applique toutes les animations actives au maillage.
        
        Args:
            mesh: Maillage à animer
            delta_time: Temps écoulé depuis la dernière frame
            
        Returns:
            Maillage animé
        """
        result = mesh.copy()
        self.current_time += delta_time
        
        # Appliquer chaque animation
        for anim in self.animations.values():
            transform = np.eye(4)
            
            if anim.type == AnimationType.ROTATION:
                # Interpoler la rotation
                rot_matrix = self._interpolate_keyframes(
                    anim.keyframes,
                    self.current_time,
                    anim.loop
                ).reshape(3, 3)
                transform[:3, :3] = rot_matrix
                
            elif anim.type == AnimationType.TRANSLATION:
                # Interpoler la translation
                translation = self._interpolate_keyframes(
                    anim.keyframes,
                    self.current_time,
                    anim.loop
                )
                transform[:3, 3] = translation
                
            elif anim.type == AnimationType.SCALE:
                # Interpoler l'échelle
                scale = self._interpolate_keyframes(
                    anim.keyframes,
                    self.current_time,
                    anim.loop
                )
                for i in range(3):
                    transform[i, i] = scale[i]
                    
            elif anim.type in [AnimationType.WAVE, AnimationType.BOUNCE]:
                # Appliquer le déplacement vertex par vertex
                offset = self._interpolate_keyframes(
                    anim.keyframes,
                    self.current_time,
                    anim.loop
                )
                result.vertices += offset
                continue
                
            # Appliquer la transformation
            result.apply_transform(transform)
            
        return result
        
    def save_animations(self, filepath: str):
        """Sauvegarde les animations au format JSON."""
        data = {
            name: {
                "type": anim.type.value,
                "duration": anim.duration,
                "loop": anim.loop,
                "reverse": anim.reverse,
                "keyframes": [
                    {
                        "time": kf.time,
                        "value": kf.value.tolist(),
                        "easing": kf.easing
                    }
                    for kf in anim.keyframes
                ]
            }
            for name, anim in self.animations.items()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load_animations(self, filepath: str):
        """Charge les animations depuis un fichier JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.animations.clear()
        
        for name, anim_data in data.items():
            keyframes = [
                AnimationKeyframe(
                    time=kf["time"],
                    value=np.array(kf["value"]),
                    easing=kf["easing"]
                )
                for kf in anim_data["keyframes"]
            ]
            
            self.animations[name] = AnimationClip(
                type=AnimationType(anim_data["type"]),
                keyframes=keyframes,
                duration=anim_data["duration"],
                loop=anim_data["loop"],
                reverse=anim_data["reverse"]
            )
            
class GPUMeshProcessor:
    """Processeur de maillages optimisé pour GPU."""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
    def process_mesh(
        self,
        mesh: trimesh.Trimesh,
        compute_normals: bool = True
    ) -> torch.Tensor:
        """
        Convertit un maillage Trimesh en tenseurs GPU.
        
        Args:
            mesh: Maillage à convertir
            compute_normals: Calculer les normales
            
        Returns:
            Tenseur des vertices
        """
        # Convertir en tenseurs
        vertices = torch.tensor(
            mesh.vertices,
            dtype=torch.float32,
            device=self.device
        )
        faces = torch.tensor(
            mesh.faces,
            dtype=torch.int64,
            device=self.device
        )
        
        if compute_normals:
            # Calculer les normales sur GPU
            vertices.requires_grad_(True)
            
            # Récupérer les vertices des triangles
            v0 = torch.index_select(vertices, 0, faces[:, 0])
            v1 = torch.index_select(vertices, 0, faces[:, 1])
            v2 = torch.index_select(vertices, 0, faces[:, 2])
            
            # Calculer les normales des faces
            face_normals = torch.cross(v1 - v0, v2 - v0)
            face_normals = torch.nn.functional.normalize(face_normals, dim=1)
            
            # Accumuler les normales par vertex
            vertex_normals = torch.zeros_like(vertices)
            for i in range(3):
                vertex_normals.index_add_(
                    0, faces[:, i],
                    face_normals
                )
                
            # Normaliser
            vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)
            
            # Ajouter les normales aux attributs
            mesh.vertex_normals = vertex_normals.detach().cpu().numpy()
            
        return vertices
        
    def apply_transform(
        self,
        vertices: torch.Tensor,
        transform: np.ndarray
    ) -> torch.Tensor:
        """Applique une transformation sur GPU."""
        transform = torch.tensor(
            transform,
            dtype=torch.float32,
            device=self.device
        )
        
        # Appliquer la rotation et l'échelle
        rotated = torch.matmul(
            vertices,
            transform[:3, :3].T
        )
        
        # Appliquer la translation
        translated = rotated + transform[:3, 3]
        
        return translated
        
    def compute_morphing(
        self,
        source_vertices: torch.Tensor,
        target_vertices: torch.Tensor,
        t: float
    ) -> torch.Tensor:
        """Calcule l'interpolation entre deux maillages sur GPU."""
        return (1 - t) * source_vertices + t * target_vertices
