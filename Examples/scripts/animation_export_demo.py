"""
Exemple d'utilisation de l'exportateur d'animations.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.text_animator import TextAnimator, GPUMeshProcessor
from ai_models.text_to_mesh import create_text_mesh
from ai_models.animation_exporter import AnimationExporter, ExportSettings


def demo_export():
    """Démontre l'exportation d'animations vers différents formats."""

    # Créer un texte 3D
    text_mesh = create_text_mesh(text="MacForge", font_size=72, depth=10, centered=True)

    # Initialiser l'animateur
    animator = TextAnimator()

    # Créer une animation de rotation complexe
    rotation = animator.create_rotation_animation(
        duration=2.0, axis=np.array([0, 1, 0]), angle_range=(0, 2 * np.pi), loop=True
    )
    animator.add_animation("rotation", rotation)

    # Ajouter une animation de rebond
    bounce = animator.create_bounce_animation(
        duration=1.0, height=5.0, num_bounces=2, damping=0.3, loop=True
    )
    animator.add_animation("bounce", bounce)

    # Exporter en glTF (format binaire .glb)
    print("\nExportation en glTF binaire...")
    gltf_settings = ExportSettings(
        format="gltf", binary=True, optimize_keyframes=True, fps=60
    )
    gltf_exporter = AnimationExporter(gltf_settings)
    gltf_exporter.export_animation(text_mesh, animator, Path("/tmp/animated_text.glb"))
    print("Animation exportée en glTF binaire: /tmp/animated_text.glb")

    # Exporter en FBX
    print("\nExportation en FBX...")
    fbx_settings = ExportSettings(
        format="fbx", embed_textures=True, optimize_keyframes=True, fps=60
    )
    fbx_exporter = AnimationExporter(fbx_settings)
    fbx_exporter.export_animation(text_mesh, animator, Path("/tmp/animated_text.fbx"))
    print("Animation exportée en FBX: /tmp/animated_text.fbx")


if __name__ == "__main__":
    demo_export()
