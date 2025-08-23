"""
Exemple d'utilisation du système d'animation pour le texte 3D.
"""

import sys
import os
import time
import numpy as np
import trimesh
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_models.text_animator import TextAnimator, GPUMeshProcessor
from ai_models.text_to_mesh import create_text_mesh

def demo_animation():
    """Démontre les différentes animations disponibles."""
    
    # Créer un texte 3D
    text_mesh = create_text_mesh(
        text="MacForge",
        font_size=72,
        depth=10,
        centered=True
    )
    
    # Initialiser l'animateur et le processeur GPU
    animator = TextAnimator()
    gpu_processor = GPUMeshProcessor()
    
    # Créer différentes animations
    
    # 1. Rotation autour de l'axe Y
    rotation = animator.create_rotation_animation(
        duration=2.0,
        axis=np.array([0, 1, 0]),
        angle_range=(0, 2*np.pi),
        loop=True
    )
    animator.add_animation("rotation", rotation)
    
    # 2. Animation de vague
    wave = animator.create_wave_animation(
        duration=1.5,
        amplitude=2.0,
        frequency=2.0,
        direction=np.array([0, 1, 0]),
        loop=True
    )
    animator.add_animation("wave", wave)
    
    # 3. Animation de rebond
    bounce = animator.create_bounce_animation(
        duration=1.0,
        height=5.0,
        num_bounces=2,
        damping=0.3,
        loop=True
    )
    animator.add_animation("bounce", bounce)
    
    # Convertir le maillage pour le GPU
    vertices = gpu_processor.process_mesh(text_mesh)
    
    # Simuler l'animation pendant quelques secondes
    start_time = time.time()
    frame_count = 0
    try:
        while frame_count < 180:  # 3 secondes à 60 FPS
            # Calculer le delta time
            current_time = time.time()
            delta_time = 1/60.0  # Fixed timestep
            
            # Appliquer les animations
            animated_mesh = animator.apply_animations(text_mesh, delta_time)
            
            # Ici, vous pouvez afficher le maillage avec votre moteur de rendu
            # Pour cet exemple, nous sauvegardons quelques frames clés
            if frame_count % 60 == 0:  # Sauvegarder une frame par seconde
                output_path = f"/tmp/animation_frame_{frame_count}.ply"
                animated_mesh.export(output_path)
                print(f"Frame {frame_count} sauvegardée dans {output_path}")
            
            frame_count += 1
            
            # Attendre pour maintenir 60 FPS
            time_to_wait = start_time + (frame_count/60.0) - time.time()
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                
    except KeyboardInterrupt:
        print("\nAnimation interrompue par l'utilisateur")
    
    print(f"\nAnimation terminée: {frame_count} frames générées")
    
if __name__ == "__main__":
    demo_animation()
