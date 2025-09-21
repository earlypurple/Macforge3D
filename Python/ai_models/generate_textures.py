"""
Script de génération de textures procédurales.
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_wood_texture(
    size: int = 1024,
    rings: int = 20,
    noise_scale: float = 0.3,
    color1: tuple = (0.6, 0.3, 0.1),
    color2: tuple = (0.8, 0.5, 0.2),
) -> Image.Image:
    """Génère une texture de bois procédurale."""
    # Créer la base
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Générer les anneaux
    R = np.sqrt(X**2 + Y**2)
    ring_pattern: np.ndarray = np.sin(R * rings * np.pi)

    # Ajouter du bruit
    noise = np.random.rand(size, size)
    texture: np.ndarray = ring_pattern + noise_scale * noise

    # Normaliser
    texture = (texture - texture.min()) / (texture.max() - texture.min())

    # Convertir en RGB
    rgb = np.zeros((size, size, 3))
    for i in range(3):
        rgb[:, :, i] = texture * color2[i] + (1 - texture) * color1[i]

    # Convertir en image
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return img


def create_holographic_texture(
    size: int = 1024, frequency: float = 20.0, saturation: float = 0.8
) -> Image.Image:
    """Génère une texture holographique."""
    # Créer la base
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    # Générer le motif de base
    pattern = np.sin(X * frequency) * np.cos(Y * frequency)

    # Créer le dégradé de couleurs
    hue = (pattern + 1) / 2  # Normaliser entre 0 et 1

    # Convertir HSV en RGB
    def hsv_to_rgb(h, s, v):
        h = h * 6
        i = int(h)
        f = h - i
        p = v * (1 - s)
        q = v * (1 - s * f)
        t = v * (1 - s * (1 - f))

        if i == 0:
            return v, t, p
        elif i == 1:
            return q, v, p
        elif i == 2:
            return p, v, t
        elif i == 3:
            return p, q, v
        elif i == 4:
            return t, p, v
        else:
            return v, p, q

    rgb = np.zeros((size, size, 3))
    for i in range(size):
        for j in range(size):
            rgb[i, j] = hsv_to_rgb(hue[i, j], saturation, 1.0)

    # Convertir en image
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    return img


def create_metal_texture(
    size: int = 1024, roughness: float = 0.2, scale: float = 50.0
) -> Image.Image:
    """Génère une texture métallique."""
    # Générer le bruit de base
    noise = np.random.rand(size, size)

    # Appliquer un flou gaussien
    from scipy.ndimage import gaussian_filter

    smooth = gaussian_filter(noise, sigma=scale)

    # Ajouter des micro-détails
    detail = np.random.rand(size, size) * roughness
    texture = smooth + detail

    # Normaliser
    texture = (texture - texture.min()) / (texture.max() - texture.min())

    # Convertir en image
    img = Image.fromarray((texture * 255).astype(np.uint8))
    return img


def generate_textures(output_dir: str = "textures") -> None:
    """Génère toutes les textures nécessaires."""
    # Convertir en Path pour utiliser les opérateurs de chemin
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Génération des textures de base...")

    # Bois
    wood = create_wood_texture()
    wood.save(str(output_path / "wood.jpg"))

    # Normal map pour le bois
    wood_normal = create_metal_texture(roughness=0.4, scale=20.0)
    wood_normal.save(str(output_path / "wood_normal.jpg"))

    # Holographique
    holo = create_holographic_texture()
    holo.save(str(output_path / "holographic.jpg"))

    # Métal
    metal = create_metal_texture()
    metal.save(str(output_path / "metal.jpg"))

    logger.info("Génération des textures terminée")


if __name__ == "__main__":
    generate_textures()
