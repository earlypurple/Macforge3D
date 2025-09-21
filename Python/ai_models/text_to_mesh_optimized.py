"""
Version optimisée du module de génération de texte 3D.
"""

import trimesh
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from shapely.ops import triangulate
import multiprocessing
import functools
from typing import Dict, List, Tuple, Optional, Any
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .mesh_enhancer import MeshEnhancer, MeshEnhancementConfig
from .text_effects import TextEffects, TextStyle, get_style, get_available_styles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache global pour les glyphes
GLYPH_CACHE: Dict[str, Any] = {}
MESH_CACHE: Dict[str, trimesh.Trimesh] = {}


class ShapelyPen(BasePen):
    """Version optimisée du ShapelyPen."""

    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.contours = []
        self.current_contour = []
        self._points_buffer = np.zeros((1000, 2), dtype=np.float64)
        self._buffer_index = 0

        # Initialiser l'améliorateur de maillage
        self.mesh_enhancer = MeshEnhancer(
            MeshEnhancementConfig(
                resolution_factor=1.5, smoothness_weight=0.3, detail_preservation=0.8
            )
        )

    def _reset_buffer(self):
        self._buffer_index = 0

    def _add_point(self, pt):
        if self._buffer_index >= len(self._points_buffer):
            # Agrandir le buffer si nécessaire
            new_buffer = np.zeros((len(self._points_buffer) * 2, 2), dtype=np.float64)
            new_buffer[: len(self._points_buffer)] = self._points_buffer
            self._points_buffer = new_buffer

        self._points_buffer[self._buffer_index] = pt
        self._buffer_index += 1

    def _moveTo(self, pt):
        if self.current_contour:
            self.contours.append(self._points_buffer[: self._buffer_index].copy())
        self._reset_buffer()
        self._add_point(pt)

    def _lineTo(self, pt):
        self._add_point(pt)

    def _curveToOne(self, pt1, pt2, pt3):
        # Version optimisée des courbes de Bézier
        steps = 10
        start = self._points_buffer[self._buffer_index - 1]
        t = np.linspace(0, 1, steps + 1)[1:]

        # Calcul vectorisé
        t1 = (1 - t) ** 3
        t2 = 3 * (1 - t) ** 2 * t
        t3 = 3 * (1 - t) * t**2
        t4 = t**3

        x = t1 * start[0] + t2 * pt1[0] + t3 * pt2[0] + t4 * pt3[0]
        y = t1 * start[1] + t2 * pt1[1] + t3 * pt2[1] + t4 * pt3[1]

        points = np.column_stack((x, y))
        for point in points:
            self._add_point(point)

    def _qCurveToOne(self, pt1, pt2):
        # Version optimisée des courbes quadratiques
        steps = 10
        start = self._points_buffer[self._buffer_index - 1]

        if pt2 is None:
            pt2 = pt1
            pt1 = ((start[0] + pt2[0]) / 2, (start[1] + pt2[1]) / 2)

        t = np.linspace(0, 1, steps + 1)[1:]
        t1 = (1 - t) ** 2
        t2 = 2 * (1 - t) * t
        t3 = t**2

        x = t1 * start[0] + t2 * pt1[0] + t3 * pt2[0]
        y = t1 * start[1] + t2 * pt1[1] + t3 * pt2[1]

        points = np.column_stack((x, y))
        for point in points:
            self._add_point(point)

        if pt2 is not None:
            self._add_point(pt2)

    def _closePath(self):
        if self._buffer_index > 0:
            if not np.array_equal(
                self._points_buffer[0], self._points_buffer[self._buffer_index - 1]
            ):
                self._add_point(self._points_buffer[0])
            self.contours.append(self._points_buffer[: self._buffer_index].copy())
        self._reset_buffer()

    def _endPath(self):
        if self._buffer_index > 0:
            self.contours.append(self._points_buffer[: self._buffer_index].copy())
        self._reset_buffer()


def process_glyph(
    glyph_data: Tuple[str, Any, float, float, str, bool],
) -> Optional[trimesh.Trimesh]:
    """
    Traite un glyphe individuel en parallèle.
    """
    char, glyph, scale, depth, glyph_name, use_cache = glyph_data

    # Vérifier le cache
    cache_key = f"{char}_{scale}_{depth}_{glyph_name}"
    if use_cache and cache_key in MESH_CACHE:
        return MESH_CACHE[cache_key]

    try:
        pen = ShapelyPen(glyph.glyphSet)
        glyph.draw(pen)

        if not pen.contours:
            return None

        # Créer les polygones
        polygons = []
        for points in pen.contours:
            if len(points) < 3:
                continue

            try:
                poly = Polygon(points)
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    poly = poly.simplify(0.1)
                    polygons.append(poly)
            except Exception:
                continue

        if not polygons:
            return None

        # Utiliser le plus grand polygone comme base
        final_polygon = max(polygons, key=lambda p: p.area)
        for poly in sorted(polygons[1:], key=lambda p: p.area, reverse=True):
            try:
                final_polygon = final_polygon.union(poly)
            except Exception:
                continue

        # Créer le maillage
        if isinstance(final_polygon, MultiPolygon):
            final_polygon = max(final_polygon.geoms, key=lambda p: p.area)

        exterior = np.array(final_polygon.exterior.coords)
        interiors = [np.array(interior.coords) for interior in final_polygon.interiors]

        vertices_2d = np.array(exterior[:, :2], dtype=np.float64)
        holes_2d = [np.array(hole[:, :2], dtype=np.float64) for hole in interiors]

        if len(vertices_2d) < 3:
            return None

        # Triangulation optimisée
        if len(vertices_2d) <= 4 or final_polygon.convex_hull.equals(final_polygon):
            # Triangulation en éventail pour les polygones simples
            triangles = np.array(
                [[0, i, i + 1] for i in range(1, len(vertices_2d) - 1)]
            )
        else:
            # Triangulation de Delaunay pour les polygones complexes
            tris = triangulate(final_polygon)
            vertices: List[Tuple[float, float]] = []
            vertex_dict = {}
            triangles = []

            for tri in tris:
                pts = np.array(tri.exterior.coords)[:-1]
                indices = []
                for pt in pts:
                    pt_tuple = tuple(pt)
                    if pt_tuple not in vertex_dict:
                        vertex_dict[pt_tuple] = len(vertices)
                        vertices.append(pt)
                    indices.append(vertex_dict[pt_tuple])
                triangles.append(indices)

            vertices_2d = np.array(vertices)
            triangles = np.array(triangles)

        # Créer le maillage 3D
        vertices_bottom = np.column_stack((vertices_2d, np.zeros(len(vertices_2d))))
        vertices_top = np.column_stack((vertices_2d, np.full(len(vertices_2d), depth)))
        vertices_3d = np.vstack((vertices_bottom, vertices_top))

        faces_bottom = triangles
        faces_top = triangles + len(vertices_2d)
        faces_top = np.fliplr(faces_top)

        # Faces latérales optimisées
        n_vertices = len(vertices_2d)
        indices = np.arange(n_vertices)
        next_indices = np.roll(indices, -1)

        v0 = indices
        v1 = next_indices
        v2 = v1 + n_vertices
        v3 = v0 + n_vertices

        faces_side_part1 = np.column_stack((v0, v1, v2))
        faces_side_part2 = np.column_stack((v2, v3, v0))
        faces_side = np.vstack((faces_side_part1, faces_side_part2))

        faces_3d = np.vstack((faces_bottom, faces_top, faces_side))

        # Créer et transformer le maillage
        mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_3d)

        if use_cache:
            MESH_CACHE[cache_key] = mesh

        return mesh

    except Exception as e:
        logger.warning(f"Erreur traitement glyphe '{char}': {str(e)}")
        return None


@lru_cache(maxsize=128)
def load_font(font_path: str) -> TTFont:
    """
    Charge une police avec mise en cache.
    """
    return TTFont(font_path)


def create_text_mesh(
    text: str,
    font: str = "Arial.ttf",
    font_size: float = 24.0,
    depth: float = 5.0,
    output_path: str = "/tmp/3d_text.ply",
    centered: bool = False,
    kerning: bool = True,
    style: str = "standard",
    use_cache: bool = True,
    n_jobs: int = -1,
    effects_config: Optional[TextStyle] = None,
) -> str:
    """
    Version optimisée de la génération de texte 3D.

    Nouveaux paramètres:
        use_cache: Active/désactive la mise en cache des glyphes
        n_jobs: Nombre de processus pour le traitement parallèle
                (-1 pour utiliser tous les cœurs)
    """
    try:
        # Trouver le fichier de police
        font_path = font
        if not os.path.exists(font_path):
            font_dirs = [
                "/System/Library/Fonts/Supplemental",
                "/usr/share/fonts/truetype/msttcorefonts",
                "/usr/share/fonts/dejavu",
            ]
            for dir_path in font_dirs:
                path_to_try = os.path.join(dir_path, font)
                if os.path.exists(path_to_try):
                    font_path = path_to_try
                    break

        # Charger la police (avec cache)
        font_obj = load_font(font_path)
        glyphSet = font_obj.getGlyphSet()

        # Facteur d'échelle
        units_per_em = font_obj["head"].unitsPerEm
        scale = font_size / units_per_em

        # Table de kerning
        kerning_table = None
        if kerning and "kern" in font_obj:
            try:
                kerning_table = font_obj["kern"].kernTables[0].kernTable
            except (AttributeError, IndexError):
                logger.warning("Table de kerning non trouvée ou invalide")

        # Suffixe de style
        style_suffix = {"bold": "Bold", "italic": "Italic"}.get(style, "")

        # Calcul de la largeur totale
        total_width = 0
        if centered:
            prev_glyph_name = None
            for char in text:
                glyph_name = font_obj.getBestCmap().get(ord(char))
                if not glyph_name:
                    total_width += font_obj["hmtx"]["a"][0]
                    continue

                if kerning and kerning_table and prev_glyph_name:
                    kern_pair = (prev_glyph_name, glyph_name)
                    if kern_pair in kerning_table:
                        total_width += kerning_table[kern_pair]

                style_glyph_name = f"{glyph_name}.{style_suffix}"
                if style != "normal" and style_glyph_name in glyphSet:
                    glyph_name = style_glyph_name

                total_width += glyphSet[glyph_name].width
                prev_glyph_name = glyph_name

        # Position initiale
        cursor_x = -total_width * scale / 2 if centered else 0

        # Préparer les données pour le traitement parallèle
        glyph_data = []
        prev_glyph_name = None

        for char in text:
            glyph_name = font_obj.getBestCmap().get(ord(char))
            if not glyph_name:
                cursor_x += font_obj["hmtx"]["a"][0] * scale
                continue

            style_glyph_name = f"{glyph_name}.{style_suffix}"
            if style != "normal" and style_glyph_name in glyphSet:
                glyph_name = style_glyph_name

            if kerning and kerning_table and prev_glyph_name:
                kern_pair = (prev_glyph_name, glyph_name)
                if kern_pair in kerning_table:
                    cursor_x += kerning_table[kern_pair] * scale

            glyph = glyphSet[glyph_name]
            glyph_data.append((char, glyph, scale, depth, glyph_name, use_cache))

            prev_glyph_name = glyph_name
            cursor_x += glyph.width * scale

        # Traitement parallèle des glyphes
        n_jobs = n_jobs if n_jobs > 0 else multiprocessing.cpu_count()
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            meshes = list(executor.map(process_glyph, glyph_data))

        # Combiner les maillages
        valid_meshes = [mesh for mesh in meshes if mesh is not None]
        if not valid_meshes:
            return "Erreur: Aucun maillage n'a été créé."

        # Positionner les maillages
        cursor_x = -total_width * scale / 2 if centered else 0
        transformed_meshes = []

        for i, (mesh, (char, glyph, scale, _, glyph_name, _)) in enumerate(
            zip(valid_meshes, glyph_data)
        ):
            if kerning and kerning_table and i > 0:
                prev_glyph = glyph_data[i - 1]
                kern_pair = (prev_glyph[4], glyph_name)
                if kern_pair in kerning_table:
                    cursor_x += kerning_table[kern_pair] * scale

            transform = np.array(
                [[1, 0, 0, cursor_x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
            )

            mesh_copy = mesh.copy()
            mesh_copy.apply_transform(transform)
            transformed_meshes.append(mesh_copy)

            cursor_x += glyph.width * scale

        # Fusionner tous les maillages
        final_mesh = trimesh.util.concatenate(transformed_meshes)

        # Améliorer le maillage avec le réseau neuronal
        try:
            logger.info("Amélioration du maillage avec IA...")
            mesh_enhancer = MeshEnhancer(
                MeshEnhancementConfig(
                    resolution_factor=1.5,
                    smoothness_weight=0.3,
                    detail_preservation=0.8,
                )
            )
            final_mesh = mesh_enhancer.enhance_mesh(final_mesh)
            logger.info("Amélioration du maillage terminée")

            # Appliquer les effets de style
            logger.info(f"Application du style '{style}'...")
            effects = TextEffects()
            style_config = effects_config or get_style(style)
            final_mesh = effects.apply_style(final_mesh, style_config)
            logger.info("Application du style terminée")

        except Exception as e:
            logger.warning(f"Erreur lors du post-traitement: {e}")

        # Sauvegarder le résultat
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        final_mesh.export(output_path)

        return output_path

    except Exception as e:
        logger.error(f"Erreur création texte 3D: {str(e)}")
        return f"Erreur: {str(e)}"
