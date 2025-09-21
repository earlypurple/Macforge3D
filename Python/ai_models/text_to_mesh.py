import trimesh
import numpy as np
import os
from shapely.geometry import Polygon, MultiPolygon, LinearRing
from shapely.geometry.base import BaseGeometry
from typing import List, Dict, Tuple, Optional, Union, Iterator, Any
from fontTools.ttLib import TTFont  # type: ignore
from fontTools.pens.basePen import BasePen  # type: ignore
import trimesh.path.util
from shapely.ops import triangulate


class ShapelyPen(BasePen):
    def __init__(self, glyphSet):
        super().__init__(glyphSet)
        self.contours: List[List[Tuple[float, float]]] = []
        self.current_contour: List[Tuple[float, float]] = []

    def _moveTo(self, pt):
        if self.current_contour:
            self.contours.append(self.current_contour)
        self.current_contour = [pt]

    def _lineTo(self, pt):
        self.current_contour.append(pt)

    def _curveToOne(self, pt1, pt2, pt3):
        # Approximate cubic bezier with points
        steps = 10
        start = self.current_contour[-1]
        for i in range(1, steps + 1):
            t = i / steps
            x = (
                (1 - t) ** 3 * start[0]
                + 3 * (1 - t) ** 2 * t * pt1[0]
                + 3 * (1 - t) * t**2 * pt2[0]
                + t**3 * pt3[0]
            )
            y = (
                (1 - t) ** 3 * start[1]
                + 3 * (1 - t) ** 2 * t * pt1[1]
                + 3 * (1 - t) * t**2 * pt2[1]
                + t**3 * pt3[1]
            )
            self.current_contour.append((x, y))

    def _qCurveToOne(self, pt1, pt2):
        # Approximate quadratic bezier with points
        steps = 10
        start = self.current_contour[-1]
        for i in range(1, steps + 1):
            t = i / steps
            # Handle both regular and implied control points
            if pt2 is None:  # Implied point
                pt2 = pt1
                pt1 = ((start[0] + pt2[0]) / 2, (start[1] + pt2[1]) / 2)

            x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * pt1[0] + t**2 * pt2[0]
            y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * pt1[1] + t**2 * pt2[1]
            self.current_contour.append((x, y))
        if pt2 is not None:
            self.current_contour.append(pt2)

    def _closePath(self):
        if self.current_contour:
            if self.current_contour[0] != self.current_contour[-1]:
                self.current_contour.append(self.current_contour[0])
            self.contours.append(self.current_contour)
        self.current_contour = []

    def _endPath(self):
        if self.current_contour:
            self.contours.append(self.current_contour)
        self.current_contour = []


import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_text_mesh(
    text: str,
    font: str = "Arial.ttf",
    font_size: float = 24.0,
    depth: float = 5.0,
    output_path: str = "/tmp/3d_text.ply",
    centered: bool = False,
    kerning: bool = True,
    style: str = "normal",  # "normal", "bold", or "italic"
) -> str:
    """
    Generates a 3D mesh from a given text string using font vector outlines.
    This approach uses fontTools to get high-quality vector paths for glyphs.

    Args:
        text: The text to convert to 3D
        font: Path to the font file (.ttf, .otf)
        font_size: Size of the text in units
        depth: Depth of the extrusion
        output_path: Where to save the resulting mesh
        centered: Whether to center the text around origin
        kerning: Whether to apply kerning adjustments
        style: Text style ("normal", "bold", or "italic")
    """
    try:
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
        font_obj = TTFont(font_path)
    except Exception as e:
        return f"Error: Could not load font '{font}'. Please provide a valid .ttf file. Details: {e}"

    glyphSet = font_obj.getGlyphSet()

    total_mesh = None
    cursor_x = 0

    # Scale factor to convert font units to model units
    units_per_em = font_obj["head"].unitsPerEm
    scale = font_size / units_per_em

    # Get kerning table if available and requested
    kerning_table = None
    if kerning and "kern" in font_obj:
        try:
            kerning_table = font_obj["kern"].kernTables[0].kernTable
        except (AttributeError, IndexError):
            print("Warning: Kerning table not found or invalid")

    # Handle style variants
    style_suffix = ""
    if style == "bold":
        style_suffix = "Bold"
    elif style == "italic":
        style_suffix = "Italic"

    # Calculate total width for centering if requested
    total_width = 0
    if centered:
        logger.info("Calculating total width for centering...")
        prev_glyph_name = None
        for char in text:
            logger.debug(f"Processing char '{char}' for width calculation")
            glyph_name = font_obj.getBestCmap().get(ord(char))
            if not glyph_name:
                total_width += font_obj["hmtx"]["a"][0]
                logger.debug(f"Using default width for char '{char}'")
                continue

            # Add kerning adjustment if available
            if kerning and kerning_table and prev_glyph_name:
                kern_pair = (prev_glyph_name, glyph_name)
                if kern_pair in kerning_table:
                    total_width += kerning_table[kern_pair]

            # Try style variant if available
            style_glyph_name = f"{glyph_name}.{style_suffix}"
            if style != "normal" and style_glyph_name in glyphSet:
                glyph_name = style_glyph_name

            total_width += glyphSet[glyph_name].width
            prev_glyph_name = glyph_name

    # Initial cursor position considering centering
    cursor_x = -total_width * scale / 2 if centered else 0
    prev_glyph_name = None

    for char in text:
        print(f"Processing glyph: {char}")

        glyph_name = font_obj.getBestCmap().get(ord(char))
        if not glyph_name:
            cursor_x += font_obj["hmtx"]["a"][0] * scale  # Advance by a default width
            prev_glyph_name = None
            continue

        # Try style variant if available
        style_glyph_name = f"{glyph_name}.{style_suffix}"
        if style != "normal" and style_glyph_name in glyphSet:
            glyph_name = style_glyph_name

        # Apply kerning adjustment if available
        if kerning and kerning_table and prev_glyph_name:
            kern_pair = (prev_glyph_name, glyph_name)
            if kern_pair in kerning_table:
                cursor_x += kerning_table[kern_pair] * scale

        glyph = glyphSet[glyph_name]
        prev_glyph_name = glyph_name
        print(f"  Glyph name: {glyph_name}")

        pen = ShapelyPen(glyphSet)
        glyph.draw(pen)

        print(f"  Number of contours: {len(pen.contours)}")
        for i, contour in enumerate(pen.contours):
            print(f"  Contour {i}: {len(contour)} points")

        if not pen.contours:
            print("  No contours found for this glyph")
            cursor_x += glyph.width * scale
            continue

        # Create polygons from contours
        polygons: List[Polygon] = []
        for points in pen.contours:
            if len(points) < 3:
                continue  # Skip invalid polygons
            try:
                print(f"  Creating polygon with {len(points)} points:")
                print("    Points:", points[:3], "..." if len(points) > 3 else "")
                # Convert points to numpy array for better handling
                points_array = np.array(points)
                # Ensure points form a closed loop
                if not np.array_equal(points_array[0], points_array[-1]):
                    points_array = np.vstack([points_array, points_array[0]])
                # Create Shapely polygon
                poly = Polygon(points_array)
                print(
                    f"    Valid: {poly.is_valid}, Empty: {poly.is_empty}, Area: {poly.area}"
                )
                if poly.is_valid and not poly.is_empty and poly.area > 0:
                    # Simplify slightly to remove redundant points while preserving shape
                    simple_poly = poly.simplify(0.1)
                    if isinstance(simple_poly, Polygon):
                        polygons.append(simple_poly)
                        print("    Added to polygons list")
            except Exception as e:
                print(f"    Warning: Could not create polygon: {str(e)}")
                continue  # Skip invalid polygons

        if not polygons:
            cursor_x += glyph.width * scale
            continue

        # Find the largest polygon by area and use it as the base
        sorted_polygons = sorted(polygons, key=lambda p: p.area, reverse=True)
        final_polygon: Union[Polygon, MultiPolygon] = sorted_polygons[0]
        print(f"  Using largest polygon as base, area: {final_polygon.area}")

        # All other polygons contribute to the shape
        for i, poly in enumerate(sorted_polygons[1:], 1):
            try:
                print(f"  Merging with polygon {i}, area: {poly.area}")
                union_result = final_polygon.union(poly)
                if isinstance(union_result, (Polygon, MultiPolygon)):
                    final_polygon = union_result
                    print(f"  After merge, area: {final_polygon.area}")
                else:
                    print(f"  Union result is not a Polygon or MultiPolygon, but {type(union_result).__name__}")
            except Exception as e:
                print(f"  Failed to merge polygon {i}: {str(e)}")
                continue

        try:
            print(f"  Creating mesh from polygon with area: {final_polygon.area}")
            # Convert polygon to the format trimesh expects
            if isinstance(final_polygon, MultiPolygon):
                print("  Converting MultiPolygon to largest Polygon")
                largest_polygon = max(final_polygon.geoms, key=lambda p: p.area)
                if isinstance(largest_polygon, Polygon):
                    final_polygon = largest_polygon
                else:
                    print(f"  Largest geometry is not a Polygon but {type(largest_polygon).__name__}")
                    continue

            # Get the polygon boundaries
            exterior = np.array(final_polygon.exterior.coords)
            interiors = []
            
            # Type annotation to help mypy
            interiors_list: List[np.ndarray] = []
            
            # LinearRing objects have a coords attribute that returns a CoordinateSequence
            # which supports iteration
            for interior in list(final_polygon.interiors):  # Convert to list to avoid iterator issues
                try:
                    # CoordinateSequence to numpy array via explicit list conversion
                    coords_list = list(interior.coords)
                    interiors_list.append(np.array(coords_list))
                except Exception as e:
                    print(f"  Failed to extract interior coordinates: {e}")
            
            # Update the main interiors variable
            interiors = interiors_list

            # Process exterior points first
            vertices_2d = np.array(
                [[x, y] for x, y in exterior[:, :2]], dtype=np.float64
            )
            # Process holes
            holes_2d = [
                np.array([[x, y] for x, y in hole[:, :2]], dtype=np.float64)
                for hole in interiors
            ]

            print(
                f"  Creating mesh with polygon having {len(vertices_2d)} exterior points and {len(holes_2d)} holes..."
            )
            if len(vertices_2d) >= 3:
                print("    First few exterior points:", vertices_2d[:3].tolist())

                try:
                    # Triangulation du polygone
                    if holes_2d:
                        triangles = []
                        # Trianguler le contour extérieur et les trous séparément
                        exterior_tri = triangulate(final_polygon.exterior)
                        for tri in exterior_tri:
                            pts = np.array(tri.exterior.coords)[:-1]
                            triangles.append(
                                [
                                    np.where(np.all(vertices_2d == pt, axis=1))[0][0]
                                    for pt in pts
                                ]
                            )
                    else:
                        # Trianguler le polygone simple
                        if len(vertices_2d) <= 4 or final_polygon.convex_hull.equals(
                            final_polygon
                        ):
                            # Si le polygone est convexe, trianguler en éventail
                            triangles = [
                                [0, i, i + 1] for i in range(1, len(vertices_2d) - 1)
                            ]
                        else:
                            # Sinon, utiliser la triangulation de Delaunay
                            triangles = []
                            tris = triangulate(final_polygon)
                            vertex_dict = {}  # Pour stocker les indices des points
                            current_index = 0

                            # Créer un nouveau tableau de vertices qui inclut tous les points nécessaires
                            new_vertices = []
                            for tri in tris:
                                pts = np.array(tri.exterior.coords)[:-1]
                                for pt in pts:
                                    pt_tuple = tuple(pt)
                                    if pt_tuple not in vertex_dict:
                                        vertex_dict[pt_tuple] = current_index
                                        new_vertices.append(pt)
                                        current_index += 1
                                indices = [vertex_dict[tuple(pt)] for pt in pts]
                                triangles.append(indices)

                            vertices_2d = np.array(new_vertices)

                    # Créer les vertices du maillage 3D
                    vertices_bottom = np.column_stack(
                        (vertices_2d, np.zeros(len(vertices_2d)))
                    )
                    vertices_top = np.column_stack(
                        (vertices_2d, np.full(len(vertices_2d), depth))
                    )
                    vertices_3d = np.vstack((vertices_bottom, vertices_top))

                    # Faces pour le bas
                    faces_bottom = np.array(triangles)

                    # Faces pour le haut (copie des triangles du bas avec indices décalés)
                    faces_top = faces_bottom + len(vertices_2d)
                    # Inverser l'ordre des sommets pour que les normales pointent vers le haut
                    faces_top = np.fliplr(faces_top)

                    # Faces latérales (quads divisés en triangles)
                    faces_side = []
                    for i in range(
                        len(exterior) - 1
                    ):  # -1 car le dernier point est une copie du premier
                        v0 = i
                        v1 = (i + 1) % (len(exterior) - 1)
                        v2 = v1 + len(vertices_2d)
                        v3 = v0 + len(vertices_2d)
                        faces_side.extend([[v0, v1, v2], [v2, v3, v0]])

                    # Combiner toutes les faces
                    faces_3d = np.vstack((faces_bottom, faces_top, faces_side))

                    # Créer le maillage final
                    char_mesh = trimesh.Trimesh(vertices=vertices_3d, faces=faces_3d)

                    # Appliquer la transformation pour déplacer le caractère
                    transform = np.array(
                        [[1, 0, 0, cursor_x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
                    )
                    char_mesh.apply_transform(transform)
                    char_mesh.apply_scale(scale)

                    if total_mesh is None:
                        total_mesh = char_mesh
                    else:
                        total_mesh = trimesh.util.concatenate([total_mesh, char_mesh])
                    print("    Mesh created successfully!")

                except Exception as e:
                    print(f"    Failed to create mesh: {str(e)}")
                    return f"Error: Failed to create mesh for character '{char}'. Details: {str(e)}"
            else:
                print("    Polygon has too few points for extrusion")
                return f"Error: Not enough points to create mesh for character '{char}'"

        except Exception as e:
            print(f"  Failed to create mesh: {str(e)}")
            return f"Error: Could not process character '{char}'. Details: {str(e)}"

        # Advance cursor
        cursor_x += glyph.width * scale

    if total_mesh is None:
        return "Error: No meshes were created successfully."
    if len(total_mesh.vertices) == 0 or len(total_mesh.faces) == 0:
        return "Error: Generated mesh has no geometry."

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Exportation du maillage
    total_mesh.export(output_path)
    return output_path


if __name__ == "__main__":
    print("Testing text_to_mesh.py script with fontTools...")
    # Test avec différents styles et options
    tests = [
        {
            "text": "Hello!",
            "style": "normal",
            "centered": False,
        },
        {
            "text": "Bold",
            "style": "bold",
            "centered": True,
        },
        {
            "text": "Italique",
            "style": "italic",
            "centered": True,
        },
        {
            "text": "Français",
            "style": "normal",
            "centered": True,
        },
    ]

    for i, test in enumerate(tests):
        print(f"\nTest {i+1}: {test['text']} ({test['style']})")
        path_or_error = create_text_mesh(
            text=str(test["text"]),
            font="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            font_size=72,
            depth=10,
            output_path=f"/tmp/test_{i+1}.ply",
            style=str(test["style"]),
            centered=bool(test["centered"]),
        )
        print(f"Result: {path_or_error}")
