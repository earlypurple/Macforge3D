import trimesh
import numpy as np
import os
from shapely.geometry import Polygon  # type: ignore


def create_text_mesh(
    text: str,
    font: str = "Arial.ttf",
    font_size: float = 24.0,
    depth: float = 5.0,
    output_path: str = "/tmp/3d_text.ply",
):
    """
    Generates a 3D mesh from a given text string using font vector outlines.
    This approach uses fontTools to get high-quality vector paths for glyphs.
    """
    try:
        from fontTools.ttLib import TTFont  # type: ignore
        from fontTools.pens.basePen import BasePen  # type: ignore
        from shapely.geometry import Polygon, MultiPolygon  # type: ignore
    except ImportError:
        return "Error: fontTools or shapely is not installed. Please run 'pip install fonttools shapely'."

    class ShapelyPen(BasePen):
        def __init__(self, glyphSet):
            super().__init__(glyphSet)
            self.contours = []
            self.current_contour = []

        def _moveTo(self, pt):
            if self.current_contour:
                self.contours.append(self.current_contour)
            self.current_contour = [pt]

        def _lineTo(self, pt):
            self.current_contour.append(pt)

        def _curveTo(self, pt1, pt2, pt3):
            # Approximate curve with line segments for simplicity
            # A more advanced implementation would handle Bezier curves properly
            steps = 10
            for i in range(1, steps + 1):
                t = i / steps
                x = (
                    (1 - t) ** 3 * self.current_contour[-1][0]
                    + 3 * (1 - t) ** 2 * t * pt1[0]
                    + 3 * (1 - t) * t**2 * pt2[0]
                    + t**3 * pt3[0]
                )
                y = (
                    (1 - t) ** 3 * self.current_contour[-1][1]
                    + 3 * (1 - t) ** 2 * t * pt1[1]
                    + 3 * (1 - t) * t**2 * pt2[1]
                    + t**3 * pt3[1]
                )
                self.current_contour.append((x, y))

        def _closePath(self):
            if self.current_contour:
                self.contours.append(self.current_contour)
                self.current_contour = []

        def get_polygons(self):
            polygons = []
            for contour in self.contours:
                if len(contour) > 2:
                    polygons.append(Polygon(contour))
            return polygons

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

    total_mesh = trimesh.Trimesh()
    cursor_x = 0

    # Scale factor to convert font units to model units
    units_per_em = font_obj["head"].unitsPerEm
    scale = font_size / units_per_em

    for char in text:
        glyph_name = font_obj.getBestCmap().get(ord(char))
        if not glyph_name:
            cursor_x += font_obj["hmtx"]["a"][0] * scale  # Advance by a default width
            continue

        glyph = glyphSet[glyph_name]

        pen = ShapelyPen(glyphSet)
        glyph.draw(pen)

        polygons = pen.get_polygons()
        if not polygons:
            cursor_x += glyph.width * scale
            continue

        # Combine polygons (e.g., for letters like 'o' with holes)
        # This is a simplified approach; robust hole handling is complex.
        outer_polygons = [p for p in polygons if p.exterior.is_ccw]
        inner_polygons = [p for p in polygons if not p.exterior.is_ccw]

        final_polygon = MultiPolygon(outer_polygons)
        for inner in inner_polygons:
            final_polygon = final_polygon.difference(inner)

        if final_polygon.is_empty:
            cursor_x += glyph.width * scale
            continue

        # Extrude and transform the character mesh
        char_mesh = trimesh.creation.extrude_polygon(final_polygon, height=depth)
        transform = np.array(
            [[1, 0, 0, cursor_x], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        char_mesh.apply_transform(transform)
        char_mesh.apply_scale(scale)

        total_mesh += char_mesh

        # Advance cursor
        cursor_x += glyph.width * scale

    if total_mesh.is_empty:
        return "Error: Generated mesh is empty. The text may not contain any renderable characters."

    total_mesh.export(output_path)
    return output_path


if __name__ == "__main__":
    print("Testing text_to_mesh.py script with fontTools...")
    path_or_error = create_text_mesh(
        text="Vector!",
        font="DejaVuSans.ttf",  # A common font on Linux systems
        font_size=72,
        depth=10,
        output_path="/tmp/vector_text.ply",
    )
    print(f"Script finished. Result: {path_or_error}")
