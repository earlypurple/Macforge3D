import vtk
import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulationView3D(QWidget):
    """Widget pour la visualisation 3D des résultats de simulation."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_vtk()

    def setup_ui(self):
        """Configure l'interface utilisateur."""
        layout = QVBoxLayout(self)

        # Créer le widget VTK
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtkWidget)

        # Initialiser le rendu
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Configurer la caméra
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().Elevation(30)
        self.ren.GetActiveCamera().Azimuth(30)
        self.ren.ResetCameraClippingRange()

        # Démarrer l'interaction
        self.iren.Initialize()

    def setup_vtk(self):
        """Configure les objets VTK."""
        # Créer une échelle de couleurs
        self.lut = vtk.vtkLookupTable()
        self.lut.SetHueRange(0.667, 0.0)  # Bleu à Rouge
        self.lut.SetTableRange(0.0, 1.0)
        self.lut.Build()

        # Créer un scalar bar
        self.scalarBar = vtk.vtkScalarBarActor()
        self.scalarBar.SetLookupTable(self.lut)
        self.scalarBar.SetTitle("Valeur")
        self.scalarBar.SetNumberOfLabels(5)
        self.ren.AddActor(self.scalarBar)

        # Préparer les acteurs
        self.meshActor = None
        self.deformActor = None
        self.vectorActor = None

    def display_results(
        self,
        mesh_data: Dict[str, Any],
        results: Dict[str, Any],
        display_type: str = "stress",
    ):
        """
        Affiche les résultats de simulation en 3D.

        Args:
            mesh_data: Données du maillage
            results: Résultats de la simulation
            display_type: Type d'affichage ("stress", "displacement", "temperature")
        """
        try:
            # Créer le maillage VTK
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()

            # Ajouter les points
            point_data = {}
            for i, triangle in enumerate(mesh_data["triangles"]):
                for vertex in ["v1", "v2", "v3"]:
                    pos = triangle[vertex]["position"]
                    point_id = points.InsertNextPoint(pos[0], pos[1], pos[2])
                    point_data[tuple(pos)] = point_id

            # Ajouter les triangles
            for i in range(0, points.GetNumberOfPoints(), 3):
                triangle = vtk.vtkTriangle()
                triangle.GetPointIds().SetId(0, i)
                triangle.GetPointIds().SetId(1, i + 1)
                triangle.GetPointIds().SetId(2, i + 2)
                triangles.InsertNextCell(triangle)

            # Créer le maillage
            mesh = vtk.vtkPolyData()
            mesh.SetPoints(points)
            mesh.SetPolys(triangles)

            # Ajouter les données scalaires selon le type d'affichage
            scalars = vtk.vtkFloatArray()
            vectors = vtk.vtkFloatArray()
            vectors.SetNumberOfComponents(3)

            if display_type == "stress":
                scalars.SetName("von_Mises")
                if "stress_distribution" in results:
                    stresses = results["stress_distribution"]
                    for stress in stresses:
                        scalars.InsertNextValue(stress)

            elif display_type == "displacement":
                vectors.SetName("Displacement")
                if "displacements" in results:
                    disps = results["displacements"]
                    for disp in disps:
                        vectors.InsertTuple3(
                            points.GetNumberOfPoints(), disp[0], disp[1], disp[2]
                        )

            elif display_type == "temperature":
                scalars.SetName("Temperature")
                if "temperature_distribution" in results:
                    temps = results["temperature_distribution"]
                    for temp in temps:
                        scalars.InsertNextValue(temp)

            # Ajouter les données au maillage
            if scalars.GetNumberOfTuples() > 0:
                mesh.GetPointData().SetScalars(scalars)
            if vectors.GetNumberOfTuples() > 0:
                mesh.GetPointData().SetVectors(vectors)

            # Créer le mapper et l'acteur
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(mesh)
            mapper.SetLookupTable(self.lut)

            if self.meshActor:
                self.ren.RemoveActor(self.meshActor)
            self.meshActor = vtk.vtkActor()
            self.meshActor.SetMapper(mapper)

            # Ajouter l'acteur au renderer
            self.ren.AddActor(self.meshActor)

            # Ajouter des glyphes pour les vecteurs si nécessaire
            if display_type == "displacement" and "displacements" in results:
                self.add_displacement_glyphs(mesh)

            # Mettre à jour le titre de l'échelle
            self.scalarBar.SetTitle(
                {
                    "stress": "Contrainte (MPa)",
                    "displacement": "Déplacement (mm)",
                    "temperature": "Température (°C)",
                }[display_type]
            )

            # Rafraîchir l'affichage
            self.vtkWidget.GetRenderWindow().Render()

            logger.info(f"Visualisation 3D mise à jour: {display_type}")

        except Exception as e:
            logger.error(f"Erreur lors de la visualisation 3D: {str(e)}")

    def add_displacement_glyphs(self, mesh: vtk.vtkPolyData):
        """Ajoute des glyphes pour visualiser les déplacements."""
        # Créer une flèche
        arrow = vtk.vtkArrowSource()

        # Créer les glyphes
        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(mesh)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetScaleModeToScaleByVector()
        glyph.SetScaleFactor(0.1)  # Ajuster selon l'échelle

        # Mapper et acteur pour les glyphes
        glyph_mapper = vtk.vtkPolyDataMapper()
        glyph_mapper.SetInputConnection(glyph.GetOutputPort())

        if self.vectorActor:
            self.ren.RemoveActor(self.vectorActor)
        self.vectorActor = vtk.vtkActor()
        self.vectorActor.SetMapper(glyph_mapper)

        self.ren.AddActor(self.vectorActor)

    def clear(self):
        """Efface tous les acteurs de la scène."""
        if self.meshActor:
            self.ren.RemoveActor(self.meshActor)
            self.meshActor = None

        if self.deformActor:
            self.ren.RemoveActor(self.deformActor)
            self.deformActor = None

        if self.vectorActor:
            self.ren.RemoveActor(self.vectorActor)
            self.vectorActor = None

        self.vtkWidget.GetRenderWindow().Render()

    def set_background_color(self, r: float, g: float, b: float):
        """Change la couleur de fond."""
        self.ren.SetBackground(r, g, b)
        self.vtkWidget.GetRenderWindow().Render()

    def set_camera_position(self, pos: Tuple[float, float, float]):
        """Définit la position de la caméra."""
        camera = self.ren.GetActiveCamera()
        camera.SetPosition(pos[0], pos[1], pos[2])
        self.ren.ResetCameraClippingRange()
        self.vtkWidget.GetRenderWindow().Render()

    def reset_view(self):
        """Réinitialise la vue."""
        self.ren.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def cleanup(self):
        """Nettoie les ressources VTK."""
        self.vtkWidget.GetRenderWindow().Finalize()
        self.iren.TerminateApp()
