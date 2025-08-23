import numpy as np
import json
from pathlib import Path
import vtk
from typing import Dict, Any, Optional, List, Tuple
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationExporter:
    """Classe pour exporter les résultats de simulation dans différents formats."""
    
    @staticmethod
    def export_to_vtk(
        mesh_data: Dict[str, Any],
        results: Dict[str, Any],
        output_path: str,
        data_name: str = "simulation_results"
    ) -> bool:
        """
        Exporte les résultats au format VTK.
        
        Args:
            mesh_data: Données du maillage
            results: Résultats de la simulation
            output_path: Chemin du fichier de sortie
            data_name: Nom du champ de données
            
        Returns:
            bool: True si l'export réussit
        """
        try:
            # Créer un maillage VTK
            points = vtk.vtkPoints()
            triangles = vtk.vtkCellArray()
            
            # Ajouter les points
            for triangle in mesh_data["triangles"]:
                for vertex in ["v1", "v2", "v3"]:
                    pos = triangle[vertex]["position"]
                    points.InsertNextPoint(pos[0], pos[1], pos[2])
            
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
            
            # Ajouter les résultats comme champs scalaires
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    scalars = vtk.vtkFloatArray()
                    scalars.SetName(key)
                    scalars.SetNumberOfComponents(1)
                    for _ in range(points.GetNumberOfPoints()):
                        scalars.InsertNextValue(float(value))
                    mesh.GetPointData().AddArray(scalars)
            
            # Sauvegarder le fichier
            writer = vtk.vtkXMLPolyDataWriter()
            writer.SetFileName(output_path)
            writer.SetInputData(mesh)
            writer.Write()
            
            logger.info(f"Export VTK réussi: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export VTK: {str(e)}")
            return False
    
    @staticmethod
    def export_to_json(
        results: Dict[str, Any],
        output_path: str,
        include_mesh: bool = False,
        mesh_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Exporte les résultats au format JSON.
        
        Args:
            results: Résultats de la simulation
            output_path: Chemin du fichier de sortie
            include_mesh: Inclure les données du maillage
            mesh_data: Données du maillage (si include_mesh=True)
            
        Returns:
            bool: True si l'export réussit
        """
        try:
            export_data = {
                "results": results,
                "timestamp": results.get("timestamp", ""),
                "material": results.get("material", ""),
                "recommendations": results.get("recommendations", [])
            }
            
            if include_mesh and mesh_data:
                export_data["mesh"] = mesh_data
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2)
                
            logger.info(f"Export JSON réussi: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'export JSON: {str(e)}")
            return False
    
    @staticmethod
    def export_report(
        results: Dict[str, Any],
        output_path: str,
        title: str = "Rapport de Simulation"
    ) -> bool:
        """
        Génère un rapport HTML des résultats.
        
        Args:
            results: Résultats de la simulation
            output_path: Chemin du fichier de sortie
            title: Titre du rapport
            
        Returns:
            bool: True si l'export réussit
        """
        try:
            # Créer le contenu HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    .section {{ margin: 20px 0; padding: 10px; background: #f8f9fa; }}
                    .warning {{ color: #e74c3c; }}
                    .success {{ color: #27ae60; }}
                </style>
            </head>
            <body>
                <h1>{title}</h1>
                
                <div class="section">
                    <h2>Informations Générales</h2>
                    <p>Matériau: {results.get('material', 'Non spécifié')}</p>
                    <p>Date: {results.get('timestamp', 'Non spécifiée')}</p>
                </div>
                
                <div class="section">
                    <h2>Résultats</h2>
            """
            
            # Ajouter les résultats numériques
            for key, value in results.items():
                if isinstance(value, (int, float)) and key not in ['timestamp']:
                    html_content += f"<p>{key}: {value}</p>\n"
            
            # Ajouter les recommandations
            if 'recommendations' in results:
                html_content += """
                <div class="section">
                    <h2>Recommandations</h2>
                    <ul>
                """
                for rec in results['recommendations']:
                    html_content += f"<li>{rec}</li>\n"
                html_content += "</ul></div>"
            
            # Fermer le document HTML
            html_content += """
                </div>
            </body>
            </html>
            """
            
            # Sauvegarder le fichier
            with open(output_path, 'w') as f:
                f.write(html_content)
                
            logger.info(f"Export du rapport réussi: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {str(e)}")
            return False
