import os
import json
import csv
import vtk
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationExporter:
    """Classe pour exporter les résultats de simulation dans différents formats."""
    
    def __init__(self, output_dir: str):
        """
        Initialise l'exportateur.
        
        Args:
            output_dir: Répertoire de sortie pour les exports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_results(
        self,
        results: Dict[str, Any],
        format: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Exporte les résultats dans le format spécifié.
        
        Args:
            results: Résultats de la simulation
            format: Format d'export ('json', 'csv', 'vtk', 'html')
            filename: Nom du fichier (optionnel)
            
        Returns:
            Chemin du fichier exporté
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_results_{timestamp}"
            
        export_path = os.path.join(self.output_dir, filename)
        
        try:
            if format == "json":
                return self._export_json(results, export_path)
            elif format == "csv":
                return self._export_csv(results, export_path)
            elif format == "vtk":
                return self._export_vtk(results, export_path)
            elif format == "html":
                return self._export_html(results, export_path)
            else:
                raise ValueError(f"Format non supporté: {format}")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'export en {format}: {str(e)}")
            raise
            
    def _export_json(self, results: Dict[str, Any], base_path: str) -> str:
        """Exporte les résultats en JSON."""
        file_path = f"{base_path}.json"
        
        # Convertir les tableaux numpy en listes
        converted_results = self._convert_numpy_arrays(results)
        
        with open(file_path, "w") as f:
            json.dump(converted_results, f, indent=2)
            
        logger.info(f"Résultats exportés en JSON: {file_path}")
        return file_path
        
    def _export_csv(self, results: Dict[str, Any], base_path: str) -> str:
        """Exporte les résultats en CSV."""
        file_path = f"{base_path}.csv"
        
        # Aplatir les résultats pour le format CSV
        flat_data = self._flatten_dict(results)
        
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Métrique", "Valeur"])
            for key, value in flat_data.items():
                writer.writerow([key, value])
                
        logger.info(f"Résultats exportés en CSV: {file_path}")
        return file_path
        
    def _export_vtk(self, results: Dict[str, Any], base_path: str) -> str:
        """Exporte les résultats en VTK."""
        file_path = f"{base_path}.vtp"
        
        if "mesh_data" not in results:
            raise ValueError("Données de maillage manquantes")
            
        # Créer le maillage VTK
        mesh = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        vertices = vtk.vtkCellArray()
        
        # Ajouter les points
        mesh_data = results["mesh_data"]
        for vertex in mesh_data.get("vertices", []):
            pos = vertex["position"]
            points.InsertNextPoint(pos[0], pos[1], pos[2])
            
        mesh.SetPoints(points)
        
        # Ajouter les triangles
        for triangle in mesh_data.get("triangles", []):
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, triangle["v1"])
            cell.GetPointIds().SetId(1, triangle["v2"])
            cell.GetPointIds().SetId(2, triangle["v3"])
            vertices.InsertNextCell(cell)
            
        mesh.SetPolys(vertices)
        
        # Ajouter les données scalaires
        for name, data in results.items():
            if isinstance(data, (list, np.ndarray)) and name != "mesh_data":
                scalars = vtk.vtkFloatArray()
                scalars.SetName(name)
                for value in data:
                    scalars.InsertNextValue(float(value))
                mesh.GetPointData().AddArray(scalars)
                
        # Sauvegarder le fichier
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file_path)
        writer.SetInputData(mesh)
        writer.Write()
        
        logger.info(f"Résultats exportés en VTK: {file_path}")
        return file_path
        
    def _export_html(self, results: Dict[str, Any], base_path: str) -> str:
        """Exporte les résultats en rapport HTML interactif."""
        file_path = f"{base_path}.html"
        
        # Générer le HTML avec des graphiques interactifs
        html_content = self._generate_html_report(results)
        
        with open(file_path, "w") as f:
            f.write(html_content)
            
        logger.info(f"Résultats exportés en HTML: {file_path}")
        return file_path
        
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Génère un rapport HTML interactif."""
        # Utiliser une template HTML avec Plotly pour les graphiques
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>Rapport de simulation</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .section {{
            margin-bottom: 30px;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }}
        h1, h2 {{
            color: #333;
        }}
        .chart {{
            margin: 20px 0;
            background: white;
            padding: 15px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Rapport de simulation</h1>
        
        <div class="section">
            <h2>Résumé</h2>
            {self._generate_summary_html(results)}
        </div>
        
        <div class="section">
            <h2>Graphiques</h2>
            {self._generate_charts_html(results)}
        </div>
        
        <div class="section">
            <h2>Données détaillées</h2>
            {self._generate_details_html(results)}
        </div>
    </div>
    
    <script>
        {self._generate_plotly_js(results)}
    </script>
</body>
</html>
"""
        
    def _generate_summary_html(self, results: Dict[str, Any]) -> str:
        """Génère la section de résumé HTML."""
        summary = results.get("summary", {})
        html = "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr><th style='text-align:left;padding:8px;'>Métrique</th><th style='text-align:right;padding:8px;'>Valeur</th></tr>"
        
        for key, value in summary.items():
            html += f"<tr><td style='padding:8px;border-top:1px solid #ddd;'>{key}</td>"
            html += f"<td style='text-align:right;padding:8px;border-top:1px solid #ddd;'>{value}</td></tr>"
            
        html += "</table>"
        return html
        
    def _generate_charts_html(self, results: Dict[str, Any]) -> str:
        """Génère la section des graphiques HTML."""
        html = "<div class='chart' id='distributionChart'></div>"
        html += "<div class='chart' id='timeSeriesChart'></div>"
        return html
        
    def _generate_details_html(self, results: Dict[str, Any]) -> str:
        """Génère la section des détails HTML."""
        flat_data = self._flatten_dict(results)
        html = "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr><th style='text-align:left;padding:8px;'>Paramètre</th><th style='text-align:right;padding:8px;'>Valeur</th></tr>"
        
        for key, value in flat_data.items():
            html += f"<tr><td style='padding:8px;border-top:1px solid #ddd;'>{key}</td>"
            html += f"<td style='text-align:right;padding:8px;border-top:1px solid #ddd;'>{value}</td></tr>"
            
        html += "</table>"
        return html
        
    def _generate_plotly_js(self, results: Dict[str, Any]) -> str:
        """Génère le code JavaScript pour les graphiques Plotly."""
        # Exemple avec un graphique de distribution
        if "distribution_data" in results:
            data = results["distribution_data"]
            return f"""
                Plotly.newPlot('distributionChart', [{{
                    x: {list(data.keys())},
                    y: {list(data.values())},
                    type: 'bar'
                }}], {{
                    title: 'Distribution des résultats'
                }});
            """
        return ""
        
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """Aplatit un dictionnaire imbriqué."""
        items: List = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
        
    def _convert_numpy_arrays(self, obj: Any) -> Any:
        """Convertit les tableaux numpy en listes Python."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {
                key: self._convert_numpy_arrays(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_numpy_arrays(item) for item in obj]
        return obj
