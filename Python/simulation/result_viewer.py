from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QTabWidget, QPushButton, QProgressBar,
    QScrollArea, QFrame, QGridLayout, QSpacerItem,
    QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
import pyqtgraph as pg
import numpy as np
from typing import Dict, Any, List
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultGraph(pg.PlotWidget):
    """Widget de graphique personnalisé basé sur pyqtgraph."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configuration du style
        self.setBackground('w')
        self.showGrid(x=True, y=True)
        self.getAxis('bottom').setPen('k')
        self.getAxis('left').setPen('k')
        
    def plot_temperature_curve(self, data: Dict[str, Any]):
        """Trace la courbe de température."""
        self.clear()
        
        times = data["temperature_curve"]["times"]
        max_temps = data["temperature_curve"]["max_temps"]
        min_temps = data["temperature_curve"]["min_temps"]
        
        # Tracer la zone de température
        self.plot(
            times, max_temps,
            pen=pg.mkPen('r', width=2),
            name='Température maximale'
        )
        self.plot(
            times, min_temps,
            pen=pg.mkPen('b', width=2),
            name='Température minimale'
        )
        
        # Ajouter les annotations
        if "glass_transition" in data:
            self.addLine(
                y=data["glass_transition"],
                pen=pg.mkPen('g', style=Qt.PenStyle.DashLine),
                name='Tg'
            )
            
    def plot_stress_distribution(self, data: Dict[str, Any]):
        """Trace la distribution des contraintes."""
        self.clear()
        
        if "stress_distribution" in data:
            stresses = data["stress_distribution"]
            y, x = np.histogram(stresses, bins=30)
            self.plot(
                x[:-1], y,
                stepMode=True,
                fillLevel=0,
                brush=(100, 150, 255, 100),
                name='Distribution des contraintes'
            )
            
            # Ajouter la limite d'élasticité
            if "yield_strength" in data:
                self.addLine(
                    x=data["yield_strength"],
                    pen=pg.mkPen('r', style=Qt.PenStyle.DashLine),
                    name='Limite élastique'
                )

class RiskIndicator(QWidget):
    """Widget personnalisé pour afficher le niveau de risque."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.risk_level = "faible"
        self.setMinimumSize(100, 100)
        
    def set_risk_level(self, level: str):
        """Définit le niveau de risque."""
        self.risk_level = level.lower()
        self.update()
        
    def paintEvent(self, event):
        """Dessine l'indicateur de risque."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Définir les couleurs
        colors = {
            "faible": QColor(46, 204, 113),
            "moyen": QColor(241, 196, 15),
            "élevé": QColor(231, 76, 60)
        }
        
        # Dessiner le cercle
        color = colors.get(self.risk_level, colors["moyen"])
        painter.setPen(QPen(color.darker(), 2))
        painter.setBrush(color)
        
        size = min(self.width(), self.height()) - 10
        x = (self.width() - size) // 2
        y = (self.height() - size) // 2
        
        painter.drawEllipse(x, y, size, size)
        
        # Ajouter le texte
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            self.risk_level.upper()
        )

class RecommendationItem(QFrame):
    """Widget pour afficher une recommandation."""
    
    def __init__(self, text: str, is_critical: bool = False, parent=None):
        super().__init__(parent)
        
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setLineWidth(1)
        
        layout = QHBoxLayout(self)
        
        # Icône (point coloré)
        icon = QLabel("●")
        icon.setStyleSheet(
            f"color: {'red' if is_critical else 'orange'}; "
            "font-size: 20px;"
        )
        layout.addWidget(icon)
        
        # Texte
        label = QLabel(text)
        label.setWordWrap(True)
        layout.addWidget(label, stretch=1)
        
        if is_critical:
            self.setStyleSheet(
                "background-color: #ffebee; border: 1px solid #ffcdd2;"
            )
        else:
            self.setStyleSheet(
                "background-color: #fff3e0; border: 1px solid #ffe0b2;"
            )

class ResultViewer(QWidget):
    """Widget principal pour la visualisation des résultats."""
    
    analysisRequested = pyqtSignal(dict)  # Signal pour demander une nouvelle analyse
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Configure l'interface utilisateur."""
        layout = QVBoxLayout(self)
        
        # En-tête avec informations générales
        header = QHBoxLayout()
        
        # Indicateur de risque
        self.risk_indicator = RiskIndicator()
        header.addWidget(self.risk_indicator)
        
        # Informations générales
        info_layout = QGridLayout()
        self.material_label = QLabel("Matériau: -")
        self.timestamp_label = QLabel("Date: -")
        self.status_label = QLabel("Statut: -")
        
        info_layout.addWidget(QLabel("Matériau:"), 0, 0)
        info_layout.addWidget(self.material_label, 0, 1)
        info_layout.addWidget(QLabel("Date:"), 1, 0)
        info_layout.addWidget(self.timestamp_label, 1, 1)
        info_layout.addWidget(QLabel("Statut:"), 2, 0)
        info_layout.addWidget(self.status_label, 2, 1)
        
        header.addLayout(info_layout)
        header.addStretch()
        
        # Bouton d'analyse
        analyze_btn = QPushButton("Nouvelle Analyse")
        analyze_btn.clicked.connect(self.request_analysis)
        header.addWidget(analyze_btn)
        
        layout.addLayout(header)
        
        # Onglets pour les différentes visualisations
        tabs = QTabWidget()
        
        # Onglet des graphiques
        graphs_widget = QWidget()
        graphs_layout = QVBoxLayout(graphs_widget)
        
        # Graphique de température
        self.temp_graph = ResultGraph()
        self.temp_graph.setTitle("Évolution de la Température")
        graphs_layout.addWidget(self.temp_graph)
        
        # Graphique des contraintes
        self.stress_graph = ResultGraph()
        self.stress_graph.setTitle("Distribution des Contraintes")
        graphs_layout.addWidget(self.stress_graph)
        
        tabs.addTab(graphs_widget, "Graphiques")
        
        # Onglet des recommandations
        recommendations_widget = QWidget()
        recommendations_layout = QVBoxLayout(recommendations_widget)
        
        self.recommendations_area = QScrollArea()
        self.recommendations_area.setWidgetResizable(True)
        self.recommendations_widget = QWidget()
        self.recommendations_layout = QVBoxLayout(self.recommendations_widget)
        self.recommendations_layout.addStretch()
        self.recommendations_area.setWidget(self.recommendations_widget)
        
        recommendations_layout.addWidget(self.recommendations_area)
        tabs.addTab(recommendations_widget, "Recommandations")
        
        # Onglet des optimisations
        optimizations_widget = QWidget()
        optimizations_layout = QVBoxLayout(optimizations_widget)
        
        self.optimizations_area = QScrollArea()
        self.optimizations_area.setWidgetResizable(True)
        self.optimizations_widget = QWidget()
        self.optimizations_layout = QVBoxLayout(self.optimizations_widget)
        self.optimizations_layout.addStretch()
        self.optimizations_area.setWidget(self.optimizations_widget)
        
        optimizations_layout.addWidget(self.optimizations_area)
        tabs.addTab(optimizations_widget, "Optimisations")
        
        layout.addWidget(tabs)
        
    def update_results(self, results: Dict[str, Any]):
        """Met à jour l'affichage avec les nouveaux résultats."""
        try:
            # Mettre à jour les informations générales
            self.material_label.setText(results.get("material", "-"))
            self.timestamp_label.setText(
                datetime.fromisoformat(results["timestamp"]).strftime("%Y-%m-%d %H:%M")
            )
            self.status_label.setText(
                "Anomalies Détectées" if results.get("anomalies_detected")
                else "Normal"
            )
            
            # Mettre à jour l'indicateur de risque
            self.risk_indicator.set_risk_level(results.get("risk_level", "moyen"))
            
            # Mettre à jour les graphiques
            if "temperature_curve" in results:
                self.temp_graph.plot_temperature_curve(results)
            if "stress_distribution" in results:
                self.stress_graph.plot_stress_distribution(results)
                
            # Mettre à jour les recommandations
            self.clear_recommendations()
            for rec in results.get("recommendations", []):
                is_critical = any(
                    kw in rec.lower()
                    for kw in ["critique", "danger", "immédiat", "risque"]
                )
                self.add_recommendation(rec, is_critical)
                
            # Mettre à jour les optimisations
            self.clear_optimizations()
            for opt in results.get("optimization_suggestions", []):
                self.add_optimization(opt)
                
            logger.info("Interface mise à jour avec succès")
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de l'interface: {str(e)}")
            
    def clear_recommendations(self):
        """Efface toutes les recommandations."""
        for i in reversed(range(self.recommendations_layout.count() - 1)):
            widget = self.recommendations_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
                
    def clear_optimizations(self):
        """Efface toutes les optimisations."""
        for i in reversed(range(self.optimizations_layout.count() - 1)):
            widget = self.optimizations_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
                
    def add_recommendation(self, text: str, is_critical: bool = False):
        """Ajoute une recommandation."""
        item = RecommendationItem(text, is_critical)
        self.recommendations_layout.insertWidget(
            self.recommendations_layout.count() - 1,
            item
        )
        
    def add_optimization(self, text: str):
        """Ajoute une suggestion d'optimisation."""
        item = RecommendationItem(text, False)
        self.optimizations_layout.insertWidget(
            self.optimizations_layout.count() - 1,
            item
        )
        
    def request_analysis(self):
        """Émet un signal pour demander une nouvelle analyse."""
        self.analysisRequested.emit({})
