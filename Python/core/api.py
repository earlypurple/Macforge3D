"""
API REST pour le système de monitoring de MacForge3D.
Utilise FastAPI pour exposer les métriques et le dashboard.
"""

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import json
import logging
from ..core.monitoring import (
    performance_monitor,
    SystemMetrics,
    SimulationMetrics,
    PerformanceAlert
)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Création de l'application FastAPI
app = FastAPI(
    title="MacForge3D Monitoring",
    description="API de monitoring pour MacForge3D",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèles Pydantic pour l'API
class MetricsResponse(BaseModel):
    timestamp: datetime
    system: Dict
    simulations: List[Dict]
    alerts: List[Dict]

class TimeRange(BaseModel):
    start: datetime
    end: datetime
    interval: str = "1m"

# Routes API
@app.get("/api/v1/metrics/current")
async def get_current_metrics():
    """Récupère les métriques actuelles."""
    try:
        system_metrics = performance_monitor.metrics_buffer.get_system_metrics(60)[-1]
        return {
            "timestamp": datetime.now(),
            "metrics": system_metrics.__dict__
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des métriques: {str(e)}"
        )

@app.get("/api/v1/metrics/history")
async def get_metrics_history(
    hours: int = 1,
    interval: str = "1m"
):
    """Récupère l'historique des métriques."""
    try:
        metrics = performance_monitor.metrics_buffer.get_system_metrics(hours * 3600)
        
        # Regrouper par intervalle
        grouped_metrics = {}
        for m in metrics:
            timestamp = m.timestamp.replace(
                second=0,
                microsecond=0
            )
            if timestamp not in grouped_metrics:
                grouped_metrics[timestamp] = []
            grouped_metrics[timestamp].append(m)
        
        # Calculer les moyennes par intervalle
        result = []
        for ts, group in grouped_metrics.items():
            avg_metrics = {
                "timestamp": ts,
                "cpu_percent": sum(m.cpu_percent for m in group) / len(group),
                "memory_percent": sum(m.memory_percent for m in group) / len(group),
                "memory_used": sum(m.memory_used for m in group) / len(group),
                "io_write_bytes": sum(m.io_write_bytes for m in group) / len(group)
            }
            result.append(avg_metrics)
        
        return sorted(result, key=lambda x: x["timestamp"])
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'historique: {str(e)}"
        )

@app.get("/api/v1/simulations")
async def get_simulations(
    hours: int = 24,
    status: Optional[str] = None
):
    """Récupère les simulations."""
    try:
        simulations = performance_monitor.metrics_buffer.get_simulation_metrics()
        
        # Filtrer par temps
        cutoff = datetime.now() - timedelta(hours=hours)
        simulations = [
            s for s in simulations
            if s.start_time >= cutoff
        ]
        
        # Filtrer par status
        if status:
            simulations = [
                s for s in simulations
                if s.status == status
            ]
            
        return [s.__dict__ for s in simulations]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des simulations: {str(e)}"
        )

@app.get("/api/v1/alerts")
async def get_alerts(
    severity: Optional[str] = None,
    hours: int = 24
):
    """Récupère les alertes."""
    try:
        return [
            a.__dict__ for a in
            performance_monitor.metrics_buffer.get_alerts(severity, hours)
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des alertes: {str(e)}"
        )

@app.get("/api/v1/report")
async def get_performance_report():
    """Récupère le rapport de performance."""
    try:
        return performance_monitor.get_performance_report()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération du rapport: {str(e)}"
        )

# WebSocket pour les mises à jour en temps réel
class MetricsWebSocket:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast_metrics(self, metrics: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(metrics)
            except Exception:
                await self.disconnect(connection)

metrics_ws = MetricsWebSocket()

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await metrics_ws.connect(websocket)
    try:
        while True:
            # Envoyer les métriques actuelles
            current_metrics = await get_current_metrics()
            await websocket.send_json(current_metrics)
            await asyncio.sleep(1)
    except Exception:
        metrics_ws.disconnect(websocket)

# Montage des fichiers statiques pour le dashboard
app.mount(
    "/dashboard",
    StaticFiles(directory="static/dashboard", html=True),
    name="dashboard"
)
