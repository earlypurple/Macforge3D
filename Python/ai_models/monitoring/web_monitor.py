"""
Interface web de monitoring pour MacForge3D.
"""

from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import asyncio
import json
import psutil
import GPUtil
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialiser FastAPI
app = FastAPI(title="MacForge3D Monitor")

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")


@dataclass
class SystemMetrics:
    """Métriques système en temps réel."""

    cpu_percent: float
    memory_percent: float
    gpu_utilization: Optional[float]
    gpu_memory_percent: Optional[float]
    disk_usage_percent: float
    timestamp: str


@dataclass
class CacheMetrics:
    """Métriques du cache."""

    size: int
    items: int
    hit_ratio: float
    compression_ratio: float
    memory_usage: float


@dataclass
class ProcessingMetrics:
    """Métriques de traitement."""

    tasks_completed: int
    tasks_failed: int
    average_processing_time: float
    success_rate: float
    queue_size: int


class MetricsCollector:
    """Collecteur de métriques en temps réel."""

    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 1000
        self._last_update = time.time()

    def collect_system_metrics(self) -> SystemMetrics:
        """Collecte les métriques système."""

        # Métriques CPU et mémoire
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()

        # Métriques GPU
        gpu_util = None
        gpu_memory = None
        if torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_util = gpu.load * 100
                    gpu_memory = gpu.memoryUtil * 100
            except Exception:
                pass

        # Métriques disque
        disk = psutil.disk_usage("/")

        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            gpu_utilization=gpu_util,
            gpu_memory_percent=gpu_memory,
            disk_usage_percent=disk.percent,
            timestamp=datetime.now().isoformat(),
        )

        # Mettre à jour l'historique
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)

        return metrics

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Retourne l'historique des métriques."""
        return [asdict(m) for m in self.metrics_history]


# Créer le collecteur global
metrics_collector = MetricsCollector()


@app.get("/")
async def get_index():
    """Page d'accueil du moniteur."""
    return HTMLResponse(
        content="""
<!DOCTYPE html>
<html>
<head>
    <title>MacForge3D Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/vue@2"></script>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
    <style>
        .metric-card {
            @apply bg-white rounded-lg shadow p-4;
        }
        .metric-value {
            @apply text-2xl font-bold text-blue-600;
        }
        .metric-label {
            @apply text-gray-600 text-sm;
        }
    </style>
</head>
<body class="bg-gray-100">
    <div id="app" class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-3xl font-bold text-gray-800">MacForge3D Monitor</h1>
        </header>
        
        <!-- Métriques système -->
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <div class="metric-card">
                <div class="metric-value">{{ systemMetrics.cpu_percent }}%</div>
                <div class="metric-label">CPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ systemMetrics.memory_percent }}%</div>
                <div class="metric-label">Memory Usage</div>
            </div>
            <div class="metric-card" v-if="systemMetrics.gpu_utilization !== null">
                <div class="metric-value">{{ systemMetrics.gpu_utilization }}%</div>
                <div class="metric-label">GPU Usage</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{{ systemMetrics.disk_usage_percent }}%</div>
                <div class="metric-label">Disk Usage</div>
            </div>
        </div>
        
        <!-- Graphiques -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-8">
            <div class="bg-white rounded-lg shadow p-4">
                <div id="cpuChart"></div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <div id="memoryChart"></div>
            </div>
        </div>
        
        <!-- Cache et traitement -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div class="bg-white rounded-lg shadow p-4">
                <h2 class="text-xl font-bold mb-4">Cache Statistics</h2>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="metric-value">{{ cacheMetrics.hit_ratio }}%</div>
                        <div class="metric-label">Cache Hit Ratio</div>
                    </div>
                    <div>
                        <div class="metric-value">{{ cacheMetrics.compression_ratio }}x</div>
                        <div class="metric-label">Compression Ratio</div>
                    </div>
                </div>
            </div>
            <div class="bg-white rounded-lg shadow p-4">
                <h2 class="text-xl font-bold mb-4">Processing Statistics</h2>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="metric-value">{{ processingMetrics.success_rate }}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div>
                        <div class="metric-value">{{ processingMetrics.queue_size }}</div>
                        <div class="metric-label">Queue Size</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        new Vue({
            el: '#app',
            data: {
                systemMetrics: {
                    cpu_percent: 0,
                    memory_percent: 0,
                    gpu_utilization: null,
                    gpu_memory_percent: null,
                    disk_usage_percent: 0
                },
                cacheMetrics: {
                    hit_ratio: 0,
                    compression_ratio: 0
                },
                processingMetrics: {
                    success_rate: 0,
                    queue_size: 0
                },
                ws: null
            },
            mounted() {
                this.connectWebSocket();
                this.initCharts();
            },
            methods: {
                connectWebSocket() {
                    this.ws = new WebSocket('ws://localhost:8000/ws');
                    this.ws.onmessage = (event) => {
                        const data = JSON.parse(event.data);
                        this.updateMetrics(data);
                        this.updateCharts(data);
                    };
                },
                updateMetrics(data) {
                    this.systemMetrics = data.system;
                    this.cacheMetrics = data.cache;
                    this.processingMetrics = data.processing;
                },
                initCharts() {
                    const layout = {
                        margin: { t: 30, r: 30, l: 50, b: 30 },
                        height: 300
                    };
                    
                    Plotly.newPlot('cpuChart', [{
                        y: [],
                        type: 'scatter',
                        name: 'CPU Usage'
                    }], {...layout, title: 'CPU Usage Over Time'});
                    
                    Plotly.newPlot('memoryChart', [{
                        y: [],
                        type: 'scatter',
                        name: 'Memory Usage'
                    }], {...layout, title: 'Memory Usage Over Time'});
                },
                updateCharts(data) {
                    Plotly.extendTraces('cpuChart', {
                        y: [[data.system.cpu_percent]]
                    }, [0]);
                    
                    Plotly.extendTraces('memoryChart', {
                        y: [[data.system.memory_percent]]
                    }, [0]);
                }
            }
        });
    </script>
</body>
</html>
    """
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Endpoint WebSocket pour les mises à jour en temps réel."""
    await websocket.accept()

    try:
        while True:
            # Collecter les métriques
            system_metrics = metrics_collector.collect_system_metrics()

            # Envoyer les métriques
            await websocket.send_json(
                {
                    "system": asdict(system_metrics),
                    "cache": {
                        "hit_ratio": 85.5,  # À remplacer par les vraies métriques
                        "compression_ratio": 2.3,
                    },
                    "processing": {"success_rate": 97.8, "queue_size": 5},
                }
            )

            await asyncio.sleep(1)  # Mise à jour chaque seconde

    except Exception as e:
        logger.error(f"Erreur WebSocket: {e}")

    finally:
        await websocket.close()


def start_monitor(host: str = "0.0.0.0", port: int = 8000):
    """Démarre le serveur de monitoring."""
    uvicorn.run(app, host=host, port=port)
