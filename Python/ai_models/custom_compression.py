"""
Module de compression personnalisée pour les maillages 3D.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from scipy.spatial import cKDTree
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json
import h5py
import math
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompressionFormat:
    """Format de compression personnalisé."""
    name: str
    version: str
    description: str
    parameters: Dict[str, Any]
    compression_type: str  # "lossy" ou "lossless"
    target_ratio: float
    preserve_attributes: List[str]

class MeshAutoencoder(nn.Module):
    """Autoencoder pour la compression de maillages."""
    
    def __init__(
        self,
        latent_dim: int = 128,
        feature_dim: int = 3,
        hidden_dims: List[int] = [256, 128]
    ):
        super().__init__()
        
        # Encodeur
        encoder_layers = []
        in_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
            
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Décodeur
        decoder_layers = []
        in_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            in_dim = hidden_dim
            
        decoder_layers.append(nn.Linear(hidden_dims[0], feature_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
        
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

class MeshCompressor:
    """Compresseur personnalisé pour les maillages."""
    
    def __init__(
        self,
        format_config: Optional[CompressionFormat] = None
    ):
        self.format = format_config or CompressionFormat(
            name="adaptive_mesh",
            version="1.0",
            description="Format de compression adaptatif pour maillages",
            parameters={
                "quantization_bits": 16,
                "use_octree": True,
                "cluster_threshold": 0.01,
                "preserve_boundaries": True
            },
            compression_type="lossy",
            target_ratio=10.0,
            preserve_attributes=["normals", "uvs"]
        )
        
        self.autoencoder = None
        self.vertex_clusters = None
        self.kdtree = None
        
    def train_autoencoder(
        self,
        meshes: List[Any],
        batch_size: int = 32,
        n_epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Entraîne l'autoencoder sur un ensemble de maillages.
        
        Args:
            meshes: Liste de maillages
            batch_size: Taille des batchs
            n_epochs: Nombre d'epochs
            learning_rate: Taux d'apprentissage
        """
        # Préparer les données
        vertices = []
        for mesh in meshes:
            vertices.append(mesh.vertices)
        vertices = np.concatenate(vertices, axis=0)
        
        # Créer le dataset
        dataset = torch.FloatTensor(vertices)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Créer le modèle
        self.autoencoder = MeshAutoencoder()
        optimizer = torch.optim.Adam(
            self.autoencoder.parameters(),
            lr=learning_rate
        )
        
        # Entraînement
        for epoch in range(n_epochs):
            total_loss = 0
            for batch in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon, _ = self.autoencoder(batch)
                
                # Calculer la perte
                loss = F.mse_loss(recon, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Loss = {avg_loss:.6f}")
                
    def compress_mesh(
        self,
        mesh: Any,
        output_path: Optional[Path] = None
    ) -> Union[bytes, Path]:
        """
        Compresse un maillage.
        
        Args:
            mesh: Maillage à compresser
            output_path: Chemin de sortie optionnel
            
        Returns:
            Données compressées ou chemin du fichier
        """
        # 1. Quantification des vertices
        vertices = mesh.vertices
        v_min = vertices.min(axis=0)
        v_max = vertices.max(axis=0)
        v_range = v_max - v_min
        
        quantization_bits = self.format.parameters["quantization_bits"]
        scale = (2**quantization_bits - 1) / v_range
        vertices_quantized = np.round((vertices - v_min) * scale)
        
        # 2. Clustering des vertices
        if self.format.parameters["use_octree"]:
            # Utiliser un octree pour le clustering
            self.kdtree = cKDTree(vertices)
            self.vertex_clusters = defaultdict(list)
            
            threshold = self.format.parameters["cluster_threshold"]
            for i, v in enumerate(vertices):
                neighbors = self.kdtree.query_ball_point(
                    v, threshold * v_range.mean()
                )
                cluster_id = min(neighbors)
                self.vertex_clusters[cluster_id].append(i)
                
        # 3. Compression des attributs préservés
        preserved_data = {}
        for attr in self.format.preserve_attributes:
            if hasattr(mesh, attr):
                data = getattr(mesh, attr)
                if isinstance(data, np.ndarray):
                    preserved_data[attr] = data
                    
        # 4. Encodage des faces
        faces = mesh.faces
        
        # 5. Optimisation de la topologie
        if self.autoencoder is not None:
            # Utiliser l'autoencoder pour la compression
            vertices_tensor = torch.FloatTensor(vertices)
            with torch.no_grad():
                _, latent = self.autoencoder(vertices_tensor)
                
        # 6. Sauvegarder les données
        if output_path:
            with h5py.File(output_path, 'w') as f:
                # Méta-données
                f.attrs['format_name'] = self.format.name
                f.attrs['format_version'] = self.format.version
                f.attrs['compression_type'] = self.format.compression_type
                
                # Données géométriques
                f.create_dataset('vertices_quantized', data=vertices_quantized)
                f.create_dataset('v_min', data=v_min)
                f.create_dataset('v_max', data=v_max)
                f.create_dataset('faces', data=faces)
                
                if self.vertex_clusters:
                    # Sauvegarder les clusters
                    cluster_data = np.array([
                        (k, len(v)) for k, v in self.vertex_clusters.items()
                    ])
                    f.create_dataset('clusters', data=cluster_data)
                    
                if self.autoencoder is not None:
                    f.create_dataset('latent', data=latent.numpy())
                    
                # Attributs préservés
                attrs_group = f.create_group('attributes')
                for name, data in preserved_data.items():
                    attrs_group.create_dataset(name, data=data)
                    
            return output_path
            
        else:
            # Retourner les données compressées en mémoire
            compressed_data = {
                'format': {
                    'name': self.format.name,
                    'version': self.format.version,
                    'type': self.format.compression_type
                },
                'geometry': {
                    'vertices_quantized': vertices_quantized.tobytes(),
                    'v_min': v_min.tobytes(),
                    'v_max': v_max.tobytes(),
                    'faces': faces.tobytes()
                },
                'attributes': {
                    name: data.tobytes()
                    for name, data in preserved_data.items()
                }
            }
            
            if self.vertex_clusters:
                compressed_data['clusters'] = {
                    str(k): v for k, v in self.vertex_clusters.items()
                }
                
            if self.autoencoder is not None:
                compressed_data['latent'] = latent.numpy().tobytes()
                
            return json.dumps(compressed_data).encode()
            
    def decompress_mesh(
        self,
        data: Union[bytes, Path]
    ) -> Any:
        """
        Décompresse un maillage.
        
        Args:
            data: Données compressées ou chemin du fichier
            
        Returns:
            Maillage décompressé
        """
        if isinstance(data, (str, Path)):
            # Charger depuis un fichier
            with h5py.File(data, 'r') as f:
                # Vérifier le format
                format_name = f.attrs['format_name']
                format_version = f.attrs['format_version']
                
                if (format_name != self.format.name or
                    format_version != self.format.version):
                    raise ValueError(f"Format incompatible: {format_name} {format_version}")
                    
                # Charger la géométrie
                vertices_quantized = f['vertices_quantized'][:]
                v_min = f['v_min'][:]
                v_max = f['v_max'][:]
                faces = f['faces'][:]
                
                # Reconstruire les vertices
                v_range = v_max - v_min
                scale = (2**self.format.parameters["quantization_bits"] - 1)
                vertices = (vertices_quantized / scale) * v_range + v_min
                
                # Charger les attributs
                attributes = {}
                if 'attributes' in f:
                    for name, dataset in f['attributes'].items():
                        attributes[name] = dataset[:]
                        
                # Reconstruire avec l'autoencoder si disponible
                if 'latent' in f and self.autoencoder is not None:
                    latent = torch.FloatTensor(f['latent'][:])
                    with torch.no_grad():
                        vertices = self.autoencoder.decode(latent).numpy()
                        
        else:
            # Décompresser depuis les données en mémoire
            compressed_data = json.loads(data.decode())
            
            # Vérifier le format
            format_info = compressed_data['format']
            if (format_info['name'] != self.format.name or
                format_info['version'] != self.format.version):
                raise ValueError(
                    f"Format incompatible: {format_info['name']} {format_info['version']}"
                )
                
            # Reconstruire la géométrie
            geometry = compressed_data['geometry']
            vertices_quantized = np.frombuffer(
                geometry['vertices_quantized'].encode(),
                dtype=np.float32
            ).reshape(-1, 3)
            
            v_min = np.frombuffer(
                geometry['v_min'].encode(),
                dtype=np.float32
            )
            v_max = np.frombuffer(
                geometry['v_max'].encode(),
                dtype=np.float32
            )
            
            faces = np.frombuffer(
                geometry['faces'].encode(),
                dtype=np.int32
            ).reshape(-1, 3)
            
            # Reconstruire les vertices
            v_range = v_max - v_min
            scale = (2**self.format.parameters["quantization_bits"] - 1)
            vertices = (vertices_quantized / scale) * v_range + v_min
            
            # Charger les attributs
            attributes = {}
            for name, data in compressed_data.get('attributes', {}).items():
                attributes[name] = np.frombuffer(
                    data.encode(),
                    dtype=np.float32
                )
                
            # Reconstruire avec l'autoencoder si disponible
            if 'latent' in compressed_data and self.autoencoder is not None:
                latent = torch.FloatTensor(
                    np.frombuffer(
                        compressed_data['latent'].encode(),
                        dtype=np.float32
                    )
                )
                with torch.no_grad():
                    vertices = self.autoencoder.decode(latent).numpy()
                    
        # Créer le maillage
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Ajouter les attributs
        for name, data in attributes.items():
            setattr(mesh, name, data)
            
        return mesh
        
    def save_format(self, path: Path):
        """Sauvegarde la configuration du format."""
        with open(path, 'w') as f:
            json.dump(vars(self.format), f, indent=2)
            
    def load_format(self, path: Path):
        """Charge la configuration du format."""
        with open(path, 'r') as f:
            config = json.load(f)
            self.format = CompressionFormat(**config)
