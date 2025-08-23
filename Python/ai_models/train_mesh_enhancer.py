"""
Script d'entraînement pour le MeshEnhancer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple
import trimesh
import os
from pathlib import Path
import logging
from tqdm import tqdm
from mesh_enhancer import MeshEnhancer, MeshEnhancementConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MeshDataset(torch.utils.data.Dataset):
    """Dataset pour l'entraînement du MeshEnhancer."""
    
    def __init__(
        self,
        data_dir: str,
        max_points: int = 10000,
        augment: bool = True
    ):
        """
        Args:
            data_dir: Dossier contenant les paires de maillages (low/high)
            max_points: Nombre maximum de points par maillage
            augment: Activer l'augmentation de données
        """
        self.data_dir = Path(data_dir)
        self.max_points = max_points
        self.augment = augment
        
        # Trouver toutes les paires de maillages
        self.pairs = []
        for low_res in self.data_dir.glob("low_res/*.ply"):
            high_res = self.data_dir / "high_res" / low_res.name
            if high_res.exists():
                self.pairs.append((low_res, high_res))
                
        logger.info(f"Trouvé {len(self.pairs)} paires de maillages")
        
    def __len__(self) -> int:
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        low_path, high_path = self.pairs[idx]
        
        # Charger les maillages
        low_mesh = trimesh.load(low_path)
        high_mesh = trimesh.load(high_path)
        
        # Normaliser
        low_mesh.vertices -= low_mesh.vertices.mean(axis=0)
        high_mesh.vertices -= high_mesh.vertices.mean(axis=0)
        scale = np.abs(low_mesh.vertices).max()
        low_mesh.vertices /= scale
        high_mesh.vertices /= scale
        
        # Augmentation de données
        if self.augment:
            # Rotation aléatoire
            angle = np.random.uniform(0, 2 * np.pi)
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            R = trimesh.transformations.rotation_matrix(angle, axis)
            low_mesh.apply_transform(R)
            high_mesh.apply_transform(R)
            
            # Bruit gaussien
            noise = np.random.normal(0, 0.01, low_mesh.vertices.shape)
            low_mesh.vertices += noise
            
        # Sous-échantillonnage si nécessaire
        if len(low_mesh.vertices) > self.max_points:
            indices = np.random.choice(
                len(low_mesh.vertices),
                self.max_points,
                replace=False
            )
            low_mesh.vertices = low_mesh.vertices[indices]
            if len(low_mesh.faces) > 0:
                valid_faces = np.all(
                    np.isin(low_mesh.faces, indices),
                    axis=1
                )
                low_mesh.faces = low_mesh.faces[valid_faces]
                
        # Convertir en tenseurs
        low_vertices = torch.FloatTensor(low_mesh.vertices)
        high_vertices = torch.FloatTensor(high_mesh.vertices)
        
        return low_vertices, high_vertices

def train(
    model: MeshEnhancer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    device: str = "cuda"
):
    """Entraîne le modèle."""
    optimizer = optim.Adam(
        list(model.encoder.parameters()) +
        list(model.decoder.parameters()),
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    criterion = nn.MSELoss()
    chamfer_loss = ChamferLoss()
    
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # Entraînement
        model.encoder.train()
        model.decoder.train()
        train_loss = 0
        
        for low_vertices, high_vertices in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}"
        ):
            low_vertices = low_vertices.to(device)
            high_vertices = high_vertices.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            latent = model.encoder(low_vertices.unsqueeze(0))
            output = model.decoder(
                latent,
                low_vertices.shape[0]
            )
            
            # Calcul de la perte
            reconstruction_loss = criterion(
                output[0].transpose(0, 1),
                high_vertices
            )
            distance_loss = chamfer_loss(
                output[0].transpose(0, 1).unsqueeze(0),
                high_vertices.unsqueeze(0)
            )
            
            loss = reconstruction_loss + 0.1 * distance_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.encoder.eval()
        model.decoder.eval()
        val_loss = 0
        
        with torch.no_grad():
            for low_vertices, high_vertices in val_loader:
                low_vertices = low_vertices.to(device)
                high_vertices = high_vertices.to(device)
                
                latent = model.encoder(low_vertices.unsqueeze(0))
                output = model.decoder(
                    latent,
                    low_vertices.shape[0]
                )
                
                reconstruction_loss = criterion(
                    output[0].transpose(0, 1),
                    high_vertices
                )
                distance_loss = chamfer_loss(
                    output[0].transpose(0, 1).unsqueeze(0),
                    high_vertices.unsqueeze(0)
                )
                
                loss = reconstruction_loss + 0.1 * distance_loss
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Mise à jour du scheduler
        scheduler.step(val_loss)
        
        # Sauvegarder le meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "encoder": model.encoder.state_dict(),
                    "decoder": model.decoder.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss
                },
                "models/mesh_enhancer_best.pth"
            )
            
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss = {train_loss:.6f}, "
            f"Val Loss = {val_loss:.6f}"
        )
        
class ChamferLoss(nn.Module):
    """Implémentation de la perte de Chamfer."""
    
    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            xyz1: (batch_size, num_points, 3)
            xyz2: (batch_size, num_points, 3)
            
        Returns:
            Chamfer loss
        """
        dist1, dist2 = self.chamfer_distance(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)
        
    def chamfer_distance(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule la distance de Chamfer."""
        B, N, _ = xyz1.shape
        _, M, _ = xyz2.shape
        
        xyz1 = xyz1.unsqueeze(2)
        xyz2 = xyz2.unsqueeze(1)
        
        dist = torch.sum(
            (xyz1 - xyz2) ** 2,
            dim=-1
        )
        
        dist1, _ = torch.min(dist, dim=2)
        dist2, _ = torch.min(dist, dim=1)
        
        return dist1, dist2

if __name__ == "__main__":
    # Configuration
    config = MeshEnhancementConfig(
        resolution_factor=1.5,
        smoothness_weight=0.3,
        detail_preservation=0.8
    )
    
    # Création des datasets
    train_dataset = MeshDataset(
        "data/train",
        max_points=10000,
        augment=True
    )
    val_dataset = MeshDataset(
        "data/val",
        max_points=10000,
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )
    
    # Création et entraînement du modèle
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MeshEnhancer(config)
    model = model.to(device)
    
    train(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.001,
        device=device
    )
