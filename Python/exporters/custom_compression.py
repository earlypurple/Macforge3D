"""
Système de compression avancé pour modèles 3D.
Optimise les performances et réduit la taille des fichiers pour rivaliser avec les meilleures applications.
"""

import io
import gzip
import bz2
import lzma
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from enum import Enum
import json
import struct

# Imports optionnels avec fallbacks
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("⚠️  trimesh not available. Some compression features will be limited.")

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    # Fallback silencieux pour HDF5

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available. Advanced mesh optimization will be limited.")


class CompressionType(Enum):
    """Types de compression disponibles."""
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"
    CUSTOM = "custom"
    ADAPTIVE = "adaptive"


class QualityLevel(Enum):
    """Niveaux de qualité pour la compression."""
    ULTRA_HIGH = "ultra_high"      # Qualité maximale
    HIGH = "high"                  # Haute qualité
    BALANCED = "balanced"          # Équilibré
    COMPRESSED = "compressed"      # Privilégie la compression
    ULTRA_COMPRESSED = "ultra_compressed"  # Compression maximale


@dataclass
class CompressionSettings:
    """Paramètres de compression personnalisables."""
    compression_type: CompressionType = CompressionType.ADAPTIVE
    quality_level: QualityLevel = QualityLevel.BALANCED
    preserve_textures: bool = True
    preserve_materials: bool = True
    preserve_animations: bool = True
    vertex_precision: int = 6      # Précision des vertices
    normal_precision: int = 4      # Précision des normales
    uv_precision: int = 4          # Précision des UV
    color_quantization: int = 256  # Quantification des couleurs
    enable_mesh_optimization: bool = True
    enable_geometry_compression: bool = True
    enable_texture_compression: bool = True


class AdvancedMeshCompressor:
    """Compresseur de mesh ultra-performant."""
    
    def __init__(self, settings: CompressionSettings):
        self.settings = settings
        
    def compress_vertices(self, vertices: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compression avancée des vertices."""
        metadata = {}
        
        if SKLEARN_AVAILABLE and self.settings.enable_geometry_compression:
            # Analyse PCA pour optimiser l'espace
            pca = PCA(n_components=3)
            vertices_transformed = pca.fit_transform(vertices)
            metadata['pca_components'] = pca.components_.tolist()
            metadata['pca_mean'] = pca.mean_.tolist()
            vertices_data = vertices_transformed
        else:
            vertices_data = vertices
            
        # Quantification adaptative
        precision = self.settings.vertex_precision
        scale = 10 ** precision
        vertices_quantized = (vertices_data * scale).astype(np.int32)
        
        metadata['scale'] = scale
        metadata['dtype'] = str(vertices_quantized.dtype)
        metadata['shape'] = vertices_quantized.shape
        
        # Compression différentielle
        if len(vertices_quantized) > 1:
            deltas = np.diff(vertices_quantized, axis=0)
            first_vertex = vertices_quantized[0:1]
            compressed_data = np.vstack([first_vertex, deltas])
        else:
            compressed_data = vertices_quantized
            
        return compressed_data.tobytes(), metadata
    
    def decompress_vertices(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Décompression des vertices."""
        shape = tuple(metadata['shape'])
        dtype = metadata['dtype']
        scale = metadata['scale']
        
        # Reconstruction des données quantifiées
        vertices_quantized = np.frombuffer(compressed_data, dtype=dtype).reshape(shape)
        
        # Décompression différentielle
        if len(vertices_quantized) > 1:
            first_vertex = vertices_quantized[0:1]
            deltas = vertices_quantized[1:]
            vertices_quantized = np.vstack([first_vertex, np.cumsum(deltas, axis=0) + first_vertex])
        
        # Dé-quantification
        vertices_data = vertices_quantized.astype(np.float32) / scale
        
        # Application PCA inverse si utilisée
        if 'pca_components' in metadata and SKLEARN_AVAILABLE:
            pca_components = np.array(metadata['pca_components'])
            pca_mean = np.array(metadata['pca_mean'])
            vertices = vertices_data @ pca_components + pca_mean
        else:
            vertices = vertices_data
            
        return vertices


class TextureCompressor:
    """Compresseur de textures avancé."""
    
    def __init__(self, settings: CompressionSettings):
        self.settings = settings
        
    def compress_texture_data(self, texture_data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compression intelligente des textures."""
        metadata = {
            'original_shape': texture_data.shape,
            'original_dtype': str(texture_data.dtype)
        }
        
        if not self.settings.enable_texture_compression:
            return texture_data.tobytes(), metadata
            
        # Quantification des couleurs
        if len(texture_data.shape) == 3 and texture_data.shape[2] >= 3:
            # Image couleur
            quantized = self._quantize_colors(texture_data)
        else:
            # Image en niveaux de gris ou autre
            quantized = texture_data
            
        metadata['quantized_shape'] = quantized.shape
        metadata['quantized_dtype'] = str(quantized.dtype)
        
        return quantized.tobytes(), metadata
    
    def _quantize_colors(self, image: np.ndarray) -> np.ndarray:
        """Quantification intelligente des couleurs."""
        if not SKLEARN_AVAILABLE:
            # Quantification simple
            levels = min(self.settings.color_quantization, 256)
            scale = 255 / (levels - 1)
            return np.round(image / scale) * scale
        
        # Quantification avancée avec K-means
        original_shape = image.shape
        pixels = image.reshape(-1, image.shape[-1])
        
        n_colors = min(self.settings.color_quantization, len(pixels))
        if n_colors >= len(pixels):
            return image
            
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        quantized_pixels = kmeans.cluster_centers_[labels]
        
        return quantized_pixels.reshape(original_shape).astype(image.dtype)


class CustomCompressionEngine:
    """Moteur de compression personnalisé ultra-performant."""
    
    def __init__(self, settings: CompressionSettings = None):
        self.settings = settings or CompressionSettings()
        self.mesh_compressor = AdvancedMeshCompressor(self.settings)
        self.texture_compressor = TextureCompressor(self.settings)
        
    def compress_model(self, model_data: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """Compression complète d'un modèle 3D."""
        compressed_sections = {}
        metadata = {
            'compression_type': self.settings.compression_type.value,
            'quality_level': self.settings.quality_level.value,
            'settings': self._serialize_settings(),
            'sections': {}
        }
        
        # Compression des vertices
        if 'vertices' in model_data:
            vertices = np.array(model_data['vertices'])
            compressed_vertices, vertex_metadata = self.mesh_compressor.compress_vertices(vertices)
            compressed_sections['vertices'] = compressed_vertices
            metadata['sections']['vertices'] = vertex_metadata
            
        # Compression des faces
        if 'faces' in model_data:
            faces = np.array(model_data['faces'], dtype=np.uint32)
            compressed_faces = self._compress_faces(faces)
            compressed_sections['faces'] = compressed_faces
            metadata['sections']['faces'] = {
                'shape': faces.shape,
                'dtype': str(faces.dtype)
            }
            
        # Compression des normales
        if 'normals' in model_data:
            normals = np.array(model_data['normals'])
            compressed_normals = self._compress_normals(normals)
            compressed_sections['normals'] = compressed_normals
            metadata['sections']['normals'] = {
                'shape': normals.shape,
                'dtype': str(normals.dtype)
            }
            
        # Compression des UV
        if 'uvs' in model_data:
            uvs = np.array(model_data['uvs'])
            compressed_uvs = self._compress_uvs(uvs)
            compressed_sections['uvs'] = compressed_uvs
            metadata['sections']['uvs'] = {
                'shape': uvs.shape,
                'dtype': str(uvs.dtype)
            }
            
        # Compression des textures
        if 'textures' in model_data:
            textures = {}
            for name, texture_data in model_data['textures'].items():
                if isinstance(texture_data, np.ndarray):
                    compressed_texture, texture_metadata = self.texture_compressor.compress_texture_data(texture_data)
                    textures[name] = compressed_texture
                    metadata['sections'][f'texture_{name}'] = texture_metadata
            compressed_sections['textures'] = textures
            
        # Assemblage final
        final_data = self._pack_compressed_data(compressed_sections, metadata)
        
        # Application de la compression finale
        return self._apply_final_compression(final_data, metadata)
    
    def decompress_model(self, compressed_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Décompression complète d'un modèle 3D."""
        # Décompression finale
        decompressed_data = self._apply_final_decompression(compressed_data, metadata)
        
        # Désassemblage
        sections, section_metadata = self._unpack_compressed_data(decompressed_data, metadata)
        
        model_data = {}
        
        # Décompression des vertices
        if 'vertices' in sections:
            model_data['vertices'] = self.mesh_compressor.decompress_vertices(
                sections['vertices'], 
                section_metadata['sections']['vertices']
            )
            
        # Décompression des autres composants
        for component in ['faces', 'normals', 'uvs']:
            if component in sections:
                component_metadata = section_metadata['sections'][component]
                shape = tuple(component_metadata['shape'])
                dtype = component_metadata['dtype']
                
                data = np.frombuffer(sections[component], dtype=dtype).reshape(shape)
                model_data[component] = data
                
        # Décompression des textures
        if 'textures' in sections:
            model_data['textures'] = {}
            for name, texture_data in sections['textures'].items():
                texture_metadata = section_metadata['sections'][f'texture_{name}']
                shape = tuple(texture_metadata['original_shape'])
                dtype = texture_metadata['original_dtype']
                
                texture_array = np.frombuffer(texture_data, dtype=dtype).reshape(shape)
                model_data['textures'][name] = texture_array
                
        return model_data
    
    def _compress_faces(self, faces: np.ndarray) -> bytes:
        """Compression optimisée des faces."""
        if len(faces) > 1:
            # Compression différentielle pour les indices
            first_face = faces[0:1]
            deltas = np.diff(faces, axis=0)
            compressed_faces = np.vstack([first_face, deltas])
        else:
            compressed_faces = faces
            
        return compressed_faces.tobytes()
    
    def _compress_normals(self, normals: np.ndarray) -> bytes:
        """Compression des normales avec quantification."""
        precision = self.settings.normal_precision
        scale = 10 ** precision
        
        # Normalisation et quantification
        normalized = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        quantized = (normalized * scale).astype(np.int16)
        
        return quantized.tobytes()
    
    def _compress_uvs(self, uvs: np.ndarray) -> bytes:
        """Compression des coordonnées UV."""
        precision = self.settings.uv_precision
        scale = 10 ** precision
        
        quantized = (uvs * scale).astype(np.int16)
        return quantized.tobytes()
    
    def _pack_compressed_data(self, sections: Dict[str, Any], metadata: Dict[str, Any]) -> bytes:
        """Assemblage des données compressées."""
        packed_data = io.BytesIO()
        
        # Écriture des métadonnées
        metadata_json = json.dumps(metadata).encode('utf-8')
        packed_data.write(struct.pack('<I', len(metadata_json)))
        packed_data.write(metadata_json)
        
        # Écriture des sections
        for name, data in sections.items():
            if isinstance(data, dict):  # Textures
                section_data = io.BytesIO()
                for tex_name, tex_data in data.items():
                    tex_name_bytes = tex_name.encode('utf-8')
                    section_data.write(struct.pack('<I', len(tex_name_bytes)))
                    section_data.write(tex_name_bytes)
                    section_data.write(struct.pack('<I', len(tex_data)))
                    section_data.write(tex_data)
                data = section_data.getvalue()
                
            name_bytes = name.encode('utf-8')
            packed_data.write(struct.pack('<I', len(name_bytes)))
            packed_data.write(name_bytes)
            packed_data.write(struct.pack('<I', len(data)))
            packed_data.write(data)
            
        return packed_data.getvalue()
    
    def _unpack_compressed_data(self, data: bytes, metadata: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Désassemblage des données compressées."""
        stream = io.BytesIO(data)
        
        # Lecture des métadonnées
        metadata_size = struct.unpack('<I', stream.read(4))[0]
        metadata_json = stream.read(metadata_size).decode('utf-8')
        section_metadata = json.loads(metadata_json)
        
        # Lecture des sections
        sections = {}
        while stream.tell() < len(data):
            name_size = struct.unpack('<I', stream.read(4))[0]
            name = stream.read(name_size).decode('utf-8')
            data_size = struct.unpack('<I', stream.read(4))[0]
            section_data = stream.read(data_size)
            
            if name == 'textures':
                # Traitement spécial pour les textures
                textures = {}
                tex_stream = io.BytesIO(section_data)
                while tex_stream.tell() < len(section_data):
                    tex_name_size = struct.unpack('<I', tex_stream.read(4))[0]
                    tex_name = tex_stream.read(tex_name_size).decode('utf-8')
                    tex_data_size = struct.unpack('<I', tex_stream.read(4))[0]
                    tex_data = tex_stream.read(tex_data_size)
                    textures[tex_name] = tex_data
                sections[name] = textures
            else:
                sections[name] = section_data
                
        return sections, section_metadata
    
    def _apply_final_compression(self, data: bytes, metadata: Dict[str, Any]) -> Tuple[bytes, Dict[str, Any]]:
        """Application de la compression finale."""
        compression_type = self.settings.compression_type
        
        if compression_type == CompressionType.ADAPTIVE:
            # Test de plusieurs algorithmes et choix du meilleur
            algorithms = [
                (CompressionType.GZIP, gzip.compress(data)),
                (CompressionType.BZIP2, bz2.compress(data)),
                (CompressionType.LZMA, lzma.compress(data))
            ]
            
            best_algo, best_data = min(algorithms, key=lambda x: len(x[1]))
            metadata['final_compression'] = best_algo.value
            return best_data, metadata
            
        elif compression_type == CompressionType.GZIP:
            metadata['final_compression'] = 'gzip'
            return gzip.compress(data), metadata
            
        elif compression_type == CompressionType.BZIP2:
            metadata['final_compression'] = 'bzip2'
            return bz2.compress(data), metadata
            
        elif compression_type == CompressionType.LZMA:
            metadata['final_compression'] = 'lzma'
            return lzma.compress(data), metadata
            
        else:
            metadata['final_compression'] = 'none'
            return data, metadata
    
    def _apply_final_decompression(self, data: bytes, metadata: Dict[str, Any]) -> bytes:
        """Application de la décompression finale."""
        compression_type = metadata.get('final_compression', 'none')
        
        if compression_type == 'gzip':
            return gzip.decompress(data)
        elif compression_type == 'bzip2':
            return bz2.decompress(data)
        elif compression_type == 'lzma':
            return lzma.decompress(data)
        else:
            return data
    
    def _serialize_settings(self) -> Dict[str, Any]:
        """Sérialisation des paramètres."""
        return {
            'compression_type': self.settings.compression_type.value,
            'quality_level': self.settings.quality_level.value,
            'preserve_textures': self.settings.preserve_textures,
            'preserve_materials': self.settings.preserve_materials,
            'preserve_animations': self.settings.preserve_animations,
            'vertex_precision': self.settings.vertex_precision,
            'normal_precision': self.settings.normal_precision,
            'uv_precision': self.settings.uv_precision,
            'color_quantization': self.settings.color_quantization,
            'enable_mesh_optimization': self.settings.enable_mesh_optimization,
            'enable_geometry_compression': self.settings.enable_geometry_compression,
            'enable_texture_compression': self.settings.enable_texture_compression,
        }


def compress_3d_model(
    model_data: Dict[str, Any], 
    settings: Optional[CompressionSettings] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Interface principale pour compresser un modèle 3D.
    
    Args:
        model_data: Données du modèle 3D
        settings: Paramètres de compression
        output_path: Chemin de sortie optionnel
        
    Returns:
        Tuple contenant les données compressées et les métadonnées
    """
    if settings is None:
        settings = CompressionSettings()
        
    engine = CustomCompressionEngine(settings)
    compressed_data, metadata = engine.compress_model(model_data)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarde des données compressées
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
            
        # Sauvegarde des métadonnées
        metadata_path = output_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    return compressed_data, metadata


def decompress_3d_model(
    compressed_data: bytes, 
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Interface principale pour décompresser un modèle 3D.
    
    Args:
        compressed_data: Données compressées
        metadata: Métadonnées de compression
        
    Returns:
        Données du modèle 3D décompressé
    """
    settings = CompressionSettings()  # Les paramètres seront récupérés des métadonnées
    engine = CustomCompressionEngine(settings)
    return engine.decompress_model(compressed_data, metadata)


def load_compressed_model(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Charge un modèle 3D compressé depuis un fichier.
    
    Args:
        file_path: Chemin vers le fichier compressé
        
    Returns:
        Données du modèle 3D décompressé
    """
    file_path = Path(file_path)
    metadata_path = file_path.with_suffix('.meta.json')
    
    # Chargement des données compressées
    with open(file_path, 'rb') as f:
        compressed_data = f.read()
        
    # Chargement des métadonnées
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return decompress_3d_model(compressed_data, metadata)


def get_compression_stats(
    original_data: Dict[str, Any], 
    compressed_data: bytes, 
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calcule les statistiques de compression.
    
    Args:
        original_data: Données originales
        compressed_data: Données compressées
        metadata: Métadonnées de compression
        
    Returns:
        Statistiques de compression
    """
    # Calcul de la taille originale
    original_size = 0
    for key, value in original_data.items():
        if isinstance(value, np.ndarray):
            original_size += value.nbytes
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    original_size += subvalue.nbytes
                    
    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    space_saved = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': compression_ratio,
        'space_saved_percent': space_saved,
        'compression_type': metadata.get('compression_type', 'unknown'),
        'quality_level': metadata.get('quality_level', 'unknown')
    }


# Fonction de test intégrée
def test_compression_system():
    """Test du système de compression."""
    print("🧪 Test du système de compression avancé...")
    
    # Création de données de test
    test_model = {
        'vertices': np.random.rand(1000, 3).astype(np.float32),
        'faces': np.random.randint(0, 1000, (500, 3)).astype(np.uint32),
        'normals': np.random.rand(1000, 3).astype(np.float32),
        'uvs': np.random.rand(1000, 2).astype(np.float32),
        'textures': {
            'diffuse': np.random.randint(0, 256, (256, 256, 3)).astype(np.uint8)
        }
    }
    
    # Test avec différents niveaux de qualité
    for quality in QualityLevel:
        settings = CompressionSettings(quality_level=quality)
        
        try:
            compressed_data, metadata = compress_3d_model(test_model, settings)
            decompressed_model = decompress_3d_model(compressed_data, metadata)
            stats = get_compression_stats(test_model, compressed_data, metadata)
            
            print(f"✓ {quality.value}: {stats['compression_ratio']:.2f}x compression, "
                  f"{stats['space_saved_percent']:.1f}% space saved")
                  
        except Exception as e:
            print(f"✗ {quality.value}: {e}")
    
    print("✅ Système de compression testé avec succès!")


if __name__ == "__main__":
    test_compression_system()
