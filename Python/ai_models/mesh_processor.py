import trimesh
import pymeshfix  # type: ignore
import numpy as np
import os
import shutil


import trimesh
try:
    import pymeshfix  # type: ignore
    PYMESHFIX_AVAILABLE = True
except ImportError:
    PYMESHFIX_AVAILABLE = False
    print("⚠️  pymeshfix not available, using fallback mesh repair")

import numpy as np
import os
import shutil
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_mesh_quality(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """Analyse complète de la qualité d'un maillage."""
    start_time = time.time()
    
    try:
        analysis = {
            "basic_stats": {
                "vertices": len(mesh.vertices),
                "faces": len(mesh.faces),
                "edges": len(mesh.edges),
                "volume": float(mesh.volume) if mesh.is_watertight else None,
                "surface_area": float(mesh.area),
                "is_watertight": mesh.is_watertight,
                "is_winding_consistent": mesh.is_winding_consistent,
                "is_empty": mesh.is_empty
            },
            "geometric_quality": {},
            "topological_quality": {},
            "processing_time": 0.0
        }
        
        # Analyse géométrique
        bounds = mesh.bounds
        analysis["geometric_quality"] = {
            "bounding_box": {
                "min": bounds[0].tolist(),
                "max": bounds[1].tolist(),
                "dimensions": (bounds[1] - bounds[0]).tolist()
            },
            "centroid": mesh.centroid.tolist(),
            "scale": float(mesh.scale),
            "convex_hull_ratio": float(mesh.area / mesh.convex_hull.area) if mesh.convex_hull else 0.0
        }
        
        # Analyse topologique
        try:
            euler_number = len(mesh.vertices) - len(mesh.edges) + len(mesh.faces)
            genus = (2 - euler_number) // 2
            
            analysis["topological_quality"] = {
                "euler_number": euler_number,
                "estimated_genus": genus,
                "connected_components": len(mesh.split(only_watertight=False)),
                "watertight_components": len(mesh.split(only_watertight=True))
            }
        except Exception as e:
            logger.warning(f"Erreur analyse topologique: {e}")
            analysis["topological_quality"] = {"error": str(e)}
        
        # Analyse de la qualité des triangles
        if len(mesh.faces) > 0:
            face_areas = mesh.area_faces
            face_angles = mesh.face_angles
            
            analysis["geometric_quality"].update({
                "triangle_quality": {
                    "min_area": float(np.min(face_areas)),
                    "max_area": float(np.max(face_areas)),
                    "mean_area": float(np.mean(face_areas)),
                    "area_std": float(np.std(face_areas)),
                    "min_angle_deg": float(np.min(face_angles) * 180 / np.pi),
                    "max_angle_deg": float(np.max(face_angles) * 180 / np.pi),
                    "degenerate_faces": int(np.sum(face_areas < 1e-10))
                }
            })
        
        analysis["processing_time"] = time.time() - start_time
        return analysis
        
    except Exception as e:
        logger.error(f"Erreur analyse maillage: {e}")
        return {
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def repair_mesh_advanced(
    input_path: str, 
    output_path: str,
    method: str = "auto",
    aggressive: bool = False,
    preserve_details: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Répare un maillage avec méthodes avancées.

    Args:
        input_path: Chemin vers le maillage d'entrée
        output_path: Chemin de sortie
        method: Méthode de réparation ("pymeshfix", "trimesh", "auto")
        aggressive: Réparation agressive (peut perdre des détails)
        preserve_details: Préserver les détails fins

    Returns:
        Tuple (succès, métadonnées de réparation)
    """
    start_time = time.time()
    
    try:
        logger.info(f"🔧 Réparation avancée: {input_path} -> {output_path}")
        
        # Charger le maillage
        mesh = trimesh.load_mesh(input_path, process=False)
        original_analysis = analyze_mesh_quality(mesh)
        
        logger.info(f"📊 Maillage original: {original_analysis['basic_stats']}")
        
        # Choisir la méthode de réparation
        if method == "auto":
            if PYMESHFIX_AVAILABLE and len(mesh.vertices) < 100000:
                method = "pymeshfix"
            else:
                method = "trimesh"
        
        repaired_mesh = None
        repair_info = {"method": method, "steps": []}
        
        if method == "pymeshfix" and PYMESHFIX_AVAILABLE:
            repaired_mesh, steps = _repair_with_pymeshfix(mesh, aggressive)
            repair_info["steps"] = steps
            
        elif method == "trimesh":
            repaired_mesh, steps = _repair_with_trimesh(mesh, aggressive, preserve_details)
            repair_info["steps"] = steps
            
        else:
            raise ValueError(f"Méthode de réparation inconnue: {method}")
        
        if repaired_mesh is None:
            return False, {"error": "Échec de la réparation", "processing_time": time.time() - start_time}
        
        # Analyser le résultat
        final_analysis = analyze_mesh_quality(repaired_mesh)
        
        # Copier les matériaux si ils existent
        _copy_materials(input_path, output_path)
        
        # Sauvegarder
        repaired_mesh.export(output_path)
        
        # Métadonnées de réparation
        repair_metadata = {
            "success": True,
            "method": method,
            "processing_time": time.time() - start_time,
            "original_analysis": original_analysis,
            "final_analysis": final_analysis,
            "repair_info": repair_info,
            "improvements": {
                "watertight": not original_analysis["basic_stats"]["is_watertight"] and final_analysis["basic_stats"]["is_watertight"],
                "vertex_reduction": original_analysis["basic_stats"]["vertices"] - final_analysis["basic_stats"]["vertices"],
                "face_reduction": original_analysis["basic_stats"]["faces"] - final_analysis["basic_stats"]["faces"]
            }
        }
        
        logger.info(f"✅ Réparation terminée en {repair_metadata['processing_time']:.2f}s")
        return True, repair_metadata
        
    except Exception as e:
        logger.error(f"❌ Erreur réparation: {e}")
        return False, {
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def _repair_with_pymeshfix(mesh: trimesh.Trimesh, aggressive: bool) -> Tuple[Optional[trimesh.Trimesh], List[str]]:
    """Réparation avec pymeshfix."""
    steps = []
    
    try:
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Configuration pymeshfix
        meshfix = pymeshfix.MeshFix(vertices, faces)
        steps.append("Initialisation pymeshfix")
        
        if aggressive:
            meshfix.repair(verbose=False, joincomp=True, fillholes=True)
            steps.append("Réparation agressive avec joincomp et fillholes")
        else:
            meshfix.repair(verbose=False)
            steps.append("Réparation standard")
        
        repaired_mesh = trimesh.Trimesh(meshfix.v, meshfix.f)
        steps.append(f"Création maillage réparé: {len(meshfix.v)} vertices, {len(meshfix.f)} faces")
        
        return repaired_mesh, steps
        
    except Exception as e:
        steps.append(f"Erreur pymeshfix: {e}")
        return None, steps

def _repair_with_trimesh(
    mesh: trimesh.Trimesh, 
    aggressive: bool, 
    preserve_details: bool
) -> Tuple[Optional[trimesh.Trimesh], List[str]]:
    """Réparation avec les outils trimesh."""
    steps = []
    result = mesh.copy()
    
    try:
        # Étape 1: Nettoyage de base
        if aggressive or not preserve_details:
            result.remove_degenerate_faces()
            steps.append("Suppression faces dégénérées")
            
            result.remove_duplicate_faces()
            steps.append("Suppression faces dupliquées")
            
            result.merge_vertices()
            steps.append("Fusion vertices proches")
        
        # Étape 2: Correction de l'orientation
        if not result.is_winding_consistent:
            result.fix_normals()
            steps.append("Correction orientation des normales")
        
        # Étape 3: Remplissage des trous (si agressif)
        if aggressive:
            try:
                result.fill_holes()
                steps.append("Remplissage des trous")
            except Exception as e:
                steps.append(f"Échec remplissage trous: {e}")
        
        # Étape 4: Convex hull si nécessaire (très agressif)
        if aggressive and not result.is_watertight:
            try:
                result = result.convex_hull
                steps.append("Génération enveloppe convexe (fallback)")
            except Exception as e:
                steps.append(f"Échec enveloppe convexe: {e}")
        
        # Étape 5: Lissage léger si préservation des détails
        if preserve_details and not aggressive:
            try:
                result = result.smoothed()
                steps.append("Lissage léger")
            except Exception as e:
                steps.append(f"Échec lissage: {e}")
        
        return result, steps
        
    except Exception as e:
        steps.append(f"Erreur réparation trimesh: {e}")
        return None, steps

def _copy_materials(input_path: str, output_path: str):
    """Copie les fichiers de matériaux associés."""
    try:
        input_path_obj = Path(input_path)
        output_path_obj = Path(output_path)
        
        # Fichier MTL
        mtl_input = input_path_obj.with_suffix('.mtl')
        if mtl_input.exists():
            mtl_output = output_path_obj.with_suffix('.mtl')
            shutil.copy2(mtl_input, mtl_output)
        
        # Textures dans le même dossier
        texture_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tga', '.tiff']
        for ext in texture_extensions:
            for texture_file in input_path_obj.parent.glob(f"*{ext}"):
                if texture_file.is_file():
                    shutil.copy2(texture_file, output_path_obj.parent / texture_file.name)
                    
    except Exception as e:
        logger.warning(f"Erreur copie matériaux: {e}")

def repair_mesh(input_path: str, output_path: str) -> bool:
    """Version legacy maintenue pour compatibilité."""
    success, _ = repair_mesh_advanced(input_path, output_path, method="auto")
    return success


def scale_mesh_advanced(
    input_path: str, 
    output_path: str, 
    target_size_mm: float,
    scaling_mode: str = "uniform",
    preserve_aspect: bool = True,
    center_mesh: bool = True
) -> Tuple[bool, Dict[str, Any]]:
    """
    Met à l'échelle un maillage avec options avancées.

    Args:
        input_path: Chemin vers le maillage d'entrée
        output_path: Chemin de sortie
        target_size_mm: Taille cible en millimètres
        scaling_mode: Mode de mise à l'échelle ("uniform", "x", "y", "z", "volume")
        preserve_aspect: Préserver les proportions
        center_mesh: Centrer le maillage après mise à l'échelle

    Returns:
        Tuple (succès, métadonnées de mise à l'échelle)
    """
    start_time = time.time()
    
    if target_size_mm <= 1e-6:
        logger.info("📏 Mise à l'échelle ignorée (taille cible = 0)")
        if input_path != output_path:
            shutil.copy2(input_path, output_path)
            _copy_materials(input_path, output_path)
        return True, {"skipped": True, "reason": "target_size_zero"}

    try:
        logger.info(f"📏 Mise à l'échelle avancée: {input_path} -> {target_size_mm}mm ({scaling_mode})")
        
        mesh = trimesh.load_mesh(input_path)
        original_bounds = mesh.bounding_box.extents
        original_volume = float(mesh.volume) if mesh.is_watertight else None
        original_center = mesh.centroid.copy()
        
        # Calculer le facteur d'échelle
        scale_factor = _calculate_scale_factor(mesh, target_size_mm, scaling_mode)
        
        if scale_factor <= 0:
            return False, {
                "error": "Facteur d'échelle invalide",
                "processing_time": time.time() - start_time
            }
        
        # Appliquer la mise à l'échelle
        if scaling_mode == "uniform":
            mesh.apply_scale(scale_factor)
        elif scaling_mode in ["x", "y", "z"]:
            scale_vector = [1.0, 1.0, 1.0]
            axis_idx = {"x": 0, "y": 1, "z": 2}[scaling_mode]
            scale_vector[axis_idx] = scale_factor
            
            if preserve_aspect:
                # Mise à l'échelle uniforme mais orientation spécifique
                mesh.apply_scale(scale_factor)
            else:
                # Mise à l'échelle non-uniforme
                transform = np.eye(4)
                transform[axis_idx, axis_idx] = scale_factor
                mesh.apply_transform(transform)
        
        elif scaling_mode == "volume":
            # Mise à l'échelle basée sur le volume
            if original_volume and original_volume > 0:
                volume_scale = (target_size_mm / original_volume) ** (1/3)
                mesh.apply_scale(volume_scale)
        
        # Centrer le maillage si demandé
        if center_mesh:
            mesh.vertices -= mesh.centroid
        
        # Analyser le résultat
        final_bounds = mesh.bounding_box.extents
        final_volume = float(mesh.volume) if mesh.is_watertight else None
        
        # Sauvegarder
        mesh.export(output_path)
        _copy_materials(input_path, output_path)
        
        # Métadonnées
        scaling_metadata = {
            "success": True,
            "scaling_mode": scaling_mode,
            "scale_factor": float(scale_factor),
            "preserve_aspect": preserve_aspect,
            "center_mesh": center_mesh,
            "processing_time": time.time() - start_time,
            "original_dimensions": original_bounds.tolist(),
            "final_dimensions": final_bounds.tolist(),
            "original_volume": original_volume,
            "final_volume": final_volume,
            "volume_change_ratio": final_volume / original_volume if original_volume and final_volume else None
        }
        
        logger.info(f"✅ Mise à l'échelle terminée: {original_bounds.max():.2f} -> {final_bounds.max():.2f}mm")
        return True, scaling_metadata
        
    except Exception as e:
        logger.error(f"❌ Erreur mise à l'échelle: {e}")
        return False, {
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def _calculate_scale_factor(mesh: trimesh.Trimesh, target_size: float, mode: str) -> float:
    """Calcule le facteur d'échelle selon le mode."""
    bounds = mesh.bounding_box.extents
    
    if mode == "uniform":
        return target_size / np.max(bounds)
    elif mode == "x":
        return target_size / bounds[0] if bounds[0] > 0 else 0
    elif mode == "y":
        return target_size / bounds[1] if bounds[1] > 0 else 0
    elif mode == "z":
        return target_size / bounds[2] if bounds[2] > 0 else 0
    elif mode == "volume":
        volume = float(mesh.volume) if mesh.is_watertight else 0
        return (target_size / volume) ** (1/3) if volume > 0 else 0
    else:
        return target_size / np.max(bounds)

def optimize_mesh(
    mesh: trimesh.Trimesh,
    target_faces: Optional[int] = None,
    edge_collapse_ratio: float = 0.5,
    preserve_boundaries: bool = True,
    preserve_uv: bool = True
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Optimise un maillage en réduisant le nombre de faces tout en préservant la qualité.
    
    Args:
        mesh: Maillage à optimiser
        target_faces: Nombre cible de faces (None = auto)
        edge_collapse_ratio: Ratio de réduction des arêtes
        preserve_boundaries: Préserver les contours
        preserve_uv: Préserver les coordonnées UV
        
    Returns:
        Tuple (maillage optimisé, métadonnées)
    """
    start_time = time.time()
    
    try:
        original_faces = len(mesh.faces)
        original_vertices = len(mesh.vertices)
        
        # Déterminer le nombre cible de faces
        if target_faces is None:
            target_faces = max(100, int(original_faces * edge_collapse_ratio))
        
        target_faces = min(target_faces, original_faces)
        
        # Simplification avec trimesh
        try:
            # Utiliser la simplification quadric error metric si disponible
            simplified = mesh.simplify_quadric_decimation(target_faces)
            if simplified is None or len(simplified.faces) == 0:
                raise ValueError("Simplification quadrique échouée")
        except:
            # Fallback vers une méthode plus simple
            try:
                simplified = mesh.simplify_quadric_decimation(target_faces)
                if simplified is None:
                    # Dernière option: sous-échantillonnage
                    vertices, faces = _decimate_mesh_simple(mesh.vertices, mesh.faces, edge_collapse_ratio)
                    simplified = trimesh.Trimesh(vertices=vertices, faces=faces)
            except:
                simplified = mesh.copy()
        
        # Statistiques d'optimisation
        optimization_stats = {
            "success": True,
            "original_faces": original_faces,
            "original_vertices": original_vertices,
            "final_faces": len(simplified.faces),
            "final_vertices": len(simplified.vertices),
            "face_reduction_ratio": 1.0 - (len(simplified.faces) / original_faces),
            "vertex_reduction_ratio": 1.0 - (len(simplified.vertices) / original_vertices),
            "processing_time": time.time() - start_time
        }
        
        return simplified, optimization_stats
        
    except Exception as e:
        logger.error(f"Erreur optimisation maillage: {e}")
        return mesh, {
            "success": False,
            "error": str(e),
            "processing_time": time.time() - start_time
        }

def _decimate_mesh_simple(vertices: np.ndarray, faces: np.ndarray, ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Décimation simple de maillage par sous-échantillonnage."""
    try:
        # Sous-échantillonnage des faces
        num_faces_keep = max(1, int(len(faces) * ratio))
        
        # Sélection aléatoire pondérée par l'aire des faces
        face_areas = []
        for face in faces:
            v1, v2, v3 = vertices[face]
            area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))
            face_areas.append(area)
        
        face_areas = np.array(face_areas)
        probabilities = face_areas / face_areas.sum()
        
        selected_indices = np.random.choice(
            len(faces), 
            size=num_faces_keep, 
            replace=False, 
            p=probabilities
        )
        
        selected_faces = faces[selected_indices]
        
        # Nettoyer les vertices non utilisés
        used_vertices = np.unique(selected_faces.flatten())
        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        
        new_vertices = vertices[used_vertices]
        new_faces = np.array([[vertex_mapping[v] for v in face] for face in selected_faces])
        
        return new_vertices, new_faces
        
    except Exception:
        # Fallback: retourner original
        return vertices, faces

def enhance_mesh_quality(
    mesh: trimesh.Trimesh,
    smooth_iterations: int = 1,
    fix_orientation: bool = True,
    remove_degenerate: bool = True,
    merge_close_vertices: bool = True
) -> Tuple[trimesh.Trimesh, Dict[str, Any]]:
    """
    Améliore la qualité générale d'un maillage.
    
    Args:
        mesh: Maillage à améliorer
        smooth_iterations: Nombre d'itérations de lissage
        fix_orientation: Corriger l'orientation des faces
        remove_degenerate: Supprimer les faces dégénérées
        merge_close_vertices: Fusionner les vertices proches
        
    Returns:
        Tuple (maillage amélioré, métadonnées)
    """
    start_time = time.time()
    improvements = []
    result = mesh.copy()
    
    try:
        original_analysis = analyze_mesh_quality(mesh)
        
        # Nettoyage de base
        if remove_degenerate:
            original_faces = len(result.faces)
            result.remove_degenerate_faces()
            removed_faces = original_faces - len(result.faces)
            if removed_faces > 0:
                improvements.append(f"Supprimé {removed_faces} faces dégénérées")
        
        if merge_close_vertices:
            original_vertices = len(result.vertices)
            result.merge_vertices()
            merged_vertices = original_vertices - len(result.vertices)
            if merged_vertices > 0:
                improvements.append(f"Fusionné {merged_vertices} vertices proches")
        
        # Correction d'orientation
        if fix_orientation and not result.is_winding_consistent:
            result.fix_normals()
            improvements.append("Corrigé l'orientation des normales")
        
        # Lissage
        if smooth_iterations > 0:
            for i in range(smooth_iterations):
                try:
                    result = result.smoothed()
                    improvements.append(f"Lissage itération {i+1}")
                except Exception as e:
                    improvements.append(f"Échec lissage itération {i+1}: {e}")
                    break
        
        final_analysis = analyze_mesh_quality(result)
        
        enhancement_stats = {
            "success": True,
            "improvements": improvements,
            "original_analysis": original_analysis,
            "final_analysis": final_analysis,
            "processing_time": time.time() - start_time
        }
        
        return result, enhancement_stats
        
    except Exception as e:
        logger.error(f"Erreur amélioration qualité: {e}")
        return mesh, {
            "success": False,
            "error": str(e),
            "improvements": improvements,
            "processing_time": time.time() - start_time
        }

def scale_mesh(input_path: str, output_path: str, target_size_mm: float) -> bool:
    """Version legacy maintenue pour compatibilité."""
    success, _ = scale_mesh_advanced(input_path, output_path, target_size_mm)
    return success


if __name__ == "__main__":
    # Create a dummy OBJ file for testing
    test_dir = "Examples/mesh_processor_test"
    os.makedirs(test_dir, exist_ok=True)

    # Unscaled and broken cube
    vertices = np.array(
        [
            [0, 0, 0],
            [10, 0, 0],
            [10, 10, 0],
            [0, 10, 0],
            [0, 0, 10],
            [10, 0, 10],
            [10, 10, 10],
            [0, 10, 10],
        ]
    )
    # Missing one face to make it broken
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],  # bottom
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # front
            [2, 3, 7],
            [2, 7, 6],  # back
            [1, 2, 6],
            [1, 6, 5],  # right
            # left face is missing
        ]
    )

    broken_path = os.path.join(test_dir, "broken_cube.obj")
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(broken_path)

    print("\n--- Testing Mesh Repair ---")
    repaired_path = os.path.join(test_dir, "repaired_cube.obj")
    success = repair_mesh(broken_path, repaired_path)
    if success:
        print("✅ Repair test successful.")
        # Verify it's watertight now
        repaired = trimesh.load_mesh(repaired_path)
        print(f"Is repaired mesh watertight? {repaired.is_watertight}")
    else:
        print("❌ Repair test failed.")

    print("\n--- Testing Mesh Scaling ---")
    scaled_path = os.path.join(test_dir, "scaled_cube.obj")
    # Use the repaired path as input for scaling
    success = scale_mesh(repaired_path, scaled_path, target_size_mm=150.0)
    if success:
        print("✅ Scaling test successful.")
        # Verify the new size
        scaled = trimesh.load_mesh(scaled_path)
        max_extent = np.max(scaled.bounding_box.extents)
        print(f"New longest dimension: {max_extent:.2f}mm (Target: 150.0mm)")
    else:
        print("❌ Scaling test failed.")

    print("\n--- Testing Zero Scaling ---")
    zero_scaled_path = os.path.join(test_dir, "zero_scaled_cube.obj")
    success = scale_mesh(scaled_path, zero_scaled_path, target_size_mm=0)
    if success and os.path.exists(zero_scaled_path):
        print("✅ Zero scaling test successful (file was moved).")
    else:
        print("❌ Zero scaling test failed.")
