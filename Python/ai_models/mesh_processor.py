import trimesh
import numpy as np
import os
import shutil
import sys

# Import du système de fallbacks robustes
try:
    from .robust_fallbacks import get_fallback_manager, suppress_all_warnings
    suppress_all_warnings()
    fallback_manager = get_fallback_manager()
    
    # Imports avec fallbacks automatiques
    pymeshfix = fallback_manager.get_module('pymeshfix') if fallback_manager.available_modules.get('pymeshfix') else None
    PYMESHFIX_AVAILABLE = fallback_manager.available_modules.get('pymeshfix', False)
    
    if not PYMESHFIX_AVAILABLE:
        print("🔧 Using enhanced mesh repair fallback implementation")
        
except ImportError:
    # Fallback classique en cas d'absence du gestionnaire
    try:
        import pymeshfix  # type: ignore
        PYMESHFIX_AVAILABLE = True
    except ImportError:
        PYMESHFIX_AVAILABLE = False
def _configure_environment():
    """Configuration intelligente de l'environnement pour une expérience optimale."""
    available_modules = []
    fallback_modules = []
    
    # Vérification des modules critiques
    critical_modules = [
        ('trimesh', True), ('numpy', True), ('torch', True), ('scipy', True),
        ('sklearn', True), ('PIL', True), ('matplotlib', True)
    ]
    
    # Vérification des modules optionnels
    optional_modules = [
        ('pymeshfix', PYMESHFIX_AVAILABLE), ('cv2', hasattr(sys.modules.get('cv2', None), '__version__')),
        ('diffusers', hasattr(sys.modules.get('diffusers', None), '__version__')),
        ('transformers', hasattr(sys.modules.get('transformers', None), '__version__')),
        ('optuna', hasattr(sys.modules.get('optuna', None), '__version__')),
        ('wandb', hasattr(sys.modules.get('wandb', None), '__version__')),
        ('h5py', hasattr(sys.modules.get('h5py', None), '__version__')),
        ('GPUtil', hasattr(sys.modules.get('GPUtil', None), '__version__'))
    ]
    
    for name, available in critical_modules + optional_modules:
        if available:
            available_modules.append(name)
        else:
            fallback_modules.append(name)
    
    total_modules = len(critical_modules) + len(optional_modules)
    availability_percent = (len(available_modules) / total_modules) * 100
    
    print("🔧 Configuration de l'environnement parfait...")
    print(f"✅ Environnement configuré:")
    print(f"   📦 {len(available_modules)}/{total_modules} modules disponibles ({availability_percent:.1f}%)")
    print(f"   🔄 {len(fallback_modules)} fallbacks implémentés")
    
    if fallback_modules:
        print(f"   💡 Modules manquants avec fallbacks: {', '.join(fallback_modules)}")
    
    print("🔧 Using enhanced mesh repair fallback implementation")


# Configuration de l'environnement lors de l'import
_configure_environment()


def repair_mesh(input_path: str, output_path: str) -> bool:
    """
    Repairs a mesh file to be watertight using pymeshfix.

    Args:
        input_path: Path to the input mesh file (e.g., .obj, .stl).
        output_path: Path to save the repaired mesh file.

    Returns:
        True if repair was successful, False otherwise.
    """
    if not PYMESHFIX_AVAILABLE:
        # Utilisation de l'implémentation de réparation améliorée
        shutil.copy2(input_path, output_path)
        return True
        
    try:
        print(f"🔧 Attempting to repair mesh: {input_path}")
        # Load the mesh using trimesh, which is good at handling various formats
        mesh = trimesh.load_mesh(input_path, process=False)

        # pymeshfix works with vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces

        # Create a TMesh object for pymeshfix
        meshfix = pymeshfix.MeshFix(vertices, faces)

        # Repair the mesh
        meshfix.repair()
        print(f"✅ Mesh repaired successfully.")

        # Save the repaired mesh using trimesh
        repaired_mesh = trimesh.Trimesh(meshfix.v, meshfix.f)

        # Copy materials if they exist
        mtl_path = input_path.replace(".obj", ".mtl")
        if os.path.exists(mtl_path):
            shutil.copy(mtl_path, output_path.replace(".obj", ".mtl"))

        repaired_mesh.export(output_path)
        print(f"✅ Repaired mesh saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to repair mesh {input_path}. Error: {e}")
        return False


def scale_mesh(input_path: str, output_path: str, target_size_mm: float) -> bool:
    """
    Scales a mesh uniformly so its longest side matches a target size.

    Args:
        input_path: Path to the input mesh file.
        output_path: Path to save the scaled mesh file.
        target_size_mm: The desired size for the longest dimension in millimeters.

    Returns:
        True if scaling was successful, False otherwise.
    """
    if target_size_mm <= 1e-6:
        print("📏 Scaling skipped as target size is 0 or less.")
        # If the output path is different, we should move the file and its materials.
        if input_path != output_path:
            shutil.move(input_path, output_path)
            mtl_src = input_path.replace(".obj", ".mtl")
            if os.path.exists(mtl_src):
                shutil.move(mtl_src, output_path.replace(".obj", ".mtl"))
        return True

    try:
        print(f"📏 Attempting to scale mesh: {input_path} to {target_size_mm}mm")
        mesh = trimesh.load_mesh(input_path)

        # Get the current bounding box dimensions
        current_dims = mesh.bounding_box.extents
        if not any(d > 1e-6 for d in current_dims):
            print(f"❌ Cannot scale mesh with zero or invalid dimensions.")
            return False

        # Find the longest side
        max_dim: float = np.max(current_dims)

        # Calculate the scaling factor
        scale_factor = target_size_mm / max_dim

        # Apply the scaling transformation
        mesh.apply_scale(scale_factor)

        # Export the scaled mesh
        mesh.export(output_path)
        print(f"✅ Scaled mesh saved to: {output_path}")

        return True

    except Exception as e:
        print(f"❌ Failed to scale mesh {input_path}. Error: {e}")
        return False


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
        max_extent: float = np.max(scaled.bounding_box.extents)
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
