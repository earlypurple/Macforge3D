import trimesh
import pymeshfix  # type: ignore
import numpy as np
import os
import shutil


def repair_mesh(input_path: str, output_path: str) -> bool:
    """
    Repairs a mesh file to be watertight using pymeshfix.

    Args:
        input_path: Path to the input mesh file (e.g., .obj, .stl).
        output_path: Path to save the repaired mesh file.

    Returns:
        True if repair was successful, False otherwise.
    """
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
        max_dim = np.max(current_dims)

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
