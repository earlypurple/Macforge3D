import trimesh
import numpy as np

def subtract(base_model, subtraction_model, output_path: str) -> str:
    """
    Subtracts one mesh from another using the manifold3d library directly
    and saves the result.

    Args:
        base_model (str or trimesh.Trimesh): The base mesh.
        subtraction_model (str or trimesh.Trimesh): The mesh to be subtracted.
        output_path (str): The path to save the resulting mesh file.

    Returns:
        str: The path to the generated mesh file, or an error message.
    """
    try:
        import manifold3d
    except ImportError:
        return "Error: manifold3d library not found. Please ensure it is installed."

    try:
        if isinstance(base_model, trimesh.Trimesh):
            base_trimesh = base_model
        else:
            base_trimesh = trimesh.load_mesh(base_model)

        if isinstance(subtraction_model, trimesh.Trimesh):
            subtraction_trimesh = subtraction_model
        else:
            subtraction_trimesh = trimesh.load_mesh(subtraction_model)

        def trimesh_to_manifold(mesh: trimesh.Trimesh) -> manifold3d.Manifold:
            verts = np.array(mesh.vertices, dtype=np.float32)
            tris = np.array(mesh.faces, dtype=np.uint32)
            manifold_mesh = manifold3d.Mesh(vert_properties=verts, tri_verts=tris)
            return manifold3d.Manifold(manifold_mesh)

        base_manifold = trimesh_to_manifold(base_trimesh)
        subtraction_manifold = trimesh_to_manifold(subtraction_trimesh)

        result_manifold = base_manifold - subtraction_manifold

        result_manifold_mesh = result_manifold.to_mesh()

        result_vertices = np.array(result_manifold_mesh.vert_properties[:, :3], dtype=np.float64)
        result_faces = np.array(result_manifold_mesh.tri_verts, dtype=np.int64)

        if len(result_vertices) == 0 or len(result_faces) == 0:
            return "Error: The boolean subtraction resulted in an empty mesh."

        result_trimesh = trimesh.Trimesh(vertices=result_vertices, faces=result_faces)

        result_trimesh.export(output_path)

        return output_path

    except Exception as e:
        error_message = f"Error during boolean subtraction: {e}"
        import traceback
        traceback.print_exc()
        return error_message

if __name__ == '__main__':
    base = trimesh.creation.box(extents=[10, 10, 10])
    sub = trimesh.creation.icosphere(radius=6)

    print("--- Testing mesh subtraction script with direct manifold3d usage ---")
    path_or_error = subtract(
        base_model=base,
        subtraction_model=sub,
        output_path='/tmp/subtracted_direct.ply'
    )
    print(f"--- Script finished. Result: {path_or_error} ---")
