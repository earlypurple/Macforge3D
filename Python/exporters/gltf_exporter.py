import json
import sys
import numpy as np
import pygltflib  # type: ignore


def create_indexed_mesh(triangles):
    """Converts a list of triangles to an indexed mesh."""
    vertices = []
    indices = []
    vertex_map = {}

    for triangle in triangles:
        for vertex_data in [triangle["v1"], triangle["v2"], triangle["v3"]]:
            pos = tuple(vertex_data["position"])
            if pos not in vertex_map:
                vertex_map[pos] = len(vertices)
                vertices.append(vertex_data)
            indices.append(vertex_map[pos])

    return vertices, indices


def export_to_gltf(mesh_data, output_path):
    """Exports mesh data from a dictionary to a GLTF 2.0 file."""
    triangles = mesh_data.get("triangles", [])
    if not triangles:
        print("No triangles found in mesh data.", file=sys.stderr)
        return

    vertices, indices = create_indexed_mesh(triangles)

    positions = np.array([v["position"] for v in vertices], dtype=np.float32)
    normals = np.array([v["normal"] for v in vertices], dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)

    # Convert numpy arrays to bytes
    positions_blob = positions.tobytes()
    normals_blob = normals.tobytes()
    indices_blob = indices.tobytes()

    gltf = pygltflib.GLTF2()

    # Scene
    gltf.scenes.append(pygltflib.Scene(nodes=[0]))
    gltf.scene = 0

    # Nodes
    gltf.nodes.append(pygltflib.Node(mesh=0))

    # Mesh
    gltf.meshes.append(
        pygltflib.Mesh(
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=0, NORMAL=1), indices=2
                )
            ]
        )
    )

    # Accessors
    gltf.accessors.extend(
        [
            pygltflib.Accessor(
                bufferView=0,
                componentType=pygltflib.FLOAT,
                count=len(positions),
                type=pygltflib.VEC3,
                min=positions.min(axis=0).tolist(),
                max=positions.max(axis=0).tolist(),
            ),
            pygltflib.Accessor(
                bufferView=1,
                componentType=pygltflib.FLOAT,
                count=len(normals),
                type=pygltflib.VEC3,
            ),
            pygltflib.Accessor(
                bufferView=2,
                componentType=pygltflib.UNSIGNED_INT,
                count=len(indices),
                type=pygltflib.SCALAR,
            ),
        ]
    )

    # BufferViews
    gltf.bufferViews.extend(
        [
            pygltflib.BufferView(
                buffer=0,
                byteOffset=0,
                byteLength=len(positions_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(positions_blob),
                byteLength=len(normals_blob),
                target=pygltflib.ARRAY_BUFFER,
            ),
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(positions_blob) + len(normals_blob),
                byteLength=len(indices_blob),
                target=pygltflib.ELEMENT_ARRAY_BUFFER,
            ),
        ]
    )

    # Buffer
    gltf.buffers.append(
        pygltflib.Buffer(
            byteLength=len(positions_blob) + len(normals_blob) + len(indices_blob)
        )
    )
    gltf.set_binary_blob(positions_blob + normals_blob + indices_blob)

    # Save to file
    gltf.save(output_path)


# This script is intended to be imported as a module.
