import Foundation
import simd

// Represents a 3D vector or a point in 3D space.
struct Vector3D: Codable, Hashable {
    var x: Float
    var y: Float
    var z: Float

    static let zero = Vector3D(x: 0, y: 0, z: 0)
}

// Represents a triangle face of a mesh, defined by three vertices.
// It can also store the normal for the face.
struct Triangle: Codable, Hashable {
    var v1: Vector3D
    var v2: Vector3D
    var v3: Vector3D
    var normal: Vector3D
}

// Represents a 3D mesh composed of triangles.
// This is the fundamental geometry representation.
class Mesh: Codable {
    var triangles: [Triangle]

    init(triangles: [Triangle]) {
        self.triangles = triangles
    }

    // A static function to create a sample cube mesh for testing.
    static func createCube(size: Float) -> Mesh {
        let s = size / 2.0
        let vertices = [
            Vector3D(x: -s, y: -s, z: -s), // 0
            Vector3D(x:  s, y: -s, z: -s), // 1
            Vector3D(x:  s, y:  s, z: -s), // 2
            Vector3D(x: -s, y:  s, z: -s), // 3
            Vector3D(x: -s, y: -s, z:  s), // 4
            Vector3D(x:  s, y: -s, z:  s), // 5
            Vector3D(x:  s, y:  s, z:  s), // 6
            Vector3D(x: -s, y:  s, z:  s)  // 7
        ]

        let triangles = [
            // Front face
            Triangle(v1: vertices[0], v2: vertices[1], v3: vertices[2], normal: Vector3D(x: 0, y: 0, z: -1)),
            Triangle(v1: vertices[0], v2: vertices[2], v3: vertices[3], normal: Vector3D(x: 0, y: 0, z: -1)),
            // Back face
            Triangle(v1: vertices[4], v2: vertices[6], v3: vertices[5], normal: Vector3D(x: 0, y: 0, z: 1)),
            Triangle(v1: vertices[4], v2: vertices[7], v3: vertices[6], normal: Vector3D(x: 0, y: 0, z: 1)),
            // Left face
            Triangle(v1: vertices[4], v2: vertices[0], v3: vertices[3], normal: Vector3D(x: -1, y: 0, z: 0)),
            Triangle(v1: vertices[4], v2: vertices[3], v3: vertices[7], normal: Vector3D(x: -1, y: 0, z: 0)),
            // Right face
            Triangle(v1: vertices[1], v2: vertices[5], v3: vertices[6], normal: Vector3D(x: 1, y: 0, z: 0)),
            Triangle(v1: vertices[1], v2: vertices[6], v3: vertices[2], normal: Vector3D(x: 1, y: 0, z: 0)),
            // Top face
            Triangle(v1: vertices[3], v2: vertices[2], v3: vertices[6], normal: Vector3D(x: 0, y: 1, z: 0)),
            Triangle(v1: vertices[3], v2: vertices[6], v3: vertices[7], normal: Vector3D(x: 0, y: 1, z: 0)),
            // Bottom face
            Triangle(v1: vertices[4], v2: vertices[5], v3: vertices[1], normal: Vector3D(x: 0, y: -1, z: 0)),
            Triangle(v1: vertices[4], v2: vertices[1], v3: vertices[0], normal: Vector3D(x: 0, y: -1, z: 0))
        ]
        return Mesh(triangles: triangles)
    }
}

// The main class representing a 3D object in the scene.
// It contains the mesh and other properties like material.
class Shape3D: Identifiable, ObservableObject {
    let id = UUID()
    @Published var name: String
    @Published var mesh: Mesh
    @Published var material: Material // Assuming Material is defined elsewhere

    init(name: String, mesh: Mesh, material: Material) {
        self.name = name
        self.mesh = mesh
        self.material = material
    }
}
