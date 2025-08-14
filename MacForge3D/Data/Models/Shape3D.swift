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
