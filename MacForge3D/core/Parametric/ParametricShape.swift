import Foundation

/// An enumeration of the available parametric shapes.
///
/// This enum defines the types of shapes that can be generated parametrically.
/// It conforms to `CaseIterable` to allow easy iteration for UI pickers and `Identifiable` for use in SwiftUI lists.
enum ParametricShapeType: String, CaseIterable, Identifiable {
    case cube = "Cube"
    case sphere = "Sphere"
    case cylinder = "Cylinder"
    case cone = "Cone"

    /// The stable identity of the entity.
    var id: String { self.rawValue }
}

/// A container for all possible parametric shape definitions.
///
/// This enum acts as a wrapper, using associated values to store the specific parameters for each shape type.
/// This provides a clean, type-safe way to handle different shape configurations.
enum ParametricShape {
    case cube(parameters: CubeParameters)
    case sphere(parameters: SphereParameters)
    case cylinder(parameters: CylinderParameters)
    case cone(parameters: ConeParameters)

    /// The `ParametricShapeType` of the current instance.
    var type: ParametricShapeType {
        switch self {
        case .cube: return .cube
        case .sphere: return .sphere
        case .cylinder: return .cylinder
        case .cone: return .cone
        }
    }
}

// MARK: - Parameter Structs

/// Parameters for creating a Cube.
struct CubeParameters: Equatable {
    /// The length of each side of the cube.
    var size: Float = 1.0
}

/// Parameters for creating a Sphere.
struct SphereParameters: Equatable {
    /// The distance from the center to any point on the surface.
    var radius: Float = 0.5
    /// The number of horizontal and vertical segments used to approximate the sphere's surface. Higher values create a smoother sphere.
    var resolution: Int = 24
}

/// Parameters for creating a Cylinder.
struct CylinderParameters: Equatable {
    /// The radius of the circular base of the cylinder.
    var radius: Float = 0.5
    /// The height of the cylinder.
    var height: Float = 1.0
    /// The number of vertices used to form the circular base. Higher values create a smoother circle.
    var resolution: Int = 32
}

/// Parameters for creating a Cone.
struct ConeParameters: Equatable {
    /// The radius of the circular base of the cone.
    var radius: Float = 0.5
    /// The height from the base to the apex of the cone.
    var height: Float = 1.0
    /// The number of vertices used to form the circular base. Higher values create a smoother circle.
    var resolution: Int = 32
}
