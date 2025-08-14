import Foundation

/// A factory for creating meshes of parametric shapes.
/// This enum serves as a namespace and cannot be instantiated.
enum ParametricMeshFactory {

    /// Generates a mesh for the given parametric shape.
    /// - Parameter shape: The parametric shape definition.
    /// - Returns: A new `Mesh` instance representing the shape.
    static func generate(shape: ParametricShape) -> Mesh {
        switch shape {
        case .cube(let params):
            return createCube(parameters: params)
        case .sphere(let params):
            return createSphere(parameters: params)
        case .cylinder(let params):
            return createCylinder(parameters: params)
        case .cone(let params):
            return createCone(parameters: params)
        }
    }

    // MARK: - Private Mesh Generation Functions

    /// Creates a cube mesh centered at the origin.
    private static func createCube(parameters: CubeParameters) -> Mesh {
        let s = parameters.size / 2.0
        let vertices = [
            Vector3D(x: -s, y: -s, z: -s), Vector3D(x:  s, y: -s, z: -s),
            Vector3D(x:  s, y:  s, z: -s), Vector3D(x: -s, y:  s, z: -s),
            Vector3D(x: -s, y: -s, z:  s), Vector3D(x:  s, y: -s, z:  s),
            Vector3D(x:  s, y:  s, z:  s), Vector3D(x: -s, y:  s, z:  s)
        ]

        let triangles = [
            // Front
            Triangle(v1: vertices[0], v2: vertices[1], v3: vertices[2], normal: .init(x: 0, y: 0, z: -1)),
            Triangle(v1: vertices[0], v2: vertices[2], v3: vertices[3], normal: .init(x: 0, y: 0, z: -1)),
            // Back
            Triangle(v1: vertices[4], v2: vertices[6], v3: vertices[5], normal: .init(x: 0, y: 0, z: 1)),
            Triangle(v1: vertices[4], v2: vertices[7], v3: vertices[6], normal: .init(x: 0, y: 0, z: 1)),
            // Left
            Triangle(v1: vertices[4], v2: vertices[0], v3: vertices[3], normal: .init(x: -1, y: 0, z: 0)),
            Triangle(v1: vertices[4], v2: vertices[3], v3: vertices[7], normal: .init(x: -1, y: 0, z: 0)),
            // Right
            Triangle(v1: vertices[1], v2: vertices[5], v3: vertices[6], normal: .init(x: 1, y: 0, z: 0)),
            Triangle(v1: vertices[1], v2: vertices[6], v3: vertices[2], normal: .init(x: 1, y: 0, z: 0)),
            // Top
            Triangle(v1: vertices[3], v2: vertices[2], v3: vertices[6], normal: .init(x: 0, y: 1, z: 0)),
            Triangle(v1: vertices[3], v2: vertices[6], v3: vertices[7], normal: .init(x: 0, y: 1, z: 0)),
            // Bottom
            Triangle(v1: vertices[4], v2: vertices[5], v3: vertices[1], normal: .init(x: 0, y: -1, z: 0)),
            Triangle(v1: vertices[4], v2: vertices[1], v3: vertices[0], normal: .init(x: 0, y: -1, z: 0))
        ]
        return Mesh(triangles: triangles)
    }

    /// Creates a UV sphere mesh centered at the origin.
    private static func createSphere(parameters: SphereParameters) -> Mesh {
        var triangles = [Triangle]()
        let segments = parameters.resolution
        let rings = parameters.resolution

        for j in 0..<rings {
            for i in 0..<segments {
                let r1 = Float(j) / Float(rings)
                let r2 = Float(j + 1) / Float(rings)
                let t1 = Float(i) / Float(segments)
                let t2 = Float(i + 1) / Float(segments)

                let p1 = pointOnSphere(radius: parameters.radius, u: t1, v: r1)
                let p2 = pointOnSphere(radius: parameters.radius, u: t2, v: r1)
                let p3 = pointOnSphere(radius: parameters.radius, u: t1, v: r2)
                let p4 = pointOnSphere(radius: parameters.radius, u: t2, v: r2)

                // For a perfect sphere, the normal is the same as the vertex position normalized.
                let n1 = Vector3D(x: p1.x, y: p1.y, z: p1.z)
                let n3 = Vector3D(x: p3.x, y: p3.y, z: p3.z)

                triangles.append(Triangle(v1: p1, v2: p2, v3: p3, normal: n1))
                triangles.append(Triangle(v1: p3, v2: p2, v3: p4, normal: n3))
            }
        }
        return Mesh(triangles: triangles)
    }

    /// Helper function to calculate a point on a sphere's surface from UV coordinates.
    private static func pointOnSphere(radius: Float, u: Float, v: Float) -> Vector3D {
        let theta = u * 2 * .pi
        let phi = v * .pi
        let x = radius * sin(phi) * cos(theta)
        let y = radius * cos(phi)
        let z = radius * sin(phi) * sin(theta)
        return Vector3D(x: x, y: y, z: z)
    }

    /// Creates a cylinder mesh centered at the origin, aligned with the Y-axis.
    private static func createCylinder(parameters: CylinderParameters) -> Mesh {
        var triangles = [Triangle]()
        let n = parameters.resolution
        let radius = parameters.radius
        let height = parameters.height
        let halfHeight = height / 2.0

        let topCenter = Vector3D(x: 0, y: halfHeight, z: 0)
        let bottomCenter = Vector3D(x: 0, y: -halfHeight, z: 0)

        for i in 0..<n {
            let angle1 = 2 * .pi * Float(i) / Float(n)
            let angle2 = 2 * .pi * Float(i + 1) / Float(n)

            let x1 = radius * cos(angle1)
            let z1 = radius * sin(angle1)
            let x2 = radius * cos(angle2)
            let z2 = radius * sin(angle2)

            let p1_top = Vector3D(x: x1, y: halfHeight, z: z1)
            let p2_top = Vector3D(x: x2, y: halfHeight, z: z2)
            let p1_bottom = Vector3D(x: x1, y: -halfHeight, z: z1)
            let p2_bottom = Vector3D(x: x2, y: -halfHeight, z: z2)

            // Create the side wall of the cylinder segment.
            let sideNormal = Vector3D(x: cos(angle1 + .pi / Float(n)), y: 0, z: sin(angle1 + .pi / Float(n)))
            triangles.append(Triangle(v1: p1_bottom, v2: p2_top, v3: p1_top, normal: sideNormal))
            triangles.append(Triangle(v1: p1_bottom, v2: p2_bottom, v3: p2_top, normal: sideNormal))

            // Create the top cap triangle for this segment.
            triangles.append(Triangle(v1: topCenter, v2: p2_top, v3: p1_top, normal: .init(x: 0, y: 1, z: 0)))
            // Create the bottom cap triangle for this segment.
            triangles.append(Triangle(v1: bottomCenter, v2: p1_bottom, v3: p2_bottom, normal: .init(x: 0, y: -1, z: 0)))
        }
        return Mesh(triangles: triangles)
    }

    /// Creates a cone mesh centered at the origin, aligned with the Y-axis.
    private static func createCone(parameters: ConeParameters) -> Mesh {
        var triangles = [Triangle]()
        let n = parameters.resolution
        let radius = parameters.radius
        let height = parameters.height

        let apex = Vector3D(x: 0, y: height / 2.0, z: 0)
        let baseCenter = Vector3D(x: 0, y: -height / 2.0, z: 0)

        for i in 0..<n {
            let angle1 = 2 * .pi * Float(i) / Float(n)
            let angle2 = 2 * .pi * Float(i + 1) / Float(n)

            let x1 = radius * cos(angle1)
            let z1 = radius * sin(angle1)
            let x2 = radius * cos(angle2)
            let z2 = radius * sin(angle2)

            let p1_base = Vector3D(x: x1, y: -height / 2.0, z: z1)
            let p2_base = Vector3D(x: x2, y: -height / 2.0, z: z2)

            // Create the sloping side surface for this segment.
            // The normal is calculated as the cross product of two edges of the triangle,
            // but a simplified approximation is used here for brevity.
            let sideNormal = Vector3D(x: cos(angle1 + .pi / Float(n)), y: radius/height, z: sin(angle1 + .pi / Float(n)))
            triangles.append(Triangle(v1: apex, v2: p2_base, v3: p1_base, normal: sideNormal))

            // Create the bottom cap triangle for this segment.
            triangles.append(Triangle(v1: baseCenter, v2: p1_base, v3: p2_base, normal: .init(x: 0, y: -1, z: 0)))
        }
        return Mesh(triangles: triangles)
    }
}
