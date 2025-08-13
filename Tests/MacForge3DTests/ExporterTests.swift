import XCTest
@testable import MacForge3D

class ExporterTests: XCTestCase {

    func testExportGLTF() throws {
        // 1. Create a simple mesh (a single triangle)
        let v1 = Vertex(position: SIMD3<Float>(0, 1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(0, 1))
        let v2 = Vertex(position: SIMD3<Float>(-1, -1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(-1, -1))
        let v3 = Vertex(position: SIMD3<Float>(1, -1, 0), normal: SIMD3<Float>(0, 0, 1), uv: SIMD2<Float>(1, -1))
        let triangle = Triangle(v1: v1, v2: v2, v3: v3, normal: SIMD3<Float>(0, 0, 1))
        let mesh = Mesh(vertices: [v1, v2, v3], triangles: [triangle])
        let shape = Shape3D(name: "TestTriangle", mesh: mesh, material: Material(name: "Default"))

        // 2. Get a temporary URL for the output file
        let temporaryDirectoryURL = FileManager.default.temporaryDirectory
        let fileURL = temporaryDirectoryURL.appendingPathComponent("test.gltf")

        // 3. Export the model
        let exporter = ModelExporter()
        let success = exporter.export(model: shape, to: fileURL, format: .gltf)

        // 4. Assert that the export was successful
        XCTAssertTrue(success, "GLTF export should succeed.")

        // 5. Assert that the file was created and is not empty
        var isDirectory: ObjCBool = false
        let fileExists = FileManager.default.fileExists(atPath: fileURL.path, isDirectory: &isDirectory)
        XCTAssertTrue(fileExists, "The GLTF file should exist.")
        XCTAssertFalse(isDirectory.boolValue, "The path should point to a file, not a directory.")

        let fileAttributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let fileSize = fileAttributes[FileAttributeKey.size] as? NSNumber
        XCTAssertNotNil(fileSize, "File size should be readable.")
        XCTAssertGreaterThan(fileSize?.intValue ?? 0, 0, "GLTF file should not be empty.")

        // Clean up the file
        try? FileManager.default.removeItem(at: fileURL)
    }
}
