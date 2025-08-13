import Foundation
import PythonKit

class MeshOperator {

    /// Asynchronously subtracts one mesh from another by calling a Python script.
    ///
    /// - Parameters:
    ///   - baseModelURL: The URL of the base mesh.
    ///   - subtractionModelURL: The URL of the mesh to subtract.
    /// - Returns: An optional `String` containing the path to the resulting .ply file, or `nil` on failure.
    static func subtract(baseModelURL: URL, subtractionModelURL: URL) async -> String? {
        // Ensure the Python environment is set up.
        PythonManager.initialize()

        return await Task.detached(priority: .userInitiated) {
            do {
                print("🐍 Importing 'mesh_operations' Python module...")
                let meshOpsModule = Python.import("ai_models.mesh_operations")

                // Define a unique output path for the generated file.
                let outputFileName = "engraved_\(UUID().uuidString).ply"
                let tempDirectory = FileManager.default.temporaryDirectory
                let outputPath = tempDirectory.appendingPathComponent(outputFileName).path

                print("🐍 Calling 'subtract' with base: '\(baseModelURL.path)' and subtraction: '\(subtractionModelURL.path)'")

                // Call the Python function with the file paths.
                let result = meshOpsModule.subtract(
                    base_model=baseModelURL.path,
                    subtraction_model=subtractionModelURL.path,
                    output_path: outputPath
                )

                let path = String(result)

                if let path = path, path.starts(with: "Error:") {
                    print("❌ Python script returned an error: \(path)")
                    return path
                }

                print("✅ Mesh subtraction script executed successfully. Result: \(path ?? "nil")")
                return path

            } catch {
                let errorMessage = "❌ Mesh subtraction script failed with a Swift-level error: \(error)"
                print(errorMessage)
                return errorMessage
            }
        }.value
    }
}
