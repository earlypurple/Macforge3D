import Foundation
import PythonKit

class TextToMeshConverter {

    /// Asynchronously generates a 3D mesh from a text string by calling a Python script.
    ///
    /// - Parameters:
    ///   - text: The text to be converted into a 3D mesh.
    ///   - font: The name of the font to use.
    ///   - fontSize: The size of the font.
    ///   - depth: The extrusion depth of the 3D text.
    /// - Returns: An optional `String` containing the path to the generated .ply file, or `nil` on failure.
    static func generate(
        text: String,
        font: String,
        fontSize: Double,
        depth: Double
    ) async -> String? {
        // Ensure the Python environment is set up.
        PythonManager.initialize()

        // Run the Python script on a background thread to avoid blocking the UI.
        return await Task.detached(priority: .userInitiated) {
            do {
                print("🐍 Importing 'text_to_mesh' Python module...")
                let textMeshModule = Python.import("ai_models.text_to_mesh")

                // Define a unique output path for the generated file.
                let outputFileName = "\(UUID().uuidString).ply"
                let tempDirectory = FileManager.default.temporaryDirectory
                let outputPath = tempDirectory.appendingPathComponent(outputFileName).path

                print("🐍 Calling 'create_text_mesh' with text: '\(text)'")

                // Call the Python function with the specified parameters.
                let result = textMeshModule.create_text_mesh(
                    text: text,
                    font: font,
                    font_size: fontSize,
                    depth: depth,
                    output_path: outputPath
                )

                // Convert the PythonObject result to a Swift String.
                let path = String(result)

                if let path = path, path.starts(with: "Error:") {
                    print("❌ Python script returned an error: \(path)")
                    return path // Propagate the error message to the UI
                }

                print("✅ 3D Text generation script executed successfully. Result: \(path ?? "nil")")
                return path

            } catch {
                let errorMessage = "❌ 3D Text generation script failed with a Swift-level error: \(error)"
                print(errorMessage)
                return errorMessage
            }
        }.value
    }
}
