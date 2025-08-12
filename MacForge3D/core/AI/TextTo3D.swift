import Foundation
import PythonKit

class TextTo3DGenerator {

    /// Asynchronously generates a 3D model from a text prompt by calling a Python script.
    ///
    /// - Parameter prompt: The text description of the model to generate.
    /// - Returns: A `String` containing the path to the generated model file, or `nil` if an error occurred.
    static func generate(prompt: String) async -> String? {
        // Ensure Python is set up before calling the script.
        PythonManager.initialize()

        // Run the Python code on a background thread to avoid blocking the UI
        return await Task.detached(priority: .userInitiated) {
            do {
                print("🐍 Importing 'text_to_3d' Python module...")
                let textTo3DModule = Python.import("text_to_3d")
                print("🐍 Calling 'generate_3d_model' function with prompt: '\(prompt)'")

                let result = textTo3DModule.generate_3d_model(prompt)

                // Convert the PythonObject result to a Swift String
                let path = String(result)
                print("✅ Python script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                print("❌ Python script execution failed with error: \(error)")
                return nil
            }
        }.value
    }
}
