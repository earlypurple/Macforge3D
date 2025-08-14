import Foundation
import PythonKit

class ImageTo3DGenerator {

    /// Asynchronously generates a 3D model from a list of image files by calling a Python script.
    ///
    /// - Parameter imagePaths: An array of file system paths to the input images.
    /// - Returns: A `String` containing the path to the generated model file, or `nil` if an error occurred.
    static func generate(imagePaths: [String]) async -> String? {
        // Ensure Python is set up before calling the script.
        PythonManager.initialize()

        // Run the Python code on a background thread to avoid blocking the UI
        return await Task.detached(priority: .userInitiated) {
            do {
                print("🐍 Importing 'image_to_3d' Python module...")
                let imageTo3DModule = Python.import("ai_models.image_to_3d")
                print("🐍 Calling 'generate_3d_model_from_images' function with \(imagePaths.count) images.")

                let result = imageTo3DModule.generate_3d_model_from_images(imagePaths)

                // Convert the PythonObject result to a Swift String
                let path = String(result)
                print("✅ Python script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                print("❌ Python script execution failed with error: \(error)")
                return "Error: Python script execution failed. See console for details."
            }
        }.value
    }
}
