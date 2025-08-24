import Foundation
import PythonKit

class ImageTo3DGenerator {

    /// Asynchronously generates a 3D model from a list of image files by calling a Python script.
    ///
    /// - Parameters:
    ///   - imagePaths: An array of file system paths to the input images.
    ///   - quality: The desired quality level ("Draft", "Default", "High").
    ///   - repairMesh: A boolean indicating whether to run mesh repair post-processing.
    ///   - targetSize: The target size in mm for the longest axis of the model. 0 means no scaling.
    /// - Returns: A `String` containing the path to the generated model file, or an error message.
    static func generate(imagePaths: [String], quality: String, repairMesh: Bool, targetSize: Float) async -> String? {
        // Ensure Python is set up before calling the script.
        PythonManager.initialize()

        // Run the Python code on a background thread to avoid blocking the UI
        return await Task.detached(priority: .userInitiated) {
            print("[PROFILING] Starting Python call: ImageTo3D.generate_3d_model_from_images")
            let startTime = Date()
            do {
                print("🐍 Importing 'image_to_3d' Python module...")
                let imageTo3DModule = Python.import("ai_models.image_to_3d")
                print("🐍 Calling 'generate_3d_model_from_images' with options: quality='\(quality)', repair=\(repairMesh), size=\(targetSize)mm")

                let result = imageTo3DModule.generate_3d_model_from_images(
                    image_paths: imagePaths,
                    quality: quality,
                    should_repair: repairMesh,
                    target_size_mm: targetSize
                )

                // Convert the PythonObject result to a Swift String
                let path = String(result)
                let timeElapsed = Date().timeIntervalSince(startTime)
                print("[PROFILING] Python call finished successfully. Time elapsed: \(String(format: "%.4f", timeElapsed)) seconds.")
                print("✅ Python script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                let timeElapsed = Date().timeIntervalSince(startTime)
                print("[PROFILING] Python call failed. Time elapsed: \(String(format: "%.4f", timeElapsed)) seconds.")
                print("❌ Python script execution failed with error: \(error)")
                // It's better to return the actual Python error for debugging.
                return "Error: Python script failed. \(error)"
            }
        }.value
    }
}
