import Foundation
import PythonKit

class FigurineGenerator {

    /// (Light Version for Testing) Asynchronously generates a 3D figurine by calling the Python script.
    ///
    /// - Parameters:
    ///   - prompt: The text description of the figurine.
    ///   - quality: The desired quality level ("standard" or "ultra_detailed").
    /// - Returns: An optional `String` containing the path to the generated .ply file, or `nil` on failure.
    static func generate(prompt: String, quality: String) async -> String? {
        // Ensure Python is ready before we proceed.
        PythonManager.initialize()

        // Offload the Python call to a background thread to keep the UI responsive.
        return await Task.detached(priority: .userInitiated) {
            do {
                // --- Using the FULL version of the script ---
                print("🐍 Importing 'figurine_generator' Python module...")
                let figurineModule = Python.import("figurine_generator")
                print("🐍 Calling 'generate_figurine' with prompt: '\(prompt)' and quality: '\(quality)'")

                // Call the Python function with both parameters.
                let result = figurineModule.generate_figurine(prompt: prompt, quality: quality)

                // Convert the PythonObject result to a Swift String.
                let path = String(result)
                print("✅ Figurine generation script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                print("❌ Figurine generation script failed with error: \(error)")
                return nil
            }
        }.value
    }
}
