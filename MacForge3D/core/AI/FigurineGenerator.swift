import Foundation
import PythonKit

class FigurineGenerator {

    /// Asynchronously generates a 3D figurine by calling the Python script.
    ///
    /// - Parameters:
    ///   - prompt: The text description of the figurine.
    ///   - quality: The desired quality level ("petit", "standard", "detailed", "ultra_realistic").
    ///   - addBase: Whether to add a base to the figurine.
    ///   - refinePetit: Whether to apply mesh refinement to the "petit" quality model.
    /// - Returns: An optional `String` containing the path to the generated .ply file, or a descriptive string for UI testing.
    static func generate(prompt: String, quality: String, addBase: Bool, refinePetit: Bool) async -> String? {
        // Ensure Python is ready before we proceed.
        PythonManager.initialize()

        // Offload the Python call to a background thread to keep the UI responsive.
        return await Task.detached(priority: .userInitiated) {
            print("[PROFILING] Starting Python call: FigurineGenerator.generate_figurine")
            let startTime = Date()
            do {
                print("🐍 Importing 'figurine_generator' Python module...")
                let figurineModule = Python.import("ai_models.figurine_generator")

                print("🐍 Calling 'generate_figurine' with prompt: '\(prompt)', quality: '\(quality)', add_base: \(addBase), refine_petit: \(refinePetit)")

                // Call the Python function with all parameters.
                let result = figurineModule.generate_figurine(
                    prompt: prompt,
                    quality: quality,
                    add_base: addBase,
                    refine_petit: refinePetit
                )

                // Convert the PythonObject result to a Swift String.
                let path = String(result)
                let timeElapsed = Date().timeIntervalSince(startTime)
                print("[PROFILING] Python call finished successfully. Time elapsed: \(String(format: "%.4f", timeElapsed)) seconds.")
                print("✅ Figurine generation script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                let timeElapsed = Date().timeIntervalSince(startTime)
                print("[PROFILING] Python call failed. Time elapsed: \(String(format: "%.4f", timeElapsed)) seconds.")
                print("❌ Figurine generation script failed with error: \(error)")
                return "Error: Python script execution failed. Check console for details."
            }
        }.value
    }
}
