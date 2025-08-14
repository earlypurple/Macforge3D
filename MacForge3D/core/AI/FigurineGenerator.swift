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
        // The Python environment is currently non-functional.
        // The following code is commented out to allow the UI to be developed independently.
        // Once the Python environment is fixed, this code can be re-enabled.

        /*
        // Ensure Python is ready before we proceed.
        PythonManager.initialize()

        // Offload the Python call to a background thread to keep the UI responsive.
        return await Task.detached(priority: .userInitiated) {
            do {
                print("🐍 Importing 'figurine_generator' Python module...")
                let figurineModule = Python.import("figurine_generator")

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
                print("✅ Figurine generation script executed successfully. Result: \(path ?? "nil")")
                return path
            } catch {
                print("❌ Figurine generation script failed with error: \(error)")
                return "Error: Python script execution failed. Check console for details."
            }
        }.value
        */

        // Return a placeholder string for UI development and testing.
        // This can be removed once the backend is functional.
        let placeholder = "Backend call disabled. \nPrompt: '\(prompt)'\nQuality: \(quality)\nAdd Base: \(addBase)\nRefine: \(refinePetit)"
        return placeholder
    }
}
