import Foundation
import PythonKit
import AppKit // To get the running application's path

class TextTo3DGenerator {

    private static var isPythonInitialized = false

    private static func initializePython() {
        guard !isPythonInitialized else { return }

        // Determine the project root directory to make relative paths work
        // This is a simple approach for development builds run from Xcode.
        // A packaged app would require a different approach (e.g., bundling the venv).
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent() // /core/AI
            .deletingLastPathComponent() // /core
            .deletingLastPathComponent() // /MacForge3D
            .path

        // Set the Python library path. This is crucial for PythonKit to find the interpreter.
        // We check for both Apple Silicon and Intel Homebrew installations.
        let appleSiliconPath = "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib"
        let intelPath = "/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib"

        let pythonLibPath = FileManager.default.fileExists(atPath: appleSiliconPath) ? appleSiliconPath : intelPath

        guard FileManager.default.fileExists(atPath: pythonLibPath) else {
            print("FATAL ERROR: Could not find Python 3.11 library. Make sure it's installed via Homebrew.")
            // In a real app, you'd want to show an alert to the user.
            return
        }

        PythonLibrary.useLibrary(at: pythonLibPath)

        let sys = Python.import("sys")

        // Add the path to our Python scripts to Python's sys.path
        let scriptsPath = URL(fileURLWithPath: projectRoot).appendingPathComponent("Python/ai_models").path
        sys.path.append(scriptsPath)

        print("✅ Python initialized successfully.")
        print("🐍 Python version: \(sys.version)")
        print("🐍 Python path: \(sys.path)")

        isPythonInitialized = true
    }

    /// Asynchronously generates a 3D model from a text prompt by calling a Python script.
    ///
    /// - Parameter prompt: The text description of the model to generate.
    /// - Returns: A `String` containing the path to the generated model file, or `nil` if an error occurred.
    static func generate(prompt: String) async -> String? {
        initializePython()

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
