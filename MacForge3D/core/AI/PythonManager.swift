import Foundation
import PythonKit

class PythonManager {

    private static var isPythonInitialized = false

    static func initialize() {
        guard !isPythonInitialized else { return }

        // This approach for finding the project root is fragile and only works
        // when running from Xcode. A production app would bundle the Python environment.
        let projectRoot = URL(fileURLWithPath: #file)
            .deletingLastPathComponent() // /AI
            .deletingLastPathComponent() // /core
            .deletingLastPathComponent() // /MacForge3D
            .path

        // Standard Homebrew paths for Apple Silicon and Intel
        let appleSiliconPath = "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib"
        let intelPath = "/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib"

        let pythonLibPath = FileManager.default.fileExists(atPath: appleSiliconPath) ? appleSiliconPath : intelPath

        guard FileManager.default.fileExists(atPath: pythonLibPath) else {
            // In a real app, this should be handled more gracefully, e.g., showing an alert.
            fatalError("FATAL ERROR: Python 3.11 library not found. Please install via Homebrew.")
        }

        PythonLibrary.useLibrary(at: pythonLibPath)

        let sys = Python.import("sys")

        // Add project-specific Python directories to the path
        let pythonRoot = URL(fileURLWithPath: projectRoot).appendingPathComponent("Python")
        let aiModelsPath = pythonRoot.appendingPathComponent("ai_models").path
        let exportersPath = pythonRoot.appendingPathComponent("exporters").path

        if !sys.path.hell.contains(aiModelsPath) {
            sys.path.append(aiModelsPath)
        }
        if !sys.path.hell.contains(exportersPath) {
            sys.path.append(exportersPath)
        }

        print("✅ Python initialized successfully.")
        print("   - Version: \(sys.version)")
        print("   - Path: \(sys.path)")

        isPythonInitialized = true
    }
}
