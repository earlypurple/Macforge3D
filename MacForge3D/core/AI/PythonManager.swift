import Foundation
import PythonKit

/// Manages the Python environment initialization and configuration.
enum PythonManager {

    private static var isPythonInitialized = false

    /// Initializes the Python environment.
    /// This function finds the Python library, sets up the Python path, and ensures
    /// that the necessary modules can be imported.
    static func initialize() {
        guard !isPythonInitialized else { return }

        do {
            // Step 1: Locate the Python library dynamically.
            let pythonLibPath = try findPythonLibrary()
            PythonLibrary.useLibrary(at: pythonLibPath)

            // Step 2: Locate the project root to find our Python source files.
            guard let projectRoot = getProjectRoot() else {
                fatalError("FATAL ERROR: Could not find the project root directory.")
            }

            // Step 3: Add project-specific Python directories to the Python path.
            let sys = Python.import("sys")
            let pythonRoot = projectRoot.appendingPathComponent("Python")

            // We append the root of the Python source tree, so imports should be relative to that.
            // e.g., `ai_models.image_to_3d`
            sys.path.append(pythonRoot.path)

            print("✅ Python initialized successfully.")
            print("   - Version: \(sys.version)")
            print("   - Path: \(sys.path)")

            isPythonInitialized = true

        } catch {
            // In a real app, this should be handled more gracefully, e.g., showing an alert.
            fatalError("FATAL ERROR: Python initialization failed. \(error)")
        }
    }

    /// Finds the path to the Python 3.11 dylib installed via Homebrew.
    /// This is more robust than hardcoding paths as it checks multiple common locations.
    ///
    /// - Throws: `PythonError.libraryNotFound` if the library cannot be located.
    /// - Returns: The path to the Python dynamic library.
    private static func findPythonLibrary() throws -> String {
        let potentialPaths = [
            "/opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib", // Apple Silicon
            "/usr/local/opt/python@3.11/Frameworks/Python.framework/Versions/3.11/lib/libpython3.11.dylib", // Intel Mac
            "/home/linuxbrew/.linuxbrew/opt/python@3.11/lib/libpython3.11.so" // Linux (for testing)
        ]

        for path in potentialPaths {
            if FileManager.default.fileExists(atPath: path) {
                print("🐍 Found Python library at: \(path)")
                return path
            }
        }

        throw PythonError.libraryNotFound
    }

    /// Determines the project's root directory.
    ///
    /// In a DEBUG environment, it searches upwards from the current file to find a directory
    /// containing a marker file (`Package.swift`). This is robust for development.
    ///
    /// In a RELEASE environment, it assumes the Python environment is bundled within
    /// the app's `Resources` directory, which is standard for production apps.
    ///
    /// - Returns: A URL pointing to the project root, or `nil` if it cannot be determined.
    private static func getProjectRoot() -> URL? {
        #if DEBUG
        // In DEBUG mode, search upwards for a root marker file.
        var currentURL = URL(fileURLWithPath: #file)

        // Search up to 10 levels deep to prevent infinite loops.
        for _ in 0..<10 {
            let markerURL = currentURL.appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: markerURL.path) {
                return currentURL
            }
            // Move to the parent directory.
            currentURL.deleteLastPathComponent()
        }
        return nil // Root marker not found.
        #else
        // In RELEASE mode, assume Python source is in the app's resource bundle.
        // The Python folder should be copied into the app bundle via a build phase.
        return Bundle.main.resourceURL
        #endif
    }

    /// Custom errors for Python-related failures.
    enum PythonError: Error, LocalizedError {
        case libraryNotFound

        var errorDescription: String? {
            switch self {
            case .libraryNotFound:
                return "Python 3.11 library not found. Please ensure it is installed via Homebrew."
            }
        }
    }
}
