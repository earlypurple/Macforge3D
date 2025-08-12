import Foundation
import PythonKit

enum TextTo3DError: Error {
    case pythonScriptNotFound
    case pythonFunctionNotFound
    case modelGenerationFailed(String)
}

class TextTo3D {

    private var textTo3DPy: PythonObject?

    init() {
        do {
            // Add the 'Python' directory to the Python path
            let sys = try Python.import("sys")
            let projectRoot = FileManager.default.currentDirectoryPath
            let pythonPath = "\(projectRoot)/Python"
            sys.path.append(pythonPath)

            // Import the python script
            self.textTo3DPy = try Python.import("ai_models.text_to_3d")
        } catch {
            print("Failed to initialize Python environment: \(error)")
            self.textTo3DPy = nil
        }
    }

    func generateModel(prompt: String, completion: @escaping (Result<String, TextTo3DError>) -> Void) {
        guard let textTo3DModule = self.textTo3DPy else {
            completion(.failure(.pythonScriptNotFound))
            return
        }

        // Run the python function in a background thread to avoid blocking the UI
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let generateFunc = textTo3DModule.generate_3d_model
                let resultPath = try generateFunc(prompt).get()

                if let path = String(resultPath) {
                    // Switch back to the main thread to return the result
                    DispatchQueue.main.async {
                        completion(.success(path))
                    }
                } else {
                    DispatchQueue.main.async {
                        completion(.failure(.modelGenerationFailed("Failed to convert result path to String")))
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    completion(.failure(.modelGenerationFailed(error.localizedDescription)))
                }
            }
        }
    }
}
