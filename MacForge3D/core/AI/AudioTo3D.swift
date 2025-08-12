import Foundation
import PythonKit

enum AudioTo3DError: Error {
    case pythonScriptNotFound
    case pythonFunctionNotFound
    case modelGenerationFailed(String)
}

class AudioTo3D {

    private var audioTo3DPy: PythonObject?

    init() {
        do {
            // The Python path is already configured by the TextTo3D class,
            // but we can ensure it's there in case this is initialized first.
            let sys = try Python.import("sys")
            let projectRoot = FileManager.default.currentDirectoryPath
            let pythonPath = "\(projectRoot)/Python"
            if !(sys.path.object.contains(pythonPath)) {
                 sys.path.append(pythonPath)
            }

            // Import the python script
            self.audioTo3DPy = try Python.import("ai_models.audio_to_3d")
        } catch {
            print("Failed to initialize Python environment for AudioTo3D: \(error)")
            self.audioTo3DPy = nil
        }
    }

    func generateModel(from audioPath: String, completion: @escaping (Result<String, AudioTo3DError>) -> Void) {
        guard let audioTo3DModule = self.audioTo3DPy else {
            completion(.failure(.pythonScriptNotFound))
            return
        }

        // Run the python function in a background thread
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let generateFunc = audioTo3DModule.generate_3d_from_audio
                let resultPath = try generateFunc(audioPath).get()

                if let path = String(resultPath) {
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
