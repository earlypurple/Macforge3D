import Foundation
import PythonKit

// Define a simple struct to decode the JSON response from Python
struct GenerationResult: Decodable {
    let status: String
    let path: String?
    let message: String?
}

class TextTo3DGenerator {

    /// Asynchronously generates a 3D model from a text prompt by calling a Python script.
    ///
    /// - Parameter prompt: The text description of the model to generate.
    /// - Parameter quality: The desired quality of the generated model.
    /// - Returns: A `Result` containing a `URL` to the generated model file, or an `Error`.
    static func generate(prompt: String, quality: String) async -> Result<URL, Error> {
        PythonManager.initialize()

        return await Task.detached(priority: .userInitiated) {
            do {
                let textTo3DModule = Python.import("ai_models.text_to_3d")
                let resultJSON = String(textTo3DModule.generate_3d_model(prompt, quality))

                guard let data = resultJSON?.data(using: .utf8) else {
                    return .failure(TextTo3DError.invalidResponse)
                }

                let decoder = JSONDecoder()
                let result = try decoder.decode(GenerationResult.self, from: data)

                if result.status == "success", let path = result.path {
                    print("✅ Generation successful. Path: \(path)")
                    return .success(URL(fileURLWithPath: path))
                } else if let message = result.message {
                    print("❌ Generation failed. Reason: \(message)")
                    return .failure(TextTo3DError.generationFailed(message: message))
                } else {
                    return .failure(TextTo3DError.invalidResponse)
                }
            } catch {
                print("❌ Python script execution failed with error: \(error)")
                return .failure(error)
            }
        }.value
    }
}
