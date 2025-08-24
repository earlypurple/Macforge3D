import Foundation
import PythonKit

class TextTo3DGenerator {
    private let pythonModule: PythonObject
    
    init() throws {
        // Initialize Python environment
        guard let pythonPath = ProcessInfo.processInfo.environment["PYTHONPATH"] else {
            throw AIError.pythonEnvironmentNotFound
        }
        
        // Set Python path
        Python.pythonPath = pythonPath
        
        // Import our custom Python module
        pythonModule = try Python.import("ai_models.text_to_3d")
    }
    
    func generate(from prompt: String, style: String = "standard") async throws -> URL {
        print("[PROFILING] Starting Python call: TextTo3DGenerator.generate_model")
        let startTime = Date()

        // Call Python function to generate 3D model
        let result = try await Task {
            let modelPath = pythonModule.generate_model(prompt: prompt, style: style)
            return URL(fileURLWithPath: String(modelPath)!)
        }.value

        let timeElapsed = Date().timeIntervalSince(startTime)
        print("[PROFILING] Python call finished. Time elapsed: \(String(format: "%.4f", timeElapsed)) seconds.")
        
        return result
    }
}

enum AIError: Error {
    case pythonEnvironmentNotFound
    case generationFailed(String)
}
