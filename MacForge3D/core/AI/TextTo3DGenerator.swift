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
        // Call Python function to generate 3D model
        let result = try await Task {
            let modelPath = pythonModule.generate_model(prompt: prompt, style: style)
            return URL(fileURLWithPath: String(modelPath)!)
        }.value
        
        return result
    }
}

enum AIError: Error {
    case pythonEnvironmentNotFound
    case generationFailed(String)
}
