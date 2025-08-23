import Foundation
import AVFoundation

class AudioTo3DGenerator {
    private let audioEngine: AVAudioEngine
    private let pythonModule: PythonObject
    
    init() throws {
        // Initialize audio engine
        audioEngine = AVAudioEngine()
        
        // Initialize Python environment
        guard let pythonPath = ProcessInfo.processInfo.environment["PYTHONPATH"] else {
            throw AudioTo3DError.pythonEnvironmentNotFound
        }
        
        Python.pythonPath = pythonPath
        pythonModule = try Python.import("ai_models.audio_to_3d")
    }
    
    func generate(from audioURL: URL, style: String = "organic") async throws -> URL {
        // Process audio file
        let audioFile = try AVAudioFile(forReading: audioURL)
        let format = audioFile.processingFormat
        
        // Convert audio to frequency data
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(audioFile.length))!
        try audioFile.read(into: buffer)
        
        // Extract frequency data
        let frequencyData = extractFrequencyData(from: buffer)
        
        // Generate 3D model using Python module
        let result = try await Task {
            let modelPath = pythonModule.generate_model_from_frequencies(
                frequencies: frequencyData,
                style: style
            )
            return URL(fileURLWithPath: String(modelPath)!)
        }.value
        
        return result
    }
    
    private func extractFrequencyData(from buffer: AVAudioPCMBuffer) -> [Float] {
        // Convert audio buffer to frequency domain data
        var frequencyData: [Float] = []
        
        if let floatChannelData = buffer.floatChannelData {
            let frameLength = Int(buffer.frameLength)
            let channelCount = Int(buffer.format.channelCount)
            
            for frame in 0..<frameLength {
                var sum: Float = 0
                for channel in 0..<channelCount {
                    sum += floatChannelData[channel][frame]
                }
                frequencyData.append(sum / Float(channelCount))
            }
        }
        
        return frequencyData
    }
}

enum AudioTo3DError: Error {
    case pythonEnvironmentNotFound
    case audioProcessingFailed
    case generationFailed(String)
}
