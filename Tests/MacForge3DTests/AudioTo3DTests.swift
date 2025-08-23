import XCTest
@testable import MacForge3D

final class AudioTo3DTests: XCTestCase {
    var generator: AudioTo3DGenerator!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        generator = try AudioTo3DGenerator()
    }
    
    override func tearDownWithError() throws {
        generator = nil
        try super.tearDownWithError()
    }
    
    func testBasicAudioGeneration() async throws {
        // Create a test audio file
        let audioURL = try createTestAudioFile()
        
        let result = try await generator.generate(from: audioURL, style: "organic")
        
        // Verify the file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.path))
        
        // Verify file extension
        XCTAssertEqual(result.pathExtension.lowercased(), "obj")
    }
    
    func testInvalidAudioFile() async {
        let invalidURL = URL(fileURLWithPath: "/nonexistent/audio.wav")
        
        do {
            _ = try await generator.generate(from: invalidURL)
            XCTFail("Should have thrown an error for invalid audio file")
        } catch {
            XCTAssertTrue(true, "Successfully caught invalid file error")
        }
    }
    
    // Helper function to create a test audio file
    private func createTestAudioFile() throws -> URL {
        let documentsPath = FileManager.default.temporaryDirectory
        let audioURL = documentsPath.appendingPathComponent("test_audio.wav")
        
        // Create a simple sine wave audio file
        let engine = AVAudioEngine()
        let mainMixer = engine.mainMixerNode
        let output = engine.outputNode
        let format = output.inputFormat(forBus: 0)
        
        // Create audio file
        let audioFile = try AVAudioFile(
            forWriting: audioURL,
            settings: format.settings
        )
        
        // Create a buffer with a sine wave
        let sampleRate = Float(format.sampleRate)
        let duration = 1.0 // 1 second
        let frequency = 440.0 // A4 note
        let frameCount = AVAudioFrameCount(duration * Double(sampleRate))
        
        let buffer = AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: frameCount
        )!
        
        let channelData = buffer.floatChannelData!
        buffer.frameLength = frameCount
        
        // Fill buffer with sine wave
        for frame in 0..<Int(frameCount) {
            let value = sin(2.0 * Float.pi * Float(frame) * Float(frequency) / sampleRate)
            channelData[0][frame] = value
        }
        
        try audioFile.write(from: buffer)
        
        return audioURL
    }
}
