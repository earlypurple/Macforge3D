import XCTest
@testable import MacForge3D

final class TextTo3DTests: XCTestCase {
    var generator: TextTo3DGenerator!
    
    override func setUpWithError() throws {
        try super.setUpWithError()
        generator = try TextTo3DGenerator()
    }
    
    override func tearDownWithError() throws {
        generator = nil
        try super.tearDownWithError()
    }
    
    func testBasicGeneration() async throws {
        let prompt = "A simple cube"
        let result = try await generator.generate(from: prompt, style: "geometric")
        
        // Verify the file exists
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.path))
        
        // Verify file extension
        XCTAssertEqual(result.pathExtension.lowercased(), "obj")
    }
    
    func testComplexGeneration() async throws {
        let prompt = "A detailed dragon figurine"
        let result = try await generator.generate(from: prompt, style: "figurine")
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.path))
        XCTAssertEqual(result.pathExtension.lowercased(), "obj")
    }
    
    func testInvalidStyle() async {
        do {
            _ = try await generator.generate(from: "test", style: "invalid_style")
            XCTFail("Should have thrown an error for invalid style")
        } catch {
            XCTAssertTrue(true, "Successfully caught invalid style error")
        }
    }
}
