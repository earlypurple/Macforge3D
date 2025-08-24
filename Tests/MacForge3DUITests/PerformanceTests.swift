import XCTest

final class PerformanceTests: XCTestCase {

    var app: XCUIApplication!

    override func setUpWithError() throws {
        try super.setUpWithError()
        continueAfterFailure = false
        app = XCUIApplication()
        app.launch()
    }

    func testTextTo3DGenerationPerformance() throws {
        // 1. Navigate to the Text to 3D view
        app.buttons["Text to 3D"].click()

        // 2. Find the UI elements
        let promptEditor = app.textViews.firstMatch
        let generateButton = app.buttons["Generate Model"]
        let previewView = app.otherElements["3DPreview"]

        // 3. Define the core actions to measure
        let performanceMetrics: [XCTMetric] = [XCTApplicationLaunchMetric(), XCTClockMetric()]

        self.measure(metrics: performanceMetrics) {
            // 4. Enter a prompt
            promptEditor.click()
            promptEditor.typeText("a detailed, futuristic motorcycle")

            // 5. Click the generate button
            generateButton.click()

            // 6. Wait for the preview view to appear, which signals completion.
            // A long timeout is necessary because the AI generation can be slow.
            let previewExists = previewView.waitForExistence(timeout: 120)
            XCTAssert(previewExists, "The 3D preview did not appear within the timeout.")

            // Note: To make this test truly isolated, we would ideally clear
            // the generated model cache between runs. For now, this measures
            // performance which may or may not include caching.
        }
    }
}
