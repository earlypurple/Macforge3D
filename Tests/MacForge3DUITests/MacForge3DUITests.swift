import XCTest

final class MacForge3DUITests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.

        // In UI tests it is usually best to stop immediately when a failure occurs.
        continueAfterFailure = false

        // In UI tests it’s important to set the initial state - such as interface orientation - required for your tests before they run. The setUp method is a good place to do this.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testAppLaunch() throws {
        // UI tests must launch the application that they test.
        let app = XCUIApplication()
        app.launch()

        // Assert that the main window exists and has the correct title.
        let window = app.windows.firstMatch
        XCTAssert(window.exists)
        XCTAssertEqual(window.title, "MacForge3D")
    }

    func testSidebarNavigation() throws {
        let app = XCUIApplication()
        app.launch()

        // Check that the "Figurine Generator" is selected by default
        let figurineGeneratorButton = app.buttons["Figurine Generator"]
        XCTAssert(figurineGeneratorButton.isSelected)

        // Click on the "Text to 3D" button
        let textTo3DButton = app.buttons["Text to 3D"]
        textTo3DButton.click()
        XCTAssert(textTo3DButton.isSelected)

        // Click on the "Audio to 3D" button
        let audioTo3DButton = app.buttons["Audio to 3D"]
        audioTo3DButton.click()
        XCTAssert(audioTo3DButton.isSelected)
    }
}
