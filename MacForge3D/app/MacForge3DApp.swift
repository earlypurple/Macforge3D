import SwiftUI

@main
struct MacForge3DApp: App {
    // We capture the app's environment to decide whether to run normally or generate screenshots.
    private let isGeneratingScreenshots: Bool

    var body: some Scene {
        WindowGroup {
            // Only show the main window if not in screenshot mode.
            if !isGeneratingScreenshots {
                ContentView()
                    .frame(minWidth: 800, minHeight: 600)
            }
        }
        .windowStyle(DefaultWindowStyle())
        .windowToolbarStyle(UnifiedCompactWindowToolbarStyle())
        .windowTitle("MacForge3D")
    }

    init() {
        // Check for the command-line argument.
        self.isGeneratingScreenshots = CommandLine.arguments.contains("--generate-screenshots")

        if isGeneratingScreenshots {
            print("📸 Launching in screenshot generation mode...")
            // Run the generator.
            Task {
                await ScreenshotGenerator.generateAll()
                // Exit the app cleanly once screenshots are generated.
                print("✅ Screenshot generation complete. Exiting.")
                exit(0)
            }
        }
    }
}
