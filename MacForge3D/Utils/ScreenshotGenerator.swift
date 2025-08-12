import SwiftUI

/// A utility to programmatically generate screenshots of SwiftUI views.
@MainActor
class ScreenshotGenerator {

    /// Generates and saves screenshots for all specified views.
    static func generateAll() async {
        print("📸 Starting screenshot generation...")

        // --- Generate main_workspace.png ---
        print("› Generating 'main_workspace.png' from ContentView...")
        await generateScreenshot(
            for: ContentView(),
            as: "main_workspace.png",
            width: 1280,
            height: 800
        )

        // --- Placeholder for text_to_3d.png ---
        // Generating this screenshot requires getting the UI into a specific state
        // after a model has been generated, which is complex to do programmatically
        // without significant refactoring of the views.
        // For now, we will skip this one and inform the user.
        print("› Skipping 'text_to_3d.png'. This requires manual generation for now.")


        print("✅ Screenshot generation process finished.")
    }

    /// Renders a given SwiftUI view and saves it as a PNG image.
    ///
    /// - Parameters:
    ///   - view: The SwiftUI view to render.
    ///   - filename: The name of the output PNG file.
    ///   - width: The width of the output image.
    ///   - height: The height of the output image.
    private static func generateScreenshot<V: View>(for view: V, as filename: String, width: CGFloat, height: CGFloat) async {
        let outputDirectory = URL(fileURLWithPath: "Documentation/screenshots", isDirectory: true)

        // Ensure the directory exists
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
        } catch {
            print("❌ Failed to create screenshot directory: \(error)")
            return
        }

        let url = outputDirectory.appendingPathComponent(filename)

        // Use ImageRenderer to get a CGImage from the view.
        let renderer = ImageRenderer(content: view.frame(width: width, height: height))
        renderer.scale = 2.0 // Render at 2x for Retina quality

        guard let cgImage = renderer.cgImage else {
            print("❌ Failed to render CGImage for \(filename).")
            return
        }

        // Use NSBitmapImageRep to convert the CGImage to PNG data.
        let bitmap = NSBitmapImageRep(cgImage: cgImage)
        bitmap.size = NSSize(width: width, height: height) // Set the bitmap size

        guard let data = bitmap.representation(using: .png, properties: [:]) else {
            print("❌ Failed to create PNG data for \(filename).")
            return
        }

        // Write the data to the file.
        do {
            try data.write(to: url)
            print("  ✅ Screenshot saved successfully to: \(url.path)")
        } catch {
            print("❌ Failed to write screenshot to disk for \(filename): \(error)")
        }
    }
}
