import SwiftUI
import SceneKit

/// A SwiftUI view that displays a 3D model from a file URL using SceneKit.
struct ThreeDPreviewView: View {
    // The URL of the 3D model file to display.
    let modelURL: URL
    // The color to apply to the model's materials.
    let modelColor: Color

    // The SceneKit scene that will be loaded.
    private let scene: SCNScene?

    // A flag to indicate if the model failed to load.
    private var didFailToLoad: Bool {
        scene == nil
    }

    init(modelURL: URL, modelColor: Color = .gray) {
        self.modelURL = modelURL
        self.modelColor = modelColor

        // Attempt to load the scene from the URL.
        do {
            let loadedScene = try SCNScene(url: modelURL, options: [
                .checkConsistency: true,
                .flattenScene: true,
                .createNormalsIfAbsent: true
            ])

            // Apply the color to all materials in the scene.
            let nsColor = NSColor(modelColor)
            self.applyColor(nsColor, to: loadedScene.rootNode)

            self.scene = loadedScene
        } catch {
            print("❌ Failed to load scene from URL: \(modelURL). Error: \(error)")
            self.scene = nil
        }
    }

    /// Recursively traverses the node hierarchy and applies the color to any node with geometry.
    private func applyColor(_ color: NSColor, to node: SCNNode) {
        if let geometry = node.geometry {
            geometry.materials.forEach { material in
                material.diffuse.contents = color
            }
        }

        for child in node.childNodes {
            applyColor(color, to: child)
        }
    }

    var body: some View {
        VStack {
            if let scene = scene {
                // If the scene loaded successfully, display it.
                SceneView(
                    scene: scene,
                    options: [
                        .allowsCameraControl, // Enables mouse/trackpad controls to rotate, pan, and zoom.
                        .autoenablesDefaultLighting // Adds a generic light source to the scene.
                    ]
                )
            } else {
                // If the scene failed to load, show an error message.
                ZStack {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.black.opacity(0.2))

                    VStack {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .font(.largeTitle)
                            .foregroundColor(.yellow)
                            .padding(.bottom, 8)
                        Text("Failed to Load Model")
                            .font(.headline)
                        Text(modelURL.lastPathComponent)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
        }
        .frame(minHeight: 300) // Ensure the view has a reasonable size.
    }
}

struct ThreeDPreviewView_Previews: PreviewProvider {
    static var previews: some View {
        // To make the preview work, we need a sample model file in our project.
        // The Python stub points to 'placeholder_figurine.ply'. We'll try to find it in the bundle.
        // Note: Previews run from a different context, so finding the project root is tricky.
        // This preview might only work when running the app.

        // A mock URL for a non-existent file to test the failure case.
        let failingURL = URL(fileURLWithPath: "/path/to/nonexistent.ply")

        // Let's create a dummy file for the success case preview
        let successURL = FileManager.default.temporaryDirectory.appendingPathComponent("preview.usdz")
        let scene = SCNScene()
        scene.rootNode.addChildNode(SCNNode(geometry: SCNSphere(radius: 1.0)))
        try? scene.write(to: successURL, options: nil, delegate: nil, progressHandler: nil)

        return Group {
            ThreeDPreviewView(modelURL: successURL, modelColor: .blue)
                .padding()
                .previewLayout(.fixed(width: 400, height: 400))
                .previewDisplayName("Success")

            ThreeDPreviewView(modelURL: failingURL)
                .padding()
                .previewLayout(.fixed(width: 400, height: 400))
                .previewDisplayName("Failure")
        }
    }
}
