import SwiftUI
import SceneKit

struct ThreeDPreviewView: View {
    var filePath: String?

    var body: some View {
        if let path = filePath {
            // Check if the file exists before trying to load it
            if FileManager.default.fileExists(atPath: path) {
                SceneKitView(scene: loadScene(from: path))
            } else {
                Text("Error: Model file not found at \(path)")
                    .foregroundColor(.red)
            }
        } else {
            // If no file path is provided, show a placeholder view
            Text("3D Preview")
                .foregroundColor(.gray)
        }
    }

    private func loadScene(from path: String) -> SCNScene? {
        do {
            let url = URL(fileURLWithPath: path)
            // SCNScene can load .ply files directly
            let scene = try SCNScene(url: url, options: nil)

            // Add a camera to the scene
            let cameraNode = SCNNode()
            cameraNode.camera = SCNCamera()
            cameraNode.position = SCNVector3(x: 0, y: 0, z: 5) // Position the camera
            scene.rootNode.addChildNode(cameraNode)

            // Add lighting to the scene
            let lightNode = SCNNode()
            lightNode.light = SCNLight()
            lightNode.light!.type = .omni
            lightNode.position = SCNVector3(x: 10, y: 10, z: 10)
            scene.rootNode.addChildNode(lightNode)

            let ambientLightNode = SCNNode()
            ambientLightNode.light = SCNLight()
            ambientLightNode.light!.type = .ambient
            ambientLightNode.light!.color = UIColor.darkGray
            scene.rootNode.addChildNode(ambientLightNode)

            return scene
        } catch {
            print("Failed to load scene: \(error)")
            return nil
        }
    }
}

struct SceneKitView: UIViewRepresentable {
    let scene: SCNScene?

    func makeUIView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.allowsCameraControl = true
        scnView.autoenablesDefaultLighting = true
        scnView.backgroundColor = UIColor.lightGray
        return scnView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        uiView.scene = scene
    }
}
