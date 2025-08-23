import SwiftUI
import SceneKit
import PythonKit

struct AnimationPreview: View {
    let scene = SCNScene()
    @State private var currentMesh: SCNNode?
    @Binding var currentTime: Double
    @Binding var isPlaying: Bool
    let animator: PythonObject
    
    init(currentTime: Binding<Double>, isPlaying: Binding<Bool>, animator: PythonObject) {
        self._currentTime = currentTime
        self._isPlaying = isPlaying
        self.animator = animator
        
        // Configuration de base de la scène
        setupScene()
    }
    
    var body: some View {
        SceneView(
            scene: scene,
            pointOfView: setupCamera(),
            options: [.allowsCameraControl]
        )
        .onAppear {
            startAnimationLoop()
        }
    }
    
    private func setupScene() {
        // Éclairage ambiant
        let ambientLight = SCNNode()
        ambientLight.light = SCNLight()
        ambientLight.light?.type = .ambient
        ambientLight.light?.intensity = 100
        scene.rootNode.addChildNode(ambientLight)
        
        // Éclairage directionnel
        let directionalLight = SCNNode()
        directionalLight.light = SCNLight()
        directionalLight.light?.type = .directional
        directionalLight.light?.intensity = 800
        directionalLight.position = SCNVector3(5, 5, 5)
        directionalLight.look(at: SCNVector3(0, 0, 0))
        scene.rootNode.addChildNode(directionalLight)
        
        // Grille de référence
        let grid = SCNNode(geometry: SCNFloor())
        grid.geometry?.firstMaterial?.diffuse.contents = NSColor.gray
        grid.opacity = 0.3
        scene.rootNode.addChildNode(grid)
    }
    
    private func setupCamera() -> SCNNode {
        let camera = SCNNode()
        camera.camera = SCNCamera()
        camera.position = SCNVector3(0, 5, 10)
        camera.look(at: SCNVector3(0, 0, 0))
        return camera
    }
    
    private func startAnimationLoop() {
        guard isPlaying else { return }
        
        let displayLink = CADisplayLink(target: self, selector: #selector(updateAnimation))
        displayLink.add(to: .current, forMode: .default)
    }
    
    @objc private func updateAnimation() {
        guard isPlaying else { return }
        
        // Mettre à jour le temps
        currentTime += 1.0 / 60.0
        if currentTime >= animator.duration {
            if animator.loop {
                currentTime = 0
            } else {
                isPlaying = false
                return
            }
        }
        
        // Appliquer l'animation
        let updatedMesh = animator.apply_animations(currentTime)
        updatePreviewMesh(from: updatedMesh)
    }
    
    private func updatePreviewMesh(from pythonMesh: PythonObject) {
        // Supprimer l'ancien maillage
        currentMesh?.removeFromParentNode()
        
        // Convertir le maillage Python en géométrie SceneKit
        let vertices = pythonMesh.vertices as? [[Float]]
        let faces = pythonMesh.faces as? [[Int32]]
        
        guard let vertices = vertices, let faces = faces else { return }
        
        // Créer les sources de données pour la géométrie
        var vertexSource = [SCNVector3]()
        for vertex in vertices {
            vertexSource.append(SCNVector3(vertex[0], vertex[1], vertex[2]))
        }
        
        // Créer les indices des faces
        var indices = [Int32]()
        for face in faces {
            indices.append(contentsOf: face)
        }
        
        // Créer la géométrie
        let geometry = SCNGeometry(
            sources: [
                SCNGeometrySource(
                    vertices: vertexSource
                )
            ],
            elements: [
                SCNGeometryElement(
                    indices: indices,
                    primitiveType: .triangles
                )
            ]
        )
        
        // Créer le nœud et l'ajouter à la scène
        let node = SCNNode(geometry: geometry)
        scene.rootNode.addChildNode(node)
        currentMesh = node
        
        // Appliquer un matériau de base
        let material = SCNMaterial()
        material.diffuse.contents = NSColor.white
        material.metalness.contents = 0.8
        material.roughness.contents = 0.2
        node.geometry?.firstMaterial = material
    }
}

struct AnimationPreview_Previews: PreviewProvider {
    static var previews: some View {
        AnimationPreview(
            currentTime: .constant(0),
            isPlaying: .constant(false),
            animator: PythonObject.import("ai_models.text_animator").TextAnimator()
        )
    }
}
