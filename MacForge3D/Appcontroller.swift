import Foundation
import SwiftUI

// Import des nouveaux modules
import plugins
import formats
import export
import render
import arvr
import scenes

class AppController: ObservableObject {
    // Managers principaux
    let pluginManager = PluginManager.shared
    let sceneManager = SceneManager.shared
    let shaderManager = ShaderManager.shared
    let vrManager = VRManager.shared
    let themeManager = ThemeManager()
    let modelImporter = ModelImporter()
    let modelExporter = ModelExporter()

    // Exemple : importer un modèle
    func importModel(from url: URL) {
        if let model = modelImporter.importModel(from: url) {
            sceneManager.addScene(model)
        }
    }

    // Exemple : exporter la scène courante
    func exportCurrentScene(to url: URL, format: ModelFormat) {
        guard let model = sceneManager.currentScene() else { return }
        _ = modelExporter.export(model: model, to: url, format: format)
    }

    func exportCurrentSceneToData(format: ModelFormat) -> Data? {
        guard let model = sceneManager.currentScene() else { return nil }

        // Create a temporary file URL
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)

        // Export the model to the temporary file
        let success = modelExporter.export(model: model, to: tempURL, format: format)

        if success {
            do {
                // Read the data from the temporary file
                let data = try Data(contentsOf: tempURL)
                // Clean up the temporary file
                try? FileManager.default.removeItem(at: tempURL)
                return data
            } catch {
                print("Error reading exported file: \(error)")
                return nil
            }
        } else {
            return nil
        }
    }

    // Exemple : activer un plugin
    func activatePlugin(named name: String) {
        pluginManager.activate(pluginNamed: name)
    }

    // Exemple : appliquer un shader
    func applyShader(named shaderName: String) {
        guard let model = sceneManager.currentScene() else { return }
        shaderManager.applyShader(named: shaderName, to: model)
    }

    // Exemple : lancer VR
    func enableVR() {
        guard let model = sceneManager.currentScene() else { return }
        vrManager.enableVR(for: model)
    }
}
