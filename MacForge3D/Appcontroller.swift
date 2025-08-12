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
