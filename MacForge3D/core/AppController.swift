import Foundation

// Import des modules principaux
import plugins
import formats
import export
import render
import arvr
import scenes

class AppController: ObservableObject {
    let pluginManager = PluginManager.shared
    let sceneManager = SceneManager.shared
    let shaderManager = ShaderManager.shared
    let vrManager = VRManager.shared
    let themeManager = ThemeManager()
    let modelImporter = ModelImporter()
    let modelExporter = ModelExporter()

    func importModel(from url: URL) {
        if let model = modelImporter.importModel(from: url) {
            sceneManager.addScene(model)
        }
    }

    func exportCurrentScene(to url: URL, format: ModelFormat) {
        guard let model = sceneManager.currentScene() else { return }
        _ = modelExporter.export(model: model, to: url, format: format)
    }

    func activatePlugin(named name: String) {
        pluginManager.activate(pluginNamed: name)
    }

    func applyShader(named shaderName: String) {
        guard let model = sceneManager.currentScene() else { return }
        shaderManager.applyShader(named: shaderName, to: model)
    }

    func enableVR() {
        guard let model = sceneManager.currentScene() else { return }
        vrManager.enableVR(for: model)
    }
}
