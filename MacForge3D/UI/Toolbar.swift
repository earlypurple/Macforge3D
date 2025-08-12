import SwiftUI

struct Toolbar: View {
    @EnvironmentObject var appController: AppController

    var body: some View {
        HStack {
            Button("Importer") {
                // Exemple d'utilisation
                let url = ... // à définir via FilePicker
                appController.importModel(from: url)
            }
            Button("Exporter") {
                let url = ... // à définir
                let format: ModelFormat = .obj // exemple
                appController.exportCurrentScene(to: url, format: format)
            }
            Button("Activer plugin") {
                appController.activatePlugin(named: "ExemplePlugin")
            }
            Button("Shader") {
                appController.applyShader(named: "ExempleShader")
            }
            Button("VR/AR") {
                appController.enableVR()
            }
            Button("Scènes") {
                appController.sceneManager.switchToScene(index: 0) // exemple
            }
            Button("Dark Mode") {
                appController.themeManager.toggleDarkMode()
            }
        }
        .padding()
        .background(Color.gray.opacity(0.2))
    }
}
