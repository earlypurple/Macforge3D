import SwiftUI

struct ContentView: View {
    @StateObject var appController = AppController()

    var body: some View {
        VStack {
            Toolbar()
            // Affichage de la scène courante
            if let scene = appController.sceneManager.currentScene() {
                SceneView(model: scene)
            }
        }
        .preferredColorScheme(appController.themeManager.darkMode ? .dark : .light)
    }
}
