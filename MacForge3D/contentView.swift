import SwiftUI

struct ContentView: View {
    @StateObject var appController = AppController()
    @State private var isLoading = false

    var body: some View {
        ZStack {
            VStack {
                // On passe une liaison (binding) vers isLoading
                Toolbar(isLoading: $isLoading)

                // Affichage de la scène courante
                if let scene = appController.sceneManager.currentScene() {
                    // Remplacer par la vue de la scène 3D
                    Text("Vue de la scène 3D")
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            // On désactive l'interface utilisateur principale pendant le chargement
            .disabled(isLoading)
            .blur(radius: isLoading ? 3 : 0)

            // Afficher une vue de progression si isLoading est vrai
            if isLoading {
                ProgressView("Génération en cours...")
                    .padding(20)
                    .background(Material.thick)
                    .cornerRadius(10)
                    .shadow(radius: 10)
            }
        }
        .preferredColorScheme(appController.themeManager.darkMode ? .dark : .light)
        .environmentObject(appController) // S'assurer que l'appController est dans l'environnement
    }
}
