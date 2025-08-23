import SwiftUI

struct ContentView: View {
    @StateObject var appController = AppController()
    @State private var isLoading = false
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            // Main 3D Modeling View
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
            .tabItem {
                Image(systemName: "cube")
                Text("3D Modeling")
            }
            .tag(0)
            
            // Modern Technology Showcase
            ModernTechShowcaseView()
                .tabItem {
                    Image(systemName: "brain.head.profile")
                    Text("Modern Tech")
                }
                .tag(1)
        }
        .preferredColorScheme(appController.themeManager.darkMode ? .dark : .light)
        .environmentObject(appController) // S'assurer que l'appController est dans l'environnement
        .onAppear {
            // Initialize modern technologies when the app starts
            appController.initializeModernTechnologies()
        }
    }
}
