import SwiftUI
import UniformTypeIdentifiers

struct Toolbar: View {
    @EnvironmentObject var appController: AppController
    @Binding var isLoading: Bool
    @State private var isImporting = false
    @State private var isExporting = false
    @State private var document: ExportedFile?

    // Définir les types de fichiers supportés pour l'importation.
    private var supportedImportTypes: [UTType] {
        ["obj", "stl", "ply", "dae", "scn", "usd", "usdz"].compactMap { UTType(filenameExtension: $0) }
    }

    var body: some View {
        HStack {
            Button(action: { isImporting = true }) {
                Image(systemName: "square.and.arrow.down")
            }
            .help("Importer un modèle")

            Button(action: {
                let placeholderData = "Ceci est un test d'exportation".data(using: .utf8) ?? Data()
                document = ExportedFile(data: placeholderData)
                isExporting = true
            }) {
                Image(systemName: "square.and.arrow.up")
            }
            .help("Exporter la scène")

            Button(action: { appController.activatePlugin(named: "ExemplePlugin") }) {
                Image(systemName: "puzzlepiece.extension")
            }
            .help("Activer un plugin")

            Button(action: { appController.applyShader(named: "ExempleShader") }) {
                Image(systemName: "paintbrush.pointed")
            }
            .help("Appliquer un shader")

            Button(action: { appController.enableVR() }) {
                Image(systemName: "arkit")
            }
            .help("Activer la VR/AR")

            Button(action: { appController.sceneManager.switchToScene(index: 0) }) {
                Image(systemName: "film.stack")
            }
            .help("Changer de scène")

            Button(action: { appController.themeManager.toggleDarkMode() }) {
                Image(systemName: "moon.circle")
            }
            .help("Changer le thème")

            // Bouton de test pour la vue de chargement
            Button(action: {
                isLoading.toggle()
            }) {
                Image(systemName: "hourglass")
            }
            .help("Tester le chargement")
        }
        .padding()
        .background(Color.gray.opacity(0.2))
        .fileImporter(
            isPresented: $isImporting,
            allowedContentTypes: supportedImportTypes,
            allowsMultipleSelection: false
        ) { result in
            do {
                if let selectedURL = try result.get().first {
                    appController.importModel(from: selectedURL)
                }
            } catch {
                print("Error importing file: \(error.localizedDescription)")
            }
        }
        .fileExporter(
            isPresented: $isExporting,
            document: document,
            contentType: .plainText, // TODO: Changer pour le type de fichier 3D approprié
            defaultFilename: "mon_modele.txt" // TODO: Changer l'extension
        ) { result in
            switch result {
            case .success(let url):
                print("Modèle exporté avec succès vers: \(url)")
            case .failure(let error):
                print("Erreur d'exportation: \(error.localizedDescription)")
            }
        }
    }
}
