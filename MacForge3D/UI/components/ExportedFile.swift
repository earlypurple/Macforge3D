import SwiftUI
import UniformTypeIdentifiers

// Ce document représente un fichier de données simple qui peut être utilisé avec le modificateur .fileExporter.
// Il encapsule les données brutes (Data) d'un modèle 3D à exporter.
struct ExportedFile: FileDocument {
    // Les types de contenu que nous pouvons lire (non utilisé pour l'exportation pure, mais requis par le protocole).
    static var readableContentTypes: [UTType] { [.data] }

    // Les données du fichier à exporter.
    var data: Data

    /// Initialise le document avec des données.
    init(data: Data) {
        self.data = data
    }

    /// Initialiseur requis par FileDocument pour la lecture (non utilisé ici).
    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw CocoaError(.fileReadCorruptFile)
        }
        self.data = data
    }

    /// Crée un FileWrapper pour écrire le document sur le disque.
    /// C'est cette méthode qui est appelée par le système lors de l'exportation.
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        return FileWrapper(regularFileWithContents: data)
    }
}
