import Foundation

protocol Plugin {
    var name: String { get }
    func activate()tu
    func deactivate()
}

class PluginManager {
    static let shared = PluginManager()
    private var plugins: [Plugin] = []

    func loadPlugins(from directory: URL) {
        // Charger dynamiquement les plugins depuis un dossier
    }

    func activate(pluginNamed name: String) {
        // Activer un plugin par son nom
    }

    func deactivate(pluginNamed name: String) {
        // Désactiver un plugin par son nom
    }

    func listPlugins() -> [String] {
        return plugins.map { $0.name }
    }
}
