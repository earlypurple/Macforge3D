import Foundation

class ExamplePlugin: Plugin {
    var name: String { "ExemplePlugin" }
    func activate() {
        print("Plugin activé")
        // Ajout de fonctionnalités
    }
    func deactivate() {
        print("Plugin désactivé")
    }
}
