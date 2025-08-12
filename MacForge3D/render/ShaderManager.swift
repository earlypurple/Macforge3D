import Foundation

class ShaderManager {
    static let shared = ShaderManager()
    private var shaders: [String: Shader] = [:]

    func loadShader(named name: String, source: String) {
        // Charger et compiler le shader
    }

    func applyShader(named name: String, to model: Model3D) {
        // Appliquer le shader au modèle
    }
}

class Shader {
    let name: String
    let source: String
    // Ajout d'autres propriétés nécessaires
    init(name: String, source: String) {
        self.name = name
        self.source = source
    }
}
