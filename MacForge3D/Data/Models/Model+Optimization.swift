import Foundation

extension Model {
    // Propriétés pour l'optimisation
    @Published var meshResolution: Int = 10000
    @Published var temperature: Double = 200.0
    @Published var pressure: Double = 1.0
    @Published var material: String = "PLA"
    
    func toDict() -> [String: Any] {
        return [
            "vertices": vertices,
            "faces": faces,
            "attributes": attributes,
            "mesh_resolution": meshResolution,
            "temperature": temperature,
            "pressure": pressure,
            "material": material
        ]
    }
    
    func fromDict(_ dict: [String: Any]) {
        if let vertices = dict["vertices"] as? [[Double]] {
            self.vertices = vertices
        }
        if let faces = dict["faces"] as? [[Int]] {
            self.faces = faces
        }
        if let attributes = dict["attributes"] as? [String: Any] {
            self.attributes = attributes
        }
        if let resolution = dict["mesh_resolution"] as? Int {
            self.meshResolution = resolution
        }
        if let temp = dict["temperature"] as? Double {
            self.temperature = temp
        }
        if let pres = dict["pressure"] as? Double {
            self.pressure = pres
        }
        if let mat = dict["material"] as? String {
            self.material = mat
        }
    }
    
    // Méthodes d'optimisation
    func optimize() {
        // Point d'entrée pour l'optimisation manuelle
        NotificationCenter.default.post(
            name: .modelOptimizationRequested,
            object: self
        )
    }
}

// Notifications
extension Notification.Name {
    static let modelOptimizationRequested = Notification.Name(
        "com.macforge3d.model.optimizationRequested"
    )
    static let modelOptimizationCompleted = Notification.Name(
        "com.macforge3d.model.optimizationCompleted"
    )
    static let modelOptimizationFailed = Notification.Name(
        "com.macforge3d.model.optimizationFailed"
    )
}
