import Foundation
import PythonKit

/// Gestionnaire de rendu distribué sur le cloud
class CloudRenderManager {
    static let shared = CloudRenderManager()
    private var cluster: [String] = []
    
    private init() {}
    
    /// Configure le cluster de rendu
    /// - Parameters:
    ///   - numNodes: Nombre de noeuds
    ///   - instanceType: Type d'instance (ex: "g4dn.xlarge" pour GPU)
    func setupRenderCluster(numNodes: Int, instanceType: String) async throws {
        cluster = try await CloudManager.shared.setupCluster(
            numNodes: numNodes,
            instanceType: instanceType
        )
    }
    
    /// Lance un rendu sur le cloud
    /// - Parameters:
    ///   - scene: Chemin de la scène
    ///   - outputPath: Chemin de sortie
    ///   - config: Configuration de rendu
    func renderScene(scene: String,
                    outputPath: String,
                    config: RenderConfig) async throws -> RenderResult {
        let jobConfig: [String: Any] = [
            "name": "render_job",
            "queue": "render_queue",
            "definition": "render_definition",
            "command": [
                "render",
                "--scene", scene,
                "--output", outputPath,
                "--width", config.width,
                "--height", config.height,
                "--samples", config.samples,
                "--device", config.device
            ]
        ]
        
        let result = try await CloudManager.shared.processRemote(
            inputPath: scene,
            outputPath: outputPath,
            jobConfig: jobConfig
        )
        
        return RenderResult(
            jobId: result["job_id"] as? String ?? "",
            status: result["status"] as? [String: Any] ?? [:],
            outputPath: result["output_path"] as? String ?? ""
        )
    }
}

/// Configuration de rendu
struct RenderConfig {
    let width: Int
    let height: Int
    let samples: Int
    let device: String  // "cpu" ou "cuda"
}

/// Résultat du rendu
struct RenderResult {
    let jobId: String
    let status: [String: Any]
    let outputPath: String
}
