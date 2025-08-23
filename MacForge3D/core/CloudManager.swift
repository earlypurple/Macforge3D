import Foundation
import PythonKit

/// Gestionnaire pour l'intégration cloud
class CloudManager {
    static let shared = CloudManager()
    private let pythonManager: PythonManager
    private var cloudModule: PythonObject?
    private var cloudManager: PythonObject?
    
    private init() {
        self.pythonManager = PythonManager.shared
        setupPythonEnvironment()
    }
    
    /// Configure l'environnement Python
    private func setupPythonEnvironment() {
        do {
            // Import du module cloud_integration
            cloudModule = try pythonManager.importModule(name: "ai_models.cloud_integration")
        } catch {
            print("Erreur lors de l'import du module cloud: \(error)")
        }
    }
    
    /// Configure le gestionnaire cloud
    /// - Parameters:
    ///   - provider: Le fournisseur cloud ("aws", "azure", "gcp")
    ///   - credentials: Les identifiants d'accès
    ///   - region: La région
    ///   - bucket: Le bucket/container
    ///   - prefix: Le préfixe pour les fichiers
    func configure(provider: String, 
                  credentials: [String: String],
                  region: String,
                  bucket: String,
                  prefix: String) {
        guard let cloudModule = cloudModule else {
            print("Module cloud non initialisé")
            return
        }
        
        let config = cloudModule.CloudConfig(
            provider: provider,
            credentials: credentials,
            region: region,
            bucket: bucket,
            prefix: prefix
        )
        
        cloudManager = cloudModule.CloudManager(config)
    }
    
    /// Synchronise un projet avec le cloud
    /// - Parameters:
    ///   - localPath: Chemin local du projet
    ///   - remotePath: Chemin distant (optionnel)
    func syncProject(localPath: String, remotePath: String? = nil) async throws {
        guard let cloudManager = cloudManager else {
            throw CloudError.notConfigured
        }
        
        let syncTask = cloudManager.sync_project(
            localPath: localPath,
            remotePath: remotePath
        )
        
        try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    try await PythonObject(syncTask.__await__())
                    continuation.resume()
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Lance un traitement sur le cloud
    /// - Parameters:
    ///   - inputPath: Chemin du fichier d'entrée
    ///   - outputPath: Chemin du fichier de sortie
    ///   - jobConfig: Configuration du job
    func processRemote(inputPath: String,
                      outputPath: String,
                      jobConfig: [String: Any]) async throws -> [String: Any] {
        guard let cloudManager = cloudManager else {
            throw CloudError.notConfigured
        }
        
        let processTask = cloudManager.process_remote(
            input_path: inputPath,
            output_path: outputPath,
            job_config: jobConfig
        )
        
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let result = try await PythonObject(processTask.__await__())
                    let swiftResult = result.toDictionary()
                    continuation.resume(returning: swiftResult)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
    
    /// Configure un cluster de calcul
    /// - Parameters:
    ///   - numNodes: Nombre de noeuds
    ///   - instanceType: Type d'instance
    func setupCluster(numNodes: Int, instanceType: String) async throws -> [String] {
        guard let cloudManager = cloudManager else {
            throw CloudError.notConfigured
        }
        
        let setupTask = cloudManager.setup_cluster(
            num_nodes: numNodes,
            instance_type: instanceType
        )
        
        return try await withCheckedThrowingContinuation { continuation in
            Task {
                do {
                    let result = try await PythonObject(setupTask.__await__())
                    let instanceIds = result.toArray().map { String(describing: $0) }
                    continuation.resume(returning: instanceIds)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

/// Erreurs possibles
enum CloudError: Error {
    case notConfigured
}

/// Extensions pour la conversion Python -> Swift
extension PythonObject {
    func toDictionary() -> [String: Any] {
        var dict = [String: Any]()
        for (key, value) in Array(self.items()) {
            dict[String(describing: key)] = convertPythonValue(value)
        }
        return dict
    }
    
    func toArray() -> [Any] {
        return Array(self).map { convertPythonValue($0) }
    }
    
    private func convertPythonValue(_ value: PythonObject) -> Any {
        if Python.isinstance(value, Python.str) {
            return String(describing: value)
        } else if Python.isinstance(value, Python.int) {
            return Int(value) ?? 0
        } else if Python.isinstance(value, Python.float) {
            return Double(value) ?? 0.0
        } else if Python.isinstance(value, Python.bool) {
            return Bool(value) ?? false
        } else if Python.isinstance(value, Python.dict) {
            return value.toDictionary()
        } else if Python.isinstance(value, Python.list) {
            return value.toArray()
        }
        return String(describing: value)
    }
}
