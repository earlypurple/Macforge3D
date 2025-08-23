import Foundation
import PythonKit

// Import des modules principaux
import plugins
import formats
import export
import render
import arvr
import scenes

class AppController: ObservableObject {
    let pluginManager = PluginManager.shared
    let sceneManager = SceneManager.shared
    let shaderManager = ShaderManager.shared
    let vrManager = VRManager.shared
    let themeManager = ThemeManager()
    let modelImporter = ModelImporter()
    let modelExporter = ModelExporter()
    
    // Gestionnaires d'optimisation
    private let smartCache: PythonObject
    private let taskDistributor: PythonObject
    private let mlOptimizer: PythonObject
    private let cacheDir: String
    
    init() {
        // Initialiser Python
        let sys = Python.import("sys")
        let os = Python.import("os")
        let workspaceDir = "/workspaces/Macforge3D"
        
        // Ajouter le dossier Python au path
        sys.path.append(Python.str(workspaceDir + "/Python"))
        
        // Importer nos modules
        let smartCacheModule = Python.import("core.smart_cache")
        let taskDistributorModule = Python.import("core.task_distributor")
        let mlOptimizerModule = Python.import("core.ml_optimizer")
        
        // Créer le dossier cache
        cacheDir = (NSHomeDirectory() as NSString).appendingPathComponent(".macforge3d/cache")
        try? FileManager.default.createDirectory(
            atPath: cacheDir,
            withIntermediateDirectories: true
        )
        
        // Initialiser les composants
        smartCache = smartCacheModule.SmartCache(
            cache_dir: Python.str(cacheDir),
            feature_names: ["mesh_resolution", "material", "temperature", "pressure"]
        )
        
        taskDistributor = taskDistributorModule.TaskDistributor(
            host: Python.str("127.0.0.1"),
            port: Python.int(5555)
        )
        
        mlOptimizer = mlOptimizerModule.MLOptimizer(
            parameter_space: [
                "mesh_resolution": (Python.int, [1000, 1000000]),
                "temperature": (Python.float, [150.0, 300.0]),
                "pressure": (Python.float, [0.1, 10.0]),
                "material": (Python.str, ["PLA", "ABS", "PETG"])
            ],
            goals: [
                mlOptimizerModule.OptimizationGoal(
                    metric_name: "quality_score",
                    direction: "maximize",
                    weight: 1.0,
                    constraint_min: 0.7
                ),
                mlOptimizerModule.OptimizationGoal(
                    metric_name: "processing_time",
                    direction: "minimize",
                    weight: 0.5,
                    constraint_max: 3600
                ),
                mlOptimizerModule.OptimizationGoal(
                    metric_name: "memory_usage",
                    direction: "minimize",
                    weight: 0.3,
                    constraint_max: 16000
                )
            ],
            history_file: Python.str(cacheDir + "/optimization_history.csv")
        )
    }
    
    deinit {
        // Arrêter proprement les composants
        taskDistributor.stop()
    }
    
    func importModel(from url: URL) {
        if let model = modelImporter.importModel(from: url) {
            // Essayer de récupérer du cache
            let parameters = [
                "mesh_resolution": model.meshResolution,
                "material": model.material
            ]
            
            if let cachedResult = smartCache.get(parameters) {
                // Utiliser le résultat en cache
                updateModelFromCache(model: model, cache: cachedResult)
            } else {
                // Ajouter la scène normalement
                sceneManager.addScene(model)
                
                // Mettre en cache pour la prochaine fois
                smartCache.put(
                    parameters,
                    [
                        "vertices": model.vertices,
                        "faces": model.faces,
                        "attributes": model.attributes
                    ]
                )
            }
        }
    }
    
    private func updateModelFromCache(model: Model, cache: PythonObject) {
        model.vertices = Array(cache["vertices"] ?? [])
        model.faces = Array(cache["faces"] ?? [])
        model.attributes = Dictionary(cache["attributes"] ?? [:])
        sceneManager.addScene(model)
    }
    
    func exportCurrentScene(to url: URL, format: ModelFormat) {
        guard let model = sceneManager.currentScene() else { return }
        
        // Optimiser les paramètres d'export
        let parameters = [
            "mesh_resolution": model.meshResolution,
            "material": model.material,
            "temperature": model.temperature,
            "pressure": model.pressure
        ]
        
        let optimizedParams = mlOptimizer.optimize(
            n_trials: 100,
            timeout: 60
        )
        
        // Appliquer les paramètres optimisés
        model.meshResolution = Int(optimizedParams["mesh_resolution"] ?? model.meshResolution)
        model.temperature = Double(optimizedParams["temperature"] ?? model.temperature)
        model.pressure = Double(optimizedParams["pressure"] ?? model.pressure)
        
        // Distribuer l'export si possible
        let taskId = taskDistributor.submit_task(
            task_type: "export",
            parameters: [
                "model": model.toDict(),
                "format": format.rawValue,
                "url": url.absoluteString
            ],
            priority: 1
        )
        
        // Attendre le résultat
        if let result = taskDistributor.get_result(taskId, timeout: 300) {
            // Mettre à jour l'optimiseur
            mlOptimizer.add_result(
                parameters,
                [
                    "quality_score": result["quality_score"],
                    "processing_time": result["processing_time"],
                    "memory_usage": result["memory_usage"]
                ]
            )
        } else {
            // Fallback: export local
            _ = modelExporter.export(model: model, to: url, format: format)
        }
    }
    
    func activatePlugin(named name: String) {
        pluginManager.activate(pluginNamed: name)
    }
    
    func applyShader(named shaderName: String) {
        guard let model = sceneManager.currentScene() else { return }
        shaderManager.applyShader(named: shaderName, to: model)
    }
    
    func enableVR() {
        guard let model = sceneManager.currentScene() else { return }
        vrManager.enableVR(for: model)
    }
    
    // MARK: - Modern Technology Integration
    
    func initializeModernTechnologies() {
        Task {
            // Initialize modern tech stack
            do {
                let pythonBridge = PythonBridge.shared
                let result = try await pythonBridge.callAsyncFunction(
                    module: "modern_tech",
                    function: "initialize_modern_technologies",
                    arguments: []
                )
                
                if let resultDict = result as? [String: Any] {
                    let successCount = resultDict.values.compactMap { $0 as? Bool }.filter { $0 }.count
                    print("🚀 Modern technologies initialized: \(successCount)/\(resultDict.count) components")
                }
            } catch {
                print("❌ Failed to initialize modern technologies: \(error)")
            }
        }
    }
    
    func generateModelWithAI(
        prompt: String,
        model: String = "gpt4v_3d",
        enableNFT: Bool = false,
        enableCollaboration: Bool = false,
        enableWebXR: Bool = false
    ) async -> Bool {
        do {
            let pythonBridge = PythonBridge.shared
            let request: [String: Any] = [
                "input_type": "text",
                "prompt": prompt,
                "model": model,
                "options": [
                    "quality": "detailed",
                    "complexity": "high",
                    "style": "realistic"
                ],
                "create_nft": enableNFT,
                "enable_collaboration": enableCollaboration,
                "enable_webxr": enableWebXR,
                "creator_id": "swift_app_controller"
            ]
            
            let result = try await pythonBridge.callAsyncFunction(
                module: "modern_tech",
                function: "create_model_with_modern_pipeline",
                arguments: [request]
            )
            
            if let resultDict = result as? [String: Any],
               let success = resultDict["success"] as? Bool,
               success {
                
                // Extract mesh data and create model
                if let meshData = resultDict["mesh_data"] as? [String: Any],
                   let vertices = meshData["vertices"] as? Int,
                   let faces = meshData["faces"] as? Int {
                    
                    // Create a new 3D model from the AI generation
                    let newModel = Model3D()
                    newModel.name = "AI Generated: \(prompt.prefix(50))"
                    newModel.meshResolution = vertices
                    newModel.material = meshData["materials"] as? [String] ?? ["default"]
                    
                    // Add to scene
                    sceneManager.addScene(newModel)
                    
                    print("✅ AI model generated successfully: \(vertices) vertices, \(faces) faces")
                    
                    // Handle additional features
                    if let nftInfo = resultDict["nft_info"] as? [String: Any] {
                        print("🎨 NFT created: \(nftInfo["token_id"] ?? "unknown")")
                    }
                    
                    if let collaborationSession = resultDict["collaboration_session"] as? String {
                        print("👥 Collaboration session: \(collaborationSession)")
                    }
                    
                    if let webxrSession = resultDict["webxr_session"] as? String {
                        print("🥽 WebXR session: \(webxrSession)")
                    }
                    
                    return true
                }
            }
            
            return false
            
        } catch {
            print("❌ AI model generation failed: \(error)")
            return false
        }
    }
    
    func executeGraphQLQuery(_ query: String) async -> [String: Any]? {
        do {
            let pythonBridge = PythonBridge.shared
            let result = try await pythonBridge.callAsyncFunction(
                module: "modern_tech.graphql_api",
                function: "execute_graphql",
                arguments: [query]
            )
            
            return result as? [String: Any]
        } catch {
            print("❌ GraphQL query failed: \(error)")
            return nil
        }
    }
    
    func createCollaborationSession(projectId: String) async -> String? {
        do {
            let pythonBridge = PythonBridge.shared
            let result = try await pythonBridge.callAsyncFunction(
                module: "modern_tech.collaboration",
                function: "create_collaboration_session",
                arguments: [projectId, "app_controller_user"]
            )
            
            return result as? String
        } catch {
            print("❌ Failed to create collaboration session: \(error)")
            return nil
        }
    }
    
    func startWebXRSession(mode: String = "immersive-vr") async -> String? {
        do {
            let pythonBridge = PythonBridge.shared
            let result = try await pythonBridge.callAsyncFunction(
                module: "modern_tech.webxr_integration",
                function: "create_xr_session",
                arguments: [mode, "app_controller_user", [:]]
            )
            
            return result as? String
        } catch {
            print("❌ Failed to start WebXR session: \(error)")
            return nil
        }
    }
    
    func getCacheStats() -> [String: Any] {
        let stats = smartCache.get_stats()
        return [
            "totalEntries": Int(stats["total_entries"] ?? 0),
            "totalSizeMB": Double(stats["total_size_mb"] ?? 0),
            "maxSizeMB": Double(stats["max_size_mb"] ?? 0),
            "usagePercent": Double(stats["usage_percent"] ?? 0),
            "oldestEntryAge": Double(stats["oldest_entry_age"] ?? 0)
        ]
    }
    
    func clearCache() {
        smartCache.clear()
    }
}
