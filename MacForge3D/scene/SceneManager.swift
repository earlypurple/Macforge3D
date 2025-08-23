import Foundation
import PythonKit
import Combine

class SceneManager: ObservableObject {
    static let shared = SceneManager()
    
    @Published private(set) var scenes: [Model3D] = []
    @Published private(set) var currentSceneIndex: Int = 0
    @Published private(set) var isProcessing: Bool = false
    @Published private(set) var processingProgress: Double = 0
    
    private let taskQueue = DispatchQueue(label: "com.macforge3d.sceneManager")
    private let taskDistributor: PythonObject
    private var cancellables = Set<AnyCancellable>()
    private var activeTask: String?
    
    private init() {
        // Initialiser Python
        let sys = Python.import("sys")
        sys.path.append(Python.str("/workspaces/Macforge3D/Python"))
        
        // Initialiser le distributeur de tâches
        let taskDistributorModule = Python.import("core.task_distributor")
        self.taskDistributor = taskDistributorModule.TaskDistributor(
            host: Python.str("127.0.0.1"),
            port: Python.int(5556)
        )
        
        // Observer les demandes d'optimisation
        NotificationCenter.default
            .publisher(for: .modelOptimizationRequested)
            .sink { [weak self] notification in
                if let model = notification.object as? Model3D {
                    self?.optimizeModel(model)
                }
            }
            .store(in: &cancellables)
    }
    
    deinit {
        taskDistributor.stop()
    }
    
    // MARK: - Gestion des scènes
    
    func addScene(_ model: Model3D) {
        scenes.append(model)
        // Distribuer le chargement si nécessaire
        if model.isHeavy {
            loadSceneDistributed(model)
        }
    }
    
    func switchToScene(index: Int) {
        guard index >= 0 && index < scenes.count else { return }
        currentSceneIndex = index
    }
    
    func currentScene() -> Model3D? {
        return scenes.isEmpty ? nil : scenes[currentSceneIndex]
    }
    
    func allScenes() -> [Model3D] {
        return scenes
    }
    
    // MARK: - Traitement distribué
    
    private func loadSceneDistributed(_ model: Model3D) {
        isProcessing = true
        processingProgress = 0
        
        taskQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Soumettre la tâche
            self.activeTask = self.taskDistributor.submit_task(
                task_type: "load_scene",
                parameters: [
                    "model": model.toDict(),
                    "optimize": true
                ],
                priority: 1,
                timeout: 600
            )
            
            // Attendre le résultat
            if let taskId = self.activeTask,
               let result = self.taskDistributor.get_result(taskId) {
                DispatchQueue.main.async {
                    if let updatedModel = result["model"] as? [String: Any] {
                        model.fromDict(updatedModel)
                    }
                    if let progress = result["progress"] as? Double {
                        self.processingProgress = progress
                    }
                    if let completed = result["completed"] as? Bool, completed {
                        self.isProcessing = false
                        self.processingProgress = 1.0
                        self.activeTask = nil
                        
                        NotificationCenter.default.post(
                            name: .modelOptimizationCompleted,
                            object: model
                        )
                    }
                }
            } else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.processingProgress = 0
                    self.activeTask = nil
                    
                    NotificationCenter.default.post(
                        name: .modelOptimizationFailed,
                        object: model
                    )
                }
            }
        }
    }
    
    private func optimizeModel(_ model: Model3D) {
        isProcessing = true
        processingProgress = 0
        
        taskQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Soumettre la tâche d'optimisation
            self.activeTask = self.taskDistributor.submit_task(
                task_type: "optimize_scene",
                parameters: [
                    "model": model.toDict(),
                    "quality_target": 0.8,
                    "max_time": 3600
                ],
                priority: 2
            )
            
            // Attendre et traiter le résultat
            if let taskId = self.activeTask,
               let result = self.taskDistributor.get_result(taskId) {
                DispatchQueue.main.async {
                    if let optimizedModel = result["model"] as? [String: Any] {
                        model.fromDict(optimizedModel)
                    }
                    if let metrics = result["metrics"] as? [String: Any] {
                        model.updateMetrics(metrics)
                    }
                    
                    self.isProcessing = false
                    self.processingProgress = 1.0
                    self.activeTask = nil
                    
                    NotificationCenter.default.post(
                        name: .modelOptimizationCompleted,
                        object: model
                    )
                }
            } else {
                DispatchQueue.main.async {
                    self.isProcessing = false
                    self.processingProgress = 0
                    self.activeTask = nil
                    
                    NotificationCenter.default.post(
                        name: .modelOptimizationFailed,
                        object: model
                    )
                }
            }
        }
    }
    
    func cancelCurrentTask() {
        if let taskId = activeTask {
            taskDistributor.cancel_task(taskId)
            activeTask = nil
            isProcessing = false
            processingProgress = 0
        }
    }
}
