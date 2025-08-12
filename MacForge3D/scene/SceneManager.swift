import Foundation

class SceneManager {
    static let shared = SceneManager()
    private var scenes: [Model3D] = []
    private var currentSceneIndex: Int = 0

    func addScene(_ model: Model3D) {
        scenes.append(model)
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
}
