import Foundation

class VRManager {
    static let shared = VRManager()
    private(set) var vrSupported: Bool = false

    func checkVRSupport() {
        // Vérifier la présence d'un périphérique VR/AR
    }

    func enableVR(for scene: Model3D) {
        // Activer la vue VR/AR sur la scène
    }

    func disableVR() {
        // Désactiver la vue VR/AR
    }
}
