import Foundation
import PythonKit

class AISlicerAdvisor {

    enum AdvisorError: Error {
        case pythonError(String)
        case pythonModuleNotFound
        case pythonFunctionNotFound
    }

    private let pythonModule: PythonObject

    init() throws {
        do {
            // Add the path to the python scripts to the python path
            let sys = try Python.import("sys")
            sys.path.append("Python")

            self.pythonModule = try Python.import("ai_models.slicer_advisor")
        } catch {
            throw AdvisorError.pythonModuleNotFound
        }
    }

    // This function takes a shape and a printer profile and returns AI-suggested settings.
    func suggestSettings(for shape: Shape3D, profile: PrinterProfile, intent: String) throws -> PrintSettings {
        guard let suggest_settings = pythonModule.suggest_settings else {
            throw AdvisorError.pythonFunctionNotFound
        }

        // We need to extract some features from the shape to pass to the AI model.
        // For now, let's just pass some basic info like the number of triangles
        // and the bounding box volume.
        let numTriangles = shape.mesh.triangles.count
        let boundingBoxVolume = profile.buildVolumeX * profile.buildVolumeY * profile.buildVolumeZ // Placeholder

        let modelFeatures: [String: PythonConvertible] = [
            "num_triangles": numTriangles,
            "bounding_box_volume": boundingBoxVolume,
            "printer_name": profile.name,
            "intent": intent // e.g., "fast", "high_quality", "strong"
        ]

        let suggestedSettingsDict = suggest_settings(modelFeatures)

        // Convert the Python dictionary back to a Swift PrintSettings struct.
        guard let settingsDict = Dictionary<String, PythonObject>(suggestedSettingsDict) else {
            throw AdvisorError.pythonError("Could not convert python dict to swift dict")
        }

        var newSettings = PrintSettings.default

        if let layerHeight = Float(settingsDict["layer_height"] ?? 0.2) {
            newSettings.layerHeight = layerHeight
        }
        if let printSpeed = Int(settingsDict["print_speed"] ?? 60) {
            newSettings.printSpeed = printSpeed
        }
        if let nozzleTemp = Int(settingsDict["nozzle_temp"] ?? 200) {
            newSettings.nozzleTemp = nozzleTemp
        }
        if let infillDensity = Int(settingsDict["infill_density"] ?? 20) {
            newSettings.infillDensity = infillDensity
        }

        return newSettings
    }
}
