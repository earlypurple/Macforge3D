import Foundation

// Represents the quality of the print.
enum PrintQuality: String, CaseIterable, Codable {
    case draft = "Brouillon"
    case normal = "Normal"
    case fine = "Fin"
    case extraFine = "Extra Fin"
}

// Defines all the settings for a 3D print job.
// These settings can be customized for each print.
struct PrintSettings: Codable {
    // Quality
    var quality: PrintQuality = .normal
    var layerHeight: Float = 0.2 // in mm
    var initialLayerHeight: Float = 0.3 // in mm

    // Infill
    var infillDensity: Int = 20 // in %
    var infillPattern: String = "Grid" // e.g., Grid, Lines, Triangles

    // Speed
    var printSpeed: Int = 60 // in mm/s
    var travelSpeed: Int = 120 // in mm/s

    // Temperature
    var nozzleTemp: Int = 200 // in Celsius
    var bedTemp: Int = 60 // in Celsius

    // Supports
    var generateSupports: Bool = true
    var supportOverhangAngle: Int = 45 // in degrees

    // Adhesion
    var buildPlateAdhesion: String = "Skirt" // e.g., Skirt, Brim, Raft
    
    // Bambu Lab specific settings
    var amsEnabled: Bool = false // Automatic Material System
    var amsSlotUsed: Int = 1 // AMS slot (1-4)
    var filamentType: String = "PLA" // For AMS material detection
    var enableLidar: Bool = true // First layer inspection
    var enableAIError: Bool = true // AI failure detection
    var silentMode: Bool = false // Silent printing mode
    var adaptiveLayerHeight: Bool = false // Variable layer height
    var smoothingMode: Bool = false // For TPU and flexible materials

    // A static default instance for convenience.
    static var `default`: PrintSettings {
        return PrintSettings()
    }
}
