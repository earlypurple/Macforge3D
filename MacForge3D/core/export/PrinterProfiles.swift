import Foundation

// Represents the physical characteristics of a 3D printer.
struct PrinterProfile: Identifiable, Codable {
    let id = UUID()
    var name: String
    var manufacturer: String

    // Build volume in mm
    var buildVolumeX: Float
    var buildVolumeY: Float
    var buildVolumeZ: Float

    // Nozzle
    var nozzleDiameter: Float = 0.4 // in mm

    // Material compatibility
    var compatibleMaterials: [String] // e.g., ["PLA", "ABS", "PETG"]

    // Default print settings for this profile
    var defaultPrintSettings: PrintSettings
    
    // Advanced features support
    var hasAMS: Bool = false // Automatic Material System
    var maxAMSSlots: Int = 0 // Number of AMS slots
    var hasLidar: Bool = false // First layer inspection
    var hasAIError: Bool = false // AI failure detection
    var maxPrintSpeed: Int = 150 // Maximum print speed in mm/s
    var hasHeatedChamber: Bool = false
    var hasFilamentRunoutSensor: Bool = false
    var gcodeFormat: String = "standard" // "bambu", "standard", "klipper"
    var slicerCompatibility: [String] = ["PrusaSlicer"] // Compatible slicers
}

// A manager class to hold and provide access to printer profiles.
class PrinterProfileManager {
    static let shared = PrinterProfileManager()

    let profiles: [PrinterProfile]

    private init() {
        self.profiles = [
            // Original Prusa i3 MK3S+
            PrinterProfile(
                name: "Original Prusa i3 MK3S+",
                manufacturer: "Prusa Research",
                buildVolumeX: 250,
                buildVolumeY: 210,
                buildVolumeZ: 210,
                compatibleMaterials: ["PLA", "ABS", "PETG", "ASA"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.15,
                    printSpeed: 80,
                    nozzleTemp: 215,
                    bedTemp: 60
                ),
                hasFilamentRunoutSensor: true,
                slicerCompatibility: ["PrusaSlicer", "SuperSlicer"]
            ),
            
            // Creality Ender 3
            PrinterProfile(
                name: "Creality Ender 3",
                manufacturer: "Creality",
                buildVolumeX: 220,
                buildVolumeY: 220,
                buildVolumeZ: 250,
                compatibleMaterials: ["PLA", "ABS", "PETG"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.2,
                    printSpeed: 50,
                    nozzleTemp: 200,
                    bedTemp: 50
                ),
                maxPrintSpeed: 80,
                slicerCompatibility: ["Cura", "PrusaSlicer"]
            ),
            
            // Bambu Lab A1 mini
            PrinterProfile(
                name: "Bambu Lab A1 mini",
                manufacturer: "Bambu Lab",
                buildVolumeX: 180,
                buildVolumeY: 180,
                buildVolumeZ: 180,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.2,
                    printSpeed: 150,
                    nozzleTemp: 220,
                    bedTemp: 65,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            ),
            
            // Bambu Lab A1
            PrinterProfile(
                name: "Bambu Lab A1",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS", "ASA", "PC", "PA-CF"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.2,
                    printSpeed: 200,
                    nozzleTemp: 230,
                    bedTemp: 70,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            ),
            
            // Bambu Lab P1P
            PrinterProfile(
                name: "Bambu Lab P1P",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS", "ASA"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.2,
                    printSpeed: 250,
                    nozzleTemp: 220,
                    bedTemp: 65,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            ),
            
            // Bambu Lab P1S
            PrinterProfile(
                name: "Bambu Lab P1S",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS", "ASA", "PC", "PA-CF", "PET-CF"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.15,
                    printSpeed: 300,
                    nozzleTemp: 240,
                    bedTemp: 80,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasHeatedChamber: true,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            ),
            
            // Bambu Lab X1
            PrinterProfile(
                name: "Bambu Lab X1",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS", "ASA", "PC", "PA-CF", "PET-CF", "PPA-CF"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.1,
                    printSpeed: 350,
                    nozzleTemp: 250,
                    bedTemp: 90,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasHeatedChamber: true,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            ),
            
            // Bambu Lab X1 Carbon
            PrinterProfile(
                name: "Bambu Lab X1 Carbon",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA", "ABS", "ASA", "PC", "PA-CF", "PET-CF", "PPA-CF", "PAHT-CF"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.08,
                    printSpeed: 400,
                    nozzleTemp: 260,
                    bedTemp: 100,
                    amsEnabled: true,
                    amsSlotUsed: 1,
                    enableLidar: true,
                    enableAIError: true,
                    adaptiveLayerHeight: true,
                    silentMode: false
                ),
                hasAMS: true,
                maxAMSSlots: 4,
                hasLidar: true,
                hasAIError: true,
                maxPrintSpeed: 500,
                hasHeatedChamber: true,
                hasFilamentRunoutSensor: true,
                gcodeFormat: "bambu",
                slicerCompatibility: ["BambuStudio", "OrcaSlicer"]
            )
        ]
    }

    func findProfile(byName name: String) -> PrinterProfile? {
        return profiles.first { $0.name == name }
    }
    
    func getBambuLabProfiles() -> [PrinterProfile] {
        return profiles.filter { $0.manufacturer == "Bambu Lab" }
    }
    
    func getProfilesWithAMS() -> [PrinterProfile] {
        return profiles.filter { $0.hasAMS }
    }
    
    func getProfilesForMaterial(_ material: String) -> [PrinterProfile] {
        return profiles.filter { $0.compatibleMaterials.contains(material) }
    }
    
    func getOptimalBambuProfile(for material: String, quality: PrintQuality = .normal) -> PrinterProfile? {
        let bambuProfiles = getBambuLabProfiles().filter { $0.compatibleMaterials.contains(material) }
        
        // Recommend best profile based on material and quality
        switch quality {
        case .extraFine:
            return bambuProfiles.first { $0.name.contains("X1 Carbon") } ?? 
                   bambuProfiles.first { $0.name.contains("X1") }
        case .fine:
            return bambuProfiles.first { $0.name.contains("P1S") } ??
                   bambuProfiles.first { $0.name.contains("X1") }
        case .normal:
            return bambuProfiles.first { $0.name.contains("P1P") } ??
                   bambuProfiles.first { $0.name.contains("A1") }
        case .draft:
            return bambuProfiles.first { $0.name.contains("A1") } ??
                   bambuProfiles.first
        }
    }
}
