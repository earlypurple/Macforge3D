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
}

// A manager class to hold and provide access to printer profiles.
class PrinterProfileManager {
    static let shared = PrinterProfileManager()

    let profiles: [PrinterProfile]

    private init() {
        self.profiles = [
            // Example Profile 1: Prusa i3 MK3S+
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
                )
            ),
            // Example Profile 2: Creality Ender 3
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
                )
            ),
            // Example Profile 3: Bambu Lab P1P
            PrinterProfile(
                name: "Bambu Lab P1P",
                manufacturer: "Bambu Lab",
                buildVolumeX: 256,
                buildVolumeY: 256,
                buildVolumeZ: 256,
                compatibleMaterials: ["PLA", "PETG", "TPU", "PVA"],
                defaultPrintSettings: PrintSettings(
                    layerHeight: 0.2,
                    printSpeed: 250, // High speed printer
                    nozzleTemp: 220,
                    bedTemp: 65
                )
            )
        ]
    }

    func findProfile(byName name: String) -> PrinterProfile? {
        return profiles.first { $0.name == name }
    }
}
