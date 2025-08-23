import Foundation

// Bambu Lab specific 3D model exporter with optimized features for Bambu Studio
class BambuLabExporter {
    
    enum ExportError: Error {
        case profileNotSupported
        case materialNotCompatible
        case invalidModel
        case exportFailed(String)
    }
    
    struct BambuExportOptions {
        var printerProfile: PrinterProfile
        var useAMS: Bool = true
        var amsSlots: [Int: String] = [:] // Slot number to material mapping
        var enableAIInspection: Bool = true
        var enableLidar: Bool = true
        var optimizeForSpeed: Bool = false
        var enableSupportPainting: Bool = false
        var customGCode: String = ""
    }
    
    // Export model optimized for Bambu Lab printers
    func exportForBambuLab(
        model: Shape3D,
        options: BambuExportOptions,
        outputURL: URL
    ) throws {
        
        // Validate Bambu Lab compatibility
        guard options.printerProfile.manufacturer == "Bambu Lab" else {
            throw ExportError.profileNotSupported
        }
        
        // Generate Bambu-specific metadata
        let metadata = generateBambuMetadata(options: options)
        
        // Optimize model for Bambu Lab printing
        let optimizedModel = optimizeModelForBambu(model, options: options)
        
        // Export with Bambu-specific settings
        try exportOptimizedModel(optimizedModel, metadata: metadata, to: outputURL)
    }
    
    // Generate Bambu Studio compatible metadata
    private func generateBambuMetadata(options: BambuExportOptions) -> [String: Any] {
        var metadata: [String: Any] = [:]
        
        // Printer configuration
        metadata["printer_model"] = options.printerProfile.name
        metadata["printer_technology"] = "FFF"
        metadata["machine_max_feedrate_x"] = options.printerProfile.maxPrintSpeed
        metadata["machine_max_feedrate_y"] = options.printerProfile.maxPrintSpeed
        
        // AMS configuration
        if options.useAMS && options.printerProfile.hasAMS {
            metadata["enable_ams"] = true
            metadata["ams_slots"] = options.amsSlots
            metadata["ams_slot_count"] = options.printerProfile.maxAMSSlots
        }
        
        // AI features
        if options.printerProfile.hasAIError {
            metadata["enable_ai_monitoring"] = options.enableAIInspection
            metadata["enable_ai_first_layer"] = options.enableLidar
        }
        
        // Build volume
        metadata["bed_size_x"] = options.printerProfile.buildVolumeX
        metadata["bed_size_y"] = options.printerProfile.buildVolumeY
        metadata["bed_size_z"] = options.printerProfile.buildVolumeZ
        
        // Advanced features
        metadata["adaptive_layer_height"] = options.printerProfile.hasAIError
        metadata["flow_calibration"] = true
        metadata["pressure_advance"] = true
        
        return metadata
    }
    
    // Optimize 3D model specifically for Bambu Lab printers
    private func optimizeModelForBambu(_ model: Shape3D, options: BambuExportOptions) -> Shape3D {
        var optimizedModel = model
        
        // Apply Bambu-specific optimizations
        if options.optimizeForSpeed {
            optimizedModel = optimizeForHighSpeed(optimizedModel)
        }
        
        // Optimize for AMS if enabled
        if options.useAMS {
            optimizedModel = optimizeForAMS(optimizedModel, slots: options.amsSlots)
        }
        
        // Apply AI-based optimizations
        if options.printerProfile.hasAIError {
            optimizedModel = applyAIOptimizations(optimizedModel)
        }
        
        return optimizedModel
    }
    
    // Optimize model for high-speed printing on Bambu Lab printers
    private func optimizeForHighSpeed(_ model: Shape3D) -> Shape3D {
        // Implementation would include:
        // - Reduce overhangs that cause speed reduction
        // - Optimize infill patterns for high-speed printing
        // - Adjust geometry for better bridging
        return model
    }
    
    // Optimize model for AMS multi-material printing
    private func optimizeForAMS(_ model: Shape3D, slots: [Int: String]) -> Shape3D {
        // Implementation would include:
        // - Color separation for multi-material
        // - Purge tower optimization
        // - Material transition planning
        return model
    }
    
    // Apply AI-based model optimizations
    private func applyAIOptimizations(_ model: Shape3D) -> Shape3D {
        // Implementation would include:
        // - Detect and fix potential print failures
        // - Optimize support placement
        // - Adjust orientation for best quality
        return model
    }
    
    // Export the optimized model with metadata
    private func exportOptimizedModel(
        _ model: Shape3D,
        metadata: [String: Any],
        to url: URL
    ) throws {
        // Implementation would create a .3mf file with:
        // - Optimized mesh data
        // - Bambu-specific metadata
        // - Print settings embedded
        // - Support structures if needed
        
        // For now, export as STL with metadata file
        let stlExporter = STLExporter()
        try stlExporter.export(shape: model, to: url)
        
        // Save metadata as JSON
        let metadataURL = url.appendingPathExtension("bambu.json")
        let jsonData = try JSONSerialization.data(withJSONObject: metadata, options: .prettyPrinted)
        try jsonData.write(to: metadataURL)
    }
    
    // Generate Bambu Studio compatible G-code header
    func generateBambuGCodeHeader(options: BambuExportOptions) -> String {
        var header = """
        ; Generated by MacForge3D for Bambu Lab
        ; Printer: \(options.printerProfile.name)
        ; Build volume: \(options.printerProfile.buildVolumeX)x\(options.printerProfile.buildVolumeY)x\(options.printerProfile.buildVolumeZ)mm
        """
        
        if options.useAMS {
            header += "\n; AMS enabled with \(options.printerProfile.maxAMSSlots) slots"
            for (slot, material) in options.amsSlots {
                header += "\n; AMS Slot \(slot): \(material)"
            }
        }
        
        if options.printerProfile.hasLidar {
            header += "\n; Lidar first layer inspection: \(options.enableLidar ? "enabled" : "disabled")"
        }
        
        if options.printerProfile.hasAIError {
            header += "\n; AI failure detection: \(options.enableAIInspection ? "enabled" : "disabled")"
        }
        
        header += "\n; MacForge3D optimizations applied\n"
        
        return header
    }
}

// Extension for Bambu Lab specific material profiles
extension BambuLabExporter {
    
    struct BambuMaterialProfile {
        let name: String
        let nozzleTemp: Int
        let bedTemp: Int
        let chamberTemp: Int?
        let printSpeed: Int
        let retraction: Float
        let pressureAdvance: Float
        let flowRate: Float
    }
    
    // Get optimal material settings for Bambu Lab printers
    func getBambuMaterialProfile(material: String, printer: PrinterProfile) -> BambuMaterialProfile? {
        switch material.uppercased() {
        case "PLA":
            return BambuMaterialProfile(
                name: "PLA Basic",
                nozzleTemp: 220,
                bedTemp: 65,
                chamberTemp: nil,
                printSpeed: printer.maxPrintSpeed,
                retraction: 0.8,
                pressureAdvance: 0.02,
                flowRate: 0.98
            )
        case "ABS":
            return BambuMaterialProfile(
                name: "ABS",
                nozzleTemp: 260,
                bedTemp: 90,
                chamberTemp: printer.hasHeatedChamber ? 40 : nil,
                printSpeed: min(printer.maxPrintSpeed, 300),
                retraction: 0.5,
                pressureAdvance: 0.025,
                flowRate: 0.96
            )
        case "PETG":
            return BambuMaterialProfile(
                name: "PETG Basic",
                nozzleTemp: 250,
                bedTemp: 80,
                chamberTemp: nil,
                printSpeed: min(printer.maxPrintSpeed, 250),
                retraction: 1.2,
                pressureAdvance: 0.03,
                flowRate: 1.0
            )
        case "TPU":
            return BambuMaterialProfile(
                name: "TPU 95A",
                nozzleTemp: 230,
                bedTemp: 50,
                chamberTemp: nil,
                printSpeed: min(printer.maxPrintSpeed, 30),
                retraction: 0.0,
                pressureAdvance: 0.0,
                flowRate: 1.05
            )
        default:
            return nil
        }
    }
}