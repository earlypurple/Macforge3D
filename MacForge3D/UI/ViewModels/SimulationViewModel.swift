import Foundation
import PythonKit
import SwiftUI

@MainActor
class SimulationViewModel: ObservableObject {
    // Paramètres de simulation
    @Published var analysisType: SimulationType = .mechanical
    @Published var material: Material = .pla
    @Published var fixedPoints: [(Double, Double, Double)] = [(0, 0, 0)]
    @Published var forces: [(Double, Double, Double, Double, Double, Double)] = [(0, 0, 0, 0, -9.81, 0)]
    
    // Paramètres thermiques
    @Published var initialTemp: Double = 200.0
    @Published var ambientTemp: Double = 25.0
    @Published var simulationTime: Double = 1800.0
    
    // État de l'interface
    @Published var isLoading = false
    @Published var hasResults = false
    @Published var errorMessage: String?
    
    // Python objects
    private var femAnalysis: PythonObject?
    private var thermalSim: PythonObject?
    private var resultAnalyzer: PythonObject?
    private(set) var resultViewer: PythonObject?
    
    init() {
        setupPython()
    }
    
    private func setupPython() {
        PythonManager.initialize()
        
        // Importer les modules Python
        let fem = Python.import("simulation.fem_analysis")
        let thermal = Python.import("simulation.thermal_sim")
        let analyzer = Python.import("simulation.result_analyzer")
        let viewer = Python.import("simulation.result_viewer")
        
        // Créer les instances
        self.resultAnalyzer = analyzer.ResultAnalyzer()
        self.resultViewer = viewer.ResultViewer()
        
        // Connecter le signal de nouvelle analyse
        if let rv = self.resultViewer {
            rv.analysisRequested.connect(self.runSimulation)
        }
    }
    
    func runSimulation() {
        Task {
            await MainActor.run {
                isLoading = true
                errorMessage = nil
            }
            
            do {
                let results = try await performSimulation()
                
                await MainActor.run {
                    if let viewer = self.resultViewer {
                        viewer.update_results(results)
                    }
                    hasResults = true
                    isLoading = false
                }
                
            } catch {
                await MainActor.run {
                    errorMessage = error.localizedDescription
                    isLoading = false
                }
            }
        }
    }
    
    private func performSimulation() async throws -> PythonObject {
        // Obtenir le chemin du modèle actif
        guard let modelPath = getActiveModelPath() else {
            throw SimulationError.noActiveModel
        }
        
        // Convertir les paramètres pour Python
        let pythonFixedPoints = fixedPoints.map { Python.tuple([$0.0, $0.1, $0.2]) }
        let pythonForces = forces.map { Python.tuple([$0.0, $0.1, $0.2, $0.3, $0.4, $0.5]) }
        
        var results: PythonObject?
        
        switch analysisType {
        case .mechanical:
            // Analyse mécanique
            results = try await performMechanicalAnalysis(
                modelPath: modelPath,
                fixedPoints: pythonFixedPoints,
                forces: pythonForces
            )
            
        case .thermal:
            // Simulation thermique
            results = try await performThermalSimulation(
                modelPath: modelPath
            )
            
        case .combined:
            // Analyse combinée
            let mechResults = try await performMechanicalAnalysis(
                modelPath: modelPath,
                fixedPoints: pythonFixedPoints,
                forces: pythonForces
            )
            
            let thermalResults = try await performThermalSimulation(
                modelPath: modelPath
            )
            
            // Fusionner les résultats
            results = mergePythonDicts(mechResults, thermalResults)
        }
        
        // Analyser les résultats
        guard let finalResults = results else {
            throw SimulationError.noResults
        }
        
        let materialProps = getMaterialProperties()
        return resultAnalyzer?.analyze_results(finalResults, material_props: materialProps) ?? finalResults
    }
    
    private func performMechanicalAnalysis(
        modelPath: String,
        fixedPoints: [PythonObject],
        forces: [PythonObject]
    ) async throws -> PythonObject {
        let fem = Python.import("simulation.fem_analysis")
        return fem.analyze_model(
            modelPath,
            material_name: material.rawValue,
            fixed_points: fixedPoints,
            forces: forces
        )
    }
    
    private func performThermalSimulation(
        modelPath: String
    ) async throws -> PythonObject {
        let thermal = Python.import("simulation.thermal_sim")
        return thermal.simulate_thermal(
            modelPath,
            material_name: material.rawValue,
            initial_temp: initialTemp,
            ambient_temp: ambientTemp,
            simulation_time: simulationTime
        )
    }
    
    private func getMaterialProperties() -> PythonObject {
        switch material {
        case .pla:
            return [
                "name": "PLA",
                "melting_point": 180,
                "glass_transition": 60,
                "yield_strength": 50e6
            ].__dict__
        case .abs:
            return [
                "name": "ABS",
                "melting_point": 230,
                "glass_transition": 105,
                "yield_strength": 40e6
            ].__dict__
        case .petg:
            return [
                "name": "PETG",
                "melting_point": 260,
                "glass_transition": 80,
                "yield_strength": 50e6
            ].__dict__
        }
    }
    
    func exportResults() {
        guard hasResults else { return }
        
        // Créer un panel de sauvegarde
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json, .text]
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        panel.title = "Exporter les Résultats"
        panel.message = "Choisissez où sauvegarder les résultats"
        panel.nameFieldLabel = "Nom du fichier:"
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                // Exporter dans le format choisi
                if url.pathExtension == "json" {
                    self.exportToJSON(url)
                } else {
                    self.exportToHTML(url)
                }
            }
        }
    }
    
    private func exportToJSON(_ url: URL) {
        guard let viewer = resultViewer else { return }
        
        let exporter = Python.import("exporters.simulation_export").SimulationExporter
        let _ = exporter.export_to_json(
            viewer.current_results,
            url.path,
            include_mesh: true
        )
    }
    
    private func exportToHTML(_ url: URL) {
        guard let viewer = resultViewer else { return }
        
        let exporter = Python.import("exporters.simulation_export").SimulationExporter
        let _ = exporter.export_report(
            viewer.current_results,
            url.path,
            title: "Rapport de Simulation MacForge3D"
        )
    }
}

// Erreurs spécifiques à la simulation
enum SimulationError: LocalizedError {
    case noActiveModel
    case noResults
    
    var errorDescription: String? {
        switch self {
        case .noActiveModel:
            return "Aucun modèle actif"
        case .noResults:
            return "La simulation n'a pas produit de résultats"
        }
    }
}

// Fonction utilitaire pour fusionner des dictionnaires Python
func mergePythonDicts(_ dict1: PythonObject, _ dict2: PythonObject) -> PythonObject {
    var merged = dict1.__dict__
    for (key, value) in dict2.__dict__.items() {
        merged[key] = value
    }
    return merged
}
