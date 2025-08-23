import SwiftUI
import PythonKit

struct SimulationView: View {
    @StateObject private var viewModel = SimulationViewModel()
    
    var body: some View {
        HSplitView {
            // Panneau de contrôle à gauche
            VStack {
                // Section des paramètres de simulation
                GroupBox("Paramètres de Simulation") {
                    Form {
                        Picker("Type d'Analyse", selection: $viewModel.analysisType) {
                            ForEach(SimulationType.allCases) { type in
                                Text(type.rawValue).tag(type)
                            }
                        }
                        
                        Picker("Matériau", selection: $viewModel.material) {
                            ForEach(Material.allCases) { material in
                                Text(material.rawValue).tag(material)
                            }
                        }
                        
                        // Conditions aux limites
                        GroupBox("Conditions aux Limites") {
                            FixedPointsEditor(points: $viewModel.fixedPoints)
                            ForcesEditor(forces: $viewModel.forces)
                        }
                        
                        // Paramètres thermiques
                        if viewModel.analysisType == .thermal {
                            GroupBox("Paramètres Thermiques") {
                                LabeledContent("Température Initiale") {
                                    TextField("°C", value: $viewModel.initialTemp, format: .number)
                                        .frame(width: 80)
                                }
                                LabeledContent("Température Ambiante") {
                                    TextField("°C", value: $viewModel.ambientTemp, format: .number)
                                        .frame(width: 80)
                                }
                                LabeledContent("Durée de Simulation") {
                                    TextField("s", value: $viewModel.simulationTime, format: .number)
                                        .frame(width: 80)
                                }
                            }
                        }
                    }
                    .padding()
                }
                
                // Boutons de contrôle
                HStack {
                    Button(action: viewModel.runSimulation) {
                        Label("Lancer la Simulation", systemImage: "play.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    
                    Button(action: viewModel.exportResults) {
                        Label("Exporter", systemImage: "square.and.arrow.up")
                    }
                    .disabled(!viewModel.hasResults)
                }
                .padding()
                
                Spacer()
            }
            .frame(minWidth: 300, maxWidth: 400)
            
            // Zone de visualisation à droite
            ZStack {
                if viewModel.isLoading {
                    ProgressView("Simulation en cours...")
                } else {
                    ResultViewerWrapper(resultViewer: viewModel.resultViewer)
                }
            }
            .frame(minWidth: 500)
        }
    }
}

// Types pour la simulation
enum SimulationType: String, CaseIterable, Identifiable {
    case mechanical = "Mécanique"
    case thermal = "Thermique"
    case combined = "Combinée"
    
    var id: String { rawValue }
}

enum Material: String, CaseIterable, Identifiable {
    case pla = "PLA"
    case abs = "ABS"
    case petg = "PETG"
    
    var id: String { rawValue }
}

// Éditeurs de conditions aux limites
struct FixedPointsEditor: View {
    @Binding var points: [(Double, Double, Double)]
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Points Fixes")
                .font(.subheadline)
            
            ForEach(points.indices, id: \.self) { index in
                HStack {
                    TextField("X", value: $points[index].0, format: .number)
                    TextField("Y", value: $points[index].1, format: .number)
                    TextField("Z", value: $points[index].2, format: .number)
                    
                    Button(action: { removePoint(at: index) }) {
                        Image(systemName: "minus.circle.fill")
                            .foregroundColor(.red)
                    }
                }
            }
            
            Button(action: addPoint) {
                Label("Ajouter un point", systemImage: "plus.circle")
            }
            .buttonStyle(.borderless)
        }
    }
    
    private func addPoint() {
        points.append((0, 0, 0))
    }
    
    private func removePoint(at index: Int) {
        points.remove(at: index)
    }
}

struct ForcesEditor: View {
    @Binding var forces: [(Double, Double, Double, Double, Double, Double)]
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Forces Appliquées")
                .font(.subheadline)
            
            ForEach(forces.indices, id: \.self) { index in
                GroupBox {
                    VStack {
                        HStack {
                            Text("Position:")
                            TextField("X", value: $forces[index].0, format: .number)
                            TextField("Y", value: $forces[index].1, format: .number)
                            TextField("Z", value: $forces[index].2, format: .number)
                        }
                        HStack {
                            Text("Force:")
                            TextField("Fx", value: $forces[index].3, format: .number)
                            TextField("Fy", value: $forces[index].4, format: .number)
                            TextField("Fz", value: $forces[index].5, format: .number)
                        }
                    }
                    
                    Button(action: { removeForce(at: index) }) {
                        Label("Supprimer", systemImage: "minus.circle")
                            .foregroundColor(.red)
                    }
                    .buttonStyle(.borderless)
                }
            }
            
            Button(action: addForce) {
                Label("Ajouter une force", systemImage: "plus.circle")
            }
            .buttonStyle(.borderless)
        }
    }
    
    private func addForce() {
        forces.append((0, 0, 0, 0, 0, 0))
    }
    
    private func removeForce(at index: Int) {
        forces.remove(at: index)
    }
}

// Wrapper SwiftUI pour le ResultViewer PyQt
struct ResultViewerWrapper: NSViewRepresentable {
    let resultViewer: PythonObject
    
    func makeNSView(context: Context) -> NSView {
        let view = NSView()
        if let pyView = resultViewer.native {
            view.addSubview(pyView.ptr.takeUnretainedValue() as! NSView)
        }
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {}
}