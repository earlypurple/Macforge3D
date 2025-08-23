import SwiftUI
import PythonKit

class Simulation3DViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var error: String?
    
    private var view3D: PythonObject?
    private let simulation = Python.import("simulation")
    
    init() {
        setupView3D()
    }
    
    private func setupView3D() {
        do {
            view3D = simulation.view3d.SimulationView3D()
        } catch {
            self.error = "Erreur d'initialisation 3D: \(error.localizedDescription)"
        }
    }
    
    func displayResults(
        meshData: [String: Any],
        results: [String: Any],
        displayType: String
    ) {
        guard let view3D = view3D else {
            error = "Vue 3D non initialisée"
            return
        }
        
        isLoading = true
        DispatchQueue.global(qos: .background).async {
            do {
                view3D.display_results(
                    mesh_data: Python.dict(meshData),
                    results: Python.dict(results),
                    display_type: displayType
                )
                DispatchQueue.main.async {
                    self.isLoading = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.error = error.localizedDescription
                    self.isLoading = false
                }
            }
        }
    }
    
    func setBackgroundColor(r: Float, g: Float, b: Float) {
        view3D?.set_background_color(r, g, b)
    }
    
    func resetView() {
        view3D?.reset_view()
    }
    
    func cleanup() {
        view3D?.cleanup()
    }
}

struct Simulation3DView: View {
    @StateObject private var viewModel = Simulation3DViewModel()
    let meshData: [String: Any]
    let results: [String: Any]
    @State private var displayType = "stress"
    @Environment(\.colorScheme) private var colorScheme
    
    var body: some View {
        VStack {
            // Zone de visualisation 3D (à implémenter avec NSViewRepresentable)
            Simulation3DRepresentable(
                viewModel: viewModel,
                meshData: meshData,
                results: results,
                displayType: displayType
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            
            // Contrôles
            VStack(spacing: 16) {
                Picker("Type d'affichage", selection: $displayType) {
                    Text("Contraintes").tag("stress")
                    Text("Déplacements").tag("displacement")
                    Text("Température").tag("temperature")
                }
                .pickerStyle(.segmented)
                .onChange(of: displayType) { newValue in
                    viewModel.displayResults(
                        meshData: meshData,
                        results: results,
                        displayType: newValue
                    )
                }
                
                HStack {
                    Button("Réinitialiser la vue") {
                        viewModel.resetView()
                    }
                    
                    Spacer()
                    
                    if viewModel.isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    
                    if let error = viewModel.error {
                        Text(error)
                            .foregroundColor(.red)
                    }
                }
            }
            .padding()
        }
        .onAppear {
            // Ajuster la couleur de fond selon le thème
            if colorScheme == .dark {
                viewModel.setBackgroundColor(r: 0.1, g: 0.1, b: 0.1)
            } else {
                viewModel.setBackgroundColor(r: 1.0, g: 1.0, b: 1.0)
            }
            
            // Afficher les résultats initiaux
            viewModel.displayResults(
                meshData: meshData,
                results: results,
                displayType: displayType
            )
        }
        .onDisappear {
            viewModel.cleanup()
        }
    }
}

// Wrapper NSViewRepresentable pour la vue VTK/PyQt
struct Simulation3DRepresentable: NSViewRepresentable {
    let viewModel: Simulation3DViewModel
    let meshData: [String: Any]
    let results: [String: Any]
    let displayType: String
    
    func makeNSView(context: Context) -> NSView {
        guard let pyQtWidget = viewModel.view3D else {
            return NSView()
        }
        
        // Créer la vue native à partir du widget PyQt
        let view = PyQtViewBridge.createView(withPyQtWidget: pyQtWidget)
        
        // Afficher les résultats initiaux
        viewModel.displayResults(
            meshData: meshData,
            results: results,
            displayType: displayType
        )
        
        return view ?? NSView()
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        // Les mises à jour sont gérées par le ViewModel
    }
    
    static func dismantleNSView(_ nsView: NSView, coordinator: ()) {
        PyQtViewBridge.destroyView(nsView)
    }
}
