import SwiftUI
import PythonKit

class BatchSimulationViewModel: ObservableObject {
    @Published var batches: [SimulationBatch] = []
    @Published var isLoading = false
    @Published var error: String?
    
    private let batchManager: PythonObject
    
    init() {
        // Initialiser le gestionnaire Python
        let simulation = Python.import("simulation")
        batchManager = simulation.batch_manager.BatchSimulationManager(
            output_dir: NSTemporaryDirectory(),
            max_workers: 4,
            progress_callback: progressCallback
        )
        
        // Charger les lots existants
        loadBatches()
    }
    
    func loadBatches() {
        DispatchQueue.global(qos: .background).async {
            self.isLoading = true
            do {
                let pythonBatches = self.batchManager.list_batches()
                DispatchQueue.main.async {
                    self.batches = pythonBatches.map { self.convertBatch($0) }
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
    
    func createBatch(
        name: String,
        description: String,
        type: String,
        parameters: [[String: Any]]
    ) {
        DispatchQueue.global(qos: .background).async {
            do {
                let batch = self.batchManager.create_batch(
                    name: name,
                    description: description,
                    sim_type: type,
                    parameters: parameters
                )
                DispatchQueue.main.async {
                    self.batches.append(self.convertBatch(batch))
                }
            } catch {
                DispatchQueue.main.async {
                    self.error = error.localizedDescription
                }
            }
        }
    }
    
    func runBatch(_ name: String) {
        DispatchQueue.global(qos: .background).async {
            do {
                let results = self.batchManager.run_batch(name)
                DispatchQueue.main.async {
                    if let index = self.batches.firstIndex(where: { $0.name == name }) {
                        self.batches[index].results = self.convertResults(results)
                        self.batches[index].status = "completed"
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.error = error.localizedDescription
                    if let index = self.batches.firstIndex(where: { $0.name == name }) {
                        self.batches[index].status = "failed"
                        self.batches[index].errorMessage = error.localizedDescription
                    }
                }
            }
        }
    }
    
    func deleteBatch(_ name: String) {
        do {
            try self.batchManager.delete_batch(name)
            DispatchQueue.main.async {
                self.batches.removeAll { $0.name == name }
            }
        } catch {
            DispatchQueue.main.async {
                self.error = error.localizedDescription
            }
        }
    }
    
    private func progressCallback(_ task: String, _ progress: Double) {
        DispatchQueue.main.async {
            // Mettre à jour la progression dans l'interface
            if let index = self.batches.firstIndex(where: { $0.status == "running" }) {
                self.batches[index].progress = progress
            }
        }
    }
    
    private func convertBatch(_ pythonBatch: PythonObject) -> SimulationBatch {
        SimulationBatch(
            name: String(pythonBatch.name) ?? "",
            description: String(pythonBatch.description) ?? "",
            type: String(pythonBatch.sim_type) ?? "",
            parameters: convertParameters(pythonBatch.parameters),
            createdAt: Date(),
            results: convertResults(pythonBatch.results),
            status: String(pythonBatch.status) ?? "pending",
            errorMessage: pythonBatch.error_message.isNone ? nil : String(pythonBatch.error_message),
            progress: 0.0
        )
    }
    
    private func convertParameters(_ pythonParams: PythonObject) -> [[String: Any]] {
        var params: [[String: Any]] = []
        for param in pythonParams {
            if let dict = param as? [String: Any] {
                params.append(dict)
            }
        }
        return params
    }
    
    private func convertResults(_ pythonResults: PythonObject) -> SimulationResults? {
        guard !pythonResults.isNone else { return nil }
        
        let simulations = pythonResults.simulations.map { sim -> [String: Any] in
            Dictionary(uniqueKeysWithValues: zip(
                sim.keys().map { String($0) ?? "" },
                sim.values().map { convertPythonValue($0) }
            ))
        }
        
        let summary = Dictionary(uniqueKeysWithValues: zip(
            pythonResults.summary.keys().map { String($0) ?? "" },
            pythonResults.summary.values().map { convertPythonValue($0) }
        ))
        
        return SimulationResults(
            simulations: simulations,
            summary: summary
        )
    }
    
    private func convertPythonValue(_ value: PythonObject) -> Any {
        if let int = Int(exactly: value) {
            return int
        } else if let double = Double(exactly: value) {
            return double
        } else if let bool = Bool(exactly: value) {
            return bool
        } else if let string = String(value) {
            return string
        } else if let array = Array<Any>(value) {
            return array
        } else if let dict = Dictionary<String, Any>(value) {
            return dict
        }
        return String(describing: value)
    }
}

struct SimulationBatch: Identifiable {
    let id = UUID()
    let name: String
    let description: String
    let type: String
    let parameters: [[String: Any]]
    let createdAt: Date
    var results: SimulationResults?
    var status: String
    var errorMessage: String?
    var progress: Double
}

struct SimulationResults {
    let simulations: [[String: Any]]
    let summary: [String: Any]
}

struct BatchSimulationView: View {
    @StateObject private var viewModel = BatchSimulationViewModel()
    @State private var showingNewBatchSheet = false
    @State private var selectedBatch: SimulationBatch?
    
    var body: some View {
        VStack {
            List(viewModel.batches) { batch in
                BatchRow(batch: batch)
                    .onTapGesture {
                        selectedBatch = batch
                    }
            }
            .refreshable {
                viewModel.loadBatches()
            }
            
            HStack {
                Button("Nouveau lot") {
                    showingNewBatchSheet = true
                }
                
                Spacer()
                
                if let error = viewModel.error {
                    Text(error)
                        .foregroundColor(.red)
                }
            }
            .padding()
        }
        .sheet(isPresented: $showingNewBatchSheet) {
            NewBatchView(viewModel: viewModel)
        }
        .sheet(item: $selectedBatch) { batch in
            BatchDetailView(batch: batch, viewModel: viewModel)
        }
    }
}

struct BatchRow: View {
    let batch: SimulationBatch
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(batch.name)
                .font(.headline)
            
            Text(batch.description)
                .font(.subheadline)
                .foregroundColor(.secondary)
            
            HStack {
                Text("Type: \(batch.type)")
                Spacer()
                StatusBadge(status: batch.status)
            }
            
            if batch.status == "running" {
                ProgressView(value: batch.progress, total: 100)
                    .progressViewStyle(.linear)
            }
        }
        .padding(.vertical, 8)
    }
}

struct StatusBadge: View {
    let status: String
    
    var color: Color {
        switch status {
        case "pending":
            return .gray
        case "running":
            return .blue
        case "completed":
            return .green
        case "failed":
            return .red
        default:
            return .gray
        }
    }
    
    var body: some View {
        Text(status)
            .font(.caption)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(color.opacity(0.2))
            .foregroundColor(color)
            .cornerRadius(8)
    }
}

struct NewBatchView: View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var viewModel: BatchSimulationViewModel
    
    @State private var name = ""
    @State private var description = ""
    @State private var type = "fem"
    @State private var parameters: [[String: Any]] = []
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Informations")) {
                    TextField("Nom", text: $name)
                    TextField("Description", text: $description)
                    Picker("Type", selection: $type) {
                        Text("FEM").tag("fem")
                        Text("Thermique").tag("thermal")
                    }
                }
                
                Section(header: Text("Paramètres")) {
                    ParameterList(parameters: $parameters)
                }
            }
            .navigationTitle("Nouveau lot")
            .navigationBarItems(
                leading: Button("Annuler") {
                    dismiss()
                },
                trailing: Button("Créer") {
                    viewModel.createBatch(
                        name: name,
                        description: description,
                        type: type,
                        parameters: parameters
                    )
                    dismiss()
                }
                .disabled(name.isEmpty)
            )
        }
    }
}

struct ParameterList: View {
    @Binding var parameters: [[String: Any]]
    
    var body: some View {
        ForEach(parameters.indices, id: \.self) { index in
            ParameterRow(parameter: binding(for: index))
        }
        
        Button("Ajouter un paramètre") {
            parameters.append([:])
        }
    }
    
    private func binding(for index: Int) -> Binding<[String: Any]> {
        Binding(
            get: { parameters[index] },
            set: { parameters[index] = $0 }
        )
    }
}

struct ParameterRow: View {
    @Binding var parameter: [String: Any]
    
    var body: some View {
        // Interface pour éditer les paramètres
        // À adapter selon les besoins spécifiques
        EmptyView()
    }
}

struct BatchDetailView: View {
    let batch: SimulationBatch
    @ObservedObject var viewModel: BatchSimulationViewModel
    @Environment(\.dismiss) var dismiss
    
    var body: some View {
        NavigationView {
            List {
                Section(header: Text("Informations")) {
                    DetailRow(label: "Nom", value: batch.name)
                    DetailRow(label: "Description", value: batch.description)
                    DetailRow(label: "Type", value: batch.type)
                    DetailRow(label: "Statut", value: batch.status)
                    if let error = batch.errorMessage {
                        DetailRow(label: "Erreur", value: error)
                            .foregroundColor(.red)
                    }
                }
                
                if let results = batch.results {
                    Section(header: Text("Résumé")) {
                        ForEach(Array(results.summary.keys.sorted()), id: \.self) { key in
                            DetailRow(
                                label: key,
                                value: String(describing: results.summary[key] ?? "")
                            )
                        }
                    }
                    
                    Section(header: Text("Simulations")) {
                        ForEach(results.simulations.indices, id: \.self) { index in
                            NavigationLink {
                                SimulationDetailView(
                                    simulation: results.simulations[index]
                                )
                            } label: {
                                Text("Simulation \(index + 1)")
                            }
                        }
                    }
                }
            }
            .navigationTitle("Détails du lot")
            .navigationBarItems(
                leading: Button("Fermer") {
                    dismiss()
                },
                trailing: HStack {
                    if batch.status == "pending" {
                        Button("Lancer") {
                            viewModel.runBatch(batch.name)
                        }
                    }
                    
                    Button("Supprimer") {
                        viewModel.deleteBatch(batch.name)
                        dismiss()
                    }
                    .foregroundColor(.red)
                }
            )
        }
    }
}

struct DetailRow: View {
    let label: String
    let value: String
    
    var body: some View {
        HStack {
            Text(label)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
        }
    }
}

struct SimulationDetailView: View {
    let simulation: [String: Any]
    
    var body: some View {
        List {
            ForEach(Array(simulation.keys.sorted()), id: \.self) { key in
                DetailRow(
                    label: key,
                    value: String(describing: simulation[key] ?? "")
                )
            }
        }
        .navigationTitle("Résultats de simulation")
    }
}
