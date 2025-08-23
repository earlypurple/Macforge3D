import SwiftUI

struct CloudView: View {
    @State private var selectedTab = 0
    @State private var showingSettings = false
    @StateObject private var clusterState = CloudClusterState()
    
    var body: some View {
        VStack {
            HStack {
                Spacer()
                Button(action: { showingSettings.toggle() }) {
                    Image(systemName: "gear")
                }
                .sheet(isPresented: $showingSettings) {
                    CloudConfigurationView()
                }
            }
            .padding(.horizontal)
            
            TabView(selection: $selectedTab) {
                // Stockage
                CloudStorageView()
                    .tabItem {
                        Image(systemName: "folder.badge.gearshape")
                        Text("Stockage")
                    }
                    .tag(0)
                
                // Rendu
                CloudRenderView(clusterState: clusterState)
                    .tabItem {
                        Image(systemName: "wand.and.rays")
                        Text("Rendu")
                    }
                    .tag(1)
                
                // Monitoring
                CloudMonitorView(clusterState: clusterState)
                    .tabItem {
                        Image(systemName: "chart.xyaxis.line")
                        Text("Monitoring")
                    }
                    .tag(2)
            }
        }
    }
}

class CloudClusterState: ObservableObject {
    @Published var isClusterActive = false
    @Published var activeNodes: Int = 0
    @Published var totalNodes: Int = 0
    @Published var currentJobs: [String: RenderJob] = [:]
    
    func activateCluster(nodes: Int) async throws {
        try await CloudRenderManager.shared.setupRenderCluster(
            numNodes: nodes,
            instanceType: "g4dn.xlarge"  // Instance GPU par défaut
        )
        
        await MainActor.run {
            isClusterActive = true
            activeNodes = nodes
            totalNodes = nodes
        }
    }
    
    func submitRenderJob(job: RenderJob) async throws {
        let config = RenderConfig(
            width: job.width,
            height: job.height,
            samples: job.samples,
            device: "cuda"
        )
        
        let result = try await CloudRenderManager.shared.renderScene(
            scene: job.scenePath,
            outputPath: job.outputPath,
            config: config
        )
        
        await MainActor.run {
            var job = job
            job.jobId = result.jobId
            job.status = .running
            currentJobs[result.jobId] = job
        }
    }
}

struct RenderJob: Identifiable {
    let id = UUID()
    var jobId: String?
    let name: String
    let scenePath: String
    let outputPath: String
    let width: Int
    let height: Int
    let samples: Int
    var status: JobStatus = .pending
    var progress: Double = 0.0
    
    enum JobStatus {
        case pending
        case running
        case completed
        case failed
    }
}

struct CloudRenderView: View {
    @ObservedObject var clusterState: CloudClusterState
    @State private var showingJobDialog = false
    @State private var newJobName = ""
    @State private var selectedWidth = 1920
    @State private var selectedHeight = 1080
    @State private var selectedSamples = 1000
    
    var body: some View {
        VStack {
            // En-tête
            HStack {
                Text("Rendu Distribué")
                    .font(.title)
                Spacer()
                Button(action: { showingJobDialog.toggle() }) {
                    Image(systemName: "plus")
                }
                .disabled(!clusterState.isClusterActive)
            }
            .padding()
            
            // État du cluster
            if !clusterState.isClusterActive {
                VStack {
                    Text("Cluster inactif")
                        .font(.headline)
                    Button("Activer le cluster (4 nœuds)") {
                        Task {
                            try await clusterState.activateCluster(nodes: 4)
                        }
                    }
                }
                .padding()
            } else {
                Text("Cluster actif: \(clusterState.activeNodes)/\(clusterState.totalNodes) nœuds")
                    .font(.headline)
                    .padding()
            }
            
            // Liste des jobs
            List(clusterState.currentJobs.values.sorted { $0.name < $1.name }) { job in
                RenderJobRow(job: job)
            }
        }
        .sheet(isPresented: $showingJobDialog) {
            NewRenderJobView(
                isPresented: $showingJobDialog,
                clusterState: clusterState
            )
        }
    }
}

struct RenderJobRow: View {
    let job: RenderJob
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(job.name)
                .font(.headline)
            
            HStack {
                Text("\(job.width)x\(job.height) • \(job.samples) samples")
                    .font(.caption)
                Spacer()
                statusView
            }
            
            if job.status == .running {
                ProgressView(value: job.progress)
                    .progressViewStyle(.linear)
            }
        }
        .padding(.vertical, 4)
    }
    
    var statusView: some View {
        HStack {
            Circle()
                .fill(statusColor)
                .frame(width: 8, height: 8)
            Text(statusText)
                .font(.caption)
        }
    }
    
    var statusColor: Color {
        switch job.status {
        case .pending:
            return .orange
        case .running:
            return .blue
        case .completed:
            return .green
        case .failed:
            return .red
        }
    }
    
    var statusText: String {
        switch job.status {
        case .pending:
            return "En attente"
        case .running:
            return "En cours"
        case .completed:
            return "Terminé"
        case .failed:
            return "Erreur"
        }
    }
}

struct NewRenderJobView: View {
    @Binding var isPresented: Bool
    @ObservedObject var clusterState: CloudClusterState
    @State private var jobName = ""
    @State private var width = 1920
    @State private var height = 1080
    @State private var samples = 1000
    @State private var selectedScene: String?
    @State private var outputPath: String?
    
    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Configuration")) {
                    TextField("Nom du job", text: $jobName)
                    
                    Stepper(value: $width, in: 640...7680) {
                        Text("Largeur: \(width)px")
                    }
                    
                    Stepper(value: $height, in: 480...4320) {
                        Text("Hauteur: \(height)px")
                    }
                    
                    Stepper(value: $samples, in: 100...10000, step: 100) {
                        Text("Échantillons: \(samples)")
                    }
                }
                
                Section(header: Text("Fichiers")) {
                    Button("Sélectionner la scène...") {
                        selectScene()
                    }
                    if let scene = selectedScene {
                        Text(scene)
                            .font(.caption)
                    }
                    
                    Button("Sélectionner la sortie...") {
                        selectOutput()
                    }
                    if let output = outputPath {
                        Text(output)
                            .font(.caption)
                    }
                }
            }
            .navigationTitle("Nouveau Job de Rendu")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Annuler") {
                        isPresented = false
                    }
                }
                
                ToolbarItem(placement: .confirmationAction) {
                    Button("Créer") {
                        createJob()
                    }
                    .disabled(!isValid)
                }
            }
        }
    }
    
    private var isValid: Bool {
        !jobName.isEmpty && selectedScene != nil && outputPath != nil
    }
    
    private func selectScene() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.model]
        
        if panel.runModal() == .OK {
            selectedScene = panel.url?.path
        }
    }
    
    private func selectOutput() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        panel.nameFieldStringValue = "\(jobName).png"
        
        if panel.runModal() == .OK {
            outputPath = panel.url?.path
        }
    }
    
    private func createJob() {
        guard let scenePath = selectedScene,
              let outputPath = outputPath else {
            return
        }
        
        let job = RenderJob(
            name: jobName,
            scenePath: scenePath,
            outputPath: outputPath,
            width: width,
            height: height,
            samples: samples
        )
        
        Task {
            try await clusterState.submitRenderJob(job: job)
            await MainActor.run {
                isPresented = false
            }
        }
    }
}

struct CloudMonitorView: View {
    @ObservedObject var clusterState: CloudClusterState
    
    var body: some View {
        VStack {
            Text("Monitoring")
                .font(.title)
                .padding()
            
            if clusterState.isClusterActive {
                // Graphique d'utilisation du cluster
                ClusterUsageGraph(clusterState: clusterState)
                    .frame(height: 200)
                    .padding()
                
                // Stats des jobs
                JobStatsView(clusterState: clusterState)
                    .padding()
                
                // Liste des jobs actifs
                ActiveJobsList(clusterState: clusterState)
            } else {
                Text("Le cluster n'est pas actif")
                    .font(.headline)
                    .foregroundColor(.secondary)
            }
        }
    }
}

struct ClusterUsageGraph: View {
    @ObservedObject var clusterState: CloudClusterState
    
    var body: some View {
        // Placeholder pour un vrai graphique
        RoundedRectangle(cornerRadius: 8)
            .stroke(Color.secondary, lineWidth: 2)
            .overlay(
                Text("Utilisation du cluster")
                    .foregroundColor(.secondary)
            )
    }
}

struct JobStatsView: View {
    @ObservedObject var clusterState: CloudClusterState
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Statistiques")
                .font(.headline)
            
            HStack {
                StatBox(
                    title: "Jobs actifs",
                    value: "\(activeJobs)"
                )
                
                StatBox(
                    title: "Jobs en attente",
                    value: "\(pendingJobs)"
                )
                
                StatBox(
                    title: "Jobs terminés",
                    value: "\(completedJobs)"
                )
            }
        }
    }
    
    private var activeJobs: Int {
        clusterState.currentJobs.values.filter { $0.status == .running }.count
    }
    
    private var pendingJobs: Int {
        clusterState.currentJobs.values.filter { $0.status == .pending }.count
    }
    
    private var completedJobs: Int {
        clusterState.currentJobs.values.filter { $0.status == .completed }.count
    }
}

struct StatBox: View {
    let title: String
    let value: String
    
    var body: some View {
        VStack {
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Text(value)
                .font(.title2)
                .bold()
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(Color.secondary.opacity(0.1))
        .cornerRadius(8)
    }
}

struct ActiveJobsList: View {
    @ObservedObject var clusterState: CloudClusterState
    
    var body: some View {
        VStack(alignment: .leading) {
            Text("Jobs Actifs")
                .font(.headline)
                .padding(.bottom)
            
            List(activeJobs) { job in
                ActiveJobRow(job: job)
            }
        }
    }
    
    private var activeJobs: [RenderJob] {
        clusterState.currentJobs.values
            .filter { $0.status == .running }
            .sorted { $0.name < $1.name }
    }
}

struct ActiveJobRow: View {
    let job: RenderJob
    
    var body: some View {
        VStack(alignment: .leading) {
            Text(job.name)
                .font(.headline)
            
            Text("\(job.width)x\(job.height) • \(Int(job.progress * 100))%")
                .font(.caption)
            
            ProgressView(value: job.progress)
                .progressViewStyle(.linear)
        }
        .padding(.vertical, 4)
    }
}
