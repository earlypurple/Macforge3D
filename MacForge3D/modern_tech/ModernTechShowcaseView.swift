import SwiftUI

/// Modern Technology Showcase View for MacForge3D
/// Displays and allows interaction with all the latest integrated technologies
struct ModernTechShowcaseView: View {
    @StateObject private var modernTech = ModernTechBridge()
    @State private var selectedTab = 0
    @State private var generatePrompt = ""
    @State private var isGenerating = false
    @State private var generationResult: ModernTechBridge.ModelGenerationResult?
    @State private var showingAdvancedOptions = false
    @State private var selectedAIModel = "gpt4v_3d"
    @State private var enableNFT = false
    @State private var enableCollaboration = false
    @State private var enableWebXR = false
    
    let aiModels = [
        ("gpt4v_3d", "GPT-4V 3D Generator", "brain.head.profile"),
        ("claude3_sculptor", "Claude-3 Sculptor", "paintbrush.pointed.fill"),
        ("gemini_pro_3d", "Gemini Pro 3D", "atom"),
        ("dalle3_3d", "DALL-E 3 to 3D", "photo.artframe"),
        ("neural_radiance_fields", "NeRF Pro", "cube.transparent"),
        ("gaussian_splatting", "Gaussian Splatting", "circle.grid.3x3.fill")
    ]
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header with initialization status
                headerView
                
                // Tab view for different sections
                TabView(selection: $selectedTab) {
                    // AI Generation Tab
                    aiGenerationView
                        .tabItem {
                            Image(systemName: "brain.head.profile")
                            Text("AI Generation")
                        }
                        .tag(0)
                    
                    // Features Overview Tab
                    featuresOverviewView
                        .tabItem {
                            Image(systemName: "star.circle.fill")
                            Text("Features")
                        }
                        .tag(1)
                    
                    // Performance Dashboard Tab
                    performanceDashboardView
                        .tabItem {
                            Image(systemName: "chart.line.uptrend.xyaxis")
                            Text("Performance")
                        }
                        .tag(2)
                    
                    // Technology Status Tab
                    technologyStatusView
                        .tabItem {
                            Image(systemName: "gear.circle.fill")
                            Text("Status")
                        }
                        .tag(3)
                }
            }
            .navigationTitle("MacForge3D Modern Tech")
            .toolbar {
                ToolbarItem(placement: .primaryAction) {
                    Button(action: {
                        Task {
                            await modernTech.performHealthCheck()
                            await modernTech.refreshStats()
                        }
                    }) {
                        Image(systemName: "arrow.clockwise.circle.fill")
                    }
                }
            }
        }
        .task {
            await modernTech.initializeModernTechnologies()
        }
    }
    
    // MARK: - Header View
    
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text("Modern Technology Stack")
                    .font(.headline)
                    .fontWeight(.bold)
                
                HStack {
                    Circle()
                        .fill(modernTech.isInitialized ? .green : .red)
                        .frame(width: 8, height: 8)
                    
                    Text(modernTech.isInitialized ? "Initialized" : "Initializing...")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Text("•")
                        .foregroundColor(.secondary)
                    
                    Text("\\(modernTech.availableFeatures.count)/8 Features")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            Spacer()
            
            // Feature indicators
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 4), spacing: 8) {
                ForEach(ModernTechBridge.ModernFeature.allCases, id: \\.self) { feature in
                    Image(systemName: feature.icon)
                        .foregroundColor(modernTech.availableFeatures.contains(feature) ? .green : .gray)
                        .font(.caption)
                }
            }
        }
        .padding()
        .background(Color(NSColor.controlBackgroundColor))
    }
    
    // MARK: - AI Generation View
    
    private var aiGenerationView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // AI Model Selection
                VStack(alignment: .leading, spacing: 12) {
                    Text("Select AI Model")
                        .font(.headline)
                    
                    LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                        ForEach(aiModels, id: \\.0) { model in
                            Button(action: {
                                selectedAIModel = model.0
                            }) {
                                HStack {
                                    Image(systemName: model.2)
                                        .foregroundColor(.accentColor)
                                    
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(model.1)
                                            .font(.caption)
                                            .fontWeight(.medium)
                                            .multilineTextAlignment(.leading)
                                        
                                        Text(model.0)
                                            .font(.caption2)
                                            .foregroundColor(.secondary)
                                    }
                                    
                                    Spacer()
                                    
                                    if selectedAIModel == model.0 {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundColor(.green)
                                    }
                                }
                                .padding(8)
                                .background(
                                    RoundedRectangle(cornerRadius: 8)
                                        .fill(selectedAIModel == model.0 ? Color.accentColor.opacity(0.1) : Color.clear)
                                        .stroke(selectedAIModel == model.0 ? Color.accentColor : Color.gray.opacity(0.3), lineWidth: 1)
                                )
                            }
                            .buttonStyle(PlainButtonStyle())
                        }
                    }
                }
                
                // Text Input
                VStack(alignment: .leading, spacing: 8) {
                    Text("Describe your 3D model")
                        .font(.headline)
                    
                    TextEditor(text: $generatePrompt)
                        .frame(minHeight: 80)
                        .padding(8)
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                        )
                        .disabled(isGenerating)
                    
                    Text("Example: Create a futuristic spaceship with sleek lines and advanced propulsion systems")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                
                // Advanced Options
                VStack(alignment: .leading, spacing: 12) {
                    Button(action: {
                        showingAdvancedOptions.toggle()
                    }) {
                        HStack {
                            Text("Advanced Options")
                                .font(.headline)
                            
                            Spacer()
                            
                            Image(systemName: showingAdvancedOptions ? "chevron.up" : "chevron.down")
                        }
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    if showingAdvancedOptions {
                        VStack(alignment: .leading, spacing: 12) {
                            Toggle("Create NFT", isOn: $enableNFT)
                            Toggle("Enable Real-time Collaboration", isOn: $enableCollaboration)
                            Toggle("Enable WebXR Preview", isOn: $enableWebXR)
                        }
                        .padding(.leading, 16)
                    }
                }
                
                // Generate Button
                Button(action: generateModel) {
                    HStack {
                        if isGenerating {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        
                        Text(isGenerating ? "Generating..." : "Generate 3D Model")
                            .fontWeight(.medium)
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(
                        RoundedRectangle(cornerRadius: 8)
                            .fill(generatePrompt.isEmpty || isGenerating ? Color.gray : Color.accentColor)
                    )
                    .foregroundColor(.white)
                }
                .disabled(generatePrompt.isEmpty || isGenerating)
                
                // Generation Result
                if let result = generationResult {
                    generationResultView(result)
                }
            }
            .padding()
        }
    }
    
    // MARK: - Features Overview View
    
    private var featuresOverviewView: some View {
        ScrollView {
            LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 16) {
                ForEach(ModernTechBridge.ModernFeature.allCases, id: \\.self) { feature in
                    FeatureCard(
                        feature: feature,
                        isAvailable: modernTech.availableFeatures.contains(feature)
                    )
                }
            }
            .padding()
        }
    }
    
    // MARK: - Performance Dashboard View
    
    private var performanceDashboardView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let stats = modernTech.currentStats {
                    // Performance Metrics
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Performance Metrics")
                            .font(.headline)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 2), spacing: 12) {
                            MetricCard(
                                title: "Requests/Second",
                                value: String(format: "%.1f", stats.performanceMetrics.requestsPerSecond),
                                icon: "speedometer",
                                color: .blue
                            )
                            
                            MetricCard(
                                title: "Response Time",
                                value: String(format: "%.2fs", stats.performanceMetrics.averageResponseTime),
                                icon: "timer",
                                color: .orange
                            )
                            
                            MetricCard(
                                title: "Cache Hit Rate",
                                value: String(format: "%.1f%%", stats.performanceMetrics.cacheHitRate * 100),
                                icon: "externaldrive.fill",
                                color: .green
                            )
                            
                            MetricCard(
                                title: "AI Success Rate",
                                value: String(format: "%.1f%%", stats.performanceMetrics.aiSuccessRate * 100),
                                icon: "brain.head.profile",
                                color: .purple
                            )
                        }
                    }
                    
                    // Component Status
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Component Status")
                            .font(.headline)
                        
                        HStack {
                            VStack {
                                Text("\\(stats.initializedComponents)")
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .foregroundColor(.green)
                                
                                Text("Initialized")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            VStack {
                                Text("\\(stats.totalComponents)")
                                    .font(.title)
                                    .fontWeight(.bold)
                                    .foregroundColor(.primary)
                                
                                Text("Total")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(
                            RoundedRectangle(cornerRadius: 8)
                                .fill(Color(NSColor.controlBackgroundColor))
                        )
                    }
                } else {
                    Text("Loading performance data...")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
            .padding()
        }
    }
    
    // MARK: - Technology Status View
    
    private var technologyStatusView: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                if let health = modernTech.healthStatus {
                    // Overall Health
                    HStack {
                        Circle()
                            .fill(health.overallHealthy ? .green : .red)
                            .frame(width: 16, height: 16)
                        
                        Text("Overall System Health")
                            .font(.headline)
                        
                        Spacer()
                        
                        Text(health.overallHealthy ? "Healthy" : "Issues Detected")
                            .foregroundColor(health.overallHealthy ? .green : .red)
                            .fontWeight(.medium)
                    }
                    
                    // Component Health
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Component Health")
                            .font(.headline)
                        
                        ForEach(Array(health.components.keys.sorted()), id: \\.self) { componentName in
                            if let componentHealth = health.components[componentName] {
                                HStack {
                                    Circle()
                                        .fill(componentHealth.healthy ? .green : .red)
                                        .frame(width: 8, height: 8)
                                    
                                    Text(componentName.capitalized)
                                        .font(.body)
                                    
                                    Spacer()
                                    
                                    Text(componentHealth.status.capitalized)
                                        .font(.caption)
                                        .foregroundColor(.secondary)
                                    
                                    if let error = componentHealth.error {
                                        Image(systemName: "exclamationmark.triangle")
                                            .foregroundColor(.red)
                                            .help(error)
                                    }
                                }
                                .padding(.vertical, 4)
                            }
                        }
                    }
                    
                    // Last Updated
                    Text("Last updated: \\(formatDate(health.timestamp))")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                } else {
                    Text("Performing health check...")
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
            .padding()
        }
    }
    
    // MARK: - Helper Views
    
    private func generationResultView(_ result: ModernTechBridge.ModelGenerationResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: result.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundColor(result.success ? .green : .red)
                
                Text("Generation Result")
                    .font(.headline)
                
                Spacer()
                
                if let processingTime = result.processingTime {
                    Text(String(format: "%.2fs", processingTime))
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }
            
            if result.success {
                if let meshData = result.meshData {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Mesh Information")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Vertices: \\(meshData.vertices)")
                                Text("Faces: \\(meshData.faces)")
                                Text("Format: \\(meshData.format)")
                            }
                            .font(.caption)
                            
                            Spacer()
                            
                            VStack(alignment: .trailing, spacing: 4) {
                                Text("Size: \\(String(format: "%.1f", meshData.fileSizeMB)) MB")
                                if let optimized = meshData.optimized, optimized {
                                    Text("✓ WebAssembly Optimized")
                                        .foregroundColor(.green)
                                }
                            }
                            .font(.caption)
                        }
                    }
                }
                
                if let technologies = result.technologiesUsed {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Technologies Used")
                            .font(.subheadline)
                            .fontWeight(.medium)
                        
                        LazyVGrid(columns: Array(repeating: GridItem(.flexible()), count: 3), spacing: 4) {
                            ForEach(technologies, id: \\.self) { tech in
                                Text(tech.capitalized)
                                    .font(.caption)
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(
                                        RoundedRectangle(cornerRadius: 4)
                                            .fill(Color.accentColor.opacity(0.2))
                                    )
                            }
                        }
                    }
                }
                
                // Additional info for special features
                if result.nftInfo != nil {
                    HStack {
                        Image(systemName: "bitcoinsign.circle.fill")
                            .foregroundColor(.orange)
                        Text("NFT Created")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
                
                if result.collaborationSession != nil {
                    HStack {
                        Image(systemName: "person.2.circle.fill")
                            .foregroundColor(.blue)
                        Text("Collaboration Session Active")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
                
                if result.webxrSession != nil {
                    HStack {
                        Image(systemName: "visionpro")
                            .foregroundColor(.purple)
                        Text("WebXR Session Ready")
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
            } else if let error = result.error {
                Text("Error: \\(error)")
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(NSColor.controlBackgroundColor))
                .stroke(result.success ? Color.green : Color.red, lineWidth: 1)
        )
    }
    
    // MARK: - Actions
    
    private func generateModel() {
        guard !generatePrompt.isEmpty else { return }
        
        isGenerating = true
        
        Task {
            let result = await modernTech.createModelWithModernPipeline(
                prompt: generatePrompt,
                inputType: "text",
                model: selectedAIModel,
                options: [
                    "quality": "detailed",
                    "complexity": "high"
                ],
                createNFT: enableNFT,
                enableCollaboration: enableCollaboration,
                enableWebXR: enableWebXR
            )
            
            await MainActor.run {
                generationResult = result
                isGenerating = false
            }
        }
    }
    
    private func formatDate(_ timestamp: String) -> String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: timestamp) {
            let displayFormatter = DateFormatter()
            displayFormatter.dateStyle = .none
            displayFormatter.timeStyle = .medium
            return displayFormatter.string(from: date)
        }
        return timestamp
    }
}

// MARK: - Supporting Views

struct FeatureCard: View {
    let feature: ModernTechBridge.ModernFeature
    let isAvailable: Bool
    
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: feature.icon)
                .font(.system(size: 32))
                .foregroundColor(isAvailable ? .accentColor : .gray)
            
            Text(feature.displayName)
                .font(.headline)
                .multilineTextAlignment(.center)
            
            Circle()
                .fill(isAvailable ? .green : .red)
                .frame(width: 8, height: 8)
        }
        .frame(maxWidth: .infinity)
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(Color(NSColor.controlBackgroundColor))
                .stroke(isAvailable ? Color.accentColor.opacity(0.3) : Color.gray.opacity(0.3), lineWidth: 1)
        )
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                
                Spacer()
                
                Text(value)
                    .font(.title2)
                    .fontWeight(.bold)
                    .foregroundColor(color)
            }
            
            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding()
        .background(
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(NSColor.controlBackgroundColor))
        )
    }
}

#Preview {
    ModernTechShowcaseView()
        .frame(width: 800, height: 600)
}