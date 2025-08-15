import SwiftUI
import Combine

// Enum for quality settings
enum GenerationQuality: String, CaseIterable, Identifiable {
    case fast = "Fast"
    case balanced = "Balanced"
    case high = "High Quality"
    var id: Self { self }
}

struct TextTo3DView: View {
    // State for the text prompt entered by the user.
    @State private var prompt: String = "a comfortable armchair, modern design"

    // State for the selected model color.
    @State private var modelColor: Color = .blue

    // State for the generation quality
    @State private var quality: GenerationQuality = .balanced

    // State to track the generation process for disabling UI elements.
    @State private var isGenerating: Bool = false

    // State to hold the result from the generation script.
    @State private var generatedModelURL: URL?

    // State for alert presentation
    @State private var showingAlert = false
    @State private var alertMessage = ""

    // Task for debounced generation.
    @State private var generationTask: Task<Void, Never>?

    var body: some View {
        Form {
            Section(header: Text("Model Description").font(.headline)) {
                TextEditor(text: $prompt)
                    .frame(height: 100)
                    .padding(4)
                    .background(Color(NSColor.textBackgroundColor))
                    .cornerRadius(8)
                    .overlay(
                        RoundedRectangle(cornerRadius: 8)
                            .stroke(Color.gray.opacity(0.3), lineWidth: 1)
                    )
                    .onChange(of: prompt) {
                        triggerGeneration()
                    }
            }

            Section(header: Text("Settings").font(.headline)) {
                ColorPicker("Model Color", selection: $modelColor)
                Picker("Quality", selection: $quality) {
                    ForEach(GenerationQuality.allCases) { quality in
                        Text(quality.rawValue).tag(quality)
                    }
                }
                .pickerStyle(.segmented)
            }

            if isGenerating {
                HStack {
                    Spacer()
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                    Text("Generating model...")
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding()
            }

            if let modelURL = generatedModelURL {
                Section(header: Text("Generated Model Preview").font(.headline)) {
                    ThreeDPreviewView(modelURL: modelURL, modelColor: modelColor)
                        .frame(height: 350)
                        .background(Color(NSColor.windowBackgroundColor))
                        .cornerRadius(10)
                }
            }
        }
        .padding()
        .navigationTitle("Text to 3D")
        .onAppear(perform: triggerGeneration) // Generate on first appearance
        .alert("Error", isPresented: $showingAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }

    private func triggerGeneration() {
        generationTask?.cancel()

        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPrompt.isEmpty {
            isGenerating = false
            return
        }

        generationTask = Task {
            do {
                await MainActor.run {
                    self.isGenerating = true
                    self.generatedModelURL = nil // Clear previous model
                }

                try await Task.sleep(for: .milliseconds(500))

                print("Starting generation for prompt: \(prompt)")
                let result = await TextTo3DGenerator.generate(prompt: prompt, quality: quality.rawValue)

                await MainActor.run {
                    switch result {
                    case .success(let url):
                        self.generatedModelURL = url
                    case .failure(let error):
                        self.alertMessage = error.localizedDescription
                        self.showingAlert = true
                    }
                    self.isGenerating = false
                }
            } catch {
                print("Generation task cancelled.")
            }
        }
    }
}

struct TextTo3DView_Previews: PreviewProvider {
    static var previews: some View {
        TextTo3DView()
            .frame(width: 500)
    }
}
