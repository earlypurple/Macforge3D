import SwiftUI
import Combine

struct TextTo3DView: View {
    // State for the text prompt entered by the user.
    @State private var prompt: String = "a comfortable armchair, modern design"

    // State for the selected model color.
    @State private var modelColor: Color = .blue

    // State to track the generation process for disabling UI elements.
    @State private var isGenerating: Bool = false

    // State to hold the result from the generation script.
    @State private var generatedModelPath: String?

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

            Section(header: Text("Customization").font(.headline)) {
                ColorPicker("Model Color", selection: $modelColor)
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

            if let modelPath = generatedModelPath {
                let modelURL = URL(fileURLWithPath: modelPath)

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
    }

    private func triggerGeneration() {
        // Cancel any previously scheduled generation task.
        generationTask?.cancel()

        // Don't generate if the prompt is empty.
        let trimmedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedPrompt.isEmpty {
            isGenerating = false
            return
        }

        // Schedule a new generation task with a debounce.
        generationTask = Task {
            do {
                // Set generating state immediately for responsiveness
                await MainActor.run {
                    self.isGenerating = true
                }

                // Wait for 500ms. If the task is cancelled, it will throw.
                try await Task.sleep(for: .milliseconds(500))

                // If we are here, the user stopped typing. Start the real generation.
                print("Starting generation for prompt: \(prompt)")
                let result = await TextTo3DGenerator.generate(prompt: prompt)

                // Update the UI on the main thread.
                await MainActor.run {
                    self.generatedModelPath = result
                    self.isGenerating = false
                }
            } catch {
                // Task was cancelled.
                print("Generation task cancelled.")
                // The isGenerating state will be handled by the next task that starts.
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
