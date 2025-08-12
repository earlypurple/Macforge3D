import SwiftUI

struct TextTo3DView: View {
    // State for the text prompt entered by the user.
    @State private var prompt: String = "a comfortable armchair, modern design"

    // State to track the generation process for disabling UI elements.
    @State private var isGenerating: Bool = false

    // State to hold the result from the generation script.
    @State private var generatedModelPath: String?

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
            }

            Section {
                HStack {
                    Spacer()
                    Button(action: {
                        print("Generate button tapped. Prompt: \(prompt)")
                        isGenerating = true
                        generatedModelPath = nil // Clear previous result

                        // Run the generation in a background task
                        Task {
                            let result = await TextTo3DGenerator.generate(prompt: prompt)

                            // Update the UI on the main thread
                            await MainActor.run {
                                self.generatedModelPath = result
                                self.isGenerating = false
                            }
                        }
                    }) {
                        HStack {
                            Image(systemName: "sparkles")
                            Text("Generate Model")
                        }
                        .padding(.horizontal)
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(isGenerating || prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    Spacer()
                }
            }
            .padding(.top)

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
                // The path from Python is relative to the project root. We create a file URL from it.
                // Note: This assumes the app is run from the project root during development.
                let modelURL = URL(fileURLWithPath: modelPath)

                Section(header: Text("Generated Model Preview").font(.headline)) {
                    ThreeDPreviewView(modelURL: modelURL)
                        .frame(height: 350)
                        .background(Color(NSColor.windowBackgroundColor))
                        .cornerRadius(10)
                }
            }
        }
        .padding()
        .navigationTitle("Text to 3D")
    }
}

struct TextTo3DView_Previews: PreviewProvider {
    static var previews: some View {
        TextTo3DView()
            .frame(width: 500)
    }
}
