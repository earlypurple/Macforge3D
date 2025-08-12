import SwiftUI

struct FigurineGeneratorView: View {
    // Enum for quality options to ensure type safety
    enum Quality: String, CaseIterable, Identifiable {
        case standard = "Standard"
        case ultraDetailed = "ultra_detailed"
        var id: Self { self }
    }

    // State for the user-entered prompt
    @State private var prompt: String = "a majestic lion"

    // State for the selected quality level
    @State private var selectedQuality: Quality = .standard

    // State to manage the generation process
    @State private var isGenerating: Bool = false
    @State private var generatedModelPath: String?

    var body: some View {
        Form {
            Section(header: Text("Figurine Description").font(.headline)) {
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

            Section(header: Text("Quality Level").font(.headline)) {
                Picker("Quality", selection: $selectedQuality) {
                    ForEach(Quality.allCases) { quality in
                        Text(quality.rawValue.capitalized).tag(quality)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .help("Ultra-detailed takes longer but produces a higher quality mesh.")
            }

            Section {
                HStack {
                    Spacer()
                    Button(action: generateFigurine) {
                        HStack {
                            Image(systemName: "person.fill.viewfinder")
                            Text("Generate Figurine")
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
                    Text("Generating Figurine...")
                        .foregroundColor(.secondary)
                    Spacer()
                }
                .padding()
            }

            if let modelPath = generatedModelPath, let modelURL = URL(string: "file://\(modelPath)") {
                 if modelPath.starts(with: "Error:") {
                    Text(modelPath)
                        .foregroundColor(.red)
                } else {
                    Section(header: Text("Generated Figurine Preview").font(.headline)) {
                        ThreeDPreviewView(modelURL: modelURL)
                            .frame(height: 350)
                            .background(Color(NSColor.windowBackgroundColor))
                            .cornerRadius(10)
                    }
                }
            }
        }
        .padding()
        .navigationTitle("Figurine Generator")
    }

    private func generateFigurine() {
        print("Generate button tapped. Prompt: \(prompt), Quality: \(selectedQuality.rawValue)")
        isGenerating = true
        generatedModelPath = nil

        Task {
            let result = await FigurineGenerator.generate(prompt: prompt, quality: selectedQuality.rawValue)

            await MainActor.run {
                self.generatedModelPath = result
                self.isGenerating = false
            }
        }
    }
}

struct FigurineGeneratorView_Previews: PreviewProvider {
    static var previews: some View {
        FigurineGeneratorView()
            .frame(width: 500)
    }
}
