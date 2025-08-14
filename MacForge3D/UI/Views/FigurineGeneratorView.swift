import SwiftUI

struct FigurineGeneratorView: View {
    // Enum for quality options to ensure type safety
    enum Quality: String, CaseIterable, Identifiable {
        case petit = "petit"
        case standard = "standard"
        case detailed = "detailed"
        case ultraRealistic = "ultra_realistic"

        var displayName: String {
            switch self {
            case .petit: return "Petit (<= 25mm)"
            case .standard: return "Standard"
            case .detailed: return "Detailed"
            case .ultraRealistic: return "Ultra-Realistic"
            }
        }

        var id: Self { self }
    }

    // State for the user-entered prompt
    @State private var prompt: String = "a majestic lion"

    // State for the selected quality level
    @State private var selectedQuality: Quality = .detailed

    // States for the new "petit" quality options
    @State private var addBase: Bool = false
    @State private var refinePetit: Bool = false

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
                        Text(quality.displayName).tag(quality)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .help("Detailed offers better quality than Standard. Ultra-Realistic uses a more advanced model for the best results but is significantly slower.")
            }

            // New section for "petit" model options
            if selectedQuality == .petit {
                Section(header: Text("Options for Petit Model").font(.headline)) {
                    Toggle("Add Base to Figurine", isOn: $addBase)
                        .help("Adds a cylindrical base to the figurine for better stability.")
                    Toggle("Improve Quality (Refine Mesh)", isOn: $refinePetit)
                        .help("Applies a smoothing filter to improve surface quality. Slightly increases generation time.")
                }
                .transition(.opacity.animation(.easeInOut))
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
        print("Generate button tapped. Prompt: \(prompt), Quality: \(selectedQuality.rawValue), Add Base: \(addBase), Refine Petit: \(refinePetit)")
        isGenerating = true
        generatedModelPath = nil

        Task {
            // --- Backend call is commented out due to environment issues ---
            // let result = await FigurineGenerator.generate(
            //     prompt: prompt,
            //     quality: selectedQuality.rawValue,
            //     addBase: addBase,
            //     refinePetit: refinePetit
            // )

            // --- Placeholder for UI testing ---
            let result = "Backend call disabled. \nPrompt: '\(prompt)'\nQuality: \(selectedQuality.rawValue)\nAdd Base: \(addBase)\nRefine: \(refinePetit)"


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
