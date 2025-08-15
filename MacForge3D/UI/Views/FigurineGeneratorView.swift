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
            case .ultraRealistic: return "Ultra-Realistic (Image)"
            }
        }

        var id: Self { self }
    }

    // State for the user-entered prompt
    @State private var prompt: String = "a majestic lion"

    // State for the selected image
    @State private var selectedImagePath: String?
    @State private var isShowingFileImporter = false

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
            if selectedQuality == .ultraRealistic {
                Section(header: Text("Input Image").font(.headline)) {
                    HStack {
                        Button(action: {
                            isShowingFileImporter = true
                        }) {
                            HStack {
                                Image(systemName: "photo")
                                Text("Select Image")
                            }
                        }
                        .fileImporter(
                            isPresented: $isShowingFileImporter,
                            allowedContentTypes: [.image],
                            allowsMultipleSelection: false
                        ) { result in
                            do {
                                let fileURL = try result.get().first!
                                self.selectedImagePath = fileURL.path
                            } catch {
                                print("Error selecting file: \(error.localizedDescription)")
                                self.selectedImagePath = nil
                            }
                        }

                        if let selectedImagePath = selectedImagePath {
                            Text(URL(fileURLWithPath: selectedImagePath).lastPathComponent)
                                .font(.footnote)
                                .foregroundColor(.secondary)
                                .padding(.leading, 10)
                        } else {
                            Text("No image selected")
                                .font(.footnote)
                                .foregroundColor(.secondary)
                                .padding(.leading, 10)
                        }
                        Spacer()
                    }
                }
            } else {
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
            }

            Section(header: Text("Quality Level").font(.headline)) {
                Picker("Quality", selection: $selectedQuality) {
                    ForEach(Quality.allCases) { quality in
                        Text(quality.displayName).tag(quality)
                    }
                }
                .pickerStyle(SegmentedPickerStyle())
                .help("Detailed offers better quality than Standard. Ultra-Realistic uses an advanced image-to-3D model.")
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
                    .disabled(isGenerating || (selectedQuality != .ultraRealistic && prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) || (selectedQuality == .ultraRealistic && selectedImagePath == nil))
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
        print("Generate button tapped. Quality: \(selectedQuality.rawValue)")
        isGenerating = true
        generatedModelPath = nil

        Task {
            let result: String
            if selectedQuality == .ultraRealistic {
                guard let imagePath = selectedImagePath else {
                    result = "Error: No image selected for Ultra-Realistic mode."
                    await MainActor.run {
                        self.generatedModelPath = result
                        self.isGenerating = false
                    }
                    return
                }
                print("Calling backend for image-to-3D with image: \(imagePath)")
                // --- Backend call is commented out due to environment issues ---
                // result = await FigurineGenerator.generate(
                //     prompt: "", // Prompt not used for image-to-3D
                //     quality: selectedQuality.rawValue,
                //     imagePath: imagePath,
                //     addBase: false, // Not applicable
                //     refinePetit: false // Not applicable
                // )
                result = "Backend call disabled. Image: \(imagePath)"
            } else {
                print("Calling backend for text-to-3D with prompt: '\(prompt)'")
                // --- Backend call is commented out due to environment issues ---
                // result = await FigurineGenerator.generate(
                //     prompt: prompt,
                //     quality: selectedQuality.rawValue,
                //     imagePath: nil,
                //     addBase: addBase,
                //     refinePetit: refinePetit
                // )
                 result = "Backend call disabled. \nPrompt: '\(prompt)'\nQuality: \(selectedQuality.rawValue)\nAdd Base: \(addBase)\nRefine: \(refinePetit)"
            }

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
