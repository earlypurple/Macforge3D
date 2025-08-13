import SwiftUI
import UniformTypeIdentifiers

struct EngravingView: View {
    // Text Generation State
    @State private var textToEngrave: String = "MacForge3D"
    @State private var fontSize: Double = 36.0
    @State private var engravingDepth: Double = 5.0
    @State private var selectedFont: String = "Arial.ttf" // More specific font name

    // Process State
    @State private var isGeneratingText: Bool = false
    @State private var isEngraving: Bool = false
    @State private var errorMessage: String?

    // Model Paths & URLs
    @State private var generatedTextModelURL: URL?
    @State private var targetModelURL: URL?
    @State private var engravedModelURL: URL?

    // File Importer State
    @State private var isShowingFileImporter = false

    private let availableFonts = ["Arial.ttf", "Helvetica.ttf", "Times New Roman.ttf", "Courier New.ttf", "Verdana.ttf"]

    var body: some View {
        Form {
            // --- Text Generation Section ---
            Section(header: Text("1. Generate 3D Text").font(.headline)) {
                TextEditor(text: $textToEngrave)
                    .frame(height: 60)
                Picker("Font", selection: $selectedFont) {
                    ForEach(availableFonts, id: \.self) { Text($0) }
                }
                .pickerStyle(.menu)
                Slider(value: $fontSize, in: 8...144, step: 1) { Text("Font Size") }
                Slider(value: $engravingDepth, in: 1...50, step: 1) { Text("Depth") }

                Button(action: generate3DText) {
                    Label("Generate Text Mesh", systemImage: "square.and.pencil")
                }
                .disabled(isGeneratingText || textToEngrave.isEmpty)

                if isGeneratingText { ProgressView() }
            }

            // --- Text Preview Section ---
            if let modelURL = generatedTextModelURL {
                Section {
                    ThreeDPreviewView(modelURL: modelURL)
                        .frame(height: 150)
                }
            }

            // --- Target Model Section ---
            Section(header: Text("2. Load Target Model").font(.headline)) {
                Button(action: { isShowingFileImporter = true }) {
                    Label("Load Model (.ply, .stl, etc.)", systemImage: "folder.badge.plus")
                }

                if let modelURL = targetModelURL {
                    ThreeDPreviewView(modelURL: modelURL)
                        .frame(height: 250)
                }
            }

            // --- Engrave Action Section ---
            Section(header: Text("3. Engrave").font(.headline)) {
                Button(action: performEngraving) {
                    Label("Engrave Text on Target", systemImage: "wand.and.stars")
                }
                .buttonStyle(.borderedProminent)
                .disabled(generatedTextModelURL == nil || targetModelURL == nil || isEngraving)

                if isEngraving { ProgressView() }
            }

            // --- Final Result Section ---
            if let modelURL = engravedModelURL {
                Section(header: Text("Result").font(.headline)) {
                    ThreeDPreviewView(modelURL: modelURL)
                        .frame(height: 300)
                }
            }

            // --- Error Message Display ---
            if let errorMessage = errorMessage {
                Section {
                    Text(errorMessage)
                        .foregroundColor(.red)
                }
            }
        }
        .padding()
        .navigationTitle("3D Engraving")
        .fileImporter(
            isPresented: $isShowingFileImporter,
            allowedContentTypes: [.item], // Allow all file types for simplicity, could restrict to 3D model types
            allowsMultipleSelection: false
        ) { result in
            do {
                guard let selectedFile: URL = try result.get().first else { return }
                // Ensure we have security-scoped access to the file
                if selectedFile.startAccessingSecurityScopedResource() {
                    self.targetModelURL = selectedFile
                }
            } catch {
                self.errorMessage = "Error loading file: \(error.localizedDescription)"
            }
        }
    }

    private func generate3DText() {
        isGeneratingText = true
        errorMessage = nil
        generatedTextModelURL = nil
        Task {
            let result = await TextToMeshConverter.generate(
                text: textToEngrave, font: selectedFont, fontSize: fontSize, depth: engravingDepth
            )
            await MainActor.run {
                if let path = result, !path.starts(with: "Error:") {
                    self.generatedTextModelURL = URL(fileURLWithPath: path)
                } else {
                    self.errorMessage = result ?? "Unknown text generation error."
                }
                isGeneratingText = false
            }
        }
    }

    private func performEngraving() {
        guard let textURL = generatedTextModelURL, let targetURL = targetModelURL else {
            self.errorMessage = "Both a text model and a target model must be available."
            return
        }

        isEngraving = true
        errorMessage = nil
        engravedModelURL = nil

        Task {
            let result = await MeshOperator.subtract(
                baseModelURL: targetURL,
                subtractionModelURL: textURL
            )

            await MainActor.run {
                if let path = result, !path.starts(with: "Error:") {
                    self.engravedModelURL = URL(fileURLWithPath: path)
                } else {
                    self.errorMessage = result ?? "Unknown engraving error."
                }
                isEngraving = false
                // It's important to stop accessing the security-scoped resource when done.
                targetURL.stopAccessingSecurityScopedResource()
            }
        }
    }
}

struct EngravingView_Previews: PreviewProvider {
    static var previews: some View {
        EngravingView()
            .frame(width: 500)
    }
}
