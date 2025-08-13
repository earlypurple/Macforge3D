import SwiftUI
import UniformTypeIdentifiers

struct ImageTo3DView: View {
    // State for the path of the selected image.
    @State private var selectedImagePath: String?

    // State for the NSImage to display a preview.
    @State private var selectedImage: NSImage?

    // State for the selected model color.
    @State private var modelColor: Color = .blue

    // State to track the generation process.
    @State private var isGenerating: Bool = false

    // State to hold the path of the generated model.
    @State private var generatedModelPath: String?

    // State to show alerts to the user.
    @State private var showAlert: Bool = false
    @State private var alertMessage: String = ""

    var body: some View {
        Form {
            Section(header: Text("Input Image").font(.headline)) {
                VStack(alignment: .center, spacing: 16) {
                    if let image = selectedImage {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(height: 250)
                            .cornerRadius(10)
                            .overlay(
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.gray.opacity(0.4), lineWidth: 1)
                            )
                    } else {
                        ZStack {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color(NSColor.windowBackgroundColor))
                                .frame(height: 250)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [10]))
                                        .foregroundColor(.gray.opacity(0.5))
                                )

                            Text("Select an image to begin")
                                .font(.title2)
                                .foregroundColor(.secondary)
                        }
                    }

                    Button(action: selectImage) {
                        HStack {
                            Image(systemName: "photo.on.rectangle.angled")
                            Text(selectedImage == nil ? "Choose Image..." : "Change Image...")
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }

            Section(header: Text("Customization").font(.headline)) {
                ColorPicker("Model Color", selection: $modelColor)
            }

            // Generate Button
            HStack {
                Spacer()
                Button(action: triggerGeneration) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("Generate Model")
                    }
                    .font(.title3)
                }
                .disabled(selectedImagePath == nil || isGenerating)
                .controlSize(.large)
                Spacer()
            }
            .padding()


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
        .navigationTitle("Image to 3D")
        .alert(isPresented: $showAlert) {
            Alert(title: Text("Error"), message: Text(alertMessage), dismissButton: .default(Text("OK")))
        }
    }

    private func selectImage() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        // Allow common image file types
        panel.allowedContentTypes = [UTType.png, UTType.jpeg, UTType.bmp, UTType.tiff]

        if panel.runModal() == .OK {
            if let url = panel.url {
                self.selectedImagePath = url.path
                self.selectedImage = NSImage(contentsOf: url)
                // Reset previous generation
                self.generatedModelPath = nil
            }
        }
    }

    private func triggerGeneration() {
        guard let imagePath = selectedImagePath else {
            alertMessage = "Please select an image first."
            showAlert = true
            return
        }

        isGenerating = true
        generatedModelPath = nil

        Task {
            print("Starting generation for image: \(imagePath)")
            // We need to create ImageTo3DGenerator first.
            // For now, let's assume it exists.
            let result = await ImageTo3DGenerator.generate(imagePath: imagePath)

            await MainActor.run {
                if let path = result, !path.starts(with: "Error:") {
                    self.generatedModelPath = path
                } else {
                    self.alertMessage = result ?? "An unknown error occurred during generation."
                    self.showAlert = true
                }
                self.isGenerating = false
            }
        }
    }
}

struct ImageTo3DView_Previews: PreviewProvider {
    static var previews: some View {
        ImageTo3DView()
            .frame(width: 500)
    }
}
