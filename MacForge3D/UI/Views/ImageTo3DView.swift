import SwiftUI
import UniformTypeIdentifiers

struct ImageTo3DView: View {
    // State for the paths of the selected images.
    @State private var selectedImagePaths: [String] = []

    // State for the NSImages to display previews.
    @State private var selectedImages: [NSImage] = []

    // State for the selected model color.
    @State private var modelColor: Color = .blue

    // State to track the generation process.
    @State private var isGenerating: Bool = false

    // State to hold the path of the generated model.
    @State private var generatedModelPath: String?

    // State to show alerts to the user.
    @State private var showAlert: Bool = false
    @State private var alertMessage: String = ""

    // States for post-processing options
    @State private var shouldRepairMesh: Bool = true
    @State private var targetSize: String = "100"

    var body: some View {
        Form {
            Section(header: Text("Input Images for Photogrammetry").font(.headline)) {
                VStack(alignment: .center, spacing: 16) {
                    if !selectedImages.isEmpty {
                        ScrollView(.horizontal, showsIndicators: true) {
                            HStack(spacing: 10) {
                                ForEach(selectedImages, id: \.self) { image in
                                    Image(nsImage: image)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(height: 150)
                                        .cornerRadius(8)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 8)
                                                .stroke(Color.gray.opacity(0.5), lineWidth: 1)
                                        )
                                }
                            }
                            .padding(.horizontal)
                        }
                        .frame(height: 160)
                    } else {
                        ZStack {
                            RoundedRectangle(cornerRadius: 10)
                                .fill(Color(NSColor.windowBackgroundColor))
                                .frame(height: 150)
                                .overlay(
                                    RoundedRectangle(cornerRadius: 10)
                                        .strokeBorder(style: StrokeStyle(lineWidth: 2, dash: [10]))
                                        .foregroundColor(.gray.opacity(0.5))
                                )

                            VStack {
                                Text("Select multiple images of an object from different angles.")
                                    .font(.title2)
                                    .foregroundColor(.secondary)
                                Text("(Minimum 5 images recommended)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }

                    Button(action: selectImages) {
                        HStack {
                            Image(systemName: "photo.on.rectangle.angled")
                            Text(selectedImages.isEmpty ? "Choose Images..." : "Choose More Images...")
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }

            Section(header: Text("Customization").font(.headline)) {
                ColorPicker("Model Color", selection: $modelColor)
            }

            Section(header: Text("Post-Processing Options").font(.headline)) {
                Toggle("Make Printable (Repair Mesh)", isOn: $shouldRepairMesh)
                    .help("Automatically repairs the mesh to make it watertight and suitable for 3D printing. Highly recommended.")

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Target Size (mm)")
                        Spacer()
                        TextField("e.g., 100", text: $targetSize)
                           .frame(maxWidth: 120)
                           .multilineTextAlignment(.trailing)
                           .textFieldStyle(RoundedBorderTextFieldStyle())
                    }
                    Text("The model will be scaled uniformly to match this size on its longest side. Set to 0 to disable.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(.vertical, 4)
            }


            // Generate Button
            HStack {
                Spacer()
                Button(action: triggerGeneration) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("Generate Photogrammetry Model")
                    }
                    .font(.title3)
                }
                .disabled(selectedImagePaths.isEmpty || isGenerating)
                .controlSize(.large)
                Spacer()
            }
            .padding()


            if isGenerating {
                HStack {
                    Spacer()
                    ProgressView()
                        .progressViewStyle(CircularProgressViewStyle())
                    Text("Generating model from images...")
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
        .navigationTitle("Image to 3D (Photogrammetry)")
        .alert(isPresented: $showAlert) {
            Alert(title: Text("Error"), message: Text(alertMessage), dismissButton: .default(Text("OK")))
        }
    }

    private func selectImages() {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = true // Allow multiple files to be selected
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [UTType.png, UTType.jpeg, UTType.bmp, UTType.tiff]

        if panel.runModal() == .OK {
            // Clear previous selections
            self.selectedImagePaths = []
            self.selectedImages = []

            for url in panel.urls {
                if let image = NSImage(contentsOf: url) {
                    self.selectedImagePaths.append(url.path)
                    self.selectedImages.append(image)
                }
            }
            // Reset previous generation
            self.generatedModelPath = nil
        }
    }

    private func triggerGeneration() {
        guard !selectedImagePaths.isEmpty else {
            alertMessage = "Please select at least one image."
            showAlert = true
            return
        }

        isGenerating = true
        generatedModelPath = nil

        // Prepare parameters for the generation task
        let shouldRepair = shouldRepairMesh
        let size = Float(targetSize) ?? 0.0

        Task {
            print("Starting photogrammetry generation for \(selectedImagePaths.count) images.")
            print("Post-processing options: Repair=\(shouldRepair), TargetSize=\(size)mm")

            // The generate function will be updated to handle multiple paths
            let result = await ImageTo3DGenerator.generate(
                imagePaths: selectedImagePaths,
                repairMesh: shouldRepair,
                targetSize: size
            )

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
