import SwiftUI

struct AudioTo3DView: View {
    @State private var modelPath: String?
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    private let audioTo3DGenerator = AudioTo3D()

    var body: some View {
        VStack {
            Text("Audio-to-3D Generation")
                .font(.title)
                .padding()

            Button(action: generateModel) {
                Text("Select Audio File & Generate")
                    .padding()
                    .background(Color.green)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .disabled(isLoading)
            .padding()

            Text("Note: This is a placeholder. No actual audio file is used.")
                .font(.caption)
                .foregroundColor(.gray)


            if isLoading {
                ProgressView("Generating model from audio...")
                    .padding()
            }

            if let errorMessage = errorMessage {
                Text(errorMessage)
                    .foregroundColor(.red)
                    .padding()
            }

            // Display the 3D preview if a model path is available
            if let modelPath = modelPath {
                ThreeDPreviewView(filePath: modelPath)
                    .frame(height: 300)
                    .cornerRadius(8)
                    .padding()
                    .shadow(radius: 5)
            }

            Spacer()
        }
        .padding()
    }

    private func generateModel() {
        isLoading = true
        errorMessage = nil
        modelPath = nil

        // In a real app, we would use a file picker to get the audio path.
        // For this scaffold, we use a dummy path.
        let dummyAudioPath = "placeholder.mp3"

        audioTo3DGenerator.generateModel(from: dummyAudioPath) { result in
            isLoading = false
            switch result {
            case .success(let path):
                modelPath = path
            case .failure(let error):
                switch error {
                case .pythonScriptNotFound:
                    errorMessage = "Error: Python script not found."
                case .pythonFunctionNotFound:
                    errorMessage = "Error: Python function not found."
                case .modelGenerationFailed(let message):
                    errorMessage = "Error: Model generation failed: \(message)"
                }
            }
        }
    }
}

struct AudioTo3DView_Previews: PreviewProvider {
    static var previews: some View {
        AudioTo3DView()
    }
}
