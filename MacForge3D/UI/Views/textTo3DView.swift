import SwiftUI

struct TextTo3DView: View {
    @State private var prompt: String = "a detailed dragon figurine"
    @State private var modelPath: String?
    @State private var isLoading: Bool = false
    @State private var errorMessage: String?

    private let textTo3DGenerator = TextTo3D()

    var body: some View {
        VStack {
            Text("Text-to-3D Generation")
                .font(.title)
                .padding()

            TextEditor(text: $prompt)
                .frame(height: 100)
                .cornerRadius(8)
                .padding()
                .shadow(radius: 5)

            Button(action: generateModel) {
                Text("Generate Model")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
            }
            .disabled(isLoading)
            .padding()

            if isLoading {
                ProgressView("Generating model...")
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

        textTo3DGenerator.generateModel(prompt: prompt) { result in
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

struct TextTo3DView_Previews: PreviewProvider {
    static var previews: some View {
        TextTo3DView()
    }
}
