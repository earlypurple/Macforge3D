import SwiftUI
import PythonKit

struct CloudStorageView: View {
    @State private var isUploading = false
    @State private var showingFilePicker = false
    @State private var selectedPath: String?
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    var body: some View {
        VStack {
            Text("Stockage Cloud")
                .font(.title)
                .padding()
            
            Group {
                Button(action: { showingFilePicker = true }) {
                    HStack {
                        Image(systemName: "icloud.and.arrow.up")
                        Text("Synchroniser un projet")
                    }
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .sheet(isPresented: $showingFilePicker) {
                    FilePickerView(selectedPath: $selectedPath) { path in
                        syncProject(path: path)
                    }
                }
                
                if isUploading {
                    ProgressView("Synchronisation en cours...")
                        .padding()
                }
            }
        }
        .alert("Synchronisation Cloud", isPresented: $showingAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }
    
    private func syncProject(path: String) {
        isUploading = true
        
        Task {
            do {
                try await CloudManager.shared.syncProject(localPath: path)
                alertMessage = "Projet synchronisé avec succès"
            } catch {
                alertMessage = "Erreur de synchronisation: \(error.localizedDescription)"
            }
            
            isUploading = false
            showingAlert = true
        }
    }
}

struct FilePickerView: View {
    @Binding var selectedPath: String?
    let onSelect: (String) -> Void
    @Environment(\.presentationMode) var presentationMode
    
    var body: some View {
        VStack {
            Text("Sélectionnez un dossier")
                .font(.headline)
                .padding()
            
            Button("Choisir") {
                let panel = NSOpenPanel()
                panel.canChooseFiles = false
                panel.canChooseDirectories = true
                panel.allowsMultipleSelection = false
                
                if panel.runModal() == .OK {
                    if let path = panel.url?.path {
                        selectedPath = path
                        onSelect(path)
                        presentationMode.wrappedValue.dismiss()
                    }
                }
            }
            .padding()
            
            Button("Annuler") {
                presentationMode.wrappedValue.dismiss()
            }
            .padding()
        }
    }
}
