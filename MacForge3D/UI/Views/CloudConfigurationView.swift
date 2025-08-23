import SwiftUI
import PythonKit

struct CloudConfigurationView: View {
    @State private var provider = "aws"
    @State private var accessKey = ""
    @State private var secretKey = ""
    @State private var region = "us-east-1"
    @State private var bucket = ""
    @State private var prefix = "models"
    
    @State private var showingAlert = false
    @State private var alertMessage = ""
    
    private let providers = ["aws", "azure", "gcp"]
    private let regions = [
        "us-east-1", "us-east-2", "us-west-1", "us-west-2",
        "eu-west-1", "eu-west-2", "eu-central-1",
        "ap-northeast-1", "ap-southeast-1", "ap-southeast-2"
    ]
    
    var body: some View {
        Form {
            Section(header: Text("Configuration Cloud")) {
                Picker("Provider", selection: $provider) {
                    ForEach(providers, id: \.self) { provider in
                        Text(provider.uppercased())
                    }
                }
                
                TextField("Access Key", text: $accessKey)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                SecureField("Secret Key", text: $secretKey)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                Picker("Region", selection: $region) {
                    ForEach(regions, id: \.self) { region in
                        Text(region)
                    }
                }
                
                TextField("Bucket/Container", text: $bucket)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                
                TextField("Prefix", text: $prefix)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
            }
            
            Section {
                Button(action: configureCloud) {
                    Text("Enregistrer la configuration")
                }
            }
        }
        .padding()
        .alert("Configuration Cloud", isPresented: $showingAlert) {
            Button("OK", role: .cancel) { }
        } message: {
            Text(alertMessage)
        }
    }
    
    private func configureCloud() {
        do {
            let credentials = [
                "access_key": accessKey,
                "secret_key": secretKey
            ]
            
            CloudManager.shared.configure(
                provider: provider,
                credentials: credentials,
                region: region,
                bucket: bucket,
                prefix: prefix
            )
            
            alertMessage = "Configuration cloud enregistrée avec succès"
            showingAlert = true
            
        } catch {
            alertMessage = "Erreur lors de la configuration: \(error.localizedDescription)"
            showingAlert = true
        }
    }
}
