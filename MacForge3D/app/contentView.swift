import SwiftUI

struct ContentView: View {
    // State to keep track of the selected panel in the sidebar.
    @State private var selection: Panel? = .figurineGenerator

    // Enum to represent the different navigation destinations.
    enum Panel: String, CaseIterable, Identifiable {
        case figurineGenerator = "Figurine Generator"
        case textTo3D = "Text to 3D"
        case audioTo3D = "Audio to 3D"
        case parametric = "Parametric"
        case simulation = "Simulation"
        case engraving = "3D Engraving"

        var id: String { self.rawValue }

        // Returns the appropriate system icon for each panel.
        var iconName: String {
            switch self {
            case .figurineGenerator:
                return "person.crop.square.fill"
            case .textTo3D:
                return "text.bubble.fill"
            case .audioTo3D:
                return "waveform.path.ecg"
            case .parametric:
                return "cube.transparent"
            case .simulation:
                return "flame.fill"
            case .engraving:
                return "square.and.pencil"
            }
        }
    }

    var body: some View {
        NavigationSplitView {
            // Sidebar with a list of navigation links.
            List(Panel.allCases, selection: $selection) { panel in
                NavigationLink(value: panel) {
                    Label(panel.rawValue, systemImage: panel.iconName)
                }
            }
            .navigationSplitViewColumnWidth(220)
            .listStyle(.sidebar)

        } detail: {
            // The detail view changes based on the sidebar selection.
            switch selection {
            case .figurineGenerator:
                FigurineGeneratorView()
            case .textTo3D:
                TextTo3DView()
            case .audioTo3D:
                AudioTo3DView()
            case .parametric:
                ParametricView()
            case .simulation:
                SimulationView()
            case .engraving:
                EngravingView()
            case .none:
                // A view to show when no selection is made.
                Text("Select a tool from the sidebar to begin.")
                    .font(.largeTitle)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .navigationTitle("MacForge3D")
    }
}

// Preview provider for Xcode Previews
struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
