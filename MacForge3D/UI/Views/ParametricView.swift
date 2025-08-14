import SwiftUI

struct ParametricView: View {
    // MARK: - State

    // The currently selected shape type.
    @State private var selectedShapeType: ParametricShapeType = .cube

    // State for all possible parameters.
    @State private var cubeParams = CubeParameters()
    @State private var sphereParams = SphereParameters()
    @State private var cylinderParams = CylinderParameters()
    @State private var coneParams = ConeParameters()

    // The URL of the temporary model file for the 3D preview.
    @State private var previewModelURL: URL?

    var body: some View {
        HSplitView {
            // Left panel for controls
            Form {
                Section(header: Text("Shape Selection").font(.headline)) {
                    Picker("Shape Type", selection: $selectedShapeType) {
                        ForEach(ParametricShapeType.allCases) { shapeType in
                            Text(shapeType.rawValue).tag(shapeType)
                        }
                    }
                    .pickerStyle(SegmentedPickerStyle())
                }

                Section(header: Text("Parameters").font(.headline)) {
                    switch selectedShapeType {
                    case .cube:
                        cubeParameterView
                    case .sphere:
                        sphereParameterView
                    case .cylinder:
                        cylinderParameterView
                    case .cone:
                        coneParameterView
                    }
                }
            }
            .padding()
            .frame(minWidth: 250, idealWidth: 300, maxWidth: 400)

            // Right panel for the 3D preview
            VStack {
                if let url = previewModelURL {
                    ThreeDPreviewView(modelURL: url, modelColor: .blue)
                } else {
                    ZStack {
                        RoundedRectangle(cornerRadius: 10).fill(Color.black.opacity(0.2))
                        Text("Select parameters to generate a preview.")
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding()
        }
        .navigationTitle("Parametric Shapes")
        .onAppear(perform: generateShape)
        .onChange(of: selectedShapeType) { _ in generateShape() }
        .onChange(of: cubeParams) { _ in generateShape() }
        .onChange(of: sphereParams) { _ in generateShape() }
        .onChange(of: cylinderParams) { _ in generateShape() }
        .onChange(of: coneParams) { _ in generateShape() }
    }

    // MARK: - Parameter Subviews

    private var cubeParameterView: some View {
        HStack {
            Text("Size")
            Spacer()
            Slider(value: $cubeParams.size, in: 0.1...10.0)
            Text(String(format: "%.2f", cubeParams.size))
        }
    }

    private var sphereParameterView: some View {
        VStack {
            HStack {
                Text("Radius")
                Spacer()
                Slider(value: $sphereParams.radius, in: 0.1...5.0)
                Text(String(format: "%.2f", sphereParams.radius))
            }
            HStack {
                Text("Resolution")
                Spacer()
                Stepper("\(sphereParams.resolution)", value: $sphereParams.resolution, in: 3...64)
            }
        }
    }

    private var cylinderParameterView: some View {
        VStack {
            HStack {
                Text("Radius")
                Spacer()
                Slider(value: $cylinderParams.radius, in: 0.1...5.0)
                Text(String(format: "%.2f", cylinderParams.radius))
            }
            HStack {
                Text("Height")
                Spacer()
                Slider(value: $cylinderParams.height, in: 0.1...10.0)
                Text(String(format: "%.2f", cylinderParams.height))
            }
            HStack {
                Text("Resolution")
                Spacer()
                Stepper("\(cylinderParams.resolution)", value: $cylinderParams.resolution, in: 3...64)
            }
        }
    }

    private var coneParameterView: some View {
        VStack {
            HStack {
                Text("Radius")
                Spacer()
                Slider(value: $coneParams.radius, in: 0.1...5.0)
                Text(String(format: "%.2f", coneParams.radius))
            }
            HStack {
                Text("Height")
                Spacer()
                Slider(value: $coneParams.height, in: 0.1...10.0)
                Text(String(format: "%.2f", coneParams.height))
            }
            HStack {
                Text("Resolution")
                Spacer()
                Stepper("\(coneParams.resolution)", value: $coneParams.resolution, in: 3...64)
            }
        }
    }

    // MARK: - Logic

    private func generateShape() {
        let shapeDefinition: ParametricShape

        switch selectedShapeType {
        case .cube:
            shapeDefinition = .cube(parameters: cubeParams)
        case .sphere:
            shapeDefinition = .sphere(parameters: sphereParams)
        case .cylinder:
            shapeDefinition = .cylinder(parameters: cylinderParams)
        case .cone:
            shapeDefinition = .cone(parameters: coneParams)
        }

        let mesh = ParametricMeshFactory.generate(shape: shapeDefinition)
        let shape = Shape3D(name: selectedShapeType.rawValue, mesh: mesh, material: Material(name: "Default", color: .blue))

        // Export the mesh to a temporary file to be used by the previewer.
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent("parametric_preview.stl")
        let exporter = STLExporter()

        do {
            try exporter.export(shape: shape, to: tempURL)
            self.previewModelURL = tempURL
            print("✅ Generated and exported preview for \(selectedShapeType.rawValue) to \(tempURL.path)")
        } catch {
            print("❌ Failed to export preview mesh: \(error)")
            self.previewModelURL = nil
        }
    }
}

struct ParametricView_Previews: PreviewProvider {
    static var previews: some View {
        ParametricView()
    }
}
