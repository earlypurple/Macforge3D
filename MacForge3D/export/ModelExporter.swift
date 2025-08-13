import Foundation
import PythonKit

// Assuming Model3D is a typealias for Shape3D for now.
typealias Model3D = Shape3D

extension Shape3D {
    func toDictionary() -> [String: [[String: [String: [Float]]]]] {
        let trianglesData = mesh.triangles.map { triangle -> [String: [String: [Float]]] in
            let v1Data = ["position": [triangle.v1.position.x, triangle.v1.position.y, triangle.v1.position.z], "normal": [triangle.normal.x, triangle.normal.y, triangle.normal.z]]
            let v2Data = ["position": [triangle.v2.position.x, triangle.v2.position.y, triangle.v2.position.z], "normal": [triangle.normal.x, triangle.normal.y, triangle.normal.z]]
            let v3Data = ["position": [triangle.v3.position.x, triangle.v3.position.y, triangle.v3.position.z], "normal": [triangle.normal.x, triangle.normal.y, triangle.normal.z]]
            return ["v1": v1Data, "v2": v2Data, "v3": v3Data]
        }
        return ["triangles": trianglesData]
    }
}


class ModelExporter {
    func export(model: Model3D, to url: URL, format: ModelFormat) -> Bool {
        // Exporter le modèle dans le format choisi
        switch format {
        case .obj:
            return exportOBJ(model, url)
        case .fbx:
            return exportFBX(model, url)
        case .gltf:
            return exportGLTF(model, url)
        case .stl:
            return exportSTL(model, url)
        }
    }

    private func exportOBJ(_ model: Model3D, _ url: URL) -> Bool {
        // Implémenter l'export OBJ
        return false
    }

    private func exportFBX(_ model: Model3D, _ url: URL) -> Bool {
        // Implémenter l'export FBX
        return false
    }

    private func exportGLTF(_ model: Model3D, _ url: URL) -> Bool {
        PythonManager.initialize()

        let meshData = model.toDictionary()

        do {
            let gltfExporter = Python.import("gltf_exporter")
            gltfExporter.export_to_gltf(meshData, url.path)
            return true
        } catch {
            print("Error exporting to GLTF: \(error)")
            return false
        }
    }

    private func exportSTL(_ model: Model3D, _ url: URL) -> Bool {
        // Implémenter l'export STL
        return false
    }
}
