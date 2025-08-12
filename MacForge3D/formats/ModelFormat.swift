import Foundation

enum ModelFormat {
    case obj
    case fbx
    case gltf
    case stl

    static func from(extension ext: String) -> ModelFormat? {
        switch ext.lowercased() {
        case "obj": return .obj
        case "fbx": return .fbx
        case "gltf", "glb": return .gltf
        case "stl": return .stl
        default: return nil
        }
    }
}
