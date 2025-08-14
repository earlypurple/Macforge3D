import SwiftUI

/// Represents the material properties of a 3D object.
struct Material: Codable, Hashable {
    /// The name of the material (e.g., "Plastic", "Metal").
    var name: String

    /// The primary color of the material.
    /// We store the components to ensure Codable conformance.
    var color: CodableColor

    /// A unique identifier for the material.
    let id: UUID

    init(name: String, color: Color, id: UUID = UUID()) {
        self.name = name
        self.color = CodableColor(color: color)
        self.id = id
    }
}

/// A Codable-compliant wrapper for `SwiftUI.Color`.
///
/// SwiftUI's `Color` type is not directly `Codable`. This struct provides a bridge by storing the
/// RGBA components of a color, which can be easily serialized and deserialized.
struct CodableColor: Codable, Hashable {
    /// The red component of the color, in the range [0, 1].
    var red: Double
    /// The green component of the color, in the range [0, 1].
    var green: Double
    /// The blue component of the color, in the range [0, 1].
    var blue: Double
    /// The opacity of the color, in the range [0, 1].
    var opacity: Double

    /// Creates a `CodableColor` instance from a `SwiftUI.Color`.
    init(color: Color) {
        let nsColor = NSColor(color)
        self.red = Double(nsColor.redComponent)
        self.green = Double(nsColor.greenComponent)
        self.blue = Double(nsColor.blueComponent)
        self.opacity = Double(nsColor.alphaComponent)
    }

    /// The SwiftUI Color representation.
    var swiftUIColor: Color {
        Color(.sRGB, red: red, green: green, blue: blue, opacity: opacity)
    }
}
