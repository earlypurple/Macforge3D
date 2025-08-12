import Foundation

class STLExporter {

    enum ExportError: Error {
        case fileWriteError(String)
    }

    // Exports a Shape3D object to a binary STL file at the given URL.
    func export(shape: Shape3D, to url: URL) throws {
        let mesh = shape.mesh
        var data = Data()

        // 1. 80-byte header (can be empty)
        let header = [UInt8](repeating: 0, count: 80)
        data.append(contentsOf: header)

        // 2. 4-byte unsigned integer for the number of triangles
        let numTriangles = UInt32(mesh.triangles.count)
        data.append(withUnsafeBytes(of: numTriangles.littleEndian) { Data($0) })

        // 3. Data for each triangle
        for triangle in mesh.triangles {
            // Normal vector (3x 4-byte floats)
            data.append(floatToBytes(triangle.normal.x))
            data.append(floatToBytes(triangle.normal.y))
            data.append(floatToBytes(triangle.normal.z))

            // Vertex 1 (3x 4-byte floats)
            data.append(floatToBytes(triangle.v1.x))
            data.append(floatToBytes(triangle.v1.y))
            data.append(floatToBytes(triangle.v1.z))

            // Vertex 2 (3x 4-byte floats)
            data.append(floatToBytes(triangle.v2.x))
            data.append(floatToBytes(triangle.v2.y))
            data.append(floatToBytes(triangle.v2.z))

            // Vertex 3 (3x 4-byte floats)
            data.append(floatToBytes(triangle.v3.x))
            data.append(floatToBytes(triangle.v3.y))
            data.append(floatToBytes(triangle.v3.z))

            // 2-byte attribute byte count (usually 0)
            let attributeByteCount = UInt16(0)
            data.append(withUnsafeBytes(of: attributeByteCount.littleEndian) { Data($0) })
        }

        // Write the data to the file
        do {
            try data.write(to: url, options: .atomic)
        } catch {
            throw ExportError.fileWriteError("Failed to write STL data to \(url.path): \(error.localizedDescription)")
        }
    }

    // Helper function to convert a Float to a Data object (4 bytes in little-endian).
    private func floatToBytes(_ value: Float) -> Data {
        var v = value.littleEndian
        return withUnsafeBytes(of: &v) { Data($0) }
    }
}
