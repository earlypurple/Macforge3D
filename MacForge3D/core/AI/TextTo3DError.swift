import Foundation

enum TextTo3DError: Error, LocalizedError {
    case generationFailed(message: String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .generationFailed(let message):
            return "3D Model Generation Failed: \(message)"
        case .invalidResponse:
            return "Received an invalid response from the generation script."
        }
    }
}
