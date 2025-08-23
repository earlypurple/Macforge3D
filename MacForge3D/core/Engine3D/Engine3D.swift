import Metal
import MetalKit

class Engine3D {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var renderPipelineState: MTLRenderPipelineState
    
    // Scene properties
    private var viewMatrix: matrix_float4x4
    private var projectionMatrix: matrix_float4x4
    
    init() throws {
        // Get the default Metal device
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw Engine3DError.metalDeviceNotFound
        }
        self.device = device
        
        // Create command queue
        guard let commandQueue = device.makeCommandQueue() else {
            throw Engine3DError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // Initialize matrices
        viewMatrix = matrix_identity_float4x4
        projectionMatrix = matrix_perspective_right_hand(fovyRadians: Float.pi / 3.0,
                                                       aspectRatio: 1.0,
                                                       nearZ: 0.1,
                                                       farZ: 100.0)
        
        // Create render pipeline state
        let library = try device.makeDefaultLibrary()
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.vertexFunction = library.makeFunction(name: "vertexShader")
        renderPipelineDescriptor.fragmentFunction = library.makeFunction(name: "fragmentShader")
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        
        renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
    }
    
    func render(in view: MTKView, mesh: Mesh) {
        guard let drawable = view.currentDrawable,
              let renderPassDescriptor = view.currentRenderPassDescriptor,
              let commandBuffer = commandQueue.makeCommandBuffer(),
              let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else {
            return
        }
        
        renderEncoder.setRenderPipelineState(renderPipelineState)
        renderEncoder.setVertexBuffer(mesh.vertexBuffer, offset: 0, index: 0)
        renderEncoder.drawIndexedPrimitives(type: .triangle,
                                          indexCount: mesh.indexCount,
                                          indexType: .uint32,
                                          indexBuffer: mesh.indexBuffer,
                                          indexBufferOffset: 0)
        
        renderEncoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

enum Engine3DError: Error {
    case metalDeviceNotFound
    case commandQueueCreationFailed
    case libraryCreationFailed
}

// Helper function for perspective matrix
func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    
    return matrix_float4x4.init(columns:(
        SIMD4<Float>(xs,  0, 0,   0),
        SIMD4<Float>(0, ys,  0,   0),
        SIMD4<Float>(0,  0, zs, -1),
        SIMD4<Float>(0,  0, zs * nearZ, 0)
    ))
}
