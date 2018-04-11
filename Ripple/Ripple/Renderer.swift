
import Metal
import MetalKit
import simd

struct Uniforms {
    var projectionMatrix: float4x4
    var modelViewMatrix: float4x4
    var modelMatrix: float4x4
}

class Renderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLRenderPipelineState
    var depthState: MTLDepthStencilState
    var drawableSize: CGSize = CGSize(width: 1, height: 1)
    var wireframe = false

    init?(device: MTLDevice, vertexDescriptor: MTLVertexDescriptor) {
        self.device = device
        self.commandQueue = self.device.makeCommandQueue()!

        do {
            pipelineState = try Renderer.buildRenderPipelineWithDevice(device: device, vertexDescriptor: vertexDescriptor)
        } catch {
            print("Unable to compile render pipeline state.  Error info: \(error)")
            return nil
        }

        let depthStateDesciptor = MTLDepthStencilDescriptor()
        depthStateDesciptor.depthCompareFunction = MTLCompareFunction.less
        depthStateDesciptor.isDepthWriteEnabled = true
        self.depthState = device.makeDepthStencilState(descriptor:depthStateDesciptor)!
    }

    class func buildRenderPipelineWithDevice(device: MTLDevice, vertexDescriptor: MTLVertexDescriptor) throws -> MTLRenderPipelineState {
        let library = device.makeDefaultLibrary()

        let vertexFunction = library?.makeFunction(name: "vertexShader")
        let fragmentFunction = library?.makeFunction(name: "fragmentShader")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "RenderPipeline"
        pipelineDescriptor.sampleCount = 1
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.vertexDescriptor = vertexDescriptor

        pipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        pipelineDescriptor.depthAttachmentPixelFormat = .depth32Float_stencil8
        pipelineDescriptor.stencilAttachmentPixelFormat = .depth32Float_stencil8

        return try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
    }

    private func currentUniforms() -> Uniforms {
        let rotation = Float(CACurrentMediaTime() * 0.1)
        
        let aspect = Float(drawableSize.width / drawableSize.height)
        let projectionMatrix = matrix_perspective_right_hand(fovyRadians: radians_from_degrees(65), aspectRatio:aspect, nearZ: 0.1, farZ: 100.0)

        let xAxis = float3(1, 0, 0)
        let yAxis = float3(0, 1, 0)
        let modelMatrix = matrix4x4_rotation(radians: 0.66, axis: xAxis) * matrix4x4_rotation(radians: rotation, axis: yAxis)
        let viewMatrix = matrix4x4_translation(0.0, 0.0, -2.0)

        return Uniforms(projectionMatrix: projectionMatrix, modelViewMatrix: simd_mul(viewMatrix, modelMatrix), modelMatrix: modelMatrix)
    }

    func draw(mesh: MTKMesh, texture: MTLTexture, commandEncoder: MTLRenderCommandEncoder) {
        commandEncoder.setCullMode(.back)
        commandEncoder.setFrontFacing(.counterClockwise)
        commandEncoder.setRenderPipelineState(pipelineState)
        commandEncoder.setDepthStencilState(depthState)
        commandEncoder.setTriangleFillMode(wireframe ? .lines : .fill)
        
        var uniforms = currentUniforms()
        
        commandEncoder.setVertexBytes(&uniforms, length:MemoryLayout<Uniforms>.stride, index: 2)
        commandEncoder.setFragmentBytes(&uniforms, length:MemoryLayout<Uniforms>.stride, index: 2)
        
        for (index, element) in mesh.vertexDescriptor.layouts.enumerated() {
            guard let layout = element as? MDLVertexBufferLayout else {
                return
            }
            
            if layout.stride != 0 {
                let buffer = mesh.vertexBuffers[index]
                commandEncoder.setVertexBuffer(buffer.buffer, offset:buffer.offset, index: index)
            }
        }
        
        commandEncoder.setFragmentTexture(texture, index: 0)
        
        for submesh in mesh.submeshes {
            commandEncoder.drawIndexedPrimitives(type: submesh.primitiveType,
                                                indexCount: submesh.indexCount,
                                                indexType: submesh.indexType,
                                                indexBuffer: submesh.indexBuffer.buffer,
                                                indexBufferOffset: submesh.indexBuffer.offset)
            
        }
    }
}

func matrix4x4_rotation(radians: Float, axis: float3) -> matrix_float4x4 {
    let unitAxis = normalize(axis)
    let ct = cosf(radians)
    let st = sinf(radians)
    let ci = 1 - ct
    let x = unitAxis.x, y = unitAxis.y, z = unitAxis.z
    return matrix_float4x4.init(columns:(vector_float4(    ct + x * x * ci, y * x * ci + z * st, z * x * ci - y * st, 0),
                                         vector_float4(x * y * ci - z * st,     ct + y * y * ci, z * y * ci + x * st, 0),
                                         vector_float4(x * z * ci + y * st, y * z * ci - x * st,     ct + z * z * ci, 0),
                                         vector_float4(                  0,                   0,                   0, 1)))
}

func matrix4x4_translation(_ translationX: Float, _ translationY: Float, _ translationZ: Float) -> matrix_float4x4 {
    return matrix_float4x4.init(columns:(vector_float4(1, 0, 0, 0),
                                         vector_float4(0, 1, 0, 0),
                                         vector_float4(0, 0, 1, 0),
                                         vector_float4(translationX, translationY, translationZ, 1)))
}

func matrix_perspective_right_hand(fovyRadians fovy: Float, aspectRatio: Float, nearZ: Float, farZ: Float) -> matrix_float4x4 {
    let ys = 1 / tanf(fovy * 0.5)
    let xs = ys / aspectRatio
    let zs = farZ / (nearZ - farZ)
    return matrix_float4x4.init(columns:(vector_float4(xs,  0, 0,   0),
                                         vector_float4( 0, ys, 0,   0),
                                         vector_float4( 0,  0, zs, -1),
                                         vector_float4( 0,  0, zs * nearZ, 0)))
}

func radians_from_degrees(_ degrees: Float) -> Float {
    return (degrees / 180) * .pi
}
