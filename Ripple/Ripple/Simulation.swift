
import Foundation
import Metal
import simd
import Quartz

class Simulation {
    static let maxRipples = 16
    
    struct Ripple {
        var centerX: Float = 0
        var centerY: Float = 0
        var time: Float = 10_000 // far out, man
    }

    var device: MTLDevice
    var computePipelineState: MTLComputePipelineState!
    var gridSize: MTLSize

    var ripples: [Ripple]
    var lastCenterIndex = 0

    init?(device: MTLDevice, gridSize: MTLSize) {
        self.device = device
        self.gridSize = gridSize
        
        ripples = [Ripple](repeating: Ripple(), count: Simulation.maxRipples)

        do {
            computePipelineState = try makeComputePipelineState()
        } catch {
            return nil
        }
    }
    
    func makeComputePipelineState() throws -> MTLComputePipelineState {
        let library = device.makeDefaultLibrary()
        let rippleFunction = library?.makeFunction(name: "ripple")
        return try device.makeComputePipelineState(function: rippleFunction!)
    }
    
    func addRippleCenter(_ position: CGPoint) {
        ripples[lastCenterIndex].centerX = Float(position.x)
        ripples[lastCenterIndex].centerY = Float(position.y)
        ripples[lastCenterIndex].time = Float(CACurrentMediaTime())
        lastCenterIndex = (lastCenterIndex + 1) % Simulation.maxRipples
    }

    func writeVertexPositions(to buffer: MTLBuffer, gridDimensions: MTLSize, time: TimeInterval, commandBuffer:MTLCommandBuffer) {
        guard let computeCommandEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        let now = Float(time)
        
        let dataSize = MemoryLayout<float4>.stride * Simulation.maxRipples
        let data = UnsafeMutableRawPointer.allocate(byteCount: dataSize, alignment: MemoryLayout<Float>.stride)
        
        for i in 0..<Simulation.maxRipples {
            var ripple = float4(ripples[i].centerX, ripples[i].centerY, now, ripples[i].time)
            memcpy(data + i * MemoryLayout<float4>.stride, &ripple, MemoryLayout<float4>.stride)
        }

        computeCommandEncoder.setComputePipelineState(computePipelineState)
        computeCommandEncoder.setBuffer(buffer, offset: 0, index: 0)
        computeCommandEncoder.setBytes(data, length: dataSize, index: 1)
        let threadsPerGrid = gridSize
        let threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
        computeCommandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeCommandEncoder.endEncoding()
        
        data.deallocate()
    }
}
