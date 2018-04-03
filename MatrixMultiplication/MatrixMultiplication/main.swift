
import Foundation
import Metal
import Accelerate

let OuterDim = 512
let InnerDim = 256

class MatrixMultiplier {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipeline: MTLComputePipelineState!
    
    var matrixA: CPUMatrix!
    var matrixB: CPUMatrix!
    var matrixC: CPUMatrix!
    
    var matrixD: GPUMatrix!
    var matrixE: GPUMatrix!
    var matrixF: GPUMatrix!

    init?() {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        guard let commandQueue = device.makeCommandQueue() else { return nil }
        
        print("Got Metal device: \(device.name)")
        
        self.device = device
        self.commandQueue = commandQueue
        
        makeResources()
        computePipeline = makeComputePipeline()
    }
    
    func makeResources() {
        matrixA = CPUMatrix(columns: InnerDim, rows: OuterDim)
        matrixA.randomize()
        matrixB = CPUMatrix(columns: OuterDim, rows: InnerDim)
        matrixB.randomize()
        matrixC = CPUMatrix(columns: OuterDim, rows: OuterDim)
        matrixC.fill(value: 0)
        
        matrixD = GPUMatrix(columns: InnerDim, rows: OuterDim, device: device)
        matrixD.fill(matrix: matrixA)
        matrixE = GPUMatrix(columns: OuterDim, rows: InnerDim, device: device)
        matrixE.fill(matrix: matrixB)
        matrixF = GPUMatrix(columns: OuterDim, rows: OuterDim, device: device)
        matrixF.fill(value: 0)
    }
    
    func makeComputePipeline() -> MTLComputePipelineState? {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default Metal library; ensure there is a .metal file in the target")
        }
        
        let kernelName = "multiply_matrices"
        guard let computeFunction = library.makeFunction(name: kernelName) else {
            fatalError("Could not find kernel function; ensure there is a function named \(kernelName) in the default library")
        }
        
        do {
            return try device.makeComputePipelineState(function: computeFunction)
        } catch {
            fatalError("Unable to create compute pipeline: \(error)")
        }
    }
    
    func runGPUMultiply() {
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            fatalError("Unable to create command buffer")
        }

        guard let commandEncoder = commandBuffer.makeComputeCommandEncoder() else {
            fatalError("Unable to create compute command encoder")
        }

        let start = Date()
        
        commandEncoder.setComputePipelineState(computePipeline)
        
        matrixF.dispatchBecomeProduct(matrixD, matrixE, commandEncoder: commandEncoder)

        commandEncoder.endEncoding()

        commandBuffer.commit()

        commandBuffer.waitUntilCompleted()
        let duration = Date().timeIntervalSince(start)

        print(String(format: "GPU execution time: %0.2fms", duration * 1000))
    }
    
    func runCPUMultiply() {
        let start = Date()
        
        matrixC.becomeProduct(matrixA, matrixB)

        let duration = Date().timeIntervalSince(start)
        print(String(format: "CPU execution time: %0.2fms", duration * 1000))
    }
    
    func verifyResults() {
        let tolerance: Float = 0.00005
        let gpuResults = matrixF.elements.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<(OuterDim * OuterDim) {
            let cpuResult = matrixC.elements[i]
            let gpuResult = gpuResults[i]
            if abs(cpuResult - gpuResult) > tolerance {
                fatalError("CPU and GPU results differ at index \(i)!")
            }
        }
        print("Verified CPU and GPU produced same result")
    }
}

if let multiplier = MatrixMultiplier() {
    multiplier.runCPUMultiply()
    multiplier.runGPUMultiply()
    multiplier.verifyResults()
}
