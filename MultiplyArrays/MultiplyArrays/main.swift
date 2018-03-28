
import Foundation
import Metal

let ArraySize = 1024 * 1024

class MultiplyArrays {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var computePipeline: MTLComputePipelineState!
    
    var inputA = [Float](repeating: 0, count: ArraySize)
    var inputB = [Float](repeating: 0, count: ArraySize)
    var output = [Float](repeating: 0, count: ArraySize)
    
    var bufferA: MTLBuffer!
    var bufferB: MTLBuffer!
    var outputBuffer: MTLBuffer!

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
        for i in 0..<ArraySize {
            inputA[i] = Float(drand48())
            inputB[i] = Float(drand48())
            output[i] = 0
        }
        
        let arrayByteCount = MemoryLayout<Float>.stride * ArraySize
        
        bufferA = device.makeBuffer(length: arrayByteCount, options: .storageModeShared)
        bufferB = device.makeBuffer(length: arrayByteCount, options: .storageModeShared)
        outputBuffer = device.makeBuffer(length: arrayByteCount, options: .storageModeShared)

        memcpy(bufferA.contents(), inputA, arrayByteCount)
        memcpy(bufferB.contents(), inputB, arrayByteCount)
        memset(outputBuffer.contents(), 0, arrayByteCount)
    }
    
    func makeComputePipeline() -> MTLComputePipelineState? {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default Metal library; ensure there is a .metal file in the target")
        }
        
        let kernelName = "multiply_arrays"
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
        
        let threadExecutionWidth = computePipeline.threadExecutionWidth
        let threadsPerGrid = MTLSizeMake(ArraySize, 1, 1)
        let threadsPerThreadgroup = MTLSizeMake(min(threadExecutionWidth, ArraySize), 1, 1)
        let threadgroupsPerGrid = MTLSizeMake(threadsPerGrid.width / threadsPerThreadgroup.width, 1, 1)

        let start = Date()

        commandEncoder.setBuffer(bufferA, offset: 0, index: 0)
        commandEncoder.setBuffer(bufferB, offset: 0, index: 1)
        commandEncoder.setBuffer(outputBuffer, offset: 0, index: 2)
        commandEncoder.setComputePipelineState(computePipeline)
        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        commandEncoder.endEncoding()
        
        commandBuffer.commit()

        commandBuffer.waitUntilCompleted()
        let duration = Date().timeIntervalSince(start)
        
        print(String(format: "GPU execution time: %0.2fms", duration * 1000))
    }
    
    func runCPUMultiply() {
        let start = Date()
        for i in 0..<ArraySize {
            output[i] = inputA[i] * inputB[i]
        }
        let duration = Date().timeIntervalSince(start)
        print(String(format: "CPU execution time: %0.2fms", duration * 1000))
    }
    
    func verifyResults() {
        let gpuResults = outputBuffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<ArraySize {
            let cpuResult = output[i]
            let gpuResult = gpuResults[i]
            if (cpuResult != gpuResult) {
                fatalError("CPU and GPU results differ at index \(i)!")
            }
        }
        print("Verified CPU and GPU produced same result")
    }
}

if let multiplier = MultiplyArrays() {
    multiplier.runCPUMultiply()
    multiplier.runGPUMultiply()
    multiplier.verifyResults()
}
