
import Foundation
import Metal
import Accelerate

protocol Matrix {
    var columns: Int { get }
    var rows: Int { get }
    func elementAt(column: Int, row: Int) -> Float
}

class CPUMatrix : Matrix {
    let elements: UnsafeMutablePointer<Float>
    let rows: Int
    let columns: Int
    
    init(columns: Int, rows: Int) {
        self.columns = columns
        self.rows = rows
        self.elements = UnsafeMutablePointer<Float>.allocate(capacity: columns * rows)
    }
    
    deinit {
        elements.deallocate()
    }
    
    func randomize() {
        for i in 0..<(columns * rows) {
            elements[i] = Float(drand48())
        }
    }
    
    func fill(value: Float) {
        for i in 0..<(columns * rows) {
            elements[i] = value
        }
    }
    
    func elementAt(column: Int, row: Int) -> Float {
        return elements[column * rows + row]
    }
    
    func setElementAt(column: Int, row: Int, value: Float) {
        elements[column * rows + row] = value
    }
    
    func becomeProduct(_ A: CPUMatrix, _ B: CPUMatrix) {
        assert(columns == A.rows)
        assert(rows == B.columns)
        assert(A.columns == B.rows)
        
        for j in 0..<rows {
            for i in 0..<columns {
                var sum: Float = 0
                for k in 0..<A.columns {
                    let a = A.elementAt(column: k, row: j)
                    let b = B.elementAt(column: i, row: k)
                    sum += a * b
                }
                setElementAt(column: i, row: j, value: sum)
            }
        }
    }
    
    func accelerateBecomeProduct(_ A: CPUMatrix, _ B: CPUMatrix) {
        cblas_sgemm(CblasColMajor,
                    CblasNoTrans, CblasNoTrans,
                    Int32(A.rows), Int32(B.columns), Int32(A.columns),
                    1.0,
                    A.elements, Int32(A.rows),
                    B.elements, Int32(B.rows),
                    1.0,
                    self.elements, Int32(self.rows))
    }
}

class GPUMatrix : Matrix {
    let elements: MTLBuffer!
    let cachedElementsPtr: UnsafeMutablePointer<Float>
    let columns: Int
    let rows: Int
    
    init(columns: Int, rows: Int, device: MTLDevice) {
        self.rows = rows
        self.columns = columns
        self.elements = device.makeBuffer(length: MemoryLayout<Float>.stride * columns * rows, options: .storageModeShared)
        cachedElementsPtr = self.elements.contents().assumingMemoryBound(to: Float.self)
    }
    
    func elementAt(column: Int, row: Int) -> Float {
        return cachedElementsPtr[column * rows + row]
    }
    
    func fill(value: Float) {
        for i in 0..<(columns * rows) {
            cachedElementsPtr[i] = value
        }
    }
    
    func fill(matrix: CPUMatrix) {
        memcpy(cachedElementsPtr, matrix.elements, MemoryLayout<Float>.stride * columns * rows)
    }
    
    func dispatchBecomeProduct(_ A: GPUMatrix, _ B: GPUMatrix, commandEncoder: MTLComputeCommandEncoder) {
        let threadsPerGrid = MTLSizeMake(OuterDim, OuterDim, 1)
        let threadsPerThreadgroup = MTLSizeMake(16, 16, 1)
        
        var dims: [UInt32] = [UInt32(A.columns), UInt32(columns), UInt32(rows)]
        
        commandEncoder.setBuffer(A.elements, offset: 0, index: 0)
        commandEncoder.setBuffer(B.elements, offset: 0, index: 1)
        commandEncoder.setBuffer(elements, offset: 0, index: 2)
        commandEncoder.setBytes(&dims, length: MemoryLayout<UInt32>.stride * 3, index: 3)
        commandEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
}
