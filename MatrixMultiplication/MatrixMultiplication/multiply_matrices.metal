
#include <metal_stdlib>
using namespace metal;

struct MatrixDims {
    uint inputAColumns;
    uint outputColumns;
    uint outputRows;
};

kernel void multiply_matrices(constant float *inputA    [[buffer(0)]],
                              constant float *inputB    [[buffer(1)]],
                              device float *output      [[buffer(2)]],
                              constant MatrixDims &dims [[buffer(3)]],
                              uint2 tpig                [[thread_position_in_grid]])
{
    uint i = tpig.x;
    uint j = tpig.y;
    float sum = 0;
    for (uint k = 0; k < dims.inputAColumns; ++k) {
        float a = inputA[k * dims.outputColumns + j];
        float b = inputB[i * dims.inputAColumns + k];
        sum = sum + (a * b);
    }
    output[i * dims.outputRows + j] = sum;
}
