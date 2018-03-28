
#include <metal_stdlib>
using namespace metal;

kernel void multiply_arrays(device float *inputA [[buffer(0)]],
                            device float *inputB [[buffer(1)]],
                            device float *output [[buffer(2)]],
                            uint tpig [[thread_position_in_grid]])
{
    output[tpig] = inputA[tpig] * inputB[tpig];
}
