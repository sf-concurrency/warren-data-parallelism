
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Uniforms {
    float4x4 projectionMatrix;
    float4x4 modelViewMatrix;
};

struct Vertex {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(2)]])
{
    VertexOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(2)]],
                               texture2d<half> colorMap [[texture(0)]])
{
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);
    half4 colorSample = colorMap.sample(colorSampler, in.texCoord.xy);
    return float4(colorSample);
}
