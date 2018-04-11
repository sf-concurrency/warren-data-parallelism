
#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Uniforms {
    float4x4 projectionMatrix;
    float4x4 modelViewMatrix;
    float4x4 modelMatrix;
};

struct Vertex {
    float3 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float4 worldPosition;
    float2 texCoord;
};

vertex VertexOut vertexShader(Vertex in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(2)]])
{
    VertexOut out;

    float4 position = float4(in.position, 1.0);
    out.position = uniforms.projectionMatrix * uniforms.modelViewMatrix * position;
    out.worldPosition = uniforms.modelMatrix * position;
    out.texCoord = in.texCoord;

    return out;
}

fragment half4 fragmentShader(VertexOut in [[stage_in]],
                               constant Uniforms &uniforms [[buffer(2)]],
                               texture2d<half> colorMap [[texture(0)]])
{
    constexpr sampler colorSampler(mip_filter::linear, mag_filter::linear, min_filter::linear);

    half4 waterColor(0.5, 0.8, 1.0, 1);
    half turbidity = 0.2;

    // Cheesy refraction effect
    float3 N = normalize(cross(dfdx(in.worldPosition.xyz), dfdy(in.worldPosition.xyz)));
    float2 uv = in.texCoord.xy;
    uv.x += N.x * 0.01;
    uv.y += N.z * 0.01;

    half4 colorSample = colorMap.sample(colorSampler, uv);    
    return turbidity * waterColor + (1 - turbidity) * colorSample;
}
