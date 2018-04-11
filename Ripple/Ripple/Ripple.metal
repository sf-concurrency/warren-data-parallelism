
#include <metal_stdlib>
using namespace metal;

#define MAX_RIPPLES 16

constant float amplitude = 0.1;
constant float frequency = 5;
constant float decayRate = 0.25;
constant float packet = 15;
constant float speed = 2;

struct Ripple {
    float centerX;
    float centerZ;
    float startTime;
    float currentTime;
};

kernel void ripple(device packed_float3 *vertices [[buffer(0)]],
                   constant Ripple *ripples       [[buffer(1)]],
                   uint2 tpig                     [[thread_position_in_grid]],
                   uint2 gridSize                 [[threads_per_grid]])
{
    uint vid = tpig.y * gridSize.x + tpig.x;
    float3 v = vertices[vid];
    
    float x = v.x;
    float y = 0;
    float z = v.z;
    
    for (int i = 0; i < MAX_RIPPLES; ++i) {
        float dx = x - ripples[i].centerX;
        float dz = z - ripples[i].centerZ;
        float d = sqrt(dx * dx + dz * dz);
        float t = ripples[i].currentTime - ripples[i].startTime;
        float r = t * speed;
        float delta = r - d;
        y += amplitude * cos(2 * frequency * M_PI_F * delta) * exp(-decayRate * r * r) * exp(-packet * delta * delta);
    }
    vertices[vid][1] = y;
}
