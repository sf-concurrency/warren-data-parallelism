
#include <metal_stdlib>
using namespace metal;

#define MAX_RIPPLES 16

constant float amplitude = 0.1;
constant float frequency = 5;
constant float speed = 0.5;
constant float wavePacket = 2;
constant float decayRate = 1;

struct SimulationData {
    float4 centersAndTimes[MAX_RIPPLES]; // center.x, center.y, startTime, now
};

kernel void ripple(device packed_float3 *vertices [[buffer(0)]],
                   constant SimulationData &data  [[buffer(1)]],
                   uint2 tpig                     [[thread_position_in_grid]],
                   uint2 gridSize                 [[threads_per_grid]])
{
    uint vid = tpig.y * gridSize.x + tpig.x;
    float3 v = vertices[vid];
    
    float x = v.x;
    float y = 0;
    float z = v.z;
    
    for (int i = 0; i < MAX_RIPPLES; ++i) {
        float dx = x - data.centersAndTimes[i][0];
        float dz = z - data.centersAndTimes[i][1];
        float d = sqrt(dx * dx + dz * dz);
        float elapsed = data.centersAndTimes[i][3] - data.centersAndTimes[i][2];
        float r = elapsed * speed;
        float delta = r - d;
        y += amplitude * exp(-decayRate * r * r) * exp(-wavePacket * delta * delta) * cos(frequency * M_PI_F * delta);
    }
    vertices[vid][1] = y;
}
