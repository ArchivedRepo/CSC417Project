#include <cuda_runtime.h>

__global__ void update_velocity(
    float3* positions,
    float3* new_positions,
    float3* velocity,
    float dt,
    int N
);