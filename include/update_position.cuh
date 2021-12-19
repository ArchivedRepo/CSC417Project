#include <cuda_runtime.h>

__global__ void update_positions(
    float3* positions,
    float3* delta_positions,
    int N
);