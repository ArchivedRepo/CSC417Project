#include <cuda_runtime.h>

__global__ void advect(
    float3* positions,
    float3* positions_star,
    float3* velocity,
    float3* accu, //vector
    float3* bottom_left, // vector
    float3* top_right, //vector
    double dt,
    int N
);