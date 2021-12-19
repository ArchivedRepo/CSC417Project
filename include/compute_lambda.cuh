#include <cuda_runtime.h>

__global__ void compute_lambda(
    float3* positions,
    float pho0,
    float mass,
    float epsilon,
    float h,
    float* lambdas,
    int* cell_start,
    int* cell_end,
    int* grid_index,
    int* particle_index,
    float3* bot_left,
    float3* up_right,
    float cube_s,
    int N
);