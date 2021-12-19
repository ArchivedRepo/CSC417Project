#include <cuda_runtime.h>


__global__ void compute_delta_position(
    float3* positions,
    float pho0,
    float h,
    float* lambdas,
    float3* delta_positions,
    int* cell_start,
    int* cell_end,
    int* grid_index,
    int* particle_index,
    float3* bot_left,
    float3* up_right,
    float cube_s,
    int N
);