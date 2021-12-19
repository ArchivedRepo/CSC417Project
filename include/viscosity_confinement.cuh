#include <cuda_runtime.h>

__global__ void viscosity_confinement(
    float3* positions,
    float3* velocity,
    float h,
    int* cell_start,
    int* cell_end,
    int* grid_index,
    int* particle_index,
    float3* bot_left,
    float3* up_right,
    float cube_s,
    int N
);