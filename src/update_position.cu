#include <update_position.cuh>

__global__ void update_positions(
    float3* positions,
    float3* delta_positions,
    int N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    positions[i].x = positions[i].x + delta_positions[i].x;
    positions[i].y = positions[i].y + delta_positions[i].y;
    positions[i].z = positions[i].z + delta_positions[i].z;
}