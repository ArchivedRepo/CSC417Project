#include <update_velocity.cuh>

__global__ void update_velocity(
    float3* positions,
    float3* new_positions,
    float3* velocity,
    float dt,
    int N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    velocity[i].x = (new_positions[i].x - positions[i].x) / dt;
    velocity[i].y = (new_positions[i].y - positions[i].y) / dt;
    velocity[i].z = (new_positions[i].z - positions[i].z) / dt;
}