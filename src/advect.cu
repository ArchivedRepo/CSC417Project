#include <advect.cuh>

#define BOUND_LIMIT 0.01

static __device__ void apply_boundry(
    float3* result,
    float3* velocity,
    float3* bottom_left,
    float3* top_right
) {
    float x = max(min(result->x, top_right->x - BOUND_LIMIT), bottom_left->x + BOUND_LIMIT);
    float y = max(min(result->y, top_right->y - BOUND_LIMIT), bottom_left->y + BOUND_LIMIT);
    float z = max(min(result->z, top_right->z - BOUND_LIMIT), bottom_left->z + BOUND_LIMIT);

    if (x != result->x){
        result->x = x;
        velocity->x = 0.0;
    }

    if (y != result->y){
        result->y = y;
        velocity->y = 0.0;
    }

    if (z != result->z){
        result->z = z;
        velocity->z = 0.0;
    }
}

__global__ void advect(
    float3* positions,
    float3* positions_star,
    float3* velocity,
    float3* accu, //vector
    float3* bottom_left, // vector
    float3* top_right, //vector
    double dt,
    int N
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    velocity[i].x += dt * accu->x;
    velocity[i].y += dt * accu->y;
    velocity[i].z += dt * accu->z;
    
    positions_star[i].x = positions[i].x + dt * velocity[i].x;
    positions_star[i].y = positions[i].y + dt * velocity[i].y;
    positions_star[i].y = positions[i].z + dt * velocity[i].z;

    float3* tmp = &positions_star[i];
    float3* tmp_v = &velocity[i];
    apply_boundry(tmp, tmp_v, bottom_left, top_right);
}