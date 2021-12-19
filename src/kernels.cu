#include <kernels.cuh>

__device__ float poly6(float r, float h) {
    if (r >= 0 && r <= h) {
        return (315.0 * powf(h*h-r*r, 3.0)) / (64.0 * M_PI * powf(h, 9.0));
    }
    return 0.0;
}

__device__ float3 spiky_grad(float3 r, float h) {
    float norm = norm3df(r.x, r.y, r.z);
    float3 normalized = make_float3(r.x / norm, r.y / norm, r.z / norm);
    if (norm > 0 && norm <= h) {
        float coeff = -(45.0 * powf(norm - h, 2.0)) / (M_PI * powf(h, 6.0));
        return make_float3(normalized.x*coeff, normalized.y*coeff, normalized.z*coeff); 
    } else {
        return make_float3(0.0, 0.0, 0.0); 
    }
}