#include <kernels.h>

__device__ float poly6(float r, float h) {
    if (r >= 0 && r <= h) {
        return (315.0 * powf(h*h-r*r, 3.0)) / (64.0 * M_PI * powf(h, 9.0));
    }
    return 0.0;
}

__device__ void spiky_grad(float3 *r, float h, float3 *grad) {
    float norm = normf(3, r);
    if (norm > 0 && norm <= h) {
        grad = r.normalized() * (-(45.0 * powf(norm - h, 2.0)) / (M_PI * powf(h, 6.0))); 
    } else {
        grad.x = 0.0;
        grad.y = 0.0;
        grad.z = 0.0;
    }
}