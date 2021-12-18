#include <cuda_runtime.h>

// r is the norm of the vector
__device__ float poly6(float r, float h);

__device__ void spiky_grad(float3 *r, float h, float3 *grad);