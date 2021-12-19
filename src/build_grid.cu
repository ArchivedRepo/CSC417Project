/**
The following code is based on:
Particle Simulation using CUDA. Simon Green
https://developer.download.nvidia.com/assets/cuda/files/particles.pdf
And used code from the corresponding source code from Nvidia cuda-samples:
https://github.com/NVIDIA/cuda-samples/blob/master/Samples/particles
*/
#include <build_grid.cuh>
#include <iostream>
#include <assert.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <vector>

static __device__ int compute_index(
    float3 position,
    float3 *bot_left,
    int num_L, int num_W, int num_H, float cube_s
) {
    float3 relative_pos = make_float3(
        position.x-bot_left->x,
        position.y-bot_left->y,
        position.z-bot_left->z
    );
    
    int3 int_pos;
    int_pos.x = (int)(relative_pos.x / cube_s);
    int_pos.y = (int)(relative_pos.y / cube_s);
    int_pos.z = (int)(relative_pos.z / cube_s);
    int h = int_pos.z * (num_W * num_L);
    int w = int_pos.y * num_L;
    return h + int_pos.x + w;
}


// static __device__ float3 index_to_xyh(int index, int L, int W, int H) {
//     int h = index / (L * W);
//     int plane_coor = index % (L * W);
//     int x = plane_coor % L;
//     int y = plane_coor / L;
//     return make_float3(x, y, h);
// }

// static __device__ int xyh_to_index(int x, int y, int h, int L, int W, int H) {
//     return h * (W * L) + y * L + x;
// }

static __global__ void compute_grid_index(
    float3* positions, int* grid_index, 
    int* particle_index, int N,
    float3* bot_left, float3* up_right, float cube_s
    ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    int num_L = ceilf((up_right->x - bot_left->x) / cube_s);
    int num_W = ceilf((up_right->y - bot_left->y) / cube_s);
    int num_H = ceilf((up_right->z - bot_left->z) / cube_s);

    grid_index[i] = compute_index(positions[i], bot_left, num_L, num_W, num_H, cube_s);
    particle_index[i] = i;

}

/*
The following code is adapthed from:
https://github.com/NVIDIA/cuda-samples/blob/11de19f00cd24e244d2f6869c64810d63aafb926/Samples/particles/particles_kernel_impl.cuh#L148
*/
static __global__ void compute_index_range(
    int *cellStart,          // output: cell start index
    int *cellEnd,            // output: cell end index
    int *gridParticleHash,   // input: sorted grid hashes
    int *gridParticleIndex,  // input: sorted particle indices
    int N
) {
    extern __shared__ uint sharedHash[];  // blockSize + 1 elements
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int hash;

    if (index < N) {
        hash = gridParticleHash[index];
        sharedHash[threadIdx.x + 1] = hash;

        if (index > 0 && threadIdx.x == 0) {
            sharedHash[0] = gridParticleHash[index - 1];
        }
    }

    __syncthreads();

    if (index < N) {
        if (index == 0 || hash != sharedHash[threadIdx.x]) {
            cellStart[hash] = index;
            if (index > 0) {
                cellEnd[sharedHash[threadIdx.x]] = index;
            }
        }

        if (index == N - 1) {
            cellEnd[hash] = index + 1;
        }
    }
}

void build_grid(
    float3* positions,
    int* result,
    float cube_s,
    float3* bot_left,
    float3* up_right,
    int* grid_index,
    int* particle_index,
    int* cell_start,
    int* cell_end,
    int N
) {
    dim3 grid_dim(64, 1, 1);
    dim3 thread_block(128, 1, 1);
    uint shared_mem_size = (128 + 1) * sizeof(int);
    compute_grid_index<<<grid_dim, thread_block>>>(positions, grid_index, particle_index,
    N, bot_left, up_right, cube_s);
    
    thrust::device_ptr<int> keys(grid_index);
    thrust::device_ptr<int> values(particle_index);
    thrust::sort_by_key(keys, keys+N, values);

    compute_index_range<<<grid_dim, thread_block, shared_mem_size>>>(cell_start, cell_end, grid_index, particle_index, N);
}