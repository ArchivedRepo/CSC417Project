#include <build_grid.cuh>
#include <iostream>
#include <assert.h>

static __device__ int compute_index(
    float3 position,
    float3 bot_left,
    int num_L, int num_W, int num_H, float cube_s
) {
    float3 relative_pos = make_float3(
        position.x-bot_left.x,
        position.y-bot_left.y,
        position.z-bot_left.z
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
    float3 bot_left, float3 up_right, float cube_s
    ) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i >= N) {
        return;
    }

    int num_L = ceilf((up_right.x - bot_left.x) / cube_s);
    int num_W = ceilf((up_right.y - bot_left.y) / cube_s);
    int num_H = ceilf((up_right.z - bot_left.z) / cube_s);

    grid_index[i] = compute_index(positions[i], bot_left, num_L, num_W, num_H, cube_s);
    particle_index[i] = i;

}

void build_grid(
    float3* positions,
    int* result,
    float cube_s,
    float3 bot_left,
    float3 up_right,
    int* grid_index,
    int* particle_index,
    int N
) {
    dim3 grid_dim(64, 1, 1);
    dim3 thread_block(128, 1, 1);
    compute_grid_index<<<grid_dim, thread_block>>>(positions, grid_index, particle_index,
    N, bot_left, up_right, cube_s);
    
    // std::sort(grid_indices.begin(), grid_indices.end());
    // result.clear();

    // int cur_grid = 0;
    // for (int i = 0; i < N; i++) {
    //     std::tuple<int, int> this_tuple = grid_indices[i];
    //     int grid_index = std::get<0>(this_tuple);
    //     assert(grid_index >= 0);
    //     // grid index
    //     // cur_grid = 5
    //     // 0, 0, 0, 1, 4, 4, 5
    //     // 0, 1, 2, 3, 4, 5, 6
    //     // 3, 4, 4, 6, 7
    //     if (grid_index == cur_grid + 1) {
    //         result.push_back(i);
    //         cur_grid += 1;
    //     } else if (grid_index != cur_grid) {
    //         // grid_index > cur_grid+1
    //         while (grid_index != cur_grid + 1) {
    //             result.push_back(i);
    //             cur_grid += 1;
    //         }
    //         result.push_back(i);
    //         cur_grid += 1;
    //     }
    //     if (i == N - 1) {
    //         while (result.size() < num_L * num_W * num_H) {
    //             result.push_back(N);
    //         }
    //     }
    // }
}