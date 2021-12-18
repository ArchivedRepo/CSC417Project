#include <EigenTypes.h>
#include <cuda_runtime.h>

// int compute_index(
//     Eigen::Vector3d position,
//     Eigen::Vector3d &bot_left,
//     int L, int W, int H, double cube_s
// );

// void index_to_xyh(int index, int L, int W, int H, int &x, int &y, int &h);

// void xyh_to_index(int x, int y, int h, int L, int W, int H, int &index);

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
);