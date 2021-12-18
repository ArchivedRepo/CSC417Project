#include <cuda_runtime.h>

__device__ void apply_boundry(
    float3* result,
    float3* velocity,
    const float3* &bottom_left,
    const float3* &top_right
);

// void update_positions(
//     Eigen::MatrixXd &positions,
//     Eigen::MatrixXd &delta_positions,
//     Eigen::MatrixXd &velocity,
//     Eigen::Vector3d &bottom_left,
//     Eigen::Vector3d &top_right
// );