#include <update_position.h>
// #include <algorithm>

#define BOUND_LIMIT 0.01

__device__ void apply_boundry(
    float3* result,
    float3* velocity,
    const float3* bottom_left,
    const float3* top_right
) {
    float x = max(min(result.x, top_right.x - BOUND_LIMIT), bottom_left.x + BOUND_LIMIT);
    float y = max(min(result.y, top_right.y - BOUND_LIMIT), bottom_left.y + BOUND_LIMIT);
    float z = max(min(result.z, top_right.z - BOUND_LIMIT), bottom_left.z + BOUND_LIMIT);

    if (x != result.x){
        result.x = x;
        velocity.x = 0.0;
    }

    if (y != result.y){
        result.y = y;
        velocity.y = 0.0;
    }

    if (z != result.z){
        result.z = z;
        velocity.z = 0.0;
    }
}


// void update_positions(
//     Eigen::MatrixXd &positions,
//     Eigen::MatrixXd &delta_positions,
//     Eigen::MatrixXd &velocity,
//     Eigen::Vector3d &bottom_left,
//     Eigen::Vector3d &top_right
// ) {
//     positions = positions + delta_positions;
//     // for (int i = 0; i < positions.rows(); i++) {
//     //     Eigen::Vector3d tmp = positions.row(i);
//     //     Eigen::Vector3d tmp_v = velocity.row(i);
//     //     apply_boundry(tmp, tmp_v, bottom_left, top_right);
//     //     positions.row(i) = tmp;
//     // }
// }