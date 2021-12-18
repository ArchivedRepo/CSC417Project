#include <update_position.cuh>
// #include <algorithm>

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