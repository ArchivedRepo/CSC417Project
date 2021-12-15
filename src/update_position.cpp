#include <update_position.h>
#include <algorithm>

#define BOUND_LIMIT 0.01

static void apply_boundry(
    Eigen::Vector3d &result,
    const Eigen::Vector3d &bottom_left,
    const Eigen::Vector3d &top_right
){
    result(0) = std::max(std::min(result(0), top_right(0) - BOUND_LIMIT), bottom_left(0) + BOUND_LIMIT);
    result(1) = std::max(std::min(result(1), top_right(1) - BOUND_LIMIT), bottom_left(1) + BOUND_LIMIT);
    result(2) = std::max(std::min(result(2), top_right(2) - BOUND_LIMIT), bottom_left(2) + BOUND_LIMIT);
}


void update_positions(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &delta_positions,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right
) {
    positions = positions + delta_positions;
    for (int i = 0; i < positions.rows(); i++) {
        Eigen::Vector3d tmp = positions.row(i);
        apply_boundry(tmp, bottom_left, top_right);
        positions.row(i) = tmp;
    }
}