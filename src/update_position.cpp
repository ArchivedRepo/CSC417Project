#include <update_position.h>
#include <algorithm>
#include <iostream>

#define BOUND_LIMIT 0.01

static void apply_boundry(
    Eigen::Vector3d &result,
    Eigen::Vector3d &velocity,
    const Eigen::Vector3d &bottom_left,
    const Eigen::Vector3d &top_right
){
    double x = std::max(std::min(result(0), top_right(0) - BOUND_LIMIT), bottom_left(0) + BOUND_LIMIT);
    double y = std::max(std::min(result(1), top_right(1) - BOUND_LIMIT), bottom_left(1) + BOUND_LIMIT);
    double z = std::max(std::min(result(2), top_right(2) - BOUND_LIMIT), bottom_left(2) + BOUND_LIMIT);
    if (x != result(0)){
        result(0) = x;
        velocity(0) = 0.0;
    }

    if (y != result(1)){
        result(1) = y;
        velocity(1) = 0.0;
    }

    if (z != result(2)){
        result(2) = z;
        velocity(2) = 0.0;
    }

}


void update_positions(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &delta_positions,
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right
) {
    positions = positions + delta_positions;
    for (int i = 0; i < positions.rows(); i++) {
        Eigen::Vector3d tmp = positions.row(i);
        Eigen::Vector3d tmp_v = positions.row(i);
        apply_boundry(tmp, tmp_v, bottom_left, top_right);
        positions.row(i) = tmp;
        velocity.row(i) = tmp_v;
    }
}