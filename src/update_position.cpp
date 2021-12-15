#include <update_position.h>
#include <algorithm>

#define BOUND_LIMIT 0.01

static void apply_boundry(
    Eigen::Vector3d &result
){
    result(0) = std::max(std::min(result(0), 3.0 - BOUND_LIMIT), 0.0 + BOUND_LIMIT);
    result(1) = std::max(std::min(result(1), 3.0 - BOUND_LIMIT), 0.0 + BOUND_LIMIT);
    result(2) = std::max(std::min(result(2), 3.0 - BOUND_LIMIT), 0.0 + BOUND_LIMIT);
}


void update_positions(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &delta_positions
) {
    positions = positions + delta_positions;
    for (int i = 0; i < positions.rows(); i++) {
        Eigen::Vector3d tmp = positions.row(i);
        apply_boundry(tmp);
        positions.row(i) = tmp;
    }
}