#include <EigenTypes.h>

void apply_boundry(
    Eigen::Vector3d &result,
    Eigen::Vector3d &velocity,
    const Eigen::Vector3d &bottom_left,
    const Eigen::Vector3d &top_right
);

void update_positions(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &delta_positions,
    Eigen::MatrixXd &velocity,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right
);