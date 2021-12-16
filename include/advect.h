#include <EigenTypes.h>

void advect(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &positions_star,
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &accu,
    Eigen::Vector3d &bottom_left,
    Eigen::Vector3d &top_right,
    double dt
);