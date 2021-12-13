#include <update_velocity.h>


void update_velocity(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &new_positions,
    double dt,
    Eigen::MatrixXd &velocity
) {
    velocity += (new_positions - positions) / dt;
}