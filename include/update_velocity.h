#include <Eigen/Dense>
#include <Eigen/Core>

void update_velocity(
    Eigen::MatrixXd &positions,
    Eigen::MatrixXd &new_positions,
    double dt,
    Eigen::MatrixXd &velocity
);