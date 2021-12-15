#include <advect.h>
#include <iostream>

#define BOUND_LIMIT 1e-3

void advect(
    Eigen::MatrixXd &velocity,
    Eigen::MatrixXd &p0,
    Eigen::MatrixXd &p1,
    Eigen::Vector3d &accu,
    float dt
) {
    // Eigen cannot auto broadcast!
    velocity = velocity + dt * Eigen::VectorXd::Ones(velocity.rows()) * accu.transpose();
    p1.resize(p0.rows(), p0.cols());
    p1 = p0 + dt * velocity;
}